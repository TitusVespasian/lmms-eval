import os
import torch
from typing import Dict, List, Tuple, Union, Optional
from transformers.utils.doc import add_start_docstrings
# The original transformer LogitsProcessor imports might not be needed if we re-implement for vllm compatibility
# But we keep them for the _calc_banned_ngram_tokens logic if we reuse it
from transformers.generation.logits_process import LogitsProcessor, _calc_banned_ngram_tokens
os.environ["VLLM_USE_V1"] = "0"
# Set multiprocessing method for vLLM
os.environ["VLLM_WORKER_MULTIPROCESS_METHOD"] = "spawn"

import asyncio
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from io import BytesIO
from multiprocessing import cpu_count

import numpy as np
from accelerate import Accelerator, DistributedType
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

NUM_SECONDS_TO_SLEEP = 5

try:
    from vllm import LLM, SamplingParams
except ImportError:
    vllm = None

# Custom LogitsProcessor adapted for vLLM
class EOSNGramLogitsProcessor:
    def __init__(self, ngram_size: int, eos_token_id: int):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        self.ngram_size = ngram_size
        self.eos_token_id = eos_token_id

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        # Compatibility wrapper for vLLM signature differences
        # To support both (input_ids, scores) and (prompt_tokens_ids, generated_tokens_ids, logits)
        return self._call_impl(*args, **kwargs)

    def _call_impl(self, *args, **kwargs) -> torch.Tensor:
        # vLLM signature adaptation
        if len(args) == 3:
             # New vLLM: (prompt_tokens_ids, generated_tokens_ids, logits)
             prompt_ids, generated_ids, scores = args
             input_ids = prompt_ids + generated_ids
        elif len(args) == 2:
             # Old vLLM / others: (input_ids, scores)
             input_ids, scores = args
        else:
             raise ValueError(f"Unexpected number of arguments for LogitsProcessor: {len(args)}")

        if not input_ids:
            return scores
            
        cur_len = len(input_ids)
        if cur_len < self.ngram_size:
            return scores

        # Convert input_ids to tensor for calc_banned_ngram_tokens compatibility
        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=scores.device)
        
        # Ensure scores has batch dim if it's 1D
        original_scores_shape = scores.shape
        if scores.dim() == 1:
            scores_processed = scores.unsqueeze(0)
        else:
            scores_processed = scores

        try:
            # num_batch_hypotheses is 1 because we process one sequence history
            banned_batch_tokens = _calc_banned_ngram_tokens(self.ngram_size, input_ids_tensor, 1, cur_len)
            banned_tokens = banned_batch_tokens[0]
        except Exception as e:
            # Fallback
            # eval_logger.warning(f"Failed to calc banned tokens: {e}")
            banned_tokens = []
        
        if banned_tokens:
            top_id = int(torch.argmax(scores_processed[0]).item())
            if top_id in banned_tokens:
                finfo_min = torch.finfo(scores.dtype).min
                scores_processed[0].fill_(finfo_min)
                scores_processed[0, self.eos_token_id] = 0.0 
        
        if len(original_scores_shape) == 1:
            return scores_processed.squeeze(0)
        return scores_processed


@register_model("vllm_eos_woinst")
class VLLM_EOS_WoInst(lmms):
    def __init__(
        self,
        model_version: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        batch_size: int = 1,
        max_frame_num: int = 32,
        threads: int = 16,
        trust_remote_code: Optional[bool] = True,
        chat_template: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.max_frame_num = max_frame_num
        self.threads = threads
        self.chat_template = chat_template

        # Convert keys passed as strings to dict
        for key, value in kwargs.items():
            if isinstance(value, str) and value.strip().startswith("{") and value.strip().endswith("}"):
                try:
                    kwargs[key] = json.loads(value)
                except json.JSONDecodeError:
                    eval_logger.warning(f"Failed to parse JSON-like string for argument '{key}': {value}")

        # mm_processor_kwargs logic
        if "mm_processor_kwargs" not in kwargs:
             kwargs["mm_processor_kwargs"] = {"min_pixels": 28*28, "max_pixels": 1280*28*28}

        self.client = LLM(
            model=self.model_version,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        
        # Get tokenizer for EOS token ID
        self.tokenizer = self.client.get_tokenizer()
        # Fallback if eos_token_id is not set in tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        if self.eos_token_id is None:
             # Try common Qwen EOS or warn
             # Qwen2-VL usually 151643 or 151645 (<|im_end|>)
             # User mentioned 151643 as example
             self.eos_token_id = 151643 

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device
        self.batch_size_per_gpu = int(batch_size)

    def encode_image(self, image: Union[Image.Image, str]):
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.copy().convert("RGB")
        return img
    
    # Function to encode the video
    def encode_video(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, self.max_frame_num, dtype=int)

        # Ensure the last frame is included
        if total_frame_num - 1 not in uniform_sampled_frames:
            uniform_sampled_frames = np.append(uniform_sampled_frames, total_frame_num - 1)

        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        pil_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            pil_frames.append(img)
            
        return pil_frames

    def flatten(self, input):
        new_list = []
        for i in input:
            if isinstance(i, list):
                for j in i:
                    new_list.append(j)
            else:
                new_list.append(i)
        return new_list

    def generate_until(self, requests) -> List[str]:
        # Clear GPU cache
        torch.cuda.empty_cache()
        if hasattr(self, "client") and hasattr(self.client, "llm_engine"):
             # vLLM might have its own cache clearing needs, but torch.cuda.empty_cache is standard
             pass

        res = []
        pbar = tqdm(total=len(requests), disable=(self._rank != 0), desc="Model Responding")

        batch_size = self.batch_size_per_gpu
        batched_requests = [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]
        
        for batch_requests in batched_requests:
            vllm_inputs = []
            
            # Determine generation args from the first request (assuming homogeneous batch)
            # or we process individually if args differ significantly?
            # vLLM generate generally supports list of inputs, but SamplingParams is usually one per batch 
            # or list of sampling params.
            # Let's extract gen_kwargs from first req for the batch settings
            
            # Common params
            gen_kwargs = batch_requests[0].arguments[1]            
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            # user specifically requested stop tokens
            stop_tokens = ["<|endoftext|>", "<|im_end|>"]
            
            # Instantiate Custom Logits Processor
            # Note: vLLM expects a list of logits processors
            
            params = {
                "temperature": gen_kwargs["temperature"],
                "max_tokens": gen_kwargs["max_new_tokens"],
                # Add other params if needed
            }
            
            # Custom EOS LogitsProcessor
            # Use self.eos_token_id we resolved in __init__
            logits_processor = EOSNGramLogitsProcessor(ngram_size=3, eos_token_id=self.eos_token_id)
            
            sampling_params = SamplingParams(
                stop=stop_tokens,
                logits_processors=[logits_processor],
                **params
            )

            for idx, req in enumerate(batch_requests):
                contexts, req_gen_kwargs, doc_to_visual, doc_id, task, split = req.arguments
                
                # Visual Processing
                visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
                imgs = []
                if None in visuals:
                    visuals = []
                else:
                    visuals = self.flatten(visuals)
                    
                    with ThreadPoolExecutor(max_workers=self.threads) as executor:
                        all_tasks = []
                        for visual in visuals:
                            if isinstance(visual, str):
                                if any(ext in visual for ext in [".mp4", ".avi", ".mov", ".flv", ".wmv"]):
                                    all_tasks.append(executor.submit(self.encode_video, visual))
                                else:
                                    all_tasks.append(executor.submit(self.encode_image, visual))
                            elif isinstance(visual, Image.Image):
                                all_tasks.append(executor.submit(self.encode_image, visual))

                        for t in all_tasks:
                            result = t.result()
                            if isinstance(result, list): # Video frames
                                imgs.extend(result)
                            else:
                                imgs.append(result)

                # Prompt Construction for Base Model
                # Manual placeholders: <|vision_start|><|image_pad|><|vision_end|>
                # If there are multiple images, Qwen2-VL usually expects one block per image.
                # Construct logic: for each image in imgs, add the block.
                
                if len(imgs) > 0:
                    image_placeholders = ""
                    for _ in imgs:
                        image_placeholders += "<|vision_start|><|image_pad|><|vision_end|>"
                    full_prompt = f"{image_placeholders}{contexts}"
                    
                    # Prepare vLLM input with multi_modal_data
                    # vLLM expects PIL images in multi_modal_data["image"]
                    
                    vllm_inputs.append({
                        "prompt": full_prompt,
                        "multi_modal_data": {"image": imgs}
                    })
                else:
                    # Text only
                    vllm_inputs.append({
                        "prompt": contexts
                    })

            # Generate
            outputs = self.client.generate(vllm_inputs, sampling_params=sampling_params)
            
            # Extract Text
            response_text = [o.outputs[0].text for o in outputs]
            
            res.extend(response_text)
            pbar.update(len(batch_requests))

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood not implemented")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
