import asyncio
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from io import BytesIO
from multiprocessing import cpu_count
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
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
    # 引入 vLLM 的 LogitsProcessor 支持
    from vllm.sampling_params import LogitsProcessor
except ImportError:
    vllm = None
    LogitsProcessor = object # Placeholder to avoid error if not installed


# ==========================================
# 1. 移植 EOSNGramLogitsProcessor
# ==========================================
# 从 transformers 源码中复用的辅助函数，计算 banned ngrams
def _calc_banned_ngram_tokens(ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int, cur_len: int) -> List[List[int]]:
    """Copied from transformers.generation.logits_process"""
    if cur_len + 1 < ngram_size:
        return [[] for _ in range(num_hypos)]

    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


class EOSNGramLogitsProcessor:
    """
    完全复刻你提供的 Qwen2_VL_EOS_woinst 里的实现。
    注意：vLLM 的 LogitsProcessor 签名是 __call__(self, prompt_tokens_ids, generated_tokens_ids) -> logits
    但实际修改 logits 时，我们需要操作的是最后一个维度的分数。
    """
    def __init__(self, ngram_size: int, eos_token_id: int):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        self.ngram_size = ngram_size
        self.eos_token_id = eos_token_id

    def __call__(self, prompt_tokens_ids: List[int], generated_tokens_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        # vLLM 传入的 logits 通常是 [vocab_size] 的一维 tensor (对于单个 sequence)
        # 或者是 [batch_size, vocab_size] (如果 vLLM 内部做了 batching，但这个接口通常是针对单个 seq 的回调)
        # vLLM 文档说明：Input logits is a tensor of shape (vocab_size,).
        
        # 为了复用 _calc_banned_ngram_tokens，我们需要构造类似 transformers 的 input_ids
        # input_ids = prompt + generated
        full_ids = prompt_tokens_ids + generated_tokens_ids
        input_ids_tensor = torch.tensor([full_ids], dtype=torch.long, device=logits.device) # [1, seq_len]
        
        # 因为是对单条处理，num_batch_hypotheses = 1
        num_batch_hypotheses = 1
        cur_len = input_ids_tensor.shape[-1]
        
        # 计算被禁止的 tokens
        # 注意：这里我们需要为了当前这一步生成做检查，所以我们需要看的是 input_ids
        # _calc_banned_ngram_tokens 会检查 input_ids 里的 ngrams
        # 这里逻辑稍有不同：transformers 是在生成这一步之前调用的。
        # _calc_banned_ngram_tokens 逻辑是：基于 prev_input_ids，找出如果下一个 token 是 X，会构成重复 ngram 的 X。
        
        banned_tokens = _calc_banned_ngram_tokens(self.ngram_size, input_ids_tensor, num_batch_hypotheses, cur_len)
        # banned_tokens 是一个 list of list，这里只有第 0 个
        banned_tokens_for_me = banned_tokens[0]

        if not banned_tokens_for_me:
            return logits

        # 核心逻辑复刻：
        # if top_id in banned_batch_tokens[i]:
        #    scores_processed[i].fill_(finfo_min)
        #    scores_processed[i, self.eos_token_id] = 0.0
        
        # 1. 找到当前分数最高的 token
        top_id = int(torch.argmax(logits).item())
        
        # 2. 如果最高分 token 在禁止列表里
        if top_id in banned_tokens_for_me:
            finfo_min = torch.finfo(logits.dtype).min
            logits.fill_(finfo_min)
            # 强制给 EOS 高分，确保被采样
            logits[self.eos_token_id] = 0.0 # 或者一个相对较大的值
            
        return logits


@register_model("vllm_eos_woinst")
class VLLM_EOS_WoInst(lmms):
    def __init__(
        self,
        model_version: str = "Qwen/Qwen2.5-VL-72B-Instruct",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        batch_size: int = 1,
        max_frame_num: int = 32,
        threads: int = 16,
        trust_remote_code: Optional[bool] = True,
        # 新增/修改参数
        ngram_size: int = 20, 
        use_chat_api: bool = False, # 默认为 False，适配 Base 模型
        chat_template: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.max_frame_num = max_frame_num
        self.threads = threads
        self.chat_template = chat_template
        self.use_chat_api = use_chat_api
        self.ngram_size = ngram_size

        # Convert args
        for key, value in kwargs.items():
            if isinstance(value, str) and value.strip().startswith("{") and value.strip().endswith("}"):
                try:
                    kwargs[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass

        # Set up vllm client
        self.client = LLM(
            model=self.model_version,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        
        # 获取 Tokenizer 用于拿到 EOS ID
        self.tokenizer = self.client.get_tokenizer()
        self.eos_token_id = self.tokenizer.eos_token_id
        # Qwen2-VL 的 EOS 有时是 <|im_end|> 或 <|endoftext|>，视配置而定
        if hasattr(self.client.llm_engine.model_config, "eos_token_id"):
             # 尝试从 vLLM 内部配置获取
             engine_eos = self.client.llm_engine.model_config.eos_token_id
             if engine_eos is not None:
                 self.eos_token_id = engine_eos

        # Accelerator setup (just for multi-process coordination in lmms-eval, actual compute is handles by vllm)
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device
        self.batch_size_per_gpu = int(batch_size)

    # Encode functions unchanged
    def encode_image(self, image: Union[Image.Image, str]):
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.copy()
        output_buffer = BytesIO()
        img.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str

    def encode_video(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, self.max_frame_num, dtype=int)
        if total_frame_num - 1 not in uniform_sampled_frames:
            uniform_sampled_frames = np.append(uniform_sampled_frames, total_frame_num - 1)
        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()
        base64_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            base64_frames.append(base64_str)
        return base64_frames

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self._rank != 0), desc="Model Responding")

        batch_size = self.batch_size_per_gpu
        batched_requests = [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]
        
        for batch_requests in batched_requests:
            
            # --- 构建 Logits Processor ---
            # 为当前 Batch 创建一个新的 Processor 实例 (虽然是无状态的，但保持逻辑清晰)
            # vLLM 要求 logits_processors 是一个 List[LogitsProcessor]
            # 注意：vLLM 的 logits_processors 是针对每个 Request 的，不是 Batch 的。
            # 下面我们会把他放到 sampling_params 里。
            
            ngram_processor = EOSNGramLogitsProcessor(ngram_size=self.ngram_size, eos_token_id=self.eos_token_id)
            
            # 预处理 Batch 数据
            batched_inputs = [] # 存放 (prompt, multi_modal_data) 或者 messages
            
            for idx in range(len(batch_requests)):
                contexts, gen_kwargs, doc_to_visual, doc_id, task, split = batch_requests[idx].arguments
                
                # --- 参数处理保持一致 ---
                if "max_new_tokens" not in gen_kwargs: gen_kwargs["max_new_tokens"] = 1024
                if gen_kwargs["max_new_tokens"] > 4096: gen_kwargs["max_new_tokens"] = 4096
                if "temperature" not in gen_kwargs: gen_kwargs["temperature"] = 0
                if "top_p" not in gen_kwargs: gen_kwargs["top_p"] = 0.95

                # 构建 SamplingParams
                # 关键点：把自定义的 logits_processors 塞进去
                sampling_params = SamplingParams(
                    temperature=gen_kwargs["temperature"],
                    max_tokens=gen_kwargs["max_new_tokens"],
                    top_p=gen_kwargs["top_p"],
                    logits_processors=[ngram_processor] # <--- 注入魔改逻辑
                )

                visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
                if None in visuals: visuals = []
                else: visuals = self.flatten(visuals)
                
                imgs = []
                with ThreadPoolExecutor(max_workers=self.threads) as executor:
                    futures = []
                    for visual in visuals:
                        if isinstance(visual, str):
                            if any(ext in visual for ext in [".mp4", ".avi", ".mov", ".flv", ".wmv"]):
                                futures.append(executor.submit(self.encode_video, visual))
                            else:
                                futures.append(executor.submit(self.encode_image, visual))
                        elif isinstance(visual, Image.Image):
                            futures.append(executor.submit(self.encode_image, visual))
                    for f in futures:
                        imgs.append(f.result())

                # --- 核心分支：Chat 模式 vs Base 补全模式 ---
                
                if self.use_chat_api:
                    # Chat 模式：构建 messages 列表
                    messages = [{"role": "user", "content": []}]
                    messages[0]["content"].append({"type": "text", "text": contexts})
                    for img in imgs:
                        messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})
                    
                    batched_inputs.append({
                        "type": "chat",
                        "messages": messages,
                        "sampling_params": sampling_params
                    })
                    
                else:
                    # Base (Woinst) 模式：
                    # 对于 Qwen2-VL Base，通常直接把 Prompt 和 Image 拼在一起
                    # 但在 vLLM API 里，如果不走 Chat Template，我们需要构建：
                    # prompt (str) + multi_modal_data (dict)
                    
                    # 1. 构建 Prompt 用于 Completion
                    # 注意：contexts 里可能包含 <image> 占位符。vLLM 可能会自动处理，也可能需要替换。
                    # Qwen2-VL 推荐的格式通常是: "<|vision_start|><|image_pad|><|vision_end|>..."
                    # 这里我们简单地把 contexts 当作纯文本 Prompt
                    
                    final_prompt = contexts
                    
                    # 2. 构建 multi_modal_data
                    # vLLM 对 Qwen2-VL 的 Multi-modal 格式要求是个 dict，key 是 "image" 或 "video"
                    # 这是一个难点：vLLM 的 generate 接口对多图支持比较挑剔。
                    # 如果有多张图，通常 key 是 "image": [img1, img2]
                    
                    multi_modal_data = {}
                    if len(imgs) > 0:
                        # 假设都是 Image，暂时没处理混合 Video/Image 的复杂情况
                        # 根据 vLLM 文档，通常是传入 PIL Image 或 base64
                        # 这里我们已经编码成 base64 string 了，需要看 vLLM 具体版本支持
                        # 为了保险，我们最好把 base64 反解回 PIL Image 给 vLLM (它内部会再处理)
                        # 或者直接尝试传 base64。为了稳妥，这里用简单的伪代码示意：
                        
                        # 还原回 PIL Image 列表 (这对 vLLM 最友好)
                        pil_images = []
                        for b64 in imgs:
                             pil_images.append(Image.open(BytesIO(base64.b64decode(b64))))

                        if len(pil_images) == 1:
                            multi_modal_data["image"] = pil_images[0]
                        else:
                            multi_modal_data["image"] = pil_images # List of images
                    
                    # Base 模式还需要处理 <image> 标签。
                    # 很多 benchmark 的 prompt 里有 <image>，直接喂给 Qwen Base 可能会有干扰
                    # 建议：去掉 <image> 标签，或者保留由 vLLM 的 Qwen2VL 插件处理
                    final_prompt = final_prompt.replace("<image>", "")
                    
                    batched_inputs.append({
                        "type": "generate",
                        "prompt": final_prompt,
                        "multi_modal_data": multi_modal_data,
                        "sampling_params": sampling_params
                    })

            # --- 执行生成 ---
            # 由于 vLLM 的 generate 和 chat 是两个接口，这里不能 batch 混用
            # 但我们在 batch 循环里，类型应该是一致的。
            
            texts_outputs = []
            
            if self.use_chat_api:
                # 批量调用 chat 接口比较麻烦，vLLM client 主要是单次调用
                # 我们这里简单循环调用（虽说是循环，但 vLLM 会在后台做 continuous batching，所以还是快的）
                for inp in batched_inputs:
                    req_msgs = inp["messages"]
                    req_params = inp["sampling_params"]
                    
                    if self.chat_template:
                        # 读取 template
                        with open(self.chat_template, "r") as f: tpl = f.read()
                        out = self.client.chat(messages=req_msgs, sampling_params=req_params, chat_template=tpl)
                    else:
                        out = self.client.chat(messages=req_msgs, sampling_params=req_params)
                    
                    texts_outputs.append(out[0].outputs[0].text)
            else:
                # Generate 接口
                for inp in batched_inputs:
                    req_prompt = inp["prompt"]
                    req_mm = inp["multi_modal_data"]
                    req_params = inp["sampling_params"]
                    
                    # 调用 generate
                    if req_mm:
                        out = self.client.generate(prompt=req_prompt, multi_modal_data=req_mm, sampling_params=req_params)
                    else:
                        out = self.client.generate(prompt=req_prompt, sampling_params=req_params)
                    
                    texts_outputs.append(out[0].outputs[0].text)

            res.extend(texts_outputs)
            pbar.update(len(batch_requests))

        pbar.close()
        return res

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        assert False, "Not supported"

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO")
