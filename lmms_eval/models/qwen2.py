from __future__ import annotations

from typing import List

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

@register_model("qwen2")
class Qwen2(lmms):
    """
    Qwen2 文本模型（CausalLM）。默认加载指令微调权重，例如："Qwen/Qwen2-7B-Instruct"。
    仅支持纯文本推理，不处理图像 / 视频 / 音频。
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2-7B-Instruct",
        device: str = "cuda",
        batch_size: int | str | None = 1,
        attn_implementation: str | None = None,
        trust_remote_code: bool = True,
        use_cache: bool = True,
        system_prompt: str = "You are a helpful assistant.",
        **kwargs,
    ) -> None:
        super().__init__()
        # 不接收额外 kwargs，防止拼写错误
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # 选择设备
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device  # 支持 "auto" 字符串

        # 加载模型
        model_kwargs = {"torch_dtype": "auto", "device_map": self.device_map, "trust_remote_code": trust_remote_code}
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = AutoModelForCausalLM.from_pretrained(pretrained, **model_kwargs).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
        # 左侧 padding 便于批量推理
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token_id is None:
            # Qwen2 tokenizer 默认没有 pad token，使用 eos 作为 pad
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._config = self._model.config
        # 设置默认最大长度
        self._max_length = 2048
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.system_prompt = system_prompt

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU], "Unsupported distributed type."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self._model)
            else:
                self._model = accelerator.prepare_model(self._model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    # ========= 属性 =========
    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # unwrap accelerate model
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    # ========= Helper =========
    def flatten(self, lst):
        return [item for sub in lst for item in sub]

    # ========= 核心接口 =========
    def loglikelihood(self, requests: List[Instance]):
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2 text model.")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """批量生成文本，直至遇到指定停用词"""
        res: list[str] = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            # 推理过程仅用文本，不关心视觉信息
            gen_kwargs = all_gen_kwargs[0]

            until = gen_kwargs.pop("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError("`until` must be str or list of str")

            # 默认生成参数
            default_gen = {
                "max_new_tokens": 512,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
            }
            current_gen = {**default_gen, **gen_kwargs}
            do_sample = current_gen["temperature"] > 0

            batched_texts = []
            for context in contexts:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context},
                ]
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                batched_texts.append(text)

            inputs = self.tokenizer(batched_texts, return_tensors="pt", padding=True).to(self.device)

            outputs = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=do_sample,
                temperature=current_gen["temperature"] if do_sample else None,
                top_p=current_gen["top_p"] if do_sample else None,
                num_beams=current_gen["num_beams"],
                max_new_tokens=current_gen["max_new_tokens"],
                use_cache=self.use_cache,
            )

            # 去掉 prompt
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)]
            answers = self.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            # 处理 until 停用词
            processed_answers: list[str] = []
            for ans in answers:
                for term in until:
                    if term:
                        ans = ans.split(term)[0]
                processed_answers.append(ans)

            for ans, context in zip(processed_answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests):
        raise NotImplementedError("Multi-round generation is not implemented for Qwen2 text model.") 