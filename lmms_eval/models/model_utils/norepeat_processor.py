import torch
from typing import Dict, List, Tuple
from transformers.utils.doc import add_start_docstrings
from transformers.generation.logits_process import LogitsProcessor, LOGITS_PROCESSOR_INPUTS_DOCSTRING, _calc_banned_ngram_tokens

class EOSNGramLogitsProcessor(LogitsProcessor):

    def __init__(self, ngram_size: int, eos_token_id: int):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        self.ngram_size = ngram_size
        self.eos_token_id = eos_token_id

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        num_batch_hypotheses = scores.shape[0]
        cur_len = input_ids.shape[-1]
        
        scores_processed = scores.clone()
        banned_batch_tokens = _calc_banned_ngram_tokens(self.ngram_size, input_ids, num_batch_hypotheses, cur_len)

        finfo_min = torch.finfo(scores.dtype).min
        for i in range(input_ids.size(0)):
            # New add
            top_id = int(torch.argmax(scores[i]).item())
            if top_id in banned_batch_tokens[i]:
                scores_processed[i].fill_(finfo_min)
                # 给 EOS 一个有限大分数（如 0.0 或 max+10），确保被采样
                scores_processed[i, self.eos_token_id] = 0.0

        return scores_processed
