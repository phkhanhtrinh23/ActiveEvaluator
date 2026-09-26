"""Hugging Face Text2SQL models: SQL, sequence confidence and prompt features."""

from __future__ import annotations

import time
from typing import List, Sequence

import numpy as np
import torch

from . import data
from .execution import extract_sql


class SQLModel:
    def __init__(self, hf_id: str, batch_size: int = 8, max_new_tokens: int = 256, max_input: int = 4096,
                 device_map: str | None = "auto"):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

        self.seq2seq = bool(getattr(AutoConfig.from_pretrained(hf_id), "is_encoder_decoder", False))
        self.tok = AutoTokenizer.from_pretrained(hf_id, padding_side="right" if self.seq2seq else "left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        cls = AutoModelForSeq2SeqLM if self.seq2seq else AutoModelForCausalLM
        self.model = cls.from_pretrained(hf_id, dtype=torch.bfloat16, device_map=device_map).eval()
        eos = self.model.generation_config.eos_token_id
        self.eos = torch.tensor(eos if isinstance(eos, list) else [eos if eos is not None else self.tok.eos_token_id])
        self.batch_size, self.max_new_tokens, self.max_input = batch_size, max_new_tokens, max_input

    def _text(self, sample: dict) -> str:
        messages = data.prompt(sample)
        if self.tok.chat_template and not self.seq2seq:
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return "\n\n".join(m["content"] for m in messages)

    def _features(self, enc) -> torch.Tensor:
        mask = enc["attention_mask"]
        if self.seq2seq:
            hidden = self.model.get_encoder()(**enc).last_hidden_state
        else:
            positions = (mask.cumsum(-1) - 1).clamp_min(0)
            hidden = self.model.base_model(**enc, position_ids=positions).last_hidden_state
        m = mask[..., None].to(hidden.dtype)
        return ((hidden * m).sum(1) / m.sum(1)).float()

    @torch.inference_mode()
    def run(self, samples: Sequence[dict]):
        """SQL strings, confidences exp(mean token log-prob), features, seconds per example."""
        texts = [self._text(s) for s in samples]
        order = np.argsort([len(t) for t in texts])[::-1]
        sqls: List[str] = [""] * len(texts)
        conf = np.zeros(len(texts), dtype=np.float32)
        feats: List[np.ndarray] = [None] * len(texts)
        seconds = np.zeros(len(texts), dtype=np.float32)
        device = self.model.device
        for i in range(0, len(order), self.batch_size):
            idx = order[i : i + self.batch_size]
            t0 = time.perf_counter()
            enc = self.tok([texts[j] for j in idx], return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_input).to(device)
            f = self._features(enc).cpu().numpy()
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      return_dict_in_generate=True, output_scores=True,
                                      pad_token_id=self.tok.pad_token_id)
            logp = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True).float()
            gen = out.sequences[:, -logp.shape[1]:]
            is_eos = torch.isin(gen, self.eos.to(gen.device))
            valid = (is_eos.cumsum(-1) - is_eos.long()) == 0
            mean_logp = (logp * valid).sum(-1) / valid.sum(-1).clamp_min(1)
            decoded = self.tok.batch_decode(gen, skip_special_tokens=True)
            elapsed = (time.perf_counter() - t0) / len(idx)
            for k, j in enumerate(idx):
                sqls[j] = extract_sql(decoded[k])
                conf[j] = float(mean_logp[k].exp())
                feats[j] = f[k]
                seconds[j] = elapsed
        return sqls, conf, np.stack(feats), seconds

    def close(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
