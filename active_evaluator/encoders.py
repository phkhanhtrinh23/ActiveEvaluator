"""Workload embedding sets Z_phi(b) = union of normalized projected token embeddings."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch


class Projection:
    """Fixed random projection W followed by row-wise L2 normalization."""

    def __init__(self, in_dim: int, out_dim: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = (rng.normal(size=(in_dim, out_dim)) / np.sqrt(out_dim)).astype(np.float32)

    def __call__(self, h: np.ndarray) -> np.ndarray:
        z = h.astype(np.float32) @ self.W
        return (z / np.linalg.norm(z, axis=1, keepdims=True).clip(1e-12)).astype(np.float16)


class TextTokenEncoder:
    """phi(x): every non-padding token of a frozen text encoder, projected and normalized."""

    def __init__(self, model_id: str, dim: int = 64, max_length: int = 128, batch_size: int = 64,
                 device: str | None = None, seed: int = 0):
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.max_length, self.batch_size = max_length, batch_size
        self.proj = Projection(self.model.config.hidden_size, dim, seed)

    @torch.inference_mode()
    def __call__(self, texts: Sequence[str]) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(list(texts[i : i + self.batch_size]), padding=True, truncation=True,
                                   max_length=self.max_length, return_tensors="pt").to(self.device)
            hidden = self.model(**batch).last_hidden_state.float().cpu().numpy()
            for h, m in zip(hidden, batch["attention_mask"].bool().cpu().numpy()):
                out.append(self.proj(h[m]))
        return out


class ImagePatchEncoder:
    """phi(x): every patch token of a frozen vision encoder, projected and normalized."""

    def __init__(self, model_id: str, dim: int = 64, batch_size: int = 64, device: str | None = None, seed: int = 0):
        from transformers import AutoImageProcessor, AutoModel

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.batch_size = batch_size
        self.proj = Projection(self.model.config.hidden_size, dim, seed)

    @torch.inference_mode()
    def __call__(self, images) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for i in range(0, len(images), self.batch_size):
            batch = self.processor(images=list(images[i : i + self.batch_size]), return_tensors="pt").to(self.device)
            hidden = self.model(**batch).last_hidden_state.float().cpu().numpy()
            out.extend(self.proj(h) for h in hidden)
        return out


def workload_set(token_sets: Sequence[np.ndarray], max_points: int | None = None, seed: int = 0) -> np.ndarray:
    """Z_phi(b) = union of phi(x) over the inputs of a workload."""
    z = np.concatenate(token_sets)
    if max_points and len(z) > max_points:
        z = z[np.random.default_rng(seed).choice(len(z), max_points, replace=False)]
    return z.astype(np.float16)
