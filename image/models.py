"""Image classifiers with task-matched heads.

`timm` / `hf` backbones use their ImageNet head on ImageNet tasks and a linear
head fit on source features elsewhere; `zeroshot` models (CLIP, SigLIP) score
the task's class names.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def _embedding(out) -> torch.Tensor:
    return out if torch.is_tensor(out) else out.pooler_output


class ImageModel:
    def __init__(self, spec: dict, device: str | None = None, batch_size: int = 128):
        self.spec, self.kind, self.batch_size = spec, spec["type"], batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.kind == "timm":
            import timm

            self.model = timm.create_model(spec["id"], pretrained=True).to(self.device).eval()
            cfg = timm.data.resolve_data_config({}, model=self.model)
            transform = timm.data.create_transform(**cfg)
            self.preprocess = lambda imgs: torch.stack([transform(i) for i in imgs])
        elif self.kind == "hf":
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            self.model = AutoModelForImageClassification.from_pretrained(spec["id"]).to(self.device).eval()
            proc = AutoImageProcessor.from_pretrained(spec["id"])
            self.preprocess = lambda imgs: proc(images=imgs, return_tensors="pt")["pixel_values"]
            self._pre_logits = None
            self.model.classifier.register_forward_hook(lambda m, inp, out: setattr(self, "_pre_logits", inp[0]))
        elif self.kind == "zeroshot":
            from transformers import AutoModel, AutoProcessor

            self.model = AutoModel.from_pretrained(spec["id"]).to(self.device).eval()
            self.proc = AutoProcessor.from_pretrained(spec["id"])
            self.preprocess = lambda imgs: self.proc(images=imgs, return_tensors="pt")["pixel_values"]
        else:
            raise ValueError(self.kind)
        self.heads: Dict[str, dict] = {}

    @torch.inference_mode()
    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Pre-logit features and native logits."""
        if self.kind == "timm":
            h = self.model.forward_head(self.model.forward_features(x), pre_logits=True)
            return h, self.model.get_classifier()(h)
        if self.kind == "hf":
            logits = self.model(pixel_values=x).logits
            return self._pre_logits.flatten(1), logits
        return _embedding(self.model.get_image_features(pixel_values=x)), None

    def _text_head(self, classes: Sequence[str]) -> torch.Tensor:
        prompts = [f"a photo of a {c}." for c in classes]
        kw = {"padding": "max_length", "max_length": 64} if "siglip" in self.model.config.model_type else {"padding": True}
        enc = self.proc(text=prompts, return_tensors="pt", **kw).to(self.device)
        with torch.inference_mode():
            return F.normalize(_embedding(self.model.get_text_features(**enc)), dim=-1)

    def features(self, images: List) -> tuple[np.ndarray, np.ndarray | None]:
        feats, logits = [], []
        for i in range(0, len(images), self.batch_size):
            h, z = self._forward(self.preprocess(images[i : i + self.batch_size]).to(self.device))
            feats.append(h.float().cpu())
            logits.append(None if z is None else z.float().cpu())
        return torch.cat(feats).numpy(), None if logits[0] is None else torch.cat(logits).numpy()

    def fit_head(self, task: str, images: List, labels: np.ndarray, cache: Path, epochs: int = 200) -> None:
        """Linear head on frozen features for non-ImageNet sources."""
        if cache.exists():
            self.heads[task] = torch.load(cache)
            return
        x = torch.as_tensor(self.features(images)[0])
        y = torch.as_tensor(labels, dtype=torch.long)
        mu, sd = x.mean(0), x.std(0).clamp_min(1e-6)
        layer = torch.nn.Linear(x.shape[1], int(y.max()) + 1)
        opt = torch.optim.Adam(layer.parameters(), lr=1e-2, weight_decay=1e-4)
        for _ in range(epochs):
            opt.zero_grad()
            F.cross_entropy(layer((x - mu) / sd), y).backward()
            opt.step()
        self.heads[task] = {"mu": mu, "sd": sd, "w": layer.weight.detach(), "b": layer.bias.detach()}
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.heads[task], cache)

    def run(self, images: List, task: str, classes: Sequence[str], native: bool, subset=None):
        """Predictions, confidences, features and seconds per image."""
        t0 = time.perf_counter()
        feats, logits = self.features(images)
        if self.kind == "zeroshot":
            text = self._text_head(classes).cpu()
            img = F.normalize(torch.as_tensor(feats), dim=-1)
            logits = (img @ text.T * 100).numpy()
        elif not native:
            h = self.heads[task]
            logits = (((torch.as_tensor(feats) - h["mu"]) / h["sd"]) @ h["w"].T + h["b"]).numpy()
        logits = torch.as_tensor(logits)
        if subset is not None:
            mask = torch.full_like(logits, float("-inf"))
            mask[:, torch.as_tensor(subset)] = 0
            logits = logits + mask
        prob = logits.softmax(-1)
        conf, pred = prob.max(-1)
        seconds = (time.perf_counter() - t0) / max(len(images), 1)
        return pred.numpy(), conf.numpy(), feats, np.full(len(images), seconds, dtype=np.float32)
