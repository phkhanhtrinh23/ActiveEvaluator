"""Image classification experiments.

    python -m image.run select --config configs/image.json --budgets 2000 4000 8000
    python -m image.run records --config configs/image.json
    python -m image.run experiment --config configs/image.json --budgets 8000 --pool-sizes 5 10 15 20 --baselines
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from active_evaluator.descriptors import ShiftDescriptor
from active_evaluator.encoders import ImagePatchEncoder, workload_set
from active_evaluator.records import ModelRecord
from active_evaluator.runner import main, slug

from . import data


class ImageClassification:
    def __init__(self, config: dict, out: Path, device: str | None = None):
        self.cfg, self.out, self.device = config, Path(out), device
        root = config["root"]
        self.tasks: Dict[str, dict] = {}
        self.target_task: Dict[str, str] = {}
        self.target_data: Dict[str, data.ImageSet] = {}
        for task, spec in config["tasks"].items():
            train, test = data.split_source(spec["source"], root)
            self.tasks[task] = {"train": train, "test": test, "native": spec["source"] == "imagenet"}
            for t, name in spec["targets"].items():
                self.target_task[t] = task
                self.target_data[t] = data.load(name, root).sample(config.get("max_target_examples"), seed=3)
        self.targets = list(self.target_data)

        w = config["workloads"]
        rng = np.random.default_rng(w.get("seed", 0))
        names = list(self.tasks)
        self.workloads = []
        for j in range(w["n"]):
            task = names[j % len(names)]
            size = int(rng.integers(w["size"][0], w["size"][1] + 1))
            base = np.sort(rng.choice(len(self.tasks[task]["test"]), size, replace=False))
            self.workloads.append((task, base, data.random_transform(int(rng.integers(2**31)))))
        self.models = {m["name"]: m for m in config["models"]}

    def num_workloads(self) -> int:
        return len(self.workloads)

    def model_names(self) -> List[str]:
        return list(self.models)

    def _images(self, j: int) -> List:
        task, base, transform = self.workloads[j]
        imgs = self.tasks[task]["test"].images(base)
        return [data.apply_transform(img, transform, seed=j * 100003 + k) for k, img in enumerate(imgs)]

    def workload_sets(self, max_points: int) -> List[np.ndarray]:
        path = self.out / "workload_sets.npz"
        if path.exists():
            z = np.load(path)
            return np.split(z["points"], z["offsets"][1:-1])
        enc = ImagePatchEncoder(self.cfg["encoder"], dim=self.cfg.get("embed_dim", 64), device=self.device)
        sets = [workload_set(enc(self._images(j)), max_points, seed=j) for j in range(len(self.workloads))]
        np.savez(path, points=np.concatenate(sets), offsets=np.cumsum([0] + [len(s) for s in sets]))
        return sets

    def update_record(self, record: ModelRecord, workload_ids: np.ndarray) -> ModelRecord:
        from .models import ImageModel

        model = ImageModel(self.models[record.name], device=self.device)
        describe: Dict[str, ShiftDescriptor] = {}
        n_src = self.cfg.get("source_size", 2000)
        for task, t in self.tasks.items():
            train = t["train"].sample(n_src, seed=4)
            imgs = train.images(range(len(train)))
            if not t["native"] and model.kind != "zeroshot":
                model.fit_head(task, imgs, train.labels, self.out / "heads" / slug(record.name) / f"{task}.pt")
            _, conf, feats, _ = model.run(imgs, task, train.classes, t["native"])
            describe[task] = ShiftDescriptor(feats, conf)

        missing = record.missing(workload_ids)
        sd, acc, sec = [], [], []
        for j in missing:
            task, base, _ = self.workloads[j]
            t = self.tasks[task]
            pred, conf, feats, seconds = model.run(self._images(j), task, t["test"].classes, t["native"])
            sd.append(describe[task](feats, conf))
            acc.append(np.mean(pred == t["test"].labels[base]))
            sec.append(seconds.sum())
        if len(missing):
            record.add_workloads(missing, np.stack(sd), acc, sec)

        for name, target in self.target_data.items():
            task = self.target_task[name]
            t = self.tasks[task]
            pred, conf, feats, _ = model.run(target.images(range(len(target))), task, target.classes,
                                             t["native"], target.subset)
            calib = t["test"].sample(self.cfg.get("calibration_size", 2000), seed=5)
            keep = np.arange(len(calib)) if target.subset is None else np.flatnonzero(np.isin(calib.labels, target.subset))
            c_pred, c_conf, _, _ = model.run(calib.images(keep), task, calib.classes, t["native"], target.subset)
            record.target_sd[name] = describe[task](feats, conf)
            record.target_acc[name] = float(np.mean(pred == target.labels))
            record.target_conf[name], record.target_pred[name] = conf, pred
            record.calib_conf[name], record.calib_correct[name], record.calib_pred[name] = c_conf, c_pred == calib.labels[keep], c_pred
        return record


if __name__ == "__main__":
    main(ImageClassification, __doc__)
