"""Text2SQL experiments.

    python -m text2sql.run select --config configs/text2sql.json --budgets 3000 6000 9000
    python -m text2sql.run records --config configs/text2sql.json
    python -m text2sql.run experiment --config configs/text2sql.json --budgets 9000 --baselines
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from active_evaluator.descriptors import ShiftDescriptor
from active_evaluator.encoders import TextTokenEncoder, workload_set
from active_evaluator.records import ModelRecord
from active_evaluator.runner import main, slug

from . import data
from .execution import Executor


class Text2SQL:
    def __init__(self, config: dict, out: Path, device: str | None = None):
        self.cfg, self.out, self.device = config, Path(out), device
        self.pool = [s for path in config["meta_pool"] for s in data.load(path)]
        w = config["workloads"]
        self.workloads = data.make_workloads(self.pool, w["n"], w["size"], w["dbs"], w.get("seed", 0))
        rng = np.random.default_rng(1)
        self.calib = [self.pool[i] for i in rng.choice(len(self.pool), config.get("calibration_size", 1000), replace=False)]
        cap = config.get("max_target_examples")
        self.target_data = {t: data.load(p, cap) for t, p in config["targets"].items() if Path(p).exists()}
        self.targets = list(self.target_data)
        self.models = {m["name"]: m for m in config["models"]}
        self.executor = Executor(config.get("db_root"), config.get("variants_root"))
        self._loaded = (None, None)

    def num_workloads(self) -> int:
        return len(self.workloads)

    def model_names(self) -> List[str]:
        return list(self.models)

    def workload_sets(self, max_points: int) -> List[np.ndarray]:
        path = self.out / "pool_tokens.npz"
        if path.exists():
            z = np.load(path)
            tokens = np.split(z["tokens"], z["offsets"][1:-1])
        else:
            enc = TextTokenEncoder(self.cfg["encoder"], dim=self.cfg.get("embed_dim", 64), device=self.device)
            tokens = enc([data.serialize(s) for s in self.pool])
            offsets = np.cumsum([0] + [len(t) for t in tokens])
            np.savez(path, tokens=np.concatenate(tokens), offsets=offsets)
        return [workload_set([tokens[i] for i in w], max_points, seed=j) for j, w in enumerate(self.workloads)]

    # --- per-model outputs, cached per example -------------------------------------
    def _outputs(self, name: str, split: str, samples: List[dict], needed: np.ndarray) -> Dict[str, np.ndarray]:
        path = self.out / "cache" / slug(name) / f"{slug(split)}.npz"
        if path.exists():
            cache = dict(np.load(path))
        else:
            cache = {"done": np.zeros(len(samples), bool), "correct": np.zeros(len(samples), bool),
                     "conf": np.zeros(len(samples), np.float32), "pred": np.zeros(len(samples), np.int64),
                     "seconds": np.zeros(len(samples), np.float32), "feat": None}
        todo = [int(i) for i in needed if not cache["done"][i]]
        if todo:
            sqls, conf, feat, seconds = self._model(name).run([samples[i] for i in todo])
            if cache["feat"] is None:
                cache["feat"] = np.zeros((len(samples), feat.shape[1]), np.float16)
            for k, i in enumerate(todo):
                _, exs, pred = self.executor.score(sqls[k], samples[i]["sql"], samples[i])
                cache["correct"][i], cache["pred"][i] = exs, pred
            cache["conf"][todo], cache["feat"][todo], cache["seconds"][todo] = conf, feat, seconds
            cache["done"][todo] = True
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **cache)
        return cache

    def _model(self, name: str):
        from .models import SQLModel

        if self._loaded[0] != name:
            if self._loaded[1] is not None:
                self._loaded[1].close()
            spec = self.models[name]
            self._loaded = (name, SQLModel(spec["hf"], batch_size=spec.get("batch_size", 8),
                                           device_map=spec.get("device_map", "auto")))
        return self._loaded[1]

    def update_record(self, record: ModelRecord, workload_ids: np.ndarray) -> ModelRecord:
        name = record.name
        spec = self.models[name]
        source = data.load(self.cfg["sources"][spec["source"]], self.cfg.get("source_size", 1000), seed=2)
        src = self._outputs(name, f"source-{spec['source']}", source, np.arange(len(source)))
        describe = ShiftDescriptor(src["feat"].astype(np.float32), src["conf"])

        missing = record.missing(workload_ids)
        if len(missing):
            needed = np.unique(np.concatenate([self.workloads[j] for j in missing]))
            pool = self._outputs(name, "pool", self.pool, needed)
            sd, acc, sec = [], [], []
            for j in missing:
                idx = self.workloads[j]
                sd.append(describe(pool["feat"][idx].astype(np.float32), pool["conf"][idx]))
                acc.append(pool["correct"][idx].mean())
                sec.append(pool["seconds"][idx].sum())
            record.add_workloads(missing, np.stack(sd), acc, sec)

        calib = self._outputs(name, "calibration", self.calib, np.arange(len(self.calib)))
        for t, samples in self.target_data.items():
            out = self._outputs(name, f"target-{t}", samples, np.arange(len(samples)))
            record.target_sd[t] = describe(out["feat"].astype(np.float32), out["conf"])
            record.target_acc[t] = float(out["correct"].mean())
            record.target_conf[t], record.target_pred[t] = out["conf"], out["pred"]
            record.calib_conf[t], record.calib_correct[t], record.calib_pred[t] = calib["conf"], calib["correct"], calib["pred"]
        return record


if __name__ == "__main__":
    main(Text2SQL, __doc__)
