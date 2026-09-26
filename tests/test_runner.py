"""End-to-end run of the shared CLI on a small in-memory modality."""

import json
import sys

import numpy as np

from active_evaluator import runner
from active_evaluator.descriptors import ShiftDescriptor
from active_evaluator.records import ModelRecord


class Toy:
    """Workloads are Gaussian clouds; model accuracy decays with the distance to its source."""

    targets = ["near", "far"]

    def __init__(self, config, out, device=None):
        rng = np.random.default_rng(0)
        self.centers = rng.normal(size=(config["n"], 3))
        self.inputs = [c + 0.3 * rng.normal(size=(20, 3)) for c in self.centers]
        self.target_inputs = {"near": 0.3 * rng.normal(size=(50, 3)), "far": 2 + 0.3 * rng.normal(size=(50, 3))}
        self.skill = {f"m{i}": s for i, s in enumerate(rng.uniform(0.5, 2, config["models"]))}

    def num_workloads(self):
        return len(self.inputs)

    def model_names(self):
        return list(self.skill)

    def workload_sets(self, max_points):
        return [(x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16) for x in self.inputs]

    def _run(self, name, x):
        d = np.linalg.norm(x, axis=1)
        conf = 1 / (1 + np.exp(d - self.skill[name]))
        return x * self.skill[name], conf, conf > 0.5

    def update_record(self, record, ids):
        feat, conf, _ = self._run(record.name, 0.3 * np.random.default_rng(1).normal(size=(100, 3)))
        describe = ShiftDescriptor(feat, conf, dim=3)
        missing = record.missing(ids)
        if len(missing):
            out = [self._run(record.name, self.inputs[j]) for j in missing]
            record.add_workloads(missing, np.stack([describe(f, c) for f, c, _ in out]),
                                 [ok.mean() for _, _, ok in out], np.ones(len(missing)))
        for t, x in self.target_inputs.items():
            f, c, ok = self._run(record.name, x)
            record.target_sd[t], record.target_acc[t] = describe(f, c), float(ok.mean())
            record.target_conf[t], record.target_pred[t] = c, ok.astype(int)
            record.calib_conf[t], record.calib_correct[t], record.calib_pred[t] = c, ok, ok.astype(int)
        return record


def _cli(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["run", *argv])
    runner.main(Toy, "toy")


def test_select_records_experiment(tmp_path, monkeypatch):
    cfg = tmp_path / "toy.json"
    cfg.write_text(json.dumps({"n": 40, "models": 10}))
    common = ["--config", str(cfg), "--out", str(tmp_path)]
    _cli(monkeypatch, "select", *common, "--budgets", "5", "10", "--device", "cpu")
    _cli(monkeypatch, "records", *common, "--full")
    _cli(monkeypatch, "experiment", *common, "--budgets", "5", "10", "--full", "--runs", "2",
         "--pool-sizes", "4", "--epochs", "30", "--baselines", "--algorithms", "cavia", "protonet")
    results = json.loads((tmp_path / "results.json").read_text())
    assert "ActiveEvaluator[hausdorff,K=10][cavia][pool=4]" in results
    assert "MetaEvaluator[full][protonet][pool=4]" in results
    assert "ATC[pool=4]" in results
    rec = ModelRecord.load(tmp_path / "records" / "m0.npz")
    assert len(rec.workload_ids) == 40
