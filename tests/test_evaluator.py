import numpy as np
import pytest

from active_evaluator import metrics, protocol
from active_evaluator.baselines import BASELINES
from active_evaluator.evaluator import ALGORITHMS, MetaEvaluator, Task
from active_evaluator.records import ModelRecord


def _records(n_models=12, n_workloads=80, n_feat=4, seed=0):
    """Accuracy = sigmoid(model skill - shift), shift observed through the descriptor."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n_feat)
    recs = {}
    for m in range(n_models):
        skill = rng.normal()
        sd = rng.normal(size=(n_workloads, n_feat)).astype(np.float32)
        acc = 1 / (1 + np.exp(-(skill - sd @ w)))
        r = ModelRecord(name=f"m{m}")
        r.add_workloads(np.arange(n_workloads), sd, acc, np.ones(n_workloads))
        for t in ("T1", "T2"):
            x = rng.normal(size=n_feat).astype(np.float32)
            r.target_sd[t] = x
            r.target_acc[t] = float(1 / (1 + np.exp(-(skill - x @ w))))
            labels = rng.integers(0, 5, 200)
            correct = rng.random(200) < r.target_acc[t]
            r.target_pred[t] = np.where(correct, labels, (labels + 1) % 5)
            r.target_conf[t] = np.clip(r.target_acc[t] + 0.1 * rng.normal(size=200), 0, 1)
            r.calib_correct[t] = rng.random(200) < 0.8
            r.calib_conf[t] = np.clip(0.8 + 0.1 * rng.normal(size=200), 0, 1)
            r.calib_pred[t] = rng.integers(0, 5, 200)
        recs[r.name] = r
    return recs


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_algorithms_train_and_predict(algorithm):
    recs = _records(n_models=6, n_workloads=40)
    tasks = [r.task(np.arange(40), np.ones(40)) for r in recs.values()]
    ev = MetaEvaluator(4, algorithm=algorithm, epochs=20, seed=0).fit(tasks)
    p = ev.predict(ev.adapt(tasks[0]), tasks[0].x[:3])
    assert p.shape == (3,) and np.all((p >= 0) & (p <= 1))


def test_cavia_learns_shift_accuracy_relation():
    recs = _records()
    names = sorted(recs)
    tasks = [recs[n].task(np.arange(80), np.ones(80)) for n in names[2:]]
    ev = MetaEvaluator(4, epochs=600, seed=0).fit(tasks)
    held = recs[names[0]]
    pred = [ev.predict(ev.adapt(held.task(np.arange(80), np.ones(80))), held.target_sd[t])[0] for t in ("T1", "T2")]
    constant = np.mean([r.acc.mean() for r in recs.values()])
    true = [held.target_acc[t] for t in ("T1", "T2")]
    assert metrics.mae(pred, true) < metrics.mae([constant] * 2, true)


def test_weights_act_as_multiplicities():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 3)).astype(np.float32)
    y = rng.random(10).astype(np.float32)
    w = rng.integers(1, 4, 10).astype(np.float32)
    rep = np.repeat(np.arange(10), w.astype(int))
    a = MetaEvaluator(3, epochs=0, seed=0).fit([Task(x, y, w)])
    b = MetaEvaluator(3, epochs=0, seed=0).fit([Task(x[rep], y[rep], np.ones(len(rep)))])
    pa = a.predict(a.adapt(Task(x, y, w)), x)
    pb = b.predict(b.adapt(Task(x[rep], y[rep], np.ones(len(rep)))), x)
    assert np.allclose(pa, pb, atol=1e-5)


def test_protocol_and_baselines(tmp_path):
    recs = _records()
    ids = np.arange(0, 80, 2)
    runs = protocol.run_meta(recs, ["T1", "T2"], ids, np.full(len(ids), 2.0), pool_size=5, runs=2, epochs=50)
    table = protocol.summarize(runs, ["T1", "T2"])
    assert set(table) == {"T1", "T2", "Avg."} and len(runs[0]["unseen_seconds"]) == 5
    for name, base in protocol.run_baselines(recs, ["T1", "T2"], pool_size=5, runs=2).items():
        assert np.isfinite(protocol.summarize(base, ["T1"])["T1"]["mae"][0]), name
    # same pools across methods
    assert protocol.split_pool(recs, 5, 1) == protocol.split_pool(recs, 5, 1)
    recs["m0"].save(tmp_path / "m0.npz")
    back = ModelRecord.load(tmp_path / "m0.npz")
    assert back.target_acc == recs["m0"].target_acc and np.array_equal(back.sd, recs["m0"].sd)


def test_atc_is_exact_when_confidence_separates_correctness():
    r = ModelRecord(name="a")
    r.calib_correct["T"] = np.r_[np.ones(70), np.zeros(30)].astype(bool)
    r.calib_conf["T"] = np.r_[np.linspace(0.8, 1, 70), np.linspace(0.1, 0.3, 30)]
    r.target_conf["T"] = np.r_[np.full(40, 0.9), np.full(60, 0.2)]
    assert BASELINES["ATC"]([r], "T")[0] == pytest.approx(0.4)


def test_ranking_metrics():
    true = [0.1, 0.5, 0.3, 0.9]
    assert metrics.pairwise_accuracy(true, true) == 1.0
    assert metrics.kendall_tau(true, true) == pytest.approx(1.0)
    assert metrics.topk_accuracy([0.9, 0.1, 0.2, 0.8], true, 1) == 0.0
    assert metrics.mae([0.5], [0.4]) == pytest.approx(10.0)
