import numpy as np
import pytest
from scipy.spatial.distance import directed_hausdorff

from active_evaluator.distances import DISTANCES, chamfer, hausdorff
from active_evaluator.selection import Selection, facility_location, objective, select, similarity


def _sets(n=12, d=5, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        z = rng.normal(size=(rng.integers(3, 9), d)) + rng.normal(size=d)
        out.append((z / np.linalg.norm(z, axis=1, keepdims=True)).astype(np.float16))
    return out


def _naive_greedy(S, k):
    cover, chosen = np.zeros(len(S)), []
    for _ in range(k):
        gains = np.maximum(S - cover[None], 0).sum(1)
        gains[chosen] = -1
        q = int(np.argmax(gains))
        chosen.append(q)
        cover = np.maximum(cover, S[q])
    return chosen


def test_hausdorff_matches_scipy():
    sets = _sets()
    D = hausdorff(sets, block=5, device="cpu")
    for u in range(len(sets)):
        for v in range(len(sets)):
            a, b = sets[u].astype(np.float64), sets[v].astype(np.float64)
            ref = max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0])
            assert D[u, v] == pytest.approx(ref, abs=2e-3)


def test_chamfer_symmetric_zero_diagonal():
    D = chamfer(_sets(), device="cpu")
    assert np.allclose(D, D.T) and np.allclose(np.diag(D), 0)


@pytest.mark.parametrize("metric", sorted(DISTANCES))
def test_all_distances_are_symmetric(metric):
    D = DISTANCES[metric](_sets(n=7), device="cpu")
    assert D.shape == (7, 7)
    assert np.allclose(D, D.T, atol=1e-4) and (D >= -1e-6).all()


def test_similarity_uses_median_bandwidth():
    D = hausdorff(_sets(), device="cpu")
    S, tau = similarity(D)
    iu = np.triu_indices(len(D), 1)
    assert tau == pytest.approx(np.median(D[iu].astype(np.float64) ** 2))
    assert np.allclose(np.diag(S), 1) and (S > 0).all() and (S <= 1).all()


def test_lazy_greedy_equals_plain_greedy():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 4))
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    S, _ = similarity(D)
    order, gains, _ = facility_location(S, [20])
    assert order.tolist() == _naive_greedy(S, 20)
    assert np.all(np.diff(gains) <= 1e-6)  # diminishing returns along the greedy path


def test_gains_sum_to_objective_and_weights_to_n():
    D = hausdorff(_sets(n=30), device="cpu")
    sel = select(D, [5, 10, 30])
    S, _ = similarity(D)
    for k in (5, 10, 30):
        ids, w = sel.subset(k)
        assert w.sum() == len(D)
        assert sel.gains[:k].sum() == pytest.approx(objective(S, ids), rel=1e-5)
    # gamma counts nearest-representative assignments
    ids, w = sel.subset(5)
    nearest = ids[np.argmax(S[ids], axis=0)]
    assert np.array_equal(w, np.array([(nearest == u).sum() for u in ids], dtype=np.float32))
    assert np.array_equal(sel.subset(30)[1], np.ones(30))


def test_selection_roundtrip(tmp_path):
    sel = select(hausdorff(_sets(n=10), device="cpu"), [3, 6])
    sel.save(tmp_path / "s.npz")
    back = Selection.load(tmp_path / "s.npz")
    assert np.array_equal(back.order, sel.order) and back.tau == sel.tau
    assert all(np.array_equal(back.weights[k], sel.weights[k]) for k in (3, 6))


def test_chunked_path_matches_batched():
    import torch

    from active_evaluator.distances import _large_pairwise

    sets = _sets(n=6)
    for reduce, fn in (("max", hausdorff), ("mean", chamfer)):
        big = _large_pairwise(sets, reduce, torch.device("cpu"), chunk=3)
        assert np.allclose(big, fn(sets, device="cpu"), atol=1e-4)
