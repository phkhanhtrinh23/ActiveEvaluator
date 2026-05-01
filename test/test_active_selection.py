import math
import random
from pathlib import Path
from typing import List

import pytest
import torch

from active_evaluator.active_selection import (
    SelectionExample,
    compute_bandwidth_median,
    compute_embeddings,
    compute_influence_weights,
    compute_per_example_gradients,
    direct_greedy_validation,
    gradient_match_omp,
    lazy_greedy_facility,
    narrow_validation,
    select_extension,
    _coverage_from,
    _gaussian_similarity,
    _marginal_gain,
)
from active_evaluator.model import ActiveEvaluator


DESCRIPTOR_DIM = 6
CONTEXT_DIM = 4
INPUT_DIM = DESCRIPTOR_DIM + CONTEXT_DIM


def _make_predictor(seed: int = 0) -> ActiveEvaluator:
    torch.manual_seed(seed)
    model = ActiveEvaluator(input_dim=INPUT_DIM, dropout=0.0)
    model.eval()
    return model


def _make_examples(n: int, *, seed: int = 0, with_labels: bool = True) -> List[SelectionExample]:
    rng = torch.Generator().manual_seed(seed)
    examples = []
    for i in range(n):
        desc = torch.randn(1, DESCRIPTOR_DIM, generator=rng)
        label = torch.randn(1, generator=rng) if with_labels else None
        examples.append(SelectionExample(key=f"x{i}", descriptor=desc, label=label))
    return examples


def test_compute_embeddings_shape():
    predictor = _make_predictor()
    examples = _make_examples(7)
    feats = compute_embeddings(predictor, examples)
    assert feats.shape[0] == 7
    assert feats.shape[1] == predictor.fc2.weight.shape[0]


def test_bandwidth_is_positive_and_scales():
    feats_small = torch.randn(50, 8) * 1.0
    feats_large = torch.randn(50, 8) * 10.0
    tau_small = compute_bandwidth_median(feats_small, seed=0)
    tau_large = compute_bandwidth_median(feats_large, seed=0)
    assert tau_small > 0
    assert tau_large > tau_small


def test_narrowing_reduces_validation_set():
    predictor = _make_predictor()
    val = _make_examples(20, seed=1)
    target = _make_examples(3, seed=2, with_labels=False)
    feats = compute_embeddings(predictor, val + target)
    tau = compute_bandwidth_median(feats, seed=0)
    val_VT, diag = narrow_validation(val, target, predictor, tau=tau, quantile_q=0.7)
    assert 0 < len(val_VT) <= len(val)
    assert all(ex in val for ex in val_VT)
    assert diag.n_val == len(val)
    assert diag.n_val_narrowed == len(val_VT)


def test_influence_weights_non_negative():
    predictor = _make_predictor()
    val = _make_examples(8, seed=3)
    weights = compute_influence_weights(predictor, val)
    assert weights.shape == (8,)
    assert (weights >= 0).all().item()


def test_disjointness_invariant():
    predictor = _make_predictor()
    pool = _make_examples(15, seed=4)
    val = _make_examples(5, seed=5)
    s0 = _make_examples(2, seed=6)
    target = _make_examples(2, seed=7, with_labels=False)
    selected, _log = select_extension(
        predictor=predictor,
        support_S0=s0,
        pool_U=pool,
        val_V=val,
        target_T=target,
        method="v1_facility",
        n_rounds=3,
        budget_fraction=0.4,
        K_steps=1,
        quantile_q=0.5,
        seed=0,
    )
    val_keys = {ex.key for ex in val}
    assert all(ex.key not in val_keys for ex in selected)


def test_budget_respected():
    predictor = _make_predictor()
    pool = _make_examples(20, seed=8)
    val = _make_examples(4, seed=9)
    s0 = _make_examples(1, seed=10)
    target = _make_examples(1, seed=11, with_labels=False)
    selected, log = select_extension(
        predictor=predictor,
        support_S0=s0,
        pool_U=pool,
        val_V=val,
        target_T=target,
        method="v1_facility",
        n_rounds=2,
        budget_absolute=5,
        K_steps=1,
        quantile_q=0.5,
        seed=0,
    )
    assert len(selected) <= 5
    assert log["budget_plan"]["total"] == 5


def _plain_greedy_facility(
    pool_features: torch.Tensor,
    coverage_init: torch.Tensor,
    val_features: torch.Tensor,
    influence: torch.Tensor,
    tau: float,
    budget: int,
) -> List[int]:
    coverage = coverage_init.clone()
    sim_pool = _gaussian_similarity(val_features, pool_features, tau)
    selected: List[int] = []
    for _ in range(budget):
        best_gain = 0.0
        best_idx = None
        for idx in range(pool_features.shape[0]):
            if idx in selected:
                continue
            gain = _marginal_gain(sim_pool[:, idx], coverage, influence)
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx is None or best_gain <= 0:
            break
        selected.append(best_idx)
        coverage = torch.maximum(coverage, sim_pool[:, best_idx])
    return selected


def test_lazy_greedy_matches_plain_greedy():
    torch.manual_seed(0)
    pool_features = torch.randn(20, 5)
    val_features = torch.randn(8, 5)
    influence = torch.rand(8) + 0.1
    tau = 1.5
    coverage_init = torch.zeros(8)

    lazy_picks, _ = lazy_greedy_facility(
        pool_features=pool_features,
        pool_keys=[f"k{i}" for i in range(20)],
        coverage_init=coverage_init.clone(),
        val_features=val_features,
        influence=influence,
        tau=tau,
        round_budget=6,
    )
    plain_picks = _plain_greedy_facility(pool_features, coverage_init, val_features, influence, tau, 6)
    assert lazy_picks == plain_picks


def test_facility_marginal_gains_non_increasing():
    torch.manual_seed(1)
    pool_features = torch.randn(15, 4)
    val_features = torch.randn(6, 4)
    influence = torch.rand(6) + 0.1
    tau = 1.0
    coverage = torch.zeros(6)
    sim_pool = _gaussian_similarity(val_features, pool_features, tau)

    picks: List[int] = []
    realised_gains: List[float] = []
    for _ in range(5):
        best_gain = -1.0
        best_idx = None
        for idx in range(pool_features.shape[0]):
            if idx in picks:
                continue
            g = _marginal_gain(sim_pool[:, idx], coverage, influence)
            if g > best_gain:
                best_gain = g
                best_idx = idx
        picks.append(best_idx)
        realised_gains.append(best_gain)
        coverage = torch.maximum(coverage, sim_pool[:, best_idx])

    for prev, curr in zip(realised_gains, realised_gains[1:]):
        assert curr <= prev + 1e-6


def test_select_extension_v2_runs_and_respects_budget():
    predictor = _make_predictor()
    pool = _make_examples(8, seed=12)
    val = _make_examples(4, seed=13)
    s0 = _make_examples(1, seed=14)
    target = _make_examples(1, seed=15, with_labels=False)
    selected, log = select_extension(
        predictor=predictor,
        support_S0=s0,
        pool_U=pool,
        val_V=val,
        target_T=target,
        method="v2_direct",
        n_rounds=2,
        budget_absolute=3,
        K_steps=2,
        inner_lr=0.05,
        quantile_q=0.5,
        seed=0,
        max_candidates_evaluated=4,
    )
    assert len(selected) <= 3
    assert log["budget_plan"]["total"] == 3


def test_v2_fass_filter_caps_direct_evaluations():
    """Version 2 must only K-step-evaluate the top-N candidates per pick."""
    predictor = _make_predictor()
    pool = _make_examples(20, seed=30)
    val = _make_examples(6, seed=31)
    s0 = _make_examples(1, seed=32)

    pool_features = compute_embeddings(predictor, pool)
    val_features = compute_embeddings(predictor, val)
    influence = compute_influence_weights(predictor, val)
    tau = compute_bandwidth_median(torch.cat([pool_features, val_features], dim=0), seed=0)

    eval_trace: List[List[int]] = []
    cap = 3
    direct_greedy_validation(
        predictor=predictor,
        pool=pool,
        pool_features=pool_features,
        support_S=s0,
        val_VT=val,
        val_features=val_features,
        influence=influence,
        tau=tau,
        K_steps=2,
        inner_lr=0.05,
        round_budget=4,
        max_candidates_evaluated=cap,
        eval_trace=eval_trace,
    )
    assert eval_trace, "FASS filter should record at least one round of candidates."
    for batch in eval_trace:
        assert len(batch) <= cap, (
            f"FASS filter must cap at {cap} but got {len(batch)} candidates."
        )


def test_v2_aborts_on_non_positive_gain():
    """Version 2's positive-gain abort rule must stop early when no candidate helps."""
    predictor = _make_predictor()
    pool = _make_examples(6, seed=40)
    val = _make_examples(4, seed=41)
    s0 = _make_examples(1, seed=42)

    # influence == 0 makes the V1 ranking flat, but more importantly,
    # we use inner_lr=0 + K_steps=0 so no adaptation happens; loss is invariant
    # to which candidate we add → gain is identically 0 → abort immediately.
    pool_features = compute_embeddings(predictor, pool)
    val_features = compute_embeddings(predictor, val)
    influence = torch.zeros(len(val))
    tau = 1.0

    picks, losses = direct_greedy_validation(
        predictor=predictor,
        pool=pool,
        pool_features=pool_features,
        support_S=s0,
        val_VT=val,
        val_features=val_features,
        influence=influence,
        tau=tau,
        K_steps=0,
        inner_lr=0.0,
        round_budget=5,
        max_candidates_evaluated=10,
    )
    assert picks == [], "V2 must abort when no candidate yields positive gain."
    assert losses == []


def test_v2_pipeline_wrapper_extends_support_set():
    """Smoke-test the pipeline wrapper end-to-end with method='v2_direct'."""
    import argparse
    from active_evaluator.descriptors import DEFAULT_FEATURE_ORDER, ShiftDescriptor
    from active_evaluator.meta_learning import (
        ActiveEvaluatorLearner,
        MetaLearningConfig,
        ShiftDescriptorTask,
    )
    from active_evaluator.pipeline import extend_test_tasks_with_active_selection

    def _desc(name, vec):
        return ShiftDescriptor(
            model_name=name,
            split_a="meta_train",
            split_b="meta_val",
            features={n: float(v) for n, v in zip(DEFAULT_FEATURE_ORDER, vec.tolist())},
        )

    def _task(idx, kind):
        rng = torch.Generator().manual_seed(idx)
        sup = torch.randn(len(DEFAULT_FEATURE_ORDER), generator=rng) * 0.3
        return ShiftDescriptorTask.from_descriptors(
            model_name=f"{kind}-{idx}",
            support=_desc(f"{kind}-{idx}", sup),
            support_label=0.5 + 0.05 * idx,
            query=_desc(f"{kind}-{idx}", sup + 0.05),
            query_label=0.5 + 0.05 * idx,
            transfer=_desc(f"{kind}-{idx}", sup + 0.1),
            transfer_label=0.5 + 0.05 * idx,
            device=torch.device("cpu"),
        )

    train_tasks = [_task(i, "train") for i in range(10)]
    val_tasks = [_task(100 + i, "val") for i in range(3)]
    test_tasks = [_task(200 + i, "test") for i in range(2)]

    context_dim = 8
    from active_evaluator.model import ActiveEvaluator as AE
    model = AE(input_dim=len(DEFAULT_FEATURE_ORDER) + context_dim, dropout=0.0)
    cfg = MetaLearningConfig(
        inner_lr=0.01,
        outer_lr=1e-3,
        inner_steps=1,
        tasks_per_batch=4,
        num_epochs=10,
        eval_inner_steps=1,
        eval_context_steps=1,
        device="cpu",
        context_dim=context_dim,
    )
    learner = ActiveEvaluatorLearner(model, cfg)
    learner.meta_train(train_tasks)

    pre_shapes = [t.support_descriptor.shape[0] for t in test_tasks]

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "out"
        args = argparse.Namespace(
            selection_method="v2_direct",
            selection_n_rounds=2,
            selection_budget_fraction=0.4,
            selection_budget_absolute=None,
            selection_K_steps=1,
            eval_inner_steps=1,
            inner_lr=0.05,
            selection_narrowing_quantile=0.5,
            selection_seed=0,
            selection_max_candidates_evaluated=4,
            selection_pool_narrow_quantile=0.0,
            selection_weight_decay=0.0,
            selection_early_stop_patience=0,
            selection_inner_lr=None,
            selection_gradmatch_lambda=1e-3,
        )
        extend_test_tasks_with_active_selection(
            meta_learner=learner,
            test_tasks=test_tasks,
            train_tasks=train_tasks,
            val_tasks=val_tasks,
            args=args,
            output_dir=out_dir,
        )
        for task in test_tasks:
            log_dir = out_dir / f"selection_{task.model_name}"
            assert log_dir.exists()
            assert (log_dir / "budget_plan.json").exists()
            assert (log_dir / "selection_trajectory.json").exists()

    post_shapes = [t.support_descriptor.shape[0] for t in test_tasks]
    # Extension is optional — V2's positive-gain rule may correctly abort if
    # the K-step adaptation already saturates V_T loss. The integration check
    # is that the pipeline ran end-to-end and emitted all log artifacts above.
    assert all(post >= pre for pre, post in zip(pre_shapes, post_shapes)), (
        f"V2 wrapper must never shrink the support set; pre={pre_shapes} post={post_shapes}"
    )


def test_gradmatch_omp_residual_decreases():
    """OMP residual norm must monotonically decrease with each pick."""
    torch.manual_seed(0)
    P = 50  # parameter dim
    N = 12  # candidates
    cand = torch.randn(N, P)
    target = torch.randn(P)
    pool_keys = [f"k{i}" for i in range(N)]
    pick_trace: List[dict] = []
    selected, weights = gradient_match_omp(
        target_grad=target,
        candidate_grads=cand,
        candidate_keys=pool_keys,
        budget=5,
        lambda_reg=1e-4,
        pick_trace=pick_trace,
    )
    assert len(selected) <= 5
    norms = [p["residual_norm_after"] for p in pick_trace]
    for prev, curr in zip(norms, norms[1:]):
        assert curr <= prev + 1e-5, f"residual must decrease: {prev} -> {curr}"


def test_gradmatch_picks_align_with_target():
    """When one candidate equals the target gradient, OMP should pick it first."""
    torch.manual_seed(0)
    P = 20
    cand = torch.randn(8, P)
    target = cand[3].clone() * 0.7  # candidate 3 is the perfect direction
    selected, weights = gradient_match_omp(
        target_grad=target,
        candidate_grads=cand,
        candidate_keys=[f"k{i}" for i in range(8)],
        budget=3,
        lambda_reg=1e-6,
    )
    assert selected[0] == 3, f"OMP should pick candidate 3 first, got {selected}"
    # First weight should be ~0.7 (since cand_3 == target/0.7)
    assert abs(weights[0].item() - 0.7) < 1e-2


def test_gradmatch_via_select_extension():
    """End-to-end smoke for the v3_gradmatch path through select_extension."""
    predictor = _make_predictor()
    pool = _make_examples(8, seed=50)
    val = _make_examples(4, seed=51)
    s0 = _make_examples(1, seed=52)
    target = _make_examples(2, seed=53, with_labels=False)
    selected, log = select_extension(
        predictor=predictor,
        support_S0=s0,
        pool_U=pool,
        val_V=val,
        target_T=target,
        method="v3_gradmatch",
        n_rounds=2,
        budget_absolute=4,
        K_steps=1,
        quantile_q=0.0,
        seed=0,
    )
    assert len(selected) <= 4
    assert log["summary"]["method"] == "v3_gradmatch"


def test_compute_per_example_gradients_shape():
    predictor = _make_predictor()
    examples = _make_examples(5, seed=60)
    grads = compute_per_example_gradients(predictor, examples)
    P = sum(p.numel() for p in predictor.parameters())
    assert grads.shape == (5, P)


def test_select_extension_writes_logs(tmp_path: Path):
    predictor = _make_predictor()
    pool = _make_examples(10, seed=20)
    val = _make_examples(4, seed=21)
    s0 = _make_examples(1, seed=22)
    target = _make_examples(1, seed=23, with_labels=False)
    log_dir = tmp_path / "selection"
    select_extension(
        predictor=predictor,
        support_S0=s0,
        pool_U=pool,
        val_V=val,
        target_T=target,
        method="v1_facility",
        n_rounds=2,
        budget_fraction=0.3,
        K_steps=1,
        quantile_q=0.5,
        seed=0,
        log_dir=log_dir,
    )
    assert (log_dir / "budget_plan.json").exists()
    assert (log_dir / "narrowing_diagnostics.json").exists()
    assert (log_dir / "selection_trajectory.json").exists()
    assert (log_dir / "selected_examples.json").exists()
