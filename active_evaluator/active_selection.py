"""Active selection of additional adaptation examples under a budget.

Two methods are supported:

- ``v1_facility``: monotone submodular, influence-weighted facility location
  with lazy greedy. Provides a (1 - 1/e) approximation guarantee under
  cardinality constraints (Nemhauser, Wolsey & Fisher 1978; Minoux 1978).
- ``v2_direct``: non-submodular oracle that runs K-step adaptation and
  measures validation-loss reduction directly. Uses plain greedy with a
  positive-gain abort rule and optional FASS-style pre-filtering by V1
  (Wei, Iyer & Bilmes 2015).

The narrowing step (Strategy A) restricts the validation set V to the subset
V_T closest in feature space to the unlabeled target T before scoring.
"""

from __future__ import annotations

import heapq
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from .budget import BudgetPlan, compute_budget
from .model import ActiveEvaluator


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data adapters
# ---------------------------------------------------------------------------


@dataclass
class SelectionExample:
    """A single (descriptor, label) example used for active selection.

    For an unlabeled target example, ``label`` may be None.
    """

    key: str
    descriptor: torch.Tensor  # shape (1, descriptor_dim) or (descriptor_dim,)
    label: Optional[torch.Tensor] = None  # scalar tensor or None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.descriptor.dim() == 1:
            self.descriptor = self.descriptor.unsqueeze(0)
        if self.label is not None and self.label.dim() == 0:
            self.label = self.label.unsqueeze(0)

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SelectionExample) and self.key == other.key


# ---------------------------------------------------------------------------
# Embedding / similarity helpers
# ---------------------------------------------------------------------------


def _zero_context(predictor: ActiveEvaluator, descriptor: torch.Tensor) -> torch.Tensor:
    """Append a zero context vector to a descriptor so the model sees its full input."""
    expected = predictor.fc1.weight.shape[1]
    have = descriptor.shape[-1]
    if have == expected:
        return descriptor
    if have > expected:
        raise ValueError(
            f"Descriptor width {have} exceeds predictor input dim {expected}."
        )
    pad = torch.zeros(*descriptor.shape[:-1], expected - have, device=descriptor.device, dtype=descriptor.dtype)
    return torch.cat([descriptor, pad], dim=-1)


def _penultimate(predictor: ActiveEvaluator, x: torch.Tensor) -> torch.Tensor:
    """Return the penultimate-layer activations of the MLP predictor."""
    h = F.relu(predictor.fc1(x))
    h = F.relu(predictor.fc2(h))
    return h


def compute_embeddings(
    predictor: ActiveEvaluator,
    examples: Sequence[SelectionExample],
    *,
    show_progress: bool = False,
    desc: str = "Embed",
) -> torch.Tensor:
    """Compute penultimate-layer features for a list of examples.

    Args:
        predictor: The trained ActiveEvaluator MLP.
        examples: Examples whose descriptors should be embedded.
        show_progress: If True, wrap the descriptor cat in a tqdm bar.
        desc: tqdm description (when ``show_progress`` is True).

    Returns:
        Tensor of shape ``(len(examples), hidden_dim)``.
    """
    if not examples:
        return torch.empty(0)
    predictor_was_training = predictor.training
    predictor.eval()
    iterator = tqdm(examples, desc=desc, leave=False) if show_progress else examples
    descriptors = torch.cat([_zero_context(predictor, ex.descriptor) for ex in iterator], dim=0)
    with torch.no_grad():
        feats = _penultimate(predictor, descriptors)
    if predictor_was_training:
        predictor.train()
    return feats.detach()


def compute_bandwidth_median(
    features: torch.Tensor,
    *,
    max_pairs: int = 2000,
    seed: int = 42,
) -> float:
    """Median-heuristic bandwidth: median of squared pairwise distances on a sample.

    Args:
        features: Tensor of shape ``(N, d)``.
        max_pairs: Cap on the number of random pairs to sample.
        seed: RNG seed for pair sampling.

    Returns:
        Positive scalar tau used in the Gaussian kernel.
    """
    n = features.shape[0]
    if n < 2:
        return 1.0
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pair_count = min(max_pairs, n * (n - 1) // 2)
    i_idx = torch.randint(0, n, (pair_count,), generator=generator)
    j_idx = torch.randint(0, n, (pair_count,), generator=generator)
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]
    if i_idx.numel() == 0:
        return 1.0
    diffs = features[i_idx] - features[j_idx]
    sq_dists = (diffs * diffs).sum(dim=-1)
    tau = float(torch.median(sq_dists).item())
    if tau <= 0.0:
        tau = float(sq_dists.mean().item()) or 1.0
    assert tau > 0.0, "Bandwidth tau must be positive."
    return tau


def _gaussian_similarity(features_a: torch.Tensor, features_b: torch.Tensor, tau: float) -> torch.Tensor:
    """Compute exp(-||a - b||^2 / tau) for every pair."""
    sq = torch.cdist(features_a, features_b, p=2.0).pow(2)
    return torch.exp(-sq / tau)


# ---------------------------------------------------------------------------
# Narrowing V to V_T (Strategy A)
# ---------------------------------------------------------------------------


@dataclass
class NarrowingDiagnostics:
    n_val: int
    n_val_narrowed: int
    quantile_q: float
    rho_threshold: float
    rho_mean: float
    rho_min: float
    rho_max: float


def narrow_validation(
    val_V: Sequence[SelectionExample],
    target_T: Sequence[SelectionExample],
    predictor: ActiveEvaluator,
    *,
    tau: float,
    quantile_q: float,
    pool_U_keys: Iterable[str] = (),
) -> Tuple[List[SelectionExample], NarrowingDiagnostics]:
    """Narrow V to the subset closest in feature space to T.

    Args:
        val_V: Full validation set with labels.
        target_T: Unlabeled target features.
        predictor: Predictor used for the embedding map.
        tau: Kernel bandwidth.
        quantile_q: Keep examples whose rho is above this quantile of rho values.
        pool_U_keys: Keys belonging to the pool U; V_T must remain disjoint.

    Returns:
        ``(V_T, diagnostics)``. V_T may be empty when V is empty.
    """
    if not val_V or not target_T:
        diag = NarrowingDiagnostics(
            n_val=len(val_V),
            n_val_narrowed=len(val_V),
            quantile_q=quantile_q,
            rho_threshold=0.0,
            rho_mean=0.0,
            rho_min=0.0,
            rho_max=0.0,
        )
        return list(val_V), diag

    feats_V = compute_embeddings(predictor, val_V)
    feats_T = compute_embeddings(predictor, target_T)
    sim = _gaussian_similarity(feats_V, feats_T, tau)
    rho = sim.mean(dim=1)
    threshold = torch.quantile(rho, quantile_q).item()
    pool_keys = set(pool_U_keys)

    selected: List[SelectionExample] = []
    for example, score in zip(val_V, rho.tolist()):
        if score >= threshold and example.key not in pool_keys:
            selected.append(example)

    if not selected and val_V:
        best_idx = int(torch.argmax(rho).item())
        selected = [val_V[best_idx]]

    diag = NarrowingDiagnostics(
        n_val=len(val_V),
        n_val_narrowed=len(selected),
        quantile_q=quantile_q,
        rho_threshold=float(threshold),
        rho_mean=float(rho.mean().item()),
        rho_min=float(rho.min().item()),
        rho_max=float(rho.max().item()),
    )
    return selected, diag


# ---------------------------------------------------------------------------
# Influence weights I(v) = ||grad_phi L(h_phi; v)||_2
# ---------------------------------------------------------------------------


def compute_influence_weights(
    predictor: ActiveEvaluator,
    val_VT: Sequence[SelectionExample],
) -> torch.Tensor:
    """Per-example gradient-norm influence weights.

    Args:
        predictor: The trained MLP.
        val_VT: Narrowed validation examples (must carry labels).

    Returns:
        Non-negative tensor of shape ``(len(val_VT),)``.
    """
    if not val_VT:
        return torch.empty(0)
    predictor.zero_grad(set_to_none=True)
    weights = []
    for v in val_VT:
        if v.label is None:
            raise ValueError(f"Validation example {v.key} is missing a label.")
        x = _zero_context(predictor, v.descriptor)
        target = v.label.to(x.device)
        for p in predictor.parameters():
            if p.grad is not None:
                p.grad = None
        pred = predictor(x)
        loss = F.mse_loss(pred, target)
        grads = torch.autograd.grad(loss, list(predictor.parameters()), retain_graph=False, create_graph=False)
        flat = torch.cat([g.reshape(-1) for g in grads if g is not None])
        weights.append(flat.norm(p=2).detach())
    out = torch.stack(weights)
    assert (out >= 0).all().item(), "Influence weights must be non-negative."
    return out


# ---------------------------------------------------------------------------
# Lazy greedy (Version 1) — influence-weighted facility location
# ---------------------------------------------------------------------------


def _coverage_from(features_VT: torch.Tensor, features_S: torch.Tensor, tau: float) -> torch.Tensor:
    """For each v, max similarity to any element of S (zero when S is empty)."""
    if features_S.numel() == 0:
        return torch.zeros(features_VT.shape[0], device=features_VT.device)
    sim = _gaussian_similarity(features_VT, features_S, tau)
    return sim.max(dim=1).values


def _marginal_gain(
    sim_to_candidate: torch.Tensor,
    coverage: torch.Tensor,
    influence: torch.Tensor,
) -> float:
    delta = (sim_to_candidate - coverage).clamp_min(0.0)
    return float((influence * delta).sum().item())


def lazy_greedy_facility(
    pool_features: torch.Tensor,
    pool_keys: Sequence[str],
    coverage_init: torch.Tensor,
    val_features: torch.Tensor,
    influence: torch.Tensor,
    tau: float,
    round_budget: int,
    cost_fn: Optional[Callable[[str], float]] = None,
    pick_trace: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[int], torch.Tensor]:
    """Lazy-greedy maximization of the influence-weighted facility-location surrogate.

    Reference: Minoux (1978), Nemhauser, Wolsey & Fisher (1978).

    Args:
        pool_features: ``(P, d)`` features of remaining candidates.
        pool_keys: Keys aligned with ``pool_features`` (used only for cost lookup).
        coverage_init: Per-validation max similarity from the current S (incl. S_0).
        val_features: ``(|V_T|, d)`` features of narrowed validation examples.
        influence: ``(|V_T|,)`` influence weights.
        tau: Kernel bandwidth.
        round_budget: Number of candidates to add this round (cardinality budget).
        cost_fn: Optional non-uniform cost function. Defaults to unit cost.

    Returns:
        ``(selected_indices, updated_coverage)``.
    """
    if round_budget <= 0 or pool_features.shape[0] == 0 or val_features.shape[0] == 0:
        return [], coverage_init.clone()

    cost = cost_fn or (lambda _key: 1.0)
    coverage = coverage_init.clone()

    sim_pool = _gaussian_similarity(val_features, pool_features, tau)  # (|V_T|, P)
    heap: List[Tuple[float, int, int]] = []
    for idx in range(pool_features.shape[0]):
        gain = _marginal_gain(sim_pool[:, idx], coverage, influence)
        c = max(cost(pool_keys[idx]), 1e-9)
        heapq.heappush(heap, (-gain / c, idx, 0))

    selected: List[int] = []
    selected_cost = 0.0
    iter_idx = 0

    while heap and selected_cost < round_budget:
        iter_idx += 1
        neg_score, candidate_idx, last_iter = heapq.heappop(heap)
        c = max(cost(pool_keys[candidate_idx]), 1e-9)
        if selected_cost + c > round_budget:
            continue

        if last_iter < iter_idx - 1:
            true_gain = _marginal_gain(sim_pool[:, candidate_idx], coverage, influence)
            true_score = true_gain / c
            if heap and true_score < -heap[0][0]:
                heapq.heappush(heap, (-true_score, candidate_idx, iter_idx))
                continue
        else:
            true_gain = -neg_score * c

        if true_gain <= 0:
            break

        selected.append(candidate_idx)
        selected_cost += c
        if pick_trace is not None:
            sim_to_VT = sim_pool[:, candidate_idx]
            top_v_idx = int(torch.argmax(influence * sim_to_VT).item()) if influence.numel() else -1
            pick_trace.append({
                "pool_index": int(candidate_idx),
                "candidate_key": pool_keys[candidate_idx],
                "marginal_gain": float(true_gain),
                "cost": float(c),
                "cumulative_cost_in_round": float(selected_cost),
                "max_sim_to_VT": float(sim_to_VT.max().item()),
                "mean_sim_to_VT": float(sim_to_VT.mean().item()),
                "most_influential_v_index": top_v_idx,
                "rank_in_round": len(selected) - 1,
            })
        coverage = torch.maximum(coverage, sim_pool[:, candidate_idx])

    return selected, coverage


# ---------------------------------------------------------------------------
# Plain greedy with adaptation oracle (Version 2)
# ---------------------------------------------------------------------------


def _clone_state_dict(predictor: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in predictor.state_dict().items()}


def _restore_state_dict(predictor: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    predictor.load_state_dict(state)


def _adapt_K_steps(
    predictor: ActiveEvaluator,
    examples: Sequence[SelectionExample],
    K_steps: int,
    inner_lr: float,
) -> None:
    """Run ``K_steps`` SGD steps on ``examples`` in-place.

    The caller is expected to clone/restore the state dict around this call.
    """
    if not examples or K_steps <= 0:
        return
    descriptors = torch.cat([_zero_context(predictor, ex.descriptor) for ex in examples], dim=0)
    labels = torch.cat([ex.label for ex in examples], dim=0)
    optim = torch.optim.SGD(predictor.parameters(), lr=inner_lr)
    predictor.train()
    for _ in tqdm(range(K_steps), desc="Adapt", leave=False, disable=K_steps <= 1):
        optim.zero_grad()
        preds = predictor(descriptors)
        loss = F.mse_loss(preds, labels)
        loss.backward()
        optim.step()
    predictor.eval()


def _val_loss(predictor: ActiveEvaluator, val: Sequence[SelectionExample]) -> float:
    if not val:
        return 0.0
    descriptors = torch.cat([_zero_context(predictor, ex.descriptor) for ex in val], dim=0)
    labels = torch.cat([ex.label for ex in val], dim=0)
    predictor.eval()
    with torch.no_grad():
        preds = predictor(descriptors)
        loss = F.mse_loss(preds, labels)
    return float(loss.item())


def direct_greedy_validation(
    predictor: ActiveEvaluator,
    pool: Sequence[SelectionExample],
    pool_features: torch.Tensor,
    support_S: Sequence[SelectionExample],
    val_VT: Sequence[SelectionExample],
    val_features: torch.Tensor,
    influence: torch.Tensor,
    tau: float,
    K_steps: int,
    inner_lr: float,
    round_budget: int,
    cost_fn: Optional[Callable[[str], float]] = None,
    max_candidates_evaluated: int = 100,
    eval_trace: Optional[List[List[int]]] = None,
    pick_trace: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[int], List[float]]:
    """Plain greedy on validation-loss reduction with FASS-style pre-filtering.

    For each round step, the V1 surrogate ranks the pool and only the top
    ``max_candidates_evaluated`` candidates are evaluated under the K-step
    adaptation oracle (Wei, Iyer & Bilmes 2015).

    Args:
        predictor: Predictor whose state will be temporarily mutated.
        pool: Candidate examples remaining for selection.
        pool_features: Features aligned with ``pool``.
        support_S: Current adaptation set (S_0 plus already-selected S).
        val_VT: Narrowed validation set.
        val_features: Features aligned with ``val_VT``.
        influence: Influence weights for ``val_VT``.
        tau: Kernel bandwidth.
        K_steps: Number of inner adaptation steps.
        inner_lr: Inner-loop learning rate.
        round_budget: Number to add this round.
        cost_fn: Optional cost function.
        max_candidates_evaluated: Cap on direct evaluations per pick.

    Returns:
        ``(selected_indices, marginal_losses)``.
    """
    if round_budget <= 0 or not pool or not val_VT:
        return [], []

    cost = cost_fn or (lambda _key: 1.0)
    state_snapshot = _clone_state_dict(predictor)

    selected: List[int] = []
    selected_cost = 0.0
    marginal_losses: List[float] = []

    coverage = _coverage_from(
        val_features,
        compute_embeddings(predictor, support_S) if support_S else torch.empty(0, val_features.shape[1], device=val_features.device),
        tau,
    )

    _restore_state_dict(predictor, state_snapshot)
    _adapt_K_steps(predictor, list(support_S), K_steps, inner_lr)
    base_loss = _val_loss(predictor, val_VT)
    _restore_state_dict(predictor, state_snapshot)

    sim_pool = _gaussian_similarity(val_features, pool_features, tau)

    while selected_cost < round_budget:
        remaining = [i for i in range(len(pool)) if i not in selected]
        if not remaining:
            break
        gains = []
        for idx in remaining:
            gain = _marginal_gain(sim_pool[:, idx], coverage, influence)
            gains.append((gain, idx))
        gains.sort(key=lambda kv: kv[0], reverse=True)
        candidate_indices = [idx for _, idx in gains[:max_candidates_evaluated]]
        if eval_trace is not None:
            eval_trace.append(list(candidate_indices))

        best_gain = 0.0
        best_idx: Optional[int] = None
        best_loss: Optional[float] = None
        for idx in candidate_indices:
            c = max(cost(pool[idx].key), 1e-9)
            if selected_cost + c > round_budget:
                continue
            extension = list(support_S) + [pool[i] for i in selected] + [pool[idx]]
            _restore_state_dict(predictor, state_snapshot)
            _adapt_K_steps(predictor, extension, K_steps, inner_lr)
            new_loss = _val_loss(predictor, val_VT)
            _restore_state_dict(predictor, state_snapshot)
            gain = (base_loss - new_loss) / c
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                best_loss = new_loss

        if best_idx is None or best_gain <= 0:
            break

        c_pick = max(cost(pool[best_idx].key), 1e-9)
        prev_loss = base_loss
        selected.append(best_idx)
        selected_cost += c_pick
        marginal_losses.append(best_loss if best_loss is not None else base_loss)
        if pick_trace is not None:
            pick_trace.append({
                "pool_index": int(best_idx),
                "candidate_key": pool[best_idx].key,
                "loss_before": float(prev_loss),
                "loss_after": float(best_loss if best_loss is not None else prev_loss),
                "loss_reduction": float(prev_loss - (best_loss if best_loss is not None else prev_loss)),
                "marginal_gain": float(best_gain),
                "cost": float(c_pick),
                "cumulative_cost_in_round": float(selected_cost),
                "n_candidates_evaluated": len(candidate_indices),
                "rank_in_round": len(selected) - 1,
            })
        coverage = torch.maximum(coverage, sim_pool[:, best_idx])
        base_loss = best_loss if best_loss is not None else base_loss

    _restore_state_dict(predictor, state_snapshot)
    return selected, marginal_losses


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def select_extension(
    predictor: ActiveEvaluator,
    support_S0: Sequence[SelectionExample],
    pool_U: Sequence[SelectionExample],
    val_V: Sequence[SelectionExample],
    target_T: Sequence[SelectionExample],
    *,
    method: str = "v1_facility",
    n_rounds: int = 5,
    budget_fraction: float = 0.10,
    budget_absolute: Optional[int] = None,
    K_steps: int = 5,
    inner_lr: float = 0.01,
    quantile_q: float = 0.7,
    cost_fn: Optional[Callable[[str], float]] = None,
    seed: int = 42,
    max_candidates_evaluated: int = 100,
    log_dir: Optional[Path] = None,
) -> Tuple[List[SelectionExample], Dict[str, Any]]:
    """Select an extension set ``S`` from ``pool_U`` to augment ``support_S0``.

    The selection always satisfies ``set(S) & set(val_V) == set()``.

    Args:
        predictor: Trained ActiveEvaluator MLP.
        support_S0: Initial adaptation set (per-task support).
        pool_U: Candidate pool, disjoint from val_V.
        val_V: Validation set with labels.
        target_T: Unlabeled target features.
        method: "v1_facility" or "v2_direct".
        n_rounds: Active-learning rounds.
        budget_fraction: Fraction of pool size as total budget.
        budget_absolute: Absolute budget; overrides budget_fraction when set.
        K_steps: Inner adaptation steps (used by V2).
        inner_lr: Inner adaptation learning rate (used by V2).
        quantile_q: Narrowing quantile for V_T.
        cost_fn: Optional candidate-cost function (default: cardinality).
        seed: Random seed.
        max_candidates_evaluated: Cap for V2 direct evaluations.
        log_dir: Directory to save diagnostics; created when missing.

    Returns:
        ``(selected_S, log)`` where log is a JSON-serializable diagnostics dict.
    """
    if method not in {"v1_facility", "v2_direct"}:
        raise ValueError(f"Unknown selection method '{method}'.")

    torch.manual_seed(seed)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

    val_keys = {ex.key for ex in val_V}
    pool_U = [ex for ex in pool_U if ex.key not in val_keys]
    pool_U_keys = {ex.key for ex in pool_U}

    plan = compute_budget(
        pool_size=len(pool_U),
        n_rounds=n_rounds,
        budget_fraction=budget_fraction,
        budget_absolute=budget_absolute,
    )
    if log_dir is not None:
        plan.save(log_dir / "budget_plan.json")

    cat_examples = list(val_V) + list(pool_U) + list(support_S0) + list(target_T)
    feat_all = compute_embeddings(predictor, cat_examples) if cat_examples else torch.empty(0)
    tau = compute_bandwidth_median(feat_all, seed=seed) if feat_all.numel() else 1.0

    val_VT, narrow_diag = narrow_validation(
        val_V,
        target_T,
        predictor,
        tau=tau,
        quantile_q=quantile_q,
        pool_U_keys=pool_U_keys,
    )
    assert all(ex.key not in pool_U_keys for ex in val_VT), "V_T must remain disjoint from U."
    if log_dir is not None:
        with (log_dir / "narrowing_diagnostics.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(narrow_diag), f, indent=2)

    selected_S: List[SelectionExample] = []
    selected_keys: set[str] = set()
    val_VT_keys = {ex.key for ex in val_VT}
    remaining_pool = list(pool_U)
    trajectory: List[Dict[str, Any]] = []
    base_state = _clone_state_dict(predictor)

    method_iter = tqdm(range(plan.n_rounds), desc=f"Active-{method}", leave=False)
    for round_idx in method_iter:
        round_budget = plan.per_round[round_idx]
        round_start = time.time()
        timings: Dict[str, float] = {}

        # Re-fit predictor on current S = S_0 ∪ selected_S so phi_round
        # reflects the latest adapted state, then re-embed V_T and pool.
        adapt_t0 = time.time()
        _restore_state_dict(predictor, base_state)
        _adapt_K_steps(predictor, list(support_S0) + selected_S, K_steps, inner_lr)
        timings["adapt_seconds"] = time.time() - adapt_t0

        val_loss_before = _val_loss(predictor, val_VT) if val_VT else 0.0

        if round_budget <= 0 or not remaining_pool or not val_VT:
            _restore_state_dict(predictor, base_state)
            trajectory.append({
                "round": round_idx,
                "round_budget": round_budget,
                "selected_keys": [],
                "val_loss_before": val_loss_before,
                "val_loss_after": val_loss_before,
                "elapsed_seconds": time.time() - round_start,
                "timings": timings,
            })
            continue

        embed_t0 = time.time()
        val_features = compute_embeddings(predictor, val_VT, show_progress=len(val_VT) > 256, desc="Embed V_T")
        pool_features = compute_embeddings(predictor, remaining_pool, show_progress=len(remaining_pool) > 256, desc="Embed U")
        support_features = compute_embeddings(predictor, list(support_S0) + selected_S)
        coverage = _coverage_from(val_features, support_features, tau)
        timings["embed_seconds"] = time.time() - embed_t0

        infl_t0 = time.time()
        influence = compute_influence_weights(predictor, val_VT)
        timings["influence_seconds"] = time.time() - infl_t0

        pool_keys = [ex.key for ex in remaining_pool]
        greedy_t0 = time.time()
        round_pick_trace: List[Dict[str, Any]] = []
        if method == "v1_facility":
            picked, coverage = lazy_greedy_facility(
                pool_features=pool_features,
                pool_keys=pool_keys,
                coverage_init=coverage,
                val_features=val_features,
                influence=influence,
                tau=tau,
                round_budget=round_budget,
                cost_fn=cost_fn,
                pick_trace=round_pick_trace,
            )
        else:
            picked, _losses = direct_greedy_validation(
                predictor=predictor,
                pool=remaining_pool,
                pool_features=pool_features,
                support_S=list(support_S0) + selected_S,
                val_VT=val_VT,
                val_features=val_features,
                influence=influence,
                tau=tau,
                K_steps=K_steps,
                inner_lr=inner_lr,
                round_budget=round_budget,
                cost_fn=cost_fn,
                max_candidates_evaluated=max_candidates_evaluated,
                pick_trace=round_pick_trace,
            )
            if picked:
                picked_features = pool_features[picked]
                coverage = torch.maximum(coverage, _gaussian_similarity(val_features, picked_features, tau).max(dim=1).values)
        timings["greedy_seconds"] = time.time() - greedy_t0

        # Augment each pick with the source training-model alias parsed
        # from the candidate key so users can read the trace directly.
        for entry in round_pick_trace:
            ck = entry.get("candidate_key", "")
            entry["source_model"] = ck.split("::", 1)[0] if "::" in ck else ck

        new_examples = [remaining_pool[i] for i in picked]
        selected_S.extend(new_examples)
        selected_keys.update(ex.key for ex in new_examples)
        remaining_pool = [ex for ex in remaining_pool if ex.key not in selected_keys]

        # Spec invariants — disjoint from V and V_T at the end of every round.
        assert len(set(ex.key for ex in selected_S) & val_keys) == 0, "S ∩ V must be empty."
        assert len(set(ex.key for ex in selected_S) & val_VT_keys) == 0, "S ∩ V_T must be empty."

        # Restore base state so val_loss_after reflects K-step adapt on the
        # newly-extended S starting from the meta-trained predictor.
        _restore_state_dict(predictor, base_state)
        val_loss_after = _val_loss_after_adaptation(
            predictor,
            list(support_S0) + selected_S,
            val_VT,
            K_steps,
            inner_lr,
        )
        _restore_state_dict(predictor, base_state)
        round_total_cost = sum(p.get("cost", 1.0) for p in round_pick_trace)
        round_total_gain = sum(p.get("marginal_gain", 0.0) for p in round_pick_trace)
        trajectory.append({
            "round": round_idx,
            "round_budget": round_budget,
            "selected_keys": [ex.key for ex in new_examples],
            "selected_source_models": [
                p.get("source_model") for p in round_pick_trace
            ],
            "picks": round_pick_trace,
            "round_cost_paid": float(round_total_cost),
            "round_gain_total": float(round_total_gain),
            "influence_stats": {
                "mean": float(influence.mean().item()) if influence.numel() else 0.0,
                "min": float(influence.min().item()) if influence.numel() else 0.0,
                "max": float(influence.max().item()) if influence.numel() else 0.0,
                "n": int(influence.numel()),
            },
            "val_loss_before": val_loss_before,
            "val_loss_after": val_loss_after,
            "elapsed_seconds": time.time() - round_start,
            "timings": timings,
        })

    # Always restore the meta-trained state so the caller sees an unmutated predictor.
    _restore_state_dict(predictor, base_state)

    # Aggregate per-pick records across rounds for the summary file.
    all_picks: List[Dict[str, Any]] = []
    for round_entry in trajectory:
        for pick in round_entry.get("picks", []):
            all_picks.append({**pick, "round": round_entry["round"]})

    cost_paid_total = sum(p.get("cost", 1.0) for p in all_picks)
    gain_total = sum(p.get("marginal_gain", 0.0) for p in all_picks)
    final_val_loss = (
        trajectory[-1]["val_loss_after"] if trajectory else None
    )
    total_time = sum(r.get("elapsed_seconds", 0.0) for r in trajectory)

    selected_records = [
        {
            "key": ex.key,
            "source_model": ex.key.split("::", 1)[0] if "::" in ex.key else ex.key,
            "true_label": float(ex.label.item()) if ex.label is not None else None,
            "cost": float((cost_fn or (lambda _k: 1.0))(ex.key)),
        }
        for ex in selected_S
    ]

    summary = {
        "method": method,
        "tau": tau,
        "cost_fn_name": plan.cost_fn_name,
        "budget_total": plan.total,
        "cost_paid_total": float(cost_paid_total),
        "cost_remaining": float(plan.total - cost_paid_total),
        "gain_total": float(gain_total),
        "n_selected": len(selected_S),
        "pool_size": len(pool_U),
        "val_size": len(val_V),
        "val_VT_size": narrow_diag.n_val_narrowed,
        "final_val_loss": final_val_loss,
        "total_selection_seconds": total_time,
        "n_rounds_executed": len(trajectory),
        "source_models_chosen": [r["source_model"] for r in selected_records],
        "rationale": (
            "V1 picks maximize the influence-weighted facility-location surrogate: "
            "each pick has the largest sum over V_T of I(v) * (sim(v, candidate) - max sim(v, S)). "
            "I(v) is the gradient-norm influence weight per validation example. "
            "Cost is c(s)=1 (cardinality) by default."
            if method == "v1_facility"
            else
            "V2 picks maximize per-cost reduction in validation loss after K-step "
            "adaptation on S_0 ∪ S ∪ {candidate}. Candidates are pre-filtered by V1 "
            "ranking (FASS) and only the top-N are K-step-evaluated. Cost is c(s)=1 by default."
        ),
    }

    log = {
        "method": method,
        "tau": tau,
        "budget_plan": asdict(plan),
        "narrowing": asdict(narrow_diag),
        "trajectory": trajectory,
        "selected": selected_records,
        "selected_keys": [ex.key for ex in selected_S],
        "selected_count": len(selected_S),
        "summary": summary,
    }
    if log_dir is not None:
        with (log_dir / "selection_trajectory.json").open("w", encoding="utf-8") as f:
            json.dump({"trajectory": trajectory, "tau": tau, "method": method}, f, indent=2)
        with (log_dir / "selected_examples.json").open("w", encoding="utf-8") as f:
            json.dump({"selected": selected_records, "selected_keys": log["selected_keys"]}, f, indent=2)
        with (log_dir / "selection_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return selected_S, log


def _val_loss_after_adaptation(
    predictor: ActiveEvaluator,
    extended_support: Sequence[SelectionExample],
    val_VT: Sequence[SelectionExample],
    K_steps: int,
    inner_lr: float,
) -> float:
    """Run a temporary K-step adaptation and report validation loss."""
    if not val_VT:
        return 0.0
    state = _clone_state_dict(predictor)
    try:
        _adapt_K_steps(predictor, list(extended_support), K_steps, inner_lr)
        return _val_loss(predictor, val_VT)
    finally:
        _restore_state_dict(predictor, state)


__all__ = [
    "SelectionExample",
    "select_extension",
    "compute_embeddings",
    "compute_bandwidth_median",
    "narrow_validation",
    "compute_influence_weights",
    "lazy_greedy_facility",
    "direct_greedy_validation",
]
