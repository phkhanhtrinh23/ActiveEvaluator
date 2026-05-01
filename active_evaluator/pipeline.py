"""End-to-end pipeline to train and evaluate ActiveEvaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import gc
import numpy as np
import torch
from shift_descriptor.config import ModelSpec, default_model_specs
from tqdm.auto import tqdm

from .accuracy import ModelAccuracySource
from .data_utils import (
    PromptSplit,
    build_prompt_splits,
    load_prompted_texts,
    load_raw_dataset,
    maybe_cap_examples,
)
from .descriptors import DEFAULT_FEATURE_ORDER, compute_shift_descriptor
from .embedding_cache import EmbeddingCache, EmbeddingCacheConfig
from .evaluation import build_prediction_records
from .generation import GenerationSettings, SQLGenerator
from .active_selection import SelectionExample, select_extension
from .meta_learning import ActiveEvaluatorLearner, MetaLearningConfig, ShiftDescriptorTask
from .model import ActiveEvaluator

def clear_cuda_cache() -> None:
    """Free Python and CUDA caches to avoid OOM between stages.

    Returns:
        None
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _compute_descriptor_stats(tasks: Sequence[ShiftDescriptorTask]) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mean/std across all descriptors to normalize features consistently.

    Args:
        tasks: Tasks whose descriptors provide statistics.

    Returns:
        (mean, std) tensors shaped like a single descriptor.
    """
    tensors = []
    for task in tasks:
        tensors.append(task.support_descriptor)
        tensors.append(task.query_descriptor)
        if task.transfer_descriptor is not None:
            tensors.append(task.transfer_descriptor)
    all_desc = torch.cat(tensors, dim=0)
    mean = all_desc.mean(dim=0, keepdim=True)
    std = all_desc.std(dim=0, keepdim=True).clamp(min=1e-6)
    return mean, std


def _apply_descriptor_norm(tasks: Sequence[ShiftDescriptorTask], mean: torch.Tensor, std: torch.Tensor) -> None:
    """Normalize task descriptors in-place using provided statistics.

    Args:
        tasks: Tasks to normalize.
        mean: Descriptor mean.
        std: Descriptor standard deviation (non-zero).
    """
    for task in tasks:
        task.support_descriptor.sub_(mean).div_(std)
        task.query_descriptor.sub_(mean).div_(std)
        if task.transfer_descriptor is not None:
            task.transfer_descriptor.sub_(mean).div_(std)


def _parse_model_entry(raw: str) -> ModelSpec:
    """Parse optional alias and remote flag from a model string.

    Args:
        raw: CLI model spec (alias=model_id or model_id[:remote]).

    Returns:
        A ModelSpec with alias populated when provided.
    """
    alias = None
    entry = raw
    if "=" in raw:
        alias, entry = raw.split("=", 1)
    if entry.endswith(":remote"):
        entry = entry[: -len(":remote")]
    return ModelSpec(model_id=entry, alias=alias, trust_remote_code=True)


def parse_model_specs(raw_list: Sequence[str] | None) -> List[ModelSpec]:
    """Convert CLI model strings into ModelSpec objects, or use defaults.

    Args:
        raw_list: Optional sequence of model strings.

    Returns:
        Parsed model specs or the default catalog.
    """
    if raw_list:
        return [_parse_model_entry(entry) for entry in raw_list]
    return default_model_specs()


def parse_args() -> argparse.Namespace:
    """Build CLI for the ActiveEvaluator pipeline and return parsed args.

    Returns:
        Parsed argparse Namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default="data/sft_spider_train_text2sql.json", help="Path to training JSON.")
    parser.add_argument("--dev-path", default="data/sft_spider_dev_text2sql.json", help="Path to dev/test JSON.")
    parser.add_argument("--output-dir", default="outputs/active_evaluator", help="Directory for ActiveEvaluator artifacts.")
    parser.add_argument("--embedding-dir", default="outputs/active_evaluator/embeddings", help="Embedding cache directory.")
    parser.add_argument("--model-ids", nargs="*", default=[
            # Llama 3.2 family (Meta)
            "meta-llama/Llama-3.2-1B",
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B",

            # TinyLlama (Llama-compatible)
            "TinyLlama/TinyLlama_v1.1",
            "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",

            # QwenCoder (XiYanSQL is a finetune of Qwen/Qwen2.x Coder)
            "Qwen/Qwen2.5-Coder-3B-Instruct",
            "Qwen/Qwen2.5-Coder-1.5B",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "XGenerationLab/XiYanSQL-QwenCoder-3B-2502",

            # StableLM-2 (Stability AI)
            "stabilityai/stablelm-2-1_6b-chat",
            "stabilityai/stablelm-2-zephyr-1_6b",

            # Lightweight Text2SQL models (causal LMs)
            "cycloneboy/SLM-SQL-0.5B",
            "cycloneboy/SLM-SQL-0.6B",
            "cycloneboy/CscSQL-Merge-Qwen2.5-Coder-0.5B-Instruct",

            # # DeepSeek Coder family
            "deepseek-ai/deepseek-coder-1.3b-base",
            "deepseek-ai/deepseek-coder-6.7b-base",
            "deepseek-ai/deepseek-coder-6.7b-instruct",
        ], help="Optional HF model ids (alias=model_id or alias=model_id:remote).")
    parser.add_argument("--test-model-ids", 
                        nargs="*",
                        default=["meta-llama/Llama-3.2-3B-Instruct",
                                 "Qwen/Qwen3-0.6B",
                                 "Gensyn/Qwen2.5-0.5B-Instruct",
                                 "Qwen/Qwen2.5-0.5B-Instruct",
                                 "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                                 "Qwen/Qwen3-0.6B-Base",
                                 "deepseek-ai/deepseek-coder-1.3b-instruct",
                                 "Qwen/Qwen2-0.5B",
                                 "unsloth/Llama-3.2-1B-Instruct"
                                ], 
                        help="Extra model ids evaluated only after meta-training.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per model during embedding extraction.")
    parser.add_argument("--max-length", type=int, default=512, help="Max token length for embeddings.")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank (<=0 disables).")
    parser.add_argument("--num-projections", type=int, default=128, help="Projection count for SWD.")
    parser.add_argument("--subsample-limit", type=int, default=0, help="Optional cap on embeddings per split before metrics.")
    parser.add_argument("--subsample-seed", type=int, default=13, help="Random seed for embedding subsampling.")
    parser.add_argument("--prompt-template", default=None, help="Path to prompt template (defaults to repo template).")
    parser.add_argument("--context-fields", nargs="*", default=("evidence", "matched_contents", "text"))
    parser.add_argument("--use-plain-text", action="store_true", help="Disable prompt templating.")
    parser.add_argument("--text-field", default="text", help="Field used when --use-plain-text is enabled.")
    parser.add_argument("--split-ratios", nargs=3, type=float, default=(0.6, 0.2, 0.2), help="Ratios for meta_train/meta_val/meta_test.")
    parser.add_argument("--split-seed", type=int, default=13, help="Seed for dataset split.")
    parser.add_argument("--max-train-samples", type=int, default=0, help="Optional cap on training samples before splitting.")
    parser.add_argument("--max-dev-samples", type=int, default=0, help="Optional cap on dev samples.")
    parser.add_argument("--inner-lr", type=float, default=0.01, help="Inner-loop lr.")
    parser.add_argument("--outer-lr", type=float, default=1e-3, help="Outer-loop lr.")
    parser.add_argument("--inner-steps", type=int, default=5, help="Inner updates per task.")
    parser.add_argument("--context-dim", type=int, default=32, help="Dimensionality of the CAVIA context vector.")
    parser.add_argument("--epochs", type=int, default=500, help="Meta-training epochs.")
    parser.add_argument("--tasks-per-batch", type=int, default=4, help="Tasks per meta-batch.")
    parser.add_argument("--device", default=None, help="Torch device override.")
    parser.add_argument("--meta-val-key", default="meta_val", help="Accuracy key for meta validation split.")
    parser.add_argument("--meta-test-key", default="meta_test", help="Accuracy key for meta testing split.")
    parser.add_argument("--dev-key", default="dev", help="Accuracy key for real dev/test split.")
    parser.add_argument("--eval-inner-steps", type=int, default=5, help="Inner-loop steps at evaluation time.")
    parser.add_argument("--gen-max-new-tokens", type=int, default=256, help="Max new tokens for SQL generation.")
    parser.add_argument("--gen-temperature", type=float, default=0.0, help="Softmax temperature during decoding.")
    parser.add_argument("--gen-top-p", type=float, default=0.9, help="Nucleus sampling top-p.")
    parser.add_argument("--gen-repetition-penalty", type=float, default=1.0, help="Repetition penalty for decoding.")
    parser.add_argument("--save-meta-preds", action="store_true", help="Whether to persist meta-test predictions (default True).")
    parser.add_argument("--no-save-meta-preds", dest="save_meta_preds", action="store_false", help=argparse.SUPPRESS)
    parser.set_defaults(save_meta_preds=True)
    parser.add_argument("--meta-reg-lambda", type=float, default=1e-4, help="L2 regularization on meta-parameters.")
    parser.add_argument("--meta-reg-beta", type=float, default=1e-3, help="KL-style meta regularizer on outer loss.")
    parser.add_argument("--use-active-selection", action="store_true",
                        help="Enable active extension of the adaptation set at test time.")
    parser.add_argument("--selection-method",
                        choices=["v1_facility", "v2_direct", "v3_gradmatch"],
                        default="v1_facility",
                        help="Active-selection algorithm to use. v3_gradmatch is "
                             "GRAD-MATCH OMP (Killamsetty et al. 2021).")
    parser.add_argument("--selection-gradmatch-lambda", type=float, default=1e-3,
                        help="L2 regularizer on OMP weights for GRAD-MATCH.")
    parser.add_argument("--selection-n-rounds", type=int, default=5,
                        help="Active-learning rounds at test time.")
    parser.add_argument("--selection-budget-fraction", type=float, default=0.10,
                        help="Total selection budget as a fraction of |U|.")
    parser.add_argument("--selection-budget-absolute", type=int, default=None,
                        help="Optional absolute selection budget; overrides fraction.")
    parser.add_argument("--selection-K-steps", type=int, default=None,
                        help="Inner adaptation steps for active selection (defaults to --eval-inner-steps).")
    parser.add_argument("--selection-narrowing-quantile", type=float, default=0.7,
                        help="Quantile q for narrowing V to V_T.")
    parser.add_argument("--selection-pool-narrow-quantile", type=float, default=0.0,
                        help="Quantile q for target-aware narrowing of pool U (0.0 disables).")
    parser.add_argument("--selection-inner-lr", type=float, default=None,
                        help="Override inner-lr inside selection's K-step adapt; defaults to --inner-lr.")
    parser.add_argument("--selection-weight-decay", type=float, default=0.0,
                        help="L2 weight decay applied during selection's K-step adapt; "
                             "regularizes V2 to prevent overfitting a tiny V_T.")
    parser.add_argument("--selection-early-stop-patience", type=int, default=0,
                        help="Stop selection's K-step adapt after this many non-improving steps "
                             "on V_T (0 disables; only used by V2's per-candidate evaluation).")
    parser.add_argument("--selection-max-candidates-evaluated", type=int, default=100,
                        help="Cap on direct evaluations per pick (Version 2 only).")
    parser.add_argument("--selection-seed", type=int, default=42,
                        help="Random seed used inside active selection.")
    return parser.parse_args()


def build_tasks(
    models: Sequence[ModelSpec],
    splits: Mapping[str, PromptSplit],
    dev_split: PromptSplit,
    *,
    cache: EmbeddingCache,
    accuracy: ModelAccuracySource,
    meta_val_key: str,
    meta_test_key: str,
    dev_key: str,
    num_projections: int,
) -> List[ShiftDescriptorTask]:
    """Construct per-model meta-learning tasks from cached descriptors and accuracy labels.

    Args:
        models: Model specs used to fetch embeddings and labels.
        splits: Prompt splits for meta-train/val/test.
        dev_split: Dev prompts treated as real test.
        cache: Embedding cache for descriptor construction.
        accuracy: Source mapping model -> accuracy.
        meta_val_key: Key to pull validation accuracy.
        meta_test_key: Key to pull meta-test accuracy.
        dev_key: Key to pull dev accuracy.
        num_projections: Projection count for SWD descriptors.

    Returns:
        List of per-model ShiftDescriptorTask entries.
    """
    tasks: List[ShiftDescriptorTask] = []
    for model in tqdm(models, desc="Build tasks", leave=False):
        model_name = model.alias or model.model_id
        emb_meta_train = cache.load_or_compute(model, "meta_train", splits["meta_train"].prompts)
        emb_meta_val = cache.load_or_compute(model, "meta_val", splits["meta_val"].prompts)
        emb_meta_test = cache.load_or_compute(model, "meta_test", splits["meta_test"].prompts)
        emb_dev = cache.load_or_compute(model, dev_split.name, dev_split.prompts)

        support_desc = compute_shift_descriptor(
            model_name,
            "meta_train",
            emb_meta_train,
            "meta_val",
            emb_meta_val,
            num_projections=num_projections,
        )
        query_desc = compute_shift_descriptor(
            model_name,
            "meta_train",
            emb_meta_train,
            "meta_test",
            emb_meta_test,
            num_projections=num_projections,
        )
        transfer_desc = compute_shift_descriptor(
            model_name,
            "meta_train",
            emb_meta_train,
            dev_split.name,
            emb_dev,
            num_projections=num_projections,
        )

        support_label = accuracy.get(model_name, meta_val_key)
        query_label = accuracy.get(model_name, meta_test_key)
        transfer_label = accuracy.get(model_name, dev_key)

        task = ShiftDescriptorTask.from_descriptors(
            model_name=model_name,
            support=support_desc,
            support_label=support_label,
            query=query_desc,
            query_label=query_label,
            transfer=transfer_desc,
            transfer_label=transfer_label,
            device=cache.device,
            feature_order=DEFAULT_FEATURE_ORDER,
        )
        tasks.append(task)
        cache.clear_extractors()
    return tasks


def save_json(path: Path, payload: Dict) -> None:
    """Persist a JSON payload, ensuring parent directories exist.

    Args:
        path: Destination file path.
        payload: Serializable object to dump.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def sanitize_model_name(spec: ModelSpec) -> str:
    """Create a filesystem-safe model identifier.

    Args:
        spec: Model spec.

    Returns:
        Alias/model id with slashes replaced by dashes.
    """
    alias = spec.alias or spec.model_id
    return alias.replace("/", "-")


def create_dev_split(prompts: List[str], samples: List[dict]) -> PromptSplit:
    """Package dev prompts into a PromptSplit with aligned indices.

    Args:
        prompts: Rendered prompts.
        samples: Original dataset entries aligned to prompts.

    Returns:
        A PromptSplit named "dev".
    """
    indices = list(range(len(prompts)))
    return PromptSplit(name="dev", prompts=prompts, indices=indices, samples=samples)


def evaluate_split_predictions(
    generator: SQLGenerator,
    split: PromptSplit,
    output_root: Path,
    model_name: str,
    *,
    db_root: Path | None = None,
) -> Dict[str, float]:
    """Generate SQL for a split and write metrics/predictions to disk.

    Args:
        generator: SQLGenerator instance.
        split: PromptSplit to evaluate.
        output_root: Directory to store split JSON.
        model_name: Human-readable model identifier.
        db_root: Optional DB path for execution evaluation.

    Returns:
        Metrics dictionary (including execution_accuracy).
    """
    preds = generator.generate(split.prompts)
    metrics, records = build_prediction_records(split.samples, preds, split.indices, db_root=db_root)
    payload = {
        "model": model_name,
        "split": split.name,
        "metrics": metrics,
        "sample_count": len(records),
        "predictions": records,
    }
    save_json(output_root / f"{split.name}.json", payload)
    return metrics


def maybe_load_cached_metrics(model_dir: Path, split_name: str, expected_samples: int) -> Dict[str, float] | None:
    """Return cached metrics when prediction count matches expectation.

    Args:
        model_dir: Directory containing cached split JSON.
        split_name: Split identifier (e.g., meta_val).
        expected_samples: Number of samples to validate cache.

    Returns:
        Metrics dict if cache is valid, otherwise None.
    """
    path = model_dir / f"{split_name}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    records = payload.get("predictions")
    sample_count = payload.get("sample_count")
    if isinstance(records, list):
        count = len(records)
    elif isinstance(sample_count, int):
        count = sample_count
    else:
        return None
    if count != expected_samples:
        return None
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return metrics


def run_inference_for_models(
    models: Sequence[ModelSpec],
    splits: Mapping[str, PromptSplit],
    dev_split: PromptSplit,
    *,
    output_dir: Path,
    gen_settings: GenerationSettings,
    device: str | None = None,
    meta_val_key: str,
    meta_test_key: str,
    dev_key: str,
    lora_r: int | None = None,
    db_root: Path | None = None,
) -> Dict[str, Dict[str, float]]:
    """Run SQL generation for all requested models and aggregate accuracy.

    Args:
        models: Model specs to evaluate.
        splits: Meta splits.
        dev_split: Dev split treated as real test.
        output_dir: Root directory for prediction artifacts.
        gen_settings: Decoding hyperparameters.
        device: Torch device override.
        meta_val_key: Accuracy key for meta validation.
        meta_test_key: Accuracy key for meta testing.
        dev_key: Accuracy key for real dev/test.
        lora_r: Optional LoRA rank for generation.
        db_root: Optional DB root for exec evaluation.

    Returns:
        Nested dict mapping model -> {split_key: execution_accuracy}.
    """
    predictions_root = output_dir / "predictions"
    predictions_root.mkdir(parents=True, exist_ok=True)
    accuracy_summary: Dict[str, Dict[str, float]] = {}
    for model in models:
        model_name = model.alias or model.model_id
        print(f"[ActiveEvaluator] Generating SQL for {model_name}...")
        model_dir = predictions_root / sanitize_model_name(model)
        model_dir.mkdir(exist_ok=True)
        model_accs: Dict[str, float] = {}
        split_plan = [
            (meta_val_key, splits["meta_val"]),
            (meta_test_key, splits["meta_test"]),
            (dev_key, dev_split),
        ]
        generator: SQLGenerator | None = None
        try:
            for key_name, split in split_plan:
                cached = maybe_load_cached_metrics(model_dir, split.name, len(split.indices))
                if cached is not None:
                    model_accs[key_name] = cached["execution_accuracy"]
                    continue
                if generator is None:
                    generator = SQLGenerator(model, gen_settings, device=device, lora_r=lora_r)
                metrics = evaluate_split_predictions(
                    generator,
                    split,
                    model_dir,
                    model_name,
                    db_root=db_root,
                )
                model_accs[key_name] = metrics["execution_accuracy"]
        finally:
            if generator is not None:
                generator.shutdown()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        accuracy_summary[model_name] = model_accs
    return accuracy_summary


def _task_to_support_example(task: ShiftDescriptorTask, *, suffix: str = "support") -> SelectionExample:
    """Wrap a task's support pair as a SelectionExample for active selection."""
    return SelectionExample(
        key=f"{task.model_name}::{suffix}",
        descriptor=task.support_descriptor.detach().clone(),
        label=task.support_label.detach().clone(),
    )


def _task_to_target_example(task: ShiftDescriptorTask) -> SelectionExample:
    """Wrap a task's query (meta-test) descriptor as an unlabeled target example."""
    return SelectionExample(
        key=f"{task.model_name}::target",
        descriptor=task.query_descriptor.detach().clone(),
        label=None,
    )


def extend_test_tasks_with_active_selection(
    *,
    meta_learner: ActiveEvaluatorLearner,
    test_tasks: Sequence[ShiftDescriptorTask],
    train_tasks: Sequence[ShiftDescriptorTask],
    val_tasks: Sequence[ShiftDescriptorTask],
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Run active selection per test task and extend its support tensors in-place.

    The pool U is built from ``train_tasks`` (their support pairs), the
    validation V from ``val_tasks``, and the unlabeled target T from each test
    task's query descriptor. The selected extension is concatenated onto the
    test task's ``support_descriptor`` / ``support_label`` tensors so that the
    standard meta-learner adaptation loop picks them up unchanged.
    """
    K_steps = args.selection_K_steps if args.selection_K_steps is not None else args.eval_inner_steps
    pool_U = [_task_to_support_example(t, suffix="train_support") for t in train_tasks]
    val_V = [_task_to_support_example(t, suffix="val_support") for t in val_tasks]
    val_keys = {ex.key for ex in val_V}
    pool_U = [ex for ex in pool_U if ex.key not in val_keys]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for task in test_tasks:
        support_S0 = [
            SelectionExample(
                key=f"{task.model_name}::S0",
                descriptor=task.support_descriptor.detach().clone(),
                label=task.support_label.detach().clone(),
            )
        ]
        target_T = [_task_to_target_example(task)]
        log_dir = output_dir / f"selection_{sanitize_alias(task.model_name)}"
        selected_S, _log = select_extension(
            predictor=meta_learner.model,
            support_S0=support_S0,
            pool_U=pool_U,
            val_V=val_V,
            target_T=target_T,
            method=args.selection_method,
            n_rounds=args.selection_n_rounds,
            budget_fraction=args.selection_budget_fraction,
            budget_absolute=args.selection_budget_absolute,
            K_steps=K_steps,
            inner_lr=args.inner_lr,
            quantile_q=args.selection_narrowing_quantile,
            pool_narrow_q=args.selection_pool_narrow_quantile,
            weight_decay=args.selection_weight_decay,
            early_stop_patience=args.selection_early_stop_patience,
            selection_inner_lr=args.selection_inner_lr,
            gradmatch_lambda=args.selection_gradmatch_lambda,
            seed=args.selection_seed,
            max_candidates_evaluated=args.selection_max_candidates_evaluated,
            log_dir=log_dir,
        )

        if not selected_S:
            continue

        device = task.support_descriptor.device
        extra_desc = torch.cat([ex.descriptor.to(device) for ex in selected_S], dim=0)
        extra_label = torch.cat([ex.label.to(device) for ex in selected_S], dim=0)
        task.support_descriptor = torch.cat([task.support_descriptor, extra_desc], dim=0)
        task.support_label = torch.cat([task.support_label, extra_label], dim=0)
        print(
            f"[ActiveEvaluator] {task.model_name}: extended support set with "
            f"{len(selected_S)} examples (method={args.selection_method})."
        )


def sanitize_alias(name: str) -> str:
    return name.replace("/", "-")


def main() -> None:
    """Entry point: run generation, build tasks, train meta-learner, and evaluate.

    Returns:
        None
    """
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_samples = load_raw_dataset(args.train_path)
    train_prompts = load_prompted_texts(
        args.train_path,
        template_path=args.prompt_template,
        context_fields=args.context_fields,
        use_plain_text=args.use_plain_text,
        text_field=args.text_field,
    )
    train_prompts, train_samples = maybe_cap_examples(train_prompts, train_samples, args.max_train_samples)

    splits = build_prompt_splits(train_prompts, samples=train_samples, ratios=args.split_ratios, seed=args.split_seed)

    dev_samples = load_raw_dataset(args.dev_path)
    dev_prompts = load_prompted_texts(
        args.dev_path,
        template_path=args.prompt_template,
        context_fields=args.context_fields,
        use_plain_text=args.use_plain_text,
        text_field=args.text_field,
    )
    dev_prompts, dev_samples = maybe_cap_examples(dev_prompts, dev_samples, args.max_dev_samples)

    dev_split = create_dev_split(dev_prompts, dev_samples)

    train_models = parse_model_specs(args.model_ids)
    test_models = parse_model_specs(args.test_model_ids) if args.test_model_ids else []
    all_models = train_models + [m for m in test_models if m not in train_models]

    gen_settings = GenerationSettings(
        max_new_tokens=args.gen_max_new_tokens,
        temperature=args.gen_temperature,
        top_p=args.gen_top_p,
        repetition_penalty=args.gen_repetition_penalty,
    )
    accuracy_map = run_inference_for_models(
        all_models,
        splits,
        dev_split,
        output_dir=output_dir,
        gen_settings=gen_settings,
        device=args.device,
        meta_val_key=args.meta_val_key,
        meta_test_key=args.meta_test_key,
        dev_key=args.dev_key,
        lora_r=args.lora_r if args.lora_r and args.lora_r > 0 else None,
        db_root=Path("data/database"),
    )
    save_json(output_dir / "model_accuracies.json", accuracy_map)
    accuracy_source = ModelAccuracySource(accuracy_map=accuracy_map)

    cache_cfg = EmbeddingCacheConfig(
        output_dir=Path(args.embedding_dir),
        batch_size=args.batch_size,
        max_length=args.max_length,
        lora_r=args.lora_r if args.lora_r > 0 else None,
        device=args.device,
        max_points_per_split=args.subsample_limit or None,
        subsample_seed=args.subsample_seed,
    )
    cache = EmbeddingCache(cache_cfg)

    tasks = build_tasks(
        train_models,
        splits,
        dev_split,
        cache=cache,
        accuracy=accuracy_source,
        meta_val_key=args.meta_val_key,
        meta_test_key=args.meta_test_key,
        dev_key=args.dev_key,
        num_projections=args.num_projections,
    )
    norm_mean, norm_std = _compute_descriptor_stats(tasks)
    _apply_descriptor_norm(tasks, norm_mean, norm_std)
    clear_cuda_cache()
    if len(tasks) > 1:
        # Hold out a meaningful chunk so the active-selection val_V (and the
        # narrowed V_T) is non-trivial. With max(3, n//3) you get >=3 val tasks
        # whenever total tasks >= 4.
        val_size = max(3, len(tasks) // 3) if len(tasks) >= 4 else 1
        val_size = min(val_size, len(tasks) - 1)  # keep at least 1 train task
        val_tasks = tasks[:val_size]
        train_tasks = tasks[val_size:]
    else:
        train_tasks = tasks
        val_tasks = tasks

    model = ActiveEvaluator(input_dim=len(DEFAULT_FEATURE_ORDER) + args.context_dim)
    meta_cfg = MetaLearningConfig(
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        inner_steps=args.inner_steps,
        tasks_per_batch=args.tasks_per_batch,
        num_epochs=args.epochs,
        device=args.device,
        eval_inner_steps=args.eval_inner_steps,
        eval_context_steps=args.eval_inner_steps,
        meta_reg_lambda=args.meta_reg_lambda,
        meta_reg_beta=args.meta_reg_beta,
        context_dim=args.context_dim,
    )
    meta_learner = ActiveEvaluatorLearner(model, meta_cfg)

    print(f"[ActiveEvaluator] Training on {len(train_tasks)} tasks for {args.epochs} epochs.")
    checkpoint_path = output_dir / "active_evaluator_model.pt"
    history = meta_learner.meta_train(train_tasks, val_tasks=val_tasks, checkpoint_path=checkpoint_path)

    context_bank = meta_learner.build_context_bank(train_tasks, context_steps=args.inner_steps)

    meta_results, meta_mae = meta_learner.evaluate(
        tasks,
        context_bank=context_bank,
        adapt_context_steps=args.eval_inner_steps,
        adapt_weight_steps=args.eval_inner_steps,
        return_mae=True,
    )
    print(f"[ActiveEvaluator] Meta-test MAE: {meta_mae:.4f}")

    transfer_results = meta_learner.evaluate_transfer(
        tasks,
        context_bank=context_bank,
        adapt_context_steps=args.eval_inner_steps,
        adapt_weight_steps=args.eval_inner_steps,
    )
    transfer_mae = float(np.mean([entry["mae"] for entry in transfer_results])) if transfer_results else None
    if transfer_mae is not None:
        print(f"[ActiveEvaluator] Real-test MAE: {transfer_mae:.4f}")

    metrics = {
        "meta_test_mae": meta_mae,
        "real_test_mae": transfer_mae,
        "epochs": args.epochs,
        "history": history,
        "num_tasks": len(tasks),
    }

    save_json(output_dir / "active_evaluator_metrics.json", metrics)
    if args.save_meta_preds:
        save_json(output_dir / "active_evaluator_meta_predictions.json", {"results": meta_results})
    save_json(output_dir / "active_evaluator_dev_predictions.json", {"results": transfer_results})
    if test_models:
        print(f"[ActiveEvaluator] Adapting to {len(test_models)} held-out model(s).")
        test_tasks = build_tasks(
            test_models,
            splits,
            dev_split,
            cache=cache,
            accuracy=accuracy_source,
            meta_val_key=args.meta_val_key,
            meta_test_key=args.meta_test_key,
            dev_key=args.dev_key,
            num_projections=args.num_projections,
        )
        _apply_descriptor_norm(test_tasks, norm_mean, norm_std)

        if args.use_active_selection:
            extend_test_tasks_with_active_selection(
                meta_learner=meta_learner,
                test_tasks=test_tasks,
                train_tasks=train_tasks,
                val_tasks=val_tasks,
                args=args,
                output_dir=output_dir,
            )

        test_meta = meta_learner.evaluate(
            test_tasks,
            context_bank=context_bank,
            adapt_context_steps=args.eval_inner_steps,
            adapt_weight_steps=args.eval_inner_steps,
        )
        test_dev = meta_learner.evaluate_transfer(
            test_tasks,
            context_bank=context_bank,
            adapt_context_steps=args.eval_inner_steps,
            adapt_weight_steps=args.eval_inner_steps,
        )
        test_meta_mae = float(np.mean([entry["mae"] for entry in test_meta])) if test_meta else None
        test_dev_mae = float(np.mean([entry["mae"] for entry in test_dev])) if test_dev else None
        if test_meta_mae is not None:
            print(f"[ActiveEvaluator] Held-out Meta-test MAE: {test_meta_mae:.4f}")
        if test_dev_mae is not None:
            print(f"[ActiveEvaluator] Held-out Real-test MAE: {test_dev_mae:.4f}")
        save_json(output_dir / "active_evaluator_test_meta_predictions.json", {"results": test_meta, "mae": test_meta_mae})
        save_json(output_dir / "active_evaluator_test_dev_predictions.json", {"results": test_dev, "mae": test_dev_mae})

if __name__ == "__main__":
    main()
