#!/bin/bash
# Re-uses outputs/shared_embeddings + cached predictions from baseline
# (the run_baseline run created predictions per-model under predictions/<alias>/<split>.json)
# Active selection only adds the test-time selection step.

# Copy predictions cache from baseline so generation is skipped
mkdir -p outputs/run_active
if [ ! -d outputs/run_active/predictions ]; then
  cp -r outputs/run_baseline/predictions outputs/run_active/predictions
fi

python -m active_evaluator.pipeline \
  --train-path data/sft_spider_train_text2sql.json \
  --dev-path data/sft_spider_dev_text2sql.json \
  --output-dir outputs/run_active \
  --embedding-dir outputs/shared_embeddings \
  --max-train-samples 1500 --max-dev-samples 300 \
  --epochs 300 --gen-max-new-tokens 96 --batch-size 4 --lora-r 0 \
  --use-active-selection \
  --selection-method v1_facility \
  --selection-n-rounds 5 \
  --selection-budget-fraction 0.30 \
  --selection-narrowing-quantile 0.5 \
  --selection-K-steps 5 \
  --model-ids \
    cycloneboy/SLM-SQL-0.5B \
    cycloneboy/SLM-SQL-0.6B \
    cycloneboy/CscSQL-Merge-Qwen2.5-Coder-0.5B-Instruct \
    Qwen/Qwen2-0.5B \
    Qwen/Qwen2.5-Coder-1.5B \
    Qwen/Qwen2.5-Coder-1.5B-Instruct \
    TinyLlama/TinyLlama_v1.1 \
    deepseek-ai/deepseek-coder-1.3b-base \
  --test-model-ids \
    Qwen/Qwen2.5-0.5B-Instruct \
    Gensyn/Qwen2.5-0.5B-Instruct \
    Qwen/Qwen3-0.6B \
    unsloth/Llama-3.2-1B-Instruct
