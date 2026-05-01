#!/bin/bash
mkdir -p outputs/run_v3
if [ ! -d outputs/run_v3/predictions ]; then
  cp -r outputs/run_baseline/predictions outputs/run_v3/predictions
fi

# V3: GRAD-MATCH via OMP (Killamsetty et al., ICML 2021).
# Pick weighted subset of pool whose summed gradient approximates the V_T
# gradient. No K-step adaptation oracle inside the picker.
python -m active_evaluator.pipeline \
  --train-path data/sft_spider_train_text2sql.json \
  --dev-path data/sft_spider_dev_text2sql.json \
  --output-dir outputs/run_v3 \
  --embedding-dir outputs/shared_embeddings \
  --max-train-samples 1500 --max-dev-samples 300 \
  --epochs 300 --gen-max-new-tokens 96 --batch-size 4 --lora-r 0 \
  --inner-lr 0.03 --eval-inner-steps 15 \
  --use-active-selection \
  --selection-method v3_gradmatch \
  --selection-n-rounds 3 \
  --selection-budget-absolute 3 \
  --selection-narrowing-quantile 0.0 \
  --selection-pool-narrow-quantile 0.5 \
  --selection-K-steps 15 \
  --selection-gradmatch-lambda 1e-3 \
  --model-ids \
    cycloneboy/SLM-SQL-0.5B \
    cycloneboy/SLM-SQL-0.6B \
    cycloneboy/CscSQL-Merge-Qwen2.5-Coder-0.5B-Instruct \
    Qwen/Qwen2-0.5B \
    Qwen/Qwen2.5-Coder-1.5B \
    TinyLlama/TinyLlama_v1.1 \
    deepseek-ai/deepseek-coder-1.3b-base \
  --test-model-ids \
    Qwen/Qwen2.5-Coder-1.5B-Instruct \
    Qwen/Qwen2.5-0.5B-Instruct \
    Gensyn/Qwen2.5-0.5B-Instruct
