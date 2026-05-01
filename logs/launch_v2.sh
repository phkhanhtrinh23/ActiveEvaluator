#!/bin/bash
mkdir -p outputs/run_v2
if [ ! -d outputs/run_v2/predictions ]; then
  cp -r outputs/run_baseline/predictions outputs/run_v2/predictions
fi

# V2 improvements over run #3:
#  --selection-weight-decay 1e-2     : regularizes K-step adapt -> prevents
#                                       overfitting tiny V_T.
#  --selection-early-stop-patience 3 : abort the K-step adapt when V_T loss
#                                       plateaus, reducing per-candidate noise.
#  --selection-inner-lr 0.1          : V2 specifically benefits from a more
#                                       aggressive lr (with sigmoid + weight
#                                       decay it stays stable) so the K-step
#                                       adapt actually moves.
python -m active_evaluator.pipeline \
  --train-path data/sft_spider_train_text2sql.json \
  --dev-path data/sft_spider_dev_text2sql.json \
  --output-dir outputs/run_v2 \
  --embedding-dir outputs/shared_embeddings \
  --max-train-samples 1500 --max-dev-samples 300 \
  --epochs 300 --gen-max-new-tokens 96 --batch-size 4 --lora-r 0 \
  --inner-lr 0.03 --eval-inner-steps 15 \
  --use-active-selection \
  --selection-method v2_direct \
  --selection-n-rounds 2 \
  --selection-budget-absolute 3 \
  --selection-narrowing-quantile 0.0 \
  --selection-pool-narrow-quantile 0.5 \
  --selection-K-steps 25 \
  --selection-max-candidates-evaluated 8 \
  --selection-weight-decay 1e-2 \
  --selection-early-stop-patience 3 \
  --selection-inner-lr 0.1 \
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
