#!/bin/bash
# Re-run #3: sigmoid output + stronger eval-time adaptation.
# Eval inner steps 5 -> 15, inner_lr 0.01 -> 0.03 to give the support pair
# (label=0 for our test models) more leverage to override the meta-training prior.
python -m active_evaluator.pipeline \
  --train-path data/sft_spider_train_text2sql.json \
  --dev-path data/sft_spider_dev_text2sql.json \
  --output-dir outputs/run_baseline \
  --embedding-dir outputs/shared_embeddings \
  --max-train-samples 1500 --max-dev-samples 300 \
  --epochs 300 --gen-max-new-tokens 96 --batch-size 4 --lora-r 0 \
  --inner-lr 0.03 --eval-inner-steps 15 \
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
