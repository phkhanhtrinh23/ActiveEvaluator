# python -m shift_descriptor.pipeline \
#   --train-path data/sft_spider_train_text2sql.json \
#   --test-path data/sft_spider_dev_text2sql.json \
#   --output-dir outputs
python -m active_evaluator.pipeline

# Example: run pipeline with Version 1 active selection
# python -m active_evaluator.pipeline \
#   --train-path data/sft_spider_train_text2sql.json \
#   --dev-path data/sft_spider_dev_text2sql.json \
#   --output-dir outputs/active_evaluator_active \
#   --use-active-selection \
#   --selection-method v1_facility \
#   --selection-n-rounds 5 \
#   --selection-budget-fraction 0.10 \
#   --selection-narrowing-quantile 0.7
