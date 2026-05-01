## Top-line MAE comparison

| Run | Seen meta-test MAE | Seen real-test MAE | Unseen meta-test MAE | Unseen real-test MAE |
| --- | --- | --- | --- | --- |
| baseline | 0.0292 | 0.3133 | 0.0690 | 0.5416 |
| v1_facility | 0.0251 | 0.2493 | 0.2237 | 0.4263 |
| v2_direct | 0.0269 | 0.3126 | 0.1615 | 0.5426 |
| v3_gradmatch | 0.0265 | 0.2731 | 0.2127 | 0.4935 |

## Per-test-model real-test predictions (unseen)

| Model | True | BL pred | BL MAE | V1 pred | V1 MAE | V2 pred | V2 MAE | V3 pred | V3 MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-Coder-1.5B-Instruct | 0.003 | 0.544 | 0.541 | 0.468 | 0.465 | 0.528 | 0.525 | 0.502 | 0.499 |
| Qwen/Qwen2.5-0.5B-Instruct | 0.000 | 0.542 | 0.542 | 0.407 | 0.407 | 0.551 | 0.551 | 0.491 | 0.491 |
| Gensyn/Qwen2.5-0.5B-Instruct | 0.000 | 0.542 | 0.542 | 0.407 | 0.407 | 0.551 | 0.551 | 0.491 | 0.491 |

## V1 vs V2 per-model winner (smaller real-test MAE wins)

| Model | V1 MAE | V2 MAE | Winner |
| --- | --- | --- | --- |
| Qwen/Qwen2.5-Coder-1.5B-Instruct | 0.4645 | 0.5250 | V1 |
| Qwen/Qwen2.5-0.5B-Instruct | 0.4072 | 0.5514 | V1 |
| Gensyn/Qwen2.5-0.5B-Instruct | 0.4072 | 0.5514 | V1 |

**V1 wins: 3 | V2 wins: 0 | Ties: 0**

## Selection economics (per test model, averaged)

| Run | Budget | Cost paid | n picked | Gain total | Final val loss | Sel. time |
| --- | --- | --- | --- | --- | --- | --- |
| v1_facility | 3.0 | 1.7 | 1.7 | 0.041 | 0.0216 | 0.04s |
| v2_direct | 3.0 | 1.7 | 1.7 | 0.012 | 0.0138 | 0.08s |
| v3_gradmatch | 3.0 | 2.0 | 2.0 | 0.000 | 0.0223 | 0.07s |

## Most-picked training models (across all test models)

### v1_facility

| Source training model | Times picked |
| --- | --- |
| Qwen/Qwen2.5-Coder-1.5B | 3 |
| Qwen/Qwen2-0.5B | 2 |

### v2_direct

| Source training model | Times picked |
| --- | --- |
| Qwen/Qwen2.5-Coder-1.5B | 2 |
| Qwen/Qwen2-0.5B | 2 |
| TinyLlama/TinyLlama_v1.1 | 1 |

### v3_gradmatch

| Source training model | Times picked |
| --- | --- |
| Qwen/Qwen2.5-Coder-1.5B | 3 |
| Qwen/Qwen2-0.5B | 2 |
| TinyLlama/TinyLlama_v1.1 | 1 |


Full comparison written to outputs/comparison.json
