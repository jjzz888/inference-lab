# inference-lab log

## Week 1 (Nov 9 → Nov 15)

### Build
| Engine | Where | Ref / commit | Install mode | Result / issues |
|---|---|---|---|---|
| vLLM | Mac CPU | | editable, CPU | |
| vLLM | Kaggle T4 | | `VLLM_USE_PRECOMPILED=1` | |
| SGLang | Kaggle T4 | | editable | |

### Tests
- vLLM `tests/v1/core`:
- SGLang:

### Serving benchmark (Qwen2.5-0.5B-Instruct, FP16, T4, random 512 in / 128 out, 200 prompts, rate=inf)
| Engine | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | Output tok/s | Req/s |
|---|---|---|---|---|---|
| vLLM | | | | | |
| SGLang | | | | | |

### Notes / possible first issues
-
