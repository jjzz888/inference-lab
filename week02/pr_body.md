<!-- markdownlint-disable -->

## Overview

`create_scheduler` (`tests/v1/core/utils.py`) and `test_kv_connector` (`tests/v1/kv_connector/unit/test_config.py`) build a `ParallelConfig` with world size > 1 and no executor backend. On CUDA hosts with fewer local GPUs than the world size, these tests fail in config validation before they test anything. This PR pins `distributed_executor_backend="mp"` when world size > 1.

## Claims

- Fixes 5 test cases on CUDA hosts with fewer GPUs than the world size:
  - `test_diffusion_read_deferral_keeps_a_longer_pp_wait` (PP=3, fails on ≤2 GPUs)
  - `test_async_scheduling_pp_allows_rescheduling_with_output_placeholders` and `test_max_loras_counts_running_request_awaiting_output` (PP=2, fail on 1 GPU)
  - `test_kv_connector[native-8.0-2-2-…]` and `test_kv_connector[lmcache-8.0-2-2-…]` (TP=2 × PP=2, fail on ≤3 GPUs)
- No behavior change for world size 1 or on hosts with enough GPUs.
- Test-only change.

## Validation

Kaggle 2× T4. "Before" is `cbe948fe0c` (current `main` at the time); "after" is this branch.

| Run | Before | After |
|---|---|---|
| 2 GPUs: `pytest tests/v1/core/test_scheduler.py -k "diffusion or pipeline"` | 1 failed, 10 passed | 11 passed |
| 2 GPUs: `pytest tests/v1/kv_connector/unit/test_config.py` | 2 failed, 19 passed | 21 passed |
| 1 GPU (`CUDA_VISIBLE_DEVICES=0`): both files in full | 5 failed, 319 passed | 324 passed |

Every "before" failure is the same error and nothing else fails, e.g.:

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for ParallelConfig
  Value error, World size (3) is larger than the number of available GPUs (2) in this node. ...
```

On a non-CUDA host (macOS), the check doesn't run, and both files produce identical results before and after the change.

## Details

- **Root cause.** In `ParallelConfig` (`vllm/config/parallel.py`), when `distributed_executor_backend` is unset and world size > 1, it raises `World size (N) is larger than the number of available GPUs (M)` on CUDA if `device_count() < world_size`. The check is skipped when a backend is set explicitly.
- **Why `mp`, and why no behavior change.** With enough local GPUs, the same code path selects `mp` on its own, so pinning it produces the same config and only skips the device-count check. These tests build configs and never launch workers, so the check doesn't apply to them. World size 1 keeps `None`, so the default path is unchanged.
- **Precedent.** `tests/v1/engine/test_kv_cache_group_metadata.py` and `tests/v1/executor/test_vllm_net_devices.py` already pass `distributed_executor_backend="mp"` when building multi-rank configs. `tests/v1/kv_connector/unit/offloading_connector/test_config.py` patches `device_count` instead.
- **Scope.** I checked every `ParallelConfig(` call in `tests/v1`. The others either set a backend, patch `device_count`, or have world size 1. The attention tests parametrize TP but always build the config with TP=1, and DP is not part of the world size.

---

<details>
<summary> Pull Request Checklist </summary>

- [ ] I used vLLM's `/pr-checklist` skill. (Mandatory for agents, optional for humans).
- [x] AI assistance was used during the creation of this PR.

- [x] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [x] **Testing and Validation:** Validates the change and ensures any added tests are meaningful and reliable, with CI coverage or documented CI resource constraints and validation performed outside CI.
- [x] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [x] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.
</details>

AI assistance (Claude) was used to find the root cause, check the scope, and draft this description. I reviewed every changed line and ran all the validation above myself.
