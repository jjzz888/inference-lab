# SGLang Unit Tests on Kaggle T4

SGLang's CPU unit tests for the scheduler and the KV (key-value) / radix cache, run through SGLang's own CI (continuous integration) runner on a Kaggle 2× T4 host. This is the SGLang counterpart of [vllm_core_tests.md](vllm_core_tests.md). Script: [sglang_tests.sh](sglang_tests.sh).

## Table of Contents
- [[#Run Details]]
- [[#Results]]
- [[#Failure Analysis]]
- [[#Next Steps]]

## Run Details
| Item | Value |
|---|---|
| Date | 2026-11-08, Kaggle background run (notebook `kaggle-sglang-test`, Version #1) |
| Environment | Kaggle, 2× T4 (sm_75), SGLang venv from `kaggle_setup.sh sglang`, no Rust extensions (`SGLANG_BUILD_RUST_EXTS=none`) |
| SGLang commit | `cbbc6b67f4` (2026-11-08 09:42 +0800) |
| Command | `test/run_suite.py --hw cpu --suite base-a-test-cpu --continue-on-error --only-file-list sglang_unit_files.txt` |
| Scope | `test/registered/unit/managers` (102 files) + `test/registered/unit/mem_cache` (194 files) = 296 files; 255 are registered in `base-a-test-cpu` and ran |
| Duration | 60 min (runner estimate 46 min) |
| Raw logs | `results/week01/logs/sglang_tests.log`, `sglang_tests_console.log`, `sglang_unit_files.txt`, `setup_sglang.log` |

`run_suite.py` runs each file in its own process and reports pass or fail per **file**, not per test.

## Results
**248 of 255 files pass (97.3%).** The 7 failing files fall into four groups. None of them points at the scheduler logic itself.

| Group | Files | Cause | Kind |
|---|---|---|---|
| Missing test dependency | `mem_cache/test_page_interleave_shard.py`, `managers/test_disagg_idle_step_counters.py` | `ModuleNotFoundError: No module named 'parameterized'` | Our setup. **Fixed:** the script now installs `parameterized` |
| No Rust toolchain | `mem_cache/test_rust_tree_core_integration.py`, `mem_cache/test_rust_tree_core_errors.py`, `mem_cache/test_tree_core_registry.py` | `FileNotFoundError: 'cargo'` | Our setup (Rust extensions skipped on purpose) |
| GPU host runs a CPU-suite test | `mem_cache/test_unified_hicache_transfer_lifecycle.py` | Test fakes assume no CUDA; exposes a real error-handling bug | Test portability + **SGLang bug** |
| T4 lacks FP8 E4M3 | `mem_cache/test_page_sharded_dsa_attention.py` (`test_two_gpu`) | Triton: `type fp8e4nv not supported in this architecture` | Test portability (missing capability check) |

## Failure Analysis
### Missing `parameterized` (2 files)
Both files import `parameterized` at module level, so they fail before any test runs. SGLang lists it in the `[test]` extra of `python/pyproject.toml`, which the script deliberately doesn't install (it pulls heavy packages such as `lm-eval`). `sglang_tests.sh` now installs `pytest expecttest parameterized`.

### No Rust toolchain (3 files)
Kaggle has no `cargo`, and `kaggle_setup.sh` builds SGLang with `SGLANG_BUILD_RUST_EXTS=none`, so the radix tree uses the Python TreeCore.
- `test_rust_tree_core_integration.py` and `test_rust_tree_core_errors.py` query the Rust toolchain and fail with `RuntimeError: failed to query the Rust toolchain with 'cargo --version --verbose'`.
- `test_tree_core_registry.py::test_registry_resolves_the_rust_backend_lazily` calls `create_tree_core(name="rust", components={})`. Without the Rust module the registry falls back to the Python factory, and `UnifiedTreeCore.__init__` then fails `assert components`. The other 22 tests in the file pass.

Expected for this environment. To cover them, rerun with `SGLANG_BUILD_RUST_EXTS=rust_tree_core` (installs rustup, builds only the Rust radix tree).

### HiCache transfer lifecycle (1 file, 1 failure + 1 error out of 4 tests)
`test_host_match_stays_pinned_while_reclaim_evicts_host_leaves` fails in two ways:
- **FAIL.** The test replaces `torch.cuda.current_stream` with a fake `_Stream` object and mocks `_cuda_host_register`. On SGLang's CPU CI there is no CUDA, so the fakes are never handed to real CUDA calls. On a GPU host, `L2TransferEngine._start_event` calls a real `torch.cuda.Event.record()` with the fake stream: `RuntimeError: expected stream to be a torch.Stream or torch.cuda.Stream object`. The write-back is then left queued, and the eviction walk ends with `AssertionError: a device-eviction walk ended with queued write-back victims not demoted`.
- **ERROR, a real bug.** At teardown, the host pool calls the real `cudaHostUnregister` on memory the mock never registered. CUDA returns error 713 (`cudaErrorHostMemoryNotRegistered`). SGLang's warning path then crashes while formatting that error:
  ```python
  # python/sglang/srt/mem_cache/pool_host/common.py, _cuda_host_unregister_ranges
  rc = int(cudart.cudaHostUnregister(ptr))
  ...
  cudart.cudaGetErrorString(rc),   # TypeError: expects torch._C._cudart.cudaError, got int 713
  ```
  So any real `cudaHostUnregister` failure raises `TypeError` instead of logging a warning and returning the failed ranges. `_cuda_host_register` has the same pattern when building its error message: a failed `cudaHostRegister` raises `TypeError` instead of its informative `RuntimeError` ("host buffer is not pinned ..."). Confirmed on the same commit in `~/code/sglang`. Likely fix: `cudart.cudaGetErrorString(cudart.cudaError(rc))` at both call sites (to verify on a CUDA host: the Mac torch has no `cudart`).

### DSA sharded attention on T4 (1 file)
`TestPageInterleaveIndexer.test_two_gpu` only checks `torch.cuda.device_count() >= 2`, so it runs on 2× T4. Its Triton kernel `_set_k_and_s_triton_kernel` stores FP8 E4M3 (`fp8e4nv`), which Triton supports only on sm ≥ 8.9; T4 is sm_75 and offers only `fp8e4b15` and `fp8e5`. The file's CUDA registration targets `4-gpu-b200`, and the CPU CI has no GPUs, so upstream CI never hits this. Other tests in the same file already gate on compute capability (`get_device_capability()[0] not in (9, 10)`); this one is missing that check.

## Next Steps
1. **Rerun the 7 files** with the fixed deps (and `SGLANG_BUILD_RUST_EXTS=rust_tree_core` for the Rust group) to confirm only the two portability failures remain. `DIRS` can't select single files, so list them in a file and pass it with `--only-file-list` to `test/run_suite.py` directly.
2. ~~**SGLang contribution candidate: the `cudaGetErrorString(int)` bug.**~~ **Already taken** (checked 2026-11-08): issue [#40926](https://github.com/sgl-project/sglang/issues/40926) and open PRs [#38509](https://github.com/sgl-project/sglang/pull/38509), [#41245](https://github.com/sgl-project/sglang/pull/41245), [#48272](https://github.com/sgl-project/sglang/pull/48272). The HiCache test's fake-stream problem is covered by PR [#46801](https://github.com/sgl-project/sglang/pull/46801). Verifying those PRs on T4 is the fallback contribution in [sglang_first_pr_plan.md](../week02/sglang_first_pr_plan.md).
3. **First SGLang PR: the DSA FP8 capability gate.** No existing issue or PR. Plan, reproduction and experiments: [sglang_first_pr_plan.md](../week02/sglang_first_pr_plan.md).
