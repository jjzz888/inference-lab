# vLLM Core Tests

What `tests/v1/core` in vLLM covers, plus the Week 1 Kaggle run: results, error analysis, and next steps.

## Table of Contents
- [[#What the Suite Covers]]
- [[#Scheduler Tests]]
- [[#KV Cache Manager Tests]]
- [[#Block Freeing Under Async Execution]]
- [[#End-to-End Tests]]
- [[#Using the Tests to Learn the Scheduler]]
- [[#Test Results]]
- [[#Error Analysis]]
- [[#Next Steps]]

## What the Suite Covers
`tests/v1/core` tests the **core scheduling layer** of the vLLM v1 engine. These are steps 5, 6 and 9 of the request lifecycle in [README](README.md) ("Debugging the request lifecycle"):

- the **Scheduler**: which requests run in this step, and how many tokens each gets
- the **KV (key-value) Cache Manager**: PagedAttention block allocation, prefix caching, freeing

It barely touches the model forward pass, so most tests need no GPU and run fast.

Size, counted on the Mac checkout (commit `a8dacd7238`; the Kaggle checkout is slightly newer):

- 27 test files and about 690 test functions
- about 1,300 tests after `@pytest.mark.parametrize` expansion, which matches the Kaggle progress bar

Most tests use a fake scheduler and fake model outputs (`create_scheduler`, `create_requests`, `create_model_runner_output` in `tests/v1/core/utils.py`), so they test scheduling logic only.

## Scheduler Tests
These decide what runs in each step.

| File | Test functions | What it tests |
|---|---|---|
| `test_scheduler.py` | 185 | **The core file.** Adding, scheduling and finishing requests. Token budget, chunked prefill, preemption, abort, stop conditions, speculative decoding, encoder inputs, and newer features such as MoE (Mixture of Experts) routed-expert tracking |
| `test_async_scheduler.py` | 12 | Async scheduling: the next step is scheduled before the previous step's output returns. Covers `max_tokens` stops, abort, and KV-pressure preemption with in-flight output |
| `test_priority_scheduler_random.py`, `test_priority_preemption_bug.py` | 1 + 1 | Priority scheduling: a randomized stress test and a regression test for a preemption bug |
| `test_mamba_align_chunk_split.py` | 10 | How Mamba (SSM, state space model) prefill chunks are split on block boundaries |
| `test_repetition_detection.py` | 19 | Detecting a generation stuck in a repetition loop and stopping early |
| `test_output.py` | 7 | Scheduler output data structures |

## KV Cache Manager Tests
These cover how PagedAttention blocks are allocated, reused and freed.

| File | Test functions | What it tests |
|---|---|---|
| `test_prefix_caching.py` | 130 | **The second core file.** Prefix caching: whether requests with a shared prefix hit and reuse existing blocks. Block hashing, reference counts and eviction, and whether results match with caching on and off |
| `prefix_cache/` (4 files) | 70 | Partial prefix hits, hit rate for hybrid models (attention + Mamba), and Mamba + EAGLE speculative decoding resuming from a checkpoint |
| `test_kv_cache_utils.py` | 75 | KV cache configuration: per-layer block size, how many blocks fit in memory, and how multiple KV groups split (e.g. SWA, sliding window attention, layers mixed with full attention) |
| `test_kv_cache_planning.py`, `test_kv_cache_config_builder.py` | 40 + 12 | Planning the KV cache layout from the model structure and memory |
| `test_single_type_kv_cache_manager.py` | 21 | Managers for a single attention type (full, sliding window, cross-attention) |
| `test_contiguous_kv_packing.py` | 26 | Packing KV into contiguous memory |
| `test_kv_sharing.py` | 2 | KV cache sharing across layers |
| `test_kv_cache_metrics.py` | 16 | Usage metrics such as KV cache utilization |
| `test_encoder_cache_manager.py` | 22 | Encoder output cache for multimodal models (e.g. image embeddings) |

## Block Freeing Under Async Execution
These check that blocks are freed at the right time when the GPU may still be using them.

| File | Test functions | What it tests |
|---|---|---|
| `test_deferred_block_free.py` | 17 | Under async scheduling, a block may still be in use by the GPU, so its free is deferred until the GPU is done |
| `test_swa_inflight_window_free.py` | 7 | With sliding window attention, blocks that slide out of the window are freed only after in-flight GPU steps finish |
| `test_prefix_replay.py` | 8 | After a prefix hit, a sliding-window model has to recompute the tail |
| `test_worker_slot_overflow.py` | 3 | Regression tests for the V2 model runner's "No free indices" assertion |
| `test_prompt_block_hash_vectors.py` | 1 | Pins prompt block hashes so the Rust frontend computes the same values as Python |

## End-to-End Tests
These start a real engine and load a model:

- `test_scheduler_e2e.py`
- `test_reset_prefix_cache_e2e.py`
- `test_priority_preemption_bug.py`

They need model weights and GPU memory, so they are the most likely to fail with `E` (error) for environment reasons.

pytest collects entries sorted by name, so `prefix_cache/` runs before every `test_*.py` file. Errors at the very start of the progress line most likely come from there.

## Using the Tests to Learn the Scheduler
`test_scheduler.py` and `test_prefix_caching.py` are the best executable documentation for the scheduler. Each test is one small scenario, for example "when two requests share a prefix, which blocks should the second one hit".

When a piece of scheduling logic is unclear:
1. Search for the matching test name.
2. Run that one test under the debugger, with a breakpoint in `Scheduler.schedule` or `KVCacheManager.allocate_slots`.

This is faster than reading the source cold. Adding a missing test is also one of the easiest first PRs (pull requests) for Week 2.

## Test Results
Kaggle run, 2026-11-07.

| Item | Value |
|---|---|
| Environment | Kaggle, 2× T4 (sm_75), torch `2.14.0+cu132` |
| vLLM commit | `1ecc9cc4ce` (`0.33.0rc1.dev1004+g1ecc9cc4c`) |
| Command | `pytest tests/v1/core -q -x --maxfail=5` |
| Summary line | `1 failed, 1284 passed, 18 warnings, 2 errors in 1588.27s` |
| Duration | 26 min 28 s |
| Log | `~/lab/logs/vllm_tests.log` |

1,284 of 1,287 tests pass (99.8%). The 3 that don't:

| Result | Test |
|---|---|
| `E` | `tests/v1/core/prefix_cache/test_hybrid_prefix_cache_hit_rate.py::test_prefix_cache_hit_rate[base]` |
| `E` | `tests/v1/core/prefix_cache/test_hybrid_prefix_cache_hit_rate.py::test_prefix_cache_hit_rate[mtp]` |
| `F` | `tests/v1/core/test_scheduler.py::test_diffusion_read_deferral_keeps_a_longer_pp_wait` |

After the environment fixes ([[#Error Analysis]]):

| Test | Final status | Cause |
|---|---|---|
| `test_prefix_cache_hit_rate[base]`, `[mtp]` | **pass** (`2 passed in 263.77s`) | Environment: missing `libcuda.so` link for FlashInfer JIT, then a cold-start Triton compile over the 480 s test timeout |
| `test_diffusion_read_deferral_keeps_a_longer_pp_wait` | fails on any CUDA host with fewer than 3 GPUs | Test portability bug (PP=3 vs 2 GPUs). Week 2 PR candidate |

Net result: 1,286 of 1,287 pass on Kaggle T4 once the environment is fixed, and the remaining failure is in the test, not in vLLM.

**Rerun 2026-11-08, first** (background run on a fresh session, vLLM `fd8f3f398e`; raw log in commit `da56775`, `results/week01/logs/vllm_tests.log`):

| Item | Value |
|---|---|
| Summary line | `1 failed, 1276 passed, 18 warnings, 1 error in 2041.55s (0:34:01)` |
| `F` | `test_diffusion_read_deferral_keeps_a_longer_pp_wait`: same portability bug, still on upstream `main` |
| `E` | `test_prefix_cache_hit_rate[base]`: `Server failed to start in time`. The JIT warm-up crashed (see [[#Next Steps]] item 2), so the cache was cold |
| `[mtp]` | passed: it ran after `[base]` had already filled part of the cache |

The total is 10 lower than on 11-07 because upstream changed the suite between commits.

**Rerun 2026-11-08, second** (Kaggle Version #2, fresh session, vLLM `ebc97ad8d0`, with the fixed warm-up; raw log in `results/week01/logs/vllm_tests.log`):

| Item | Value |
|---|---|
| Warm-up | `ENGINE READY` in 592 s (cold Triton cache, no time limit) |
| Summary line | `1 failed, 1277 passed, 18 warnings in 1458.51s (0:24:18)` |
| `F` | `test_diffusion_read_deferral_keeps_a_longer_pp_wait`: same portability bug |
| Hybrid `[base]`, `[mtp]` | **both passed** |

1,277 of 1,278 pass with no manual steps. The 18 warnings are `DeprecationWarning`s from FlashInfer's CUTLASS imports (`cutlass.utils.LayoutEnum`), not from vLLM. The suite is also 10 minutes faster than the first rerun, because the hybrid test no longer waits out its 480 s timeout.

## Error Analysis
For each problem, record the test node id, the root cause from its traceback, and the category: environment (T4/sm_75, missing optional dependency, model download), or a real bug that reproduces on current `main`.

### Hybrid Prefix Cache Hit Rate (2 errors)
- **What the test does.** A module-scoped fixture starts a real OpenAI-compatible server (`RemoteOpenAIServer`) for `Qwen/Qwen3.5-0.8B`, a hybrid model (linear attention + full attention). It then replays multi-turn conversations and checks the prefix-cache hit counters from `/metrics`. `[mtp]` adds MTP (multi-token prediction) speculative decoding with 1 token.
- **What the log says.** `RuntimeError: Server exited unexpectedly.` (once per parametrization). The server process died during startup; its own error is in the server output, not in this line.
- **Why `E`, not `F`.** The server fixture failed, so neither test body ran. Both parametrizations share the fixture, so they fail together.
- **Root cause (confirmed from the server log, `~/lab/logs/hybrid_rerun.log`).** A FlashInfer JIT **link** failure on the Kaggle image, not a vLLM bug.
  - FlashInfer has no prebuilt kernels for sm_75, so on first use it JIT-compiles its RMSNorm op into `~/.cache/flashinfer/0.7.3/75/cached_ops/norm`.
  - nvcc 13.0 compiles both `.cuda.o` files. The link step `c++ ... -L/usr/local/cuda/lib64/stubs -lcuda` then fails: `/usr/bin/ld: cannot find -lcuda`. The image's CUDA toolkit has no `libcuda.so` in `stubs/`, while the driver's `libcuda.so.1` is mounted elsewhere.
  - Chain: `Ninja build failed` → `Engine core initialization failed` → `Server exited unexpectedly` → fixture error → `E` for both parametrizations.
  - **Ruled out: BF16 (bfloat16).** vLLM logs `Your device 'Tesla T4' (with compute capability 7.5) doesn't support torch.bfloat16. Falling back to torch.float16` and continues.
  - **Not fatal:** `Cannot use FA version 2 ... FA2 is only supported on devices with compute capability >= 8`. vLLM falls back to another attention backend.
- **Fix.** The driver's `libcuda.so.1` is at `/usr/local/nvidia/lib64` (reached via `LD_LIBRARY_PATH`, not in `ldconfig`). `/usr/local/cuda` is a read-only mount, so the stubs dir can't be written. `kaggle_setup.sh` (`fix_libcuda_link`) symlinks `/usr/local/lib/libcuda.so` to the driver library instead (a default `ld` search dir), and verifies it by linking a test program with `-lcuda`. Verified on Kaggle: the link test prints `LINK OK`.
- **Second problem after the fix: startup timeout.** The rerun no longer crashes, but fails with `RuntimeError: Server failed to start in time.` after 503 s. `RemoteOpenAIServer` waits at most 480 s (`max_wait_seconds or 480` in `tests/utils.py`). Server log timeline (`~/lab/logs/hybrid_rerun2.log`):

  | Time | Stage | Took |
  |---|---|---|
  | 14:42:19 → 14:42:50 | start, load model (1.63 GiB weights) | 31 s |
  | 14:42:58 → 14:49:36 | **memory profiling**: a dummy forward pass (including one max-size image, since the model is multimodal) to size the KV cache | **6 min 38 s** |
  | 14:49:36 → 14:50:05 | warm up GDN (gated delta net) Triton kernels, vision rotary kernels | 29 s |
  | 14:50:05 → ~14:50:19 | CUDA graph capture starts, then the 480 s deadline hits | cut off |

  The profiling pass is the first real forward, so **Triton JIT-compiles** every kernel then: GDN linear attention, the Triton attention backend used on T4, and the vision encoder. There are no ninja/nvcc lines. Kaggle gives the engine 2 CPU threads (`Reducing Torch threads from 2 to 1`), so compilation is slow.
- **Workaround.** Warm the caches once (`LLM('Qwen/Qwen3.5-0.8B')` in the same session), then rerun the test. Triton's cache in `~/.triton/cache` persists for the session.
- **Result (verified).**
  - The warm-up `LLM('Qwen/Qwen3.5-0.8B')` reached `ENGINE READY` in **1 min 52 s**: the timed-out run had already filled the Triton cache.
  - Rerunning the test file then gave **`2 passed in 263.77s`**, both `[base]` and `[mtp]`.
- **Category.** Environment: Kaggle image (missing `libcuda.so` link) plus a slow cold start on T4 with 2 CPU threads, against a hard-coded 480 s test timeout. Not a vLLM bug. On upstream CI GPUs these kernels are prebuilt or compile faster.
- **Rerun pitfall.** Rerunning this test with `.venv/bin/python -m pytest ...` without the venv on `PATH` fails differently: `FileNotFoundError: ... 'vllm'`, because the fixture spawns `vllm serve` as a subprocess. Prefix the command with `PATH=/root/lab/vllm/.venv/bin:$PATH`.
- **Category.** Probably environment (T4). Confirm from the traceback under `= ERRORS =`.

### Diffusion Read Deferral (1 failure)
- **What the test does.** Pure scheduler logic, no GPU. A diffusion "read-only" request runs with pipeline parallelism 3 on the V2 model runner. The test checks that a read is deferred until the pipeline-parallel (PP) wait allows it (`next_decode_eligible_step` 4, then 7), and that a later, shorter deferral doesn't shorten that wait.
- **On the Mac (commit `a8dacd7238`, same test and scheduler code as Kaggle).** It also fails, but for a different reason: `CPU Model Runner V2 without Triton does not yet support: diffusion models`, raised while building `VllmConfig` before any scheduling. That is a Mac/CPU limitation, so it does not explain the Kaggle failure.
- **Where it fails on Kaggle.** Inside `_diffusion_scheduler` → `create_scheduler` → `VllmConfig(...)` (`tests/v1/core/utils.py`), the same spot as on the Mac. No scheduling ran, and no `next_decode_eligible_step` assertion was reached.
- **Root cause (confirmed by the Kaggle log).** `pydantic_core.ValidationError: 1 validation error for ParallelConfig` → `World size (3) is larger than the number of available GPUs (2) in this node`. It is the only diffusion test with `pipeline_parallel_size=3`, so world size is 3. `ParallelConfig` (`vllm/config/parallel.py`, executor-backend selection) raises `ValueError: World size (3) is larger than the number of available GPUs (2) in this node` when the platform is CUDA and `device_count() < world_size`. Kaggle has 2 T4s. Every other diffusion test uses PP 1 and passes.
- **Why upstream CI passes.** The fixture says "These CPU tests only exercise scheduling", so the test assumes a CPU-only runner, or a GPU runner with at least 3 GPUs. It only patches Triton and the V2 runner flag, not the GPU count. On a CUDA host with 1–2 GPUs it fails before testing anything.
- **Category.** Test portability bug, not a scheduler bug. **A good Week 2 first-PR candidate:** make the test independent of the host's GPU count, e.g. pin a single-process executor backend in the config, monkeypatch `current_platform.device_count`, or skip when `is_cuda() and device_count() < 3`. Before filing, search vLLM issues/PRs for `test_diffusion_read_deferral_keeps_a_longer_pp_wait` and "World size is larger than the number of available GPUs" in tests.

## Next Steps
Draft, from this run:

1. **Week 2 first PR: the diffusion test.** Search vLLM issues and PRs for `test_diffusion_read_deferral_keeps_a_longer_pp_wait` and "World size is larger than the number of available GPUs". If nothing exists, make the test independent of the host's GPU count. Reproduce first on Kaggle (2× T4), and run the other diffusion tests in `test_scheduler.py` to make sure they still pass. **Planned in [first_pr_plan.md](../week02/first_pr_plan.md).** No existing issue or PR. The bug also hits two PP=2 tests on 1-GPU hosts, so the fix goes in the shared `create_scheduler` helper.
2. **Full-suite reruns on a fresh Kaggle session.** **Automated.** `kaggle_setup.sh` now fixes the `libcuda.so` link, and before `tests/v1/core` it warms the JIT caches by starting the hybrid test's model once with no time limit. The model name is read from `MODEL = "..."` in the test file, so it follows upstream. The log goes to `~/lab/logs/vllm_warmup.log`, a failed warm-up only warns, and `WARMUP_HYBRID=0` skips it.
   - **Verified on a fresh session (2026-11-08):**
     - ✅ The `libcuda.so` auto-fix worked: `created: ... (verified: -lcuda links)`.
     - ➖ The wheel fallback didn't trigger: `fd8f3f398e` already had a published wheel.
     - ❌ The warm-up crashed after 66 s with `FileNotFoundError: '/root/lab/vllm/<stdin>'`. It fed the script through stdin (`python - <<PY`), and vLLM starts EngineCore with multiprocessing `spawn`, which re-imports `__main__` from its path. `<stdin>` isn't a file. **Fixed:** the warm-up now uses `python -c`, which has no path, so spawn skips the re-import. Reproduced and verified with a plain-Python spawn test and a fake vLLM that spawns a child.
   - ✅ **Confirmed (2026-11-08, second rerun):** the warm-up reached `ENGINE READY` in 592 s, and both hybrid tests passed (`1 failed, 1277 passed`). The only failure left is item 1.
3. ~~**Benchmarks.** Run `run_bench.sh both` in a session where setup has finished.~~ **Done (2026-11-07).** Results, backend findings (vLLM `TRITON_ATTN` vs SGLang FlashInfer) and the follow-up `SGLANG_ATTN=triton` rerun are in [LOG.md](../LOG.md).
4. **Read the scheduler through tests.** Step through `test_scheduler.py` and `test_prefix_caching.py` cases on the Mac (pytest is now in `~/code/vllm/.venv`), alongside the request-lifecycle diagram.
