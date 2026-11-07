# inference-lab log

## Week 1 (Nov 9 → Nov 15)

### Build
| Engine | Where | Ref / commit | Install mode | Result / issues |
|---|---|---|---|---|
| vLLM | Mac CPU | `a8dacd7238` | editable, CPU | OK. `smoke_offline.py` runs; breakpoint in `Scheduler.schedule` hits. Needed: proxy, uv ≥ 0.8, CPU KV cache capped at 2 GiB |
| vLLM | Kaggle T4 | `1ecc9cc4ce` (`0.33.0rc1.dev1004`) | `VLLM_USE_PRECOMPILED=1`, torch `2.14.0+cu132` | OK. Needed: pin uv to the venv (`UV_SYSTEM_PYTHON=true` in image), `libcuda.so` link for FlashInfer JIT |
| SGLang | Kaggle T4 | `ba5906471` (`0.5.24.dev910`) | editable, `SGLANG_BUILD_RUST_EXTS=none`, torch `2.14.1+cu130`, FlashInfer 0.7.3 | OK. No Rust toolchain → Python TreeCore |

### Tests
- vLLM `tests/v1/core` (Kaggle T4): `1 failed, 1284 passed, 2 errors` in 26 min. After the environment fixes, 1,286/1,287 pass. The remaining failure is a test portability bug (PP=3 on a 2-GPU host). Details: [week01/vllm_core_tests.md](week01/vllm_core_tests.md).
- SGLang: not run yet.

### Serving benchmark (Qwen2.5-0.5B-Instruct, FP16, T4, random 512 in / 128 out, 200 prompts, rate=inf)
| Engine | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | Output tok/s | Req/s |
|---|---|---|---|---|---|
| vLLM | 5190.60 | 9470.35 | 71.74 | 1716.97 | 13.41 |
| SGLang | 2259.82 | 4118.24 | 50.90 | 2897.28 | 22.63 |

Both runs: Kaggle 2× T4 (one GPU used), driver 580.178.04, 2026-11-07, one run each, default server settings except `--dtype half`.

| Metric | vLLM | SGLang | SGLang vs vLLM |
|---|---|---|---|
| Benchmark duration (s) | 14.91 | 8.84 | 1.69× faster |
| Request throughput (req/s) | 13.41 | 22.63 | 1.69× |
| Output token throughput (tok/s) | 1,716.97 | 2,897.28 | 1.69× |
| Total token throughput (tok/s) | 8,584.86 | 14,486.39 | 1.69× |
| Peak output throughput (tok/s) | 4,000 | 6,467 | 1.62× |
| Mean / median TTFT (ms) | 5,190.60 / 5,050.34 | 2,259.82 / 2,220.70 | 2.3× lower |
| P99 TTFT (ms) | 9,470.35 | 4,118.24 | 2.3× lower |
| Mean / median TPOT (ms) | 71.74 / 73.60 | 50.90 / 51.19 | 1.4× lower |
| P99 TPOT (ms) | 95.06 | 64.02 | 1.5× lower |
| Median / P99 ITL (ms) | 50.73 / 347.98 | 34.55 / 176.41 | 1.5× / 2.0× lower |

Run directories:
- vLLM: `results/week01/vllm-20261107-151619` (commit `1ecc9cc4ce`, torch `2.14.0+cu132`)
- SGLang: `results/week01/sglang-20261107-151822` (commit `ba5906471`, torch `2.14.1+cu130`). Attention backend: FlashInfer (default). Prefix cache: Python TreeCore (no Rust extensions). Server cold start about 4 min.

Reading the numbers:
- `rate=inf` sends all 200 requests at t=0 (102,400 prompt tokens). TTFT is mostly queueing behind prefill, which is chunked under a per-step token budget. SGLang's log shows 2,048-token prefill batches of about 5 sequences, so 200 prompts need about 50 prefill steps. TPOT and ITL rise while prefill chunks share steps with running decodes, which explains P99 ITL ≫ median for both engines.
- This measures peak throughput under a burst, not latency under realistic load.

Caveats before drawing conclusions:
- **Attention backend differs, and is the likely main cause.** T4 has no FlashAttention 2 (needs sm ≥ 8.0).
  - vLLM: `Using TRITON_ATTN attention backend out of potential backends: ['TRITON_ATTN', 'FLEX_ATTENTION']`.
  - SGLang: FlashInfer (its default).
- **CUDA graphs point the other way.** vLLM captured 51 PIECEWISE and 35 FULL CUDA graph sizes, so its 200-request decode most likely ran on a graph. SGLang's decode log shows `cuda graph: False`: 200 running requests exceed its default CUDA-graph batch-size cap. SGLang still had 1.4× lower TPOT without graphs, which points at the attention kernel (Triton vs FlashInfer) rather than graph capture. **Test:** rerun SGLang with `SGLANG_ATTN=triton` so both use Triton attention.
- **Defaults differ.** Per-step token budget (`max_num_batched_tokens` vs SGLang's chunked prefill size), max sequences, and memory fraction are not aligned.
- **One run each, on a shared cloud GPU.** Repeat at least 3× and report the spread before calling the gap real.

### Next for the benchmark
- ~~Record vLLM's attention backend and CUDA graph capture sizes~~: TRITON_ATTN, 51 PIECEWISE + 35 FULL graph sizes (see caveats).
- Isolate the attention kernel: `SGLANG_ATTN=triton bash week01/run_bench.sh sglang`, then compare against the vLLM run.
  - **Run 1, cold (`sglang-20261107-154655`): not usable.** 4.81 req/s, 616 output tok/s, mean TTFT 21,028 ms (P90 33,023), mean TPOT 160.85 ms, median ITL 44.49 ms, **max ITL 35,147 ms**. Median ITL is normal, so decode itself was fine. The 35 s stall and the TTFT tail point to Triton JIT-compiling kernels for shapes first seen during the benchmark (server warm-up didn't cover them), on 2 CPU threads. _Rerun warm; max ITL should drop to a few seconds before the numbers count._
  - **Run 2, warm cache (`sglang-20261107-155359`): same stall.** 5.65 req/s, 723 output tok/s, mean TTFT 14,904 ms (P90 26,865), mean TPOT 160.66 ms, median ITL 45.05 ms, **max ITL 28,958 ms**. The Triton cache was warm, so first-time JIT compilation doesn't explain it.
  - **Not a stall.** The three largest timestamp gaps in `server.log` are all at startup: prefill CUDA graph capture 21 s, init 12 s, idle before the first request. During the benchmark no gap exceeds 8 s, which also rules out autotune.
  - **Likely cause: SGLang's prefill-first scheduling plus a slow Triton prefill kernel.** By default SGLang runs prefill whenever requests are queued and pauses running decodes (no mixed chunked prefill; vLLM mixes by default). With 200 requests arriving at once, the first requests get their first token, then wait until all 200 prefills finish. So **max ITL ≈ the length of the whole prefill phase**, and TTFT is spread evenly over it:

    | Run | Max ITL (s) | P99 TTFT (s) | Mean TTFT (s) |
    |---|---|---|---|
    | FlashInfer | 3.8 | 4.1 | 2.3 |
    | Triton, run 2 | 29.0 | 29.3 | 14.9 (≈ half, as for a uniform queue) |

  - **Confirmed from `server.log`.** Burst prefill batches are the `Prefill batch` lines with `#new-seq` ≥ 2; warm-up requests have 1 sequence. Timestamps have 1 s resolution.

    | Run | Burst prefill start → end | Prefill phase | Batches (2,048 tokens each) | Max ITL | Prefill throughput |
    |---|---|---|---|---|---|
    | SGLang + FlashInfer | 15:23:25 → 15:23:29 | ~4 s | 50 | 3.8 s | ~25,600 tok/s |
    | SGLang + Triton | 15:55:19 → 15:55:48 | ~29 s | 49 | 29.0 s | ~3,500 tok/s |

    - In both runs, the first `Decode batch, #running-req: 200` appears in the same second as the last prefill batch. So max ITL is exactly "wait for all prefills".
    - Same number of batches, same chunk size. Only per-batch speed differs: **SGLang's Triton prefill (extend) attention is ~7× slower than FlashInfer on T4**, while decode is only ~23% slower (median ITL 45.05 vs 34.55 ms).
  - **Takeaways.**
    - On T4, FlashInfer's main win for SGLang is prefill, not decode.
    - Prefill-first scheduling turns slow prefill directly into a decode stall for every running request. Under bursts, mixing prefill chunks into decode steps (vLLM's default; SGLang's `--enable-mixed-chunk`) trades some TPOT for no stall.
    - For a fair vLLM-vs-SGLang comparison on T4, keep SGLang on FlashInfer and compare scheduling knobs explicitly.
  - **What is usable: steady-state decode (median ITL), which a single stall barely moves.**

    | Config | Median ITL (ms) |
    |---|---|
    | SGLang + FlashInfer | 34.55 |
    | SGLang + Triton (runs 1, 2) | 44.49, 45.05 |
    | vLLM + Triton | 50.73 |

    - Swapping Triton → FlashInfer inside SGLang makes each decode step ~23% faster.
    - With both on Triton attention, SGLang's decode step is still ~11% faster than vLLM's. That gap comes from outside the attention kernel: scheduling overhead, sampling, CUDA graph use.
    - Throughput and TTFT from the Triton runs are not comparable until the stall is explained.
- Rerun both at fixed concurrency (e.g. `--max-concurrency 16, 64`) and at finite request rates, which is closer to real serving than a single burst.
- Align the knobs that differ by default (per-step token budget, max running requests) and repeat 3×.

### Notes / possible first issues
- vLLM `test_diffusion_read_deferral_keeps_a_longer_pp_wait` fails on any CUDA host with < 3 GPUs (`World size (3) is larger than the number of available GPUs (2)`). Week 2 PR candidate.
