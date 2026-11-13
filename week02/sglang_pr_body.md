<!-- markdownlint-disable -->
## Motivation

`TestPageInterleaveIndexer.test_two_gpu` / `test_four_gpu` in `test/registered/unit/mem_cache/test_page_sharded_dsa_attention.py` only check the GPU count. On multi-GPU hosts below SM 8.9 (T4, V100, A100, ...) they error instead of skipping, because the DSA indexer K cache stores FP8 E4M3 and Triton can't compile `fp8e4nv` there:

```
triton.compiler.errors.CompilationError: at 1:0:
def _set_k_and_s_triton_kernel(
^
ValueError("type fp8e4nv not supported in this architecture. The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')")
```

Both `packed` iterations of `_run_indexer` go through `SetKAndS.triton`, so the test can't run at all below SM 8.9. The file is registered in `base-a-test-cpu`, so running that suite on such a host reports a false failure. Upstream CI doesn't hit this: the CPU runners have no GPU, and the CUDA registration targets `4-gpu-b200`.

## Modifications

In `TestPageInterleaveIndexer._test_world`, skip when `torch.cuda.get_device_capability() < (8, 9)`. The check runs in the parent process before `mp.spawn`, so no workers start and no process group is created on unsupported GPUs. This mirrors the existing capability gate in `TestShardedDSAAttention` in the same file. The threshold is SM 8.9 rather than `major in (9, 10)` because this path only needs FP8 E4M3, so Ada keeps its coverage.

Test-only change, 2 lines. No production code changes.

## Accuracy Tests

N/A (test-only change).

## Speed Tests and Profiling

N/A (test-only change).

## Validation

Kaggle 2× Tesla T4 (SM 7.5), driver 580.178.04, torch 2.14.1+cu130, Triton 3.8.0. Base is `c24285bde9` (`main` at the time); "PR" is this branch.

| Check | Base | PR |
|---|---|---|
| `TestPageInterleaveIndexer.test_two_gpu` | `ERROR`: `fp8e4nv not supported` | `skipped 'DSA indexer K cache stores FP8 E4M3, which needs SM 8.9+'` |
| Whole file (6 tests) | `FAILED (errors=1, skipped=2)` | `OK (skipped=3)`; the other 4 tests are unchanged |
| `test/run_suite.py --hw cpu --suite base-a-test-cpu`, `unit/mem_cache` | file fails | file passes |
| PR branch with `get_device_capability` patched to `(9, 0)` | | test runs again and hits the `fp8e4nv` kernel error, so the gate is the only thing that changed and SM ≥ 8.9 coverage is kept |
| PR branch, `CUDA_VISIBLE_DEVICES=0` | | still skipped by the GPU-count check (`requires 2 GPUs`), which runs first |

I also looked through the other `unit/mem_cache` tests that check `device_count()` (`test_page_interleave_gather_multi_gpu.py`, `test_dsa_layer_split_broadcast.py`) for the same problem. Neither uses FP8, so neither needs a capability gate.

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests). (This change is itself a test fix.)
- [ ] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations). (N/A)
- [ ] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed). (N/A)
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).

AI assistance: drafted with Claude (Claude Code); I reproduced and validated every result above on the hardware listed.

As a new contributor I can't trigger CI. Could a maintainer run `/tag-and-rerun-ci`? Thanks!
