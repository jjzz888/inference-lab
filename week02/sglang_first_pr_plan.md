# First SGLang PR Plan

Plan for a first SGLang PR (pull request), found in the Week 1 Kaggle unit-test run ([sglang_tests.md](../week01/sglang_tests.md)). The vLLM counterpart is [first_pr_plan.md](first_pr_plan.md).

Short version: the obvious candidate, the `cudaGetErrorString(int)` bug, is **already taken** (one issue, three open PRs). The plan targets the other T4 finding, which nobody has reported: the DSA (DeepSeek Sparse Attention) indexer sharding test runs an FP8 E4M3 Triton kernel on GPUs that can't compile it.

## Table of Contents
- [[#Candidates]]
- [[#Problem]]
- [[#Findings]]
- [[#Analysis]]
- [[#Reproduction]]
- [[#Fix]]
- [[#Experiment Plan]]
- [[#Steps]]
- [[#PR Outline]]
- [[#Fallback Contribution on the Taken Bug]]
- [[#Status]]

## Candidates
Checked on 2026-11-08 against upstream `main` `3bcde0c80e`, searching issues and PRs on `sgl-project/sglang`.

| Week 1 finding | Existing work | Decision |
|---|---|---|
| `pool_host/common.py` passes an `int` to `cudart.cudaGetErrorString`, so CUDA host-register errors raise `TypeError` | Issue [#40926](https://github.com/sgl-project/sglang/issues/40926) (open, detailed); PRs [#38509](https://github.com/sgl-project/sglang/pull/38509) (approved 2026-09, not merged), [#41245](https://github.com/sgl-project/sglang/pull/41245), [#48272](https://github.com/sgl-project/sglang/pull/48272) (2026-11-04) | **Taken.** A fourth PR would be a duplicate. Contribute by verifying instead ([[#Fallback Contribution on the Taken Bug]]) |
| HiCache lifecycle test's fake stream breaks on a GPU host | PR [#46801](https://github.com/sgl-project/sglang/pull/46801) runs those cases on a fake device | **Taken** |
| `test_page_sharded_dsa_attention.py::TestPageInterleaveIndexer::test_two_gpu` fails on T4 with `fp8e4nv not supported` | None found (searched the file name, the class name and `fp8e4nv`) | **Chosen** |
| 2 files need `parameterized`, 3 need Rust | Our environment | Not an upstream bug |

## Problem
On a CUDA host with at least 2 GPUs of compute capability below 8.9 (T4, V100, A100, A10, RTX 30xx), `test_two_gpu` errors instead of skipping:

```
triton.compiler.errors.CompilationError: at 1:0:
def _set_k_and_s_triton_kernel(
^
ValueError("type fp8e4nv not supported in this architecture. The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')")
```

The same holds for `test_four_gpu` on 4 such GPUs. The file is registered in SGLang's CPU suite (`base-a-test-cpu`), so anyone running that suite on a multi-GPU pre-Ada machine, as we did on Kaggle 2× T4, gets a red result that says nothing about their change.

## Findings
- **Where.** `test/registered/unit/mem_cache/test_page_sharded_dsa_attention.py`, class `TestPageInterleaveIndexer`, added in `20c49f12c9` "Support kv cache sharding for DSA indexer ([#40925](https://github.com/sgl-project/sglang/pull/40925))" (2026-10-12). Unchanged on `main` `3bcde0c80e`.
- **The only gate is the GPU count:**
  ```python
  class TestPageInterleaveIndexer(CustomTestCase):
      def _test_world(self, world):
          if torch.cuda.device_count() < world:
              self.skipTest(f"indexer KV sharding test requires {world} GPUs")
          ...mp.spawn(_run_indexer, ...)
  ```
- **The failing path.** `_run_indexer` → `_IndexerCacheCheck.seed` → `store` → `DSATokenToKVPool.set_index_k_scale_buffer` → `store_quantized` → `dsa_index_k_cache.SetKAndS.triton` → `_set_k_and_s_triton_kernel`. The indexer key cache always stores FP8 E4M3 keys. `_run_indexer` loops over `packed in (False, True)`, but "packed" only changes the layout, so both iterations need FP8 E4M3 and the whole test is unrunnable below SM 8.9.
- **Why upstream CI doesn't see it.** The file's CUDA registration is `register_cuda_ci(est_time=120, stage="extra-b", runner_config="4-gpu-b200")` (B200, SM 10.0), and the CPU runners have no GPU, so the test skips there.
- **Precedent in the same file.** `TestShardedDSAAttention` already gates on hardware: `if torch.cuda.get_device_capability()[0] not in (9, 10): self.skipTest("FlashMLA sparse requires Hopper or Blackwell")`.
- **Precedent elsewhere.** Open PR [#45488](https://github.com/sgl-project/sglang/pull/45488) "[Fix] [NVIDIA] kvcache: Gate fused FP8 prefix-valid commit kernel on SM89+" applies the same SM 8.9 threshold to production code.

## Analysis
- **Triton's rule.** Triton compiles `fp8e4nv` (FP8 E4M3, NVIDIA flavour) only for SM ≥ 8.9 (Ada, Hopper, Blackwell). On older GPUs it offers `fp8e4b15` and `fp8e5`, hence the error message.
- **Test bug, not a product bug.** DSA models (DeepSeek V3.2/V4 style) aren't expected to run on T4, so the kernel needs no fallback. The test should state its hardware requirement and skip, as its neighbour does.
- **Right threshold.** Gate on capability `< (8, 9)`, not on `major in (9, 10)`. The indexer path itself only needs FP8 E4M3, so Ada (8.9) and future archs should keep running it.
- **ROCm.** The file registers no AMD CI, and on ROCm `get_device_capability()` returns the gfx major/minor (e.g. (9, 4) on MI300), which passes the gate. ROCm behaviour stays exactly as today.
- **Size.** One condition plus one skip message in one test file. Small, but it fixes a real false failure in the CPU suite. To avoid a "busywork" PR, the experiment below also sweeps the rest of `unit/mem_cache` for the same pattern and folds any other hit into the same PR.

## Reproduction
Environment: Kaggle 2× T4 (SM 7.5), `kaggle_setup.sh sglang`, SGLang venv at `/root/lab/sglang/.venv`.

| Case | Command (from `/root/lab/sglang`) | Expected before fix |
|---|---|---|
| 2 GPUs, T4 | `.venv/bin/python test/registered/unit/mem_cache/test_page_sharded_dsa_attention.py TestPageInterleaveIndexer.test_two_gpu` | `ERROR` with `fp8e4nv not supported` (seen on 2026-11-08 at `cbbc6b67f4`) |
| 1 GPU, T4 | Same, with `CUDA_VISIBLE_DEVICES=0` | `skipped: indexer KV sharding test requires 2 GPUs` |
| CI-style file run | `.venv/bin/python test/run_suite.py --hw cpu --suite base-a-test-cpu --only-file-list <file containing the test path>` | `✗ FAILED ... returned exit code 1` |

A failure that reproduces at the PR's base commit is the precondition for opening the PR. If upstream has meanwhile fixed it, stop and pick another candidate.

## Fix
In `TestPageInterleaveIndexer._test_world`, after the GPU-count check:

```python
def _test_world(self, world):
    if torch.cuda.device_count() < world:
        self.skipTest(f"indexer KV sharding test requires {world} GPUs")
    if torch.cuda.get_device_capability() < (8, 9):
        self.skipTest("DSA indexer K cache stores FP8 E4M3, which needs SM 8.9+")
    ...
```

- Checked in the parent process, before `mp.spawn`, so no workers start and no NCCL group is created on unsupported GPUs.
- `get_device_capability()` reads device 0. Kaggle and CI hosts are homogeneous; mixed-GPU hosts are out of scope.
- Alternative considered: a shared helper such as `sglang.srt.utils.device.get_device_sm()`. It returns an int like 75 and can answer from NVML without initializing CUDA, but a plain `torch` call matches the neighbouring `TestShardedDSAAttention` gate. Follow the file's existing style unless a reviewer asks otherwise.

## Experiment Plan
All on Kaggle 2× T4 after `kaggle_setup.sh sglang`, on the PR branch. Every result goes into the PR description.

| # | Experiment | Expected |
|---|---|---|
| E1 | Reproduce at the base commit (table above) | `ERROR`, `fp8e4nv not supported` |
| E2 | Same test on the PR branch | `skipped 'DSA indexer K cache stores FP8 E4M3, which needs SM 8.9+'` |
| E3 | Whole file on the PR branch | No new failures; other classes pass or skip exactly as on the base commit |
| E4 | Gate is the only change: on the PR branch, patch `torch.cuda.get_device_capability` to return `(9, 0)` and run `test_two_gpu` | Back to the `fp8e4nv` error, which shows the test still runs on SM ≥ 8.9 |
| E5 | 1 GPU (`CUDA_VISIBLE_DEVICES=0`) | Still skipped by the GPU-count check, which runs first |
| E6 | Sweep for the same pattern: `grep -rln "device_count()" test/registered/unit/mem_cache` and, for each hit, check whether its CUDA path needs FP8 E4M3, SM 9.0 or similar without a capability gate. Run the default `unit/mem_cache` scope with `sglang_tests.sh` on the PR branch | Ideally only this file; fold any further hits into the PR, with the same E1–E3 evidence |
| E7 | `/run-cpu-ci` result on the PR (a maintainer triggers it) | Green |

E4 command. It goes through a script file, not `python - <<PY`: `mp.spawn` re-imports `__main__` by path, and stdin has none (the Week 1 warm-up crash).
```bash
cd /root/lab/sglang && cat > /tmp/e4_gate.py <<'PY'
import sys, unittest
from unittest import mock
import torch
sys.path.insert(0, "test/registered/unit/mem_cache")
import test_page_sharded_dsa_attention as t

if __name__ == "__main__":
    with mock.patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
        r = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromName("TestPageInterleaveIndexer.test_two_gpu", t))
    print("errors:", len(r.errors), "skipped:", len(r.skipped))
PY
.venv/bin/python /tmp/e4_gate.py
```
The spawned workers don't inherit the patch, but they don't need it: the patch only gets the parent past the gate, and the workers then hit the real kernel error.

## Steps
1. **Re-check for duplicates** right before starting. Search issues and PRs for `test_page_sharded_dsa_attention`, `TestPageInterleaveIndexer` and `fp8e4nv`. Also read the latest history of the test file.
2. **Branch from upstream** on the Mac:
   ```bash
   cd ~/code/sglang && git fetch origin && git switch -c test/dsa-indexer-shard-fp8-gate origin/main
   ```
3. **Make the edit** and run `pre-commit run --files test/registered/unit/mem_cache/test_page_sharded_dsa_attention.py` (install with `pip3 install pre-commit && pre-commit install` once).
4. **Commit and push to the fork** (remote `fork`, created by `mac_init.sh`):
   ```bash
   git commit -s -m "[Test] mem_cache: Skip DSA indexer sharding test below SM 8.9"
   git push -u fork test/dsa-indexer-shard-fp8-gate
   ```
5. **Validate on Kaggle**: E1–E6. In the Kaggle setup, check out the branch from the fork (`SGLANG_REF` only takes upstream refs, so clone normally, then `git fetch https://github.com/jjzz888/sglang.git test/dsa-indexer-shard-fp8-gate && git checkout FETCH_HEAD` before installing). Save the logs to `results/week02/`.
6. **Open the PR** with `gh pr create --repo sgl-project/sglang --base main --head jjzz888:test/dsa-indexer-shard-fp8-gate` and the outline below. Ask in the PR, or on the SGLang Slack, for `/run-cpu-ci`: new contributors can't trigger CI.
7. **Follow up**: answer review comments, rebase if `main` moves, and log the outcome in `LOG.md`.

## PR Outline
Title (SGLang format `[Type] scope: title`): `[Test] mem_cache: Skip DSA indexer sharding test below SM 8.9`

```markdown
## Motivation
`TestPageInterleaveIndexer.test_two_gpu` / `test_four_gpu` in
`test/registered/unit/mem_cache/test_page_sharded_dsa_attention.py` only check the GPU count.
On multi-GPU hosts below SM 8.9 (T4, V100, A100, ...) they error instead of skipping, because the
DSA indexer K cache stores FP8 E4M3 and Triton can't compile `fp8e4nv` there:
`ValueError("type fp8e4nv not supported in this architecture ...")`.
The file is in `base-a-test-cpu`, so running that suite on such a host reports a false failure.
Upstream CI doesn't hit this (CPU runners have no GPU; the CUDA registration targets B200).

## Modifications
Skip in `_test_world` when `torch.cuda.get_device_capability() < (8, 9)`, before spawning workers,
mirroring the existing capability gate in `TestShardedDSAAttention`. No production code changes.

## Accuracy Tests / Speed Tests
N/A (test-only change).

## Validation (Kaggle 2x T4, SM 7.5)
- Base commit: ERROR, `fp8e4nv not supported` (log)
- This PR: skipped with the new reason; rest of the file unchanged (log)
- Capability patched to (9, 0): the test runs again and hits the kernel, so SM >= 8.9 coverage is unchanged
- 1 GPU: still skipped by the GPU-count check

AI assistance: drafted with Claude (Claude Code); I reproduced and validated every result above.
```

Check SGLang's current rules on AI-assisted PRs before submitting. If there is none, keep the disclosure line and add the `Co-authored-by` trailer only if they accept it.

## Fallback Contribution on the Taken Bug
The `cudaGetErrorString(int)` bug is the more serious finding, but it already has owners. Useful work that doesn't duplicate them:
1. Read issue [#40926](https://github.com/sgl-project/sglang/issues/40926) and PRs [#38509](https://github.com/sgl-project/sglang/pull/38509), [#41245](https://github.com/sgl-project/sglang/pull/41245), [#48272](https://github.com/sgl-project/sglang/pull/48272) to see which parts each covers and why [#38509](https://github.com/sgl-project/sglang/pull/38509), approved since 2026-09, isn't merged (its reviewer asked for a lint fix).
2. On Kaggle T4, check out each PR and reproduce the original `TypeError`: call `_cuda_host_unregister_ranges(torch.cuda.cudart(), [(ptr, 4096)], operation="probe")` on a pointer that was never registered (CUDA error 713, `cudaErrorHostMemoryNotRegistered`). Then confirm the PR turns it into a warning.
3. Post the results as a review comment: "verified on T4 (SM 7.5), CUDA 13.0, torch 2.14.1". A real-hardware check from a non-H100 machine is a contribution maintainers value, and it's the right first interaction with the HiCache owners.

## Status
- [x] Week 1 run found the failure (2026-11-08, `cbbc6b67f4`)
- [x] Duplicate search, history and root cause (2026-11-08, `main` `3bcde0c80e`)
- [x] Step 1 re-check (2026-11-08, `main` `c24285bde9`): still no issue/PR; file last changed at `cbbc6b67f4`; gate unchanged. [#47578](https://github.com/sgl-project/sglang/pull/47578) touches `page_interleave` but not this file
- [x] E1 reproduce at the PR base (`c24285bde9`): `ERROR`, `fp8e4nv not supported`
- [x] Branch, edit, pre-commit (all passed), commit `61918f024d` (signed-off + Claude `Co-authored-by`, common in upstream history), pushed to `fork/test/dsa-indexer-shard-fp8-gate`
- [x] E2–E6 on Kaggle 2× T4 (2026-11-08, notebook `tiramisooo/sglang-first-pr`); logs in `results/week02/logs/`. All as expected:
  - E2 skipped with the new reason; E3 whole file `FAILED (errors=1, skipped=2)` → `OK (skipped=3)`, the other 4 tests unchanged; E4 patched `(9, 0)` → back to `fp8e4nv` (`errors: 1`); E5 `requires 2 GPUs`
  - E6a sweep: `test_page_interleave_gather_multi_gpu.py` and `test_dsa_layer_split_broadcast.py` check `device_count()` but use no FP8, so this file is the only hit
  - E6b `unit/mem_cache` via `sglang_tests.sh`: this file now passes. Compared with Week 1, `test_page_interleave_shard.py` also passes now (unrelated); new failure `test_unified_scheduler_hooks.py` (see below)
  - `ci_base.log` / `ci_pr.log` are **invalid**: the cell ran `run_suite.py` without `.venv/bin` on `PATH`, so the child `python3` had no `sglang`. E6b is the valid CI-style evidence
- [x] PR body drafted: [sglang_pr_body.md](sglang_pr_body.md)
- [x] PR opened: [#49164](https://github.com/sgl-project/sglang/pull/49164) (2026-11-08, head `61918f024d`); mergeable, review required
- [ ] CI triggered: the `pr-gate` checks fail with `Missing required label 'run-ci'` (and the `*-finish` jobs fail after them). Needs a maintainer to add `run-ci` / comment `/tag-and-rerun-ci`; asked in the PR body. If nobody responds in a few days, ask on the SGLang Slack
- [ ] 2026-11-13 check: no labels, comments, reviews or review requests since opening (only the bot note that automated review is off for forks). Still merges cleanly into `main` `4f8b686753`; upstream changed the file twice (`4a99de617f`, `1edbe8f6f5`) but not `TestPageInterleaveIndexer`, and the gate is still missing. Pinged for `run-ci` in a [PR comment](https://github.com/sgl-project/sglang/pull/49164#issuecomment-6842615998) (2026-11-13), then [cc'd hnyls2002](https://github.com/sgl-project/sglang/pull/49164#issuecomment-6842667199) (`mem_cache` CODEOWNER, author of #49144). If still silent in a few days: SGLang Slack
- [ ] Candidate for a 2nd PR: `test_unified_scheduler_hooks.py::TestOpportunisticCompactionErrors::test_an_error_after_processing_a_result_early_propagates` fails with `AttributeError: 'Scheduler' object has no attribute 'pause_state'`. [#49144](https://github.com/sgl-project/sglang/pull/49144) (`b070051032`) moved `_engine_paused` into `PauseState`, but the test still sets `scheduler._engine_paused`. Not fixed on `main` `3825fcd397`; no issue or PR found. CPU CI should hit it too, so someone may fix it soon: re-check before taking it. Plan: [sglang_second_pr_plan.md](sglang_second_pr_plan.md)
- [ ] Optional: verification comment on the `cudaGetErrorString` PRs
