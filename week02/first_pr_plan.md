# First vLLM PR Plan

Plan for the Week 2 first PR (pull request): make vLLM's scheduler unit tests that use PP (pipeline parallelism) pass on CUDA hosts with fewer GPUs than the PP size. Found in the Week 1 Kaggle run ([vllm_core_tests.md](../week01/vllm_core_tests.md)).

## Table of Contents
- [[#Findings]]
- [[#Reproduction]]
- [[#Fix]]
- [[#Steps]]
- [[#PR Description Outline]]
- [[#Status]]

## Findings
Checked on 2026-11-08 against upstream `main` `d1879210a4`.

- **Not reported yet.** Searching vLLM issues and PRs for the test name, `diffusion_read_deferral`, and the error text found nothing related. The search itself works: "DiffusionGemma" returns results.
- **Still present on upstream `main`.**
  - Upstream changed the test: it dropped the V2-runner fixture.
  - It still calls `_diffusion_scheduler(pipeline_parallel_size=3)`.
  - `create_scheduler` in `tests/v1/core/utils.py` still builds `ParallelConfig(pipeline_parallel_size=..., data_parallel_size=...)` with no executor backend.
- **Broader than one test.** Two more tests in `tests/v1/core/test_scheduler.py` use `pipeline_parallel_size=2`, so they fail on a 1-GPU CUDA host. Fix the shared helper, not just the diffusion test. This also meets vLLM's rule against one-off "busywork" PRs.
- **Root cause.** In `vllm/config/parallel.py`, when `distributed_executor_backend is None` and the world size is > 1, `ParallelConfig` raises `World size (N) is larger than the number of available GPUs (M) in this node` if the platform is CUDA and `device_count() < world_size`. The check is skipped when a backend is set explicitly.
- **Precedent in the repo.** `tests/v1/engine/test_kv_cache_group_metadata.py` already passes `distributed_executor_backend="mp"`.
- **Origin.** The diffusion test came in with `1b3b88ec2b` "[Core] structured generation mode for DiffusionGemma model (Jev-like) (#57250)".

## Reproduction
Simulated on the Mac by faking a CUDA host (`current_platform.is_cuda = lambda: True`, `current_platform.device_count = lambda: N`) and building the `ParallelConfig` that `create_scheduler` builds:

| Simulated host | No backend | `distributed_executor_backend="mp"` |
|---|---|---|
| 2 GPUs, PP=2 | ✅ OK, backend auto-selected as `mp` | — |
| 2 GPUs, PP=3 | ❌ `World size (3) is larger than the number of available GPUs (2)` | ✅ OK |
| 1 GPU, PP=2 | ❌ `World size (2) is larger than the number of available GPUs (1)` | ✅ OK |

Real failure on Kaggle 2× T4 (Week 1): `pydantic_core._pydantic_core.ValidationError: 1 validation error for ParallelConfig` → `World size (3) is larger than the number of available GPUs (2) in this node`.

**No behavior change.** When there are enough GPUs, vLLM picks `mp` on its own (first row). Pinning `mp` gives the same config and only skips the device-count check. Scheduler tests build configs and never launch workers, so the check does not apply to them.

## Fix
In `tests/v1/core/utils.py`, `create_scheduler`, pin the backend only when PP > 1. PP = 1 stays unchanged.

```python
parallel_config=ParallelConfig(
    pipeline_parallel_size=pipeline_parallel_size,
    data_parallel_size=data_parallel_size,
    # Scheduler tests only build configs and never launch workers. Pin the
    # backend so the world-size vs local-GPU-count check doesn't fail on
    # CUDA hosts with fewer GPUs than pipeline_parallel_size.
    distributed_executor_backend="mp" if pipeline_parallel_size > 1 else None,
),
```

Before writing it, run `git grep -n "ParallelConfig(" tests/v1` and check whether other test helpers under `tests/v1` have the same problem. Include them if so: a complete, systematic scope makes a stronger PR.

## Steps
vLLM requires the human submitter to review every changed line and run the tests personally; "pure agent" PRs are not accepted. Do steps 3–6 yourself.

### 1. Prepare (once)
Read `docs/contributing/README.md`. The hard rules are:
- **DCO (Developer Certificate of Origin).** Every commit needs `Signed-off-by`. `git commit -s` adds it.
- **pre-commit.** Lint and format checks run on each commit.
- **PR title.** It starts with bracketed tags, e.g. `[Tests]`.
- **AI-assisted contributions.** Add a `Co-authored-by: Claude` trailer to commits, say so in the PR description, and tick "AI assistance was used" in the checklist.

```bash
cd ~/code/vllm && uv pip install "pre-commit>=4.5.1" && pre-commit install
```

### 2. Branch from the latest upstream `main`
```bash
cd ~/code/vllm && git fetch origin && git switch -c fix/scheduler-tests-pp-gpu-count origin/main
```
Then `source week01/mac_session.sh`: it rebuilds automatically if `csrc/` changed.

### 3. Change the code
Edit `create_scheduler` as in [[#Fix]], plus any other helpers found by the `git grep`.

### 4. Validate on Kaggle 2× T4
Only a real CUDA host reproduces it. The change is test-only Python, so no rebuild is needed.

```bash
cd /root/lab/vllm && git fetch https://github.com/jjzz888/vllm fix/scheduler-tests-pp-gpu-count && git checkout FETCH_HEAD
pytest tests/v1/core/test_scheduler.py -q -k "diffusion or pipeline" -p no:cacheprovider
CUDA_VISIBLE_DEVICES=0 pytest tests/v1/core/test_scheduler.py -q -p no:cacheprovider
```
- The second command uses a single GPU and runs the whole `test_scheduler.py`, which covers the two PP=2 tests.
- Run both before and after the change, and keep both outputs for the PR's Validation section.
- The branch must be pushed to the fork first (step 5) for `git fetch` to see it. Push, validate, then open the PR.

### 5. Commit and push to the fork
```bash
pre-commit run --files tests/v1/core/utils.py
git commit -s -m "[Tests] Make PP scheduler tests independent of the host GPU count"
git push fork fix/scheduler-tests-pp-gpu-count
```
Add the `Co-authored-by: Claude` trailer to the commit message.

### 6. Open the PR
```bash
gh pr create
```
It targets upstream by default (`gh repo set-default vllm-project/vllm` was set by `mac_init.sh`). Fill in the PR template as in [[#PR Description Outline]].

## PR Description Outline
- **Overview.** `create_scheduler` builds `ParallelConfig` without an executor backend, so on CUDA hosts with fewer GPUs than PP, every scheduler test with PP > 1 fails in config validation before testing anything.
- **Claims.** Fixes 3 tests on such hosts (the PP=3 diffusion test and two PP=2 tests). No behavior change on other hosts.
- **Validation.**
  - Kaggle 2× T4 before/after for the `-k "diffusion or pipeline"` run.
  - Single-GPU (`CUDA_VISIBLE_DEVICES=0`) before/after for the full `test_scheduler.py`.
  - The simulated table from [[#Reproduction]].
- **Details.**
  - Root cause: `ParallelConfig` only runs the device-count check when no backend is set.
  - Why there is no behavior change: `mp` is what vLLM selects anyway when GPUs suffice.
  - Precedent: `tests/v1/engine/test_kv_cache_group_metadata.py` already uses `distributed_executor_backend="mp"`.
- **Checklist.** Tick "AI assistance was used".

## Status
- [x] Root cause found and reproduced (simulated on the Mac, real on Kaggle)
- [x] Checked for existing issues and PRs (none, 2026-11-08)
- [ ] Branch created from upstream `main`
- [ ] Fix written, plus any other affected helpers
- [ ] Validated on Kaggle (2 GPUs and 1 GPU), before and after
- [ ] Pushed to the fork and PR opened
- [ ] Review feedback addressed
