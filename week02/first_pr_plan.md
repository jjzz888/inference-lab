# First vLLM PR Plan

Plan for the Week 2 first PR (pull request): make vLLM's config-only unit tests with world size > 1 (scheduler tests with PP (pipeline parallelism), plus a kv_connector config test with TP×PP = 4) pass on CUDA hosts with fewer GPUs than the world size. Found in the Week 1 Kaggle run ([vllm_core_tests.md](../week01/vllm_core_tests.md)).

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
- **Scope sweep (`git grep -n "ParallelConfig(" tests/v1`, branch base `cbe948fe0c`).** One more affected site: `tests/v1/kv_connector/unit/test_config.py::test_kv_connector` builds `ParallelConfig(tensor_parallel_size=tp, pipeline_parallel_size=pp)` with no backend, and two parameter sets use `tp=2, pp=2` (world size 4), so they fail on 2× T4 and on 1 GPU. Everything else is safe:
  - Backend already set: `ec_connector/unit/utils.py`, `executor/test_vllm_net_devices.py`, `worker/test_dspark_utils.py`, `worker/test_gpu_pcp_manager.py` (`ray`), `engine/test_kv_cache_group_metadata.py`.
  - `device_count` patched: `kv_connector/unit/offloading_connector/test_config.py`, `core/test_kv_cache_planning.py`.
  - World size 1: the attention tests parametrize TP but only shrink head counts and always build the config with TP=1. DP-only configs don't count, because world size = PP × TP × PCP.
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

In `tests/v1/kv_connector/unit/test_config.py`, `test_kv_connector`, the same treatment (chosen 2026-11-08 after the scope sweep):

```python
        parallel_config=ParallelConfig(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            # Config-only test; skip the world-size vs local-GPU-count check.
            distributed_executor_backend="mp" if tp * pp > 1 else None,
        ),
```

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
Edit `create_scheduler` and `test_kv_connector` as in [[#Fix]].

### 4. Commit and push to the fork
Pushing to your own fork notifies no one; only opening the PR (step 6) is visible upstream. So push first, then validate. If validation finds a problem, amend and force-push to the fork.

```bash
pre-commit run --files tests/v1/core/utils.py tests/v1/kv_connector/unit/test_config.py
git add tests/v1/core/utils.py tests/v1/kv_connector/unit/test_config.py
git commit -s -m "[Tests] Make multi-GPU config tests independent of the host GPU count" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
git log -1 && git push fork fix/scheduler-tests-pp-gpu-count
```
- `-s` adds `Signed-off-by` (DCO); the second `-m` adds the AI-assistance trailer. Check that `git log -1` shows both.
- Before committing, run the two test files on the Mac and compare against unmodified code (`git stash`). The Mac can't reproduce the bug (it isn't CUDA), but before and after must have identical failure lists.

### 5. Validate on Kaggle 2× T4
Only a real CUDA host reproduces it. The change is test-only Python, so no rebuild is needed. These are the cells actually used on 2026-11-08, in a fresh notebook (Settings: **Accelerator = GPU T4 x2**, **Internet = On**).

**Cell 1: install vLLM** (foreground, ~10–15 min). A fresh session has no `/root/lab/vllm`. `RUN_TESTS=0` skips the Week 1 test run and the JIT warm-up. Check for `in venv: True` and `gpus 2`. A `falling back to ...` line is harmless: the change is Python-only, and Cell 2 checks out the exact commits.
```bash
%%bash
git clone -q https://github.com/jjzz888/inference-lab.git /kaggle/working/inference-lab
VLLM_REF=cbe948fe0c RUN_TESTS=0 bash /kaggle/working/inference-lab/week01/kaggle_setup.sh vllm 2>&1 | tee /kaggle/working/setup_vllm.log | grep -E "==>|installed at|falling back|in venv|gpus|ERROR|WARN"
```

**Cell 2: before/after validation** (background, ~7 min). It activates the venv; otherwise pytest runs on Kaggle's system Python. It also installs the test deps that `RUN_TESTS=0` skipped, runs the same three commands on the base commit and then on the pushed branch, and saves each run to a file.
```bash
%%bash
cat > /kaggle/working/pr_validate.sh <<'EOF'
cd /root/lab/vllm
source .venv/bin/activate
export UV_PYTHON="$VIRTUAL_ENV/bin/python" UV_SYSTEM_PYTHON=false
uv pip install -q pytest pytest-asyncio tblib
run_all () {
  echo "===== 2 GPUs: scheduler (diffusion or pipeline) ====="
  pytest tests/v1/core/test_scheduler.py -q -k "diffusion or pipeline" -p no:cacheprovider 2>&1 | tail -15
  echo "===== 2 GPUs: kv_connector test_config ====="
  pytest tests/v1/kv_connector/unit/test_config.py -q -p no:cacheprovider 2>&1 | tail -15
  echo "===== 1 GPU: both files ====="
  CUDA_VISIBLE_DEVICES=0 pytest tests/v1/core/test_scheduler.py tests/v1/kv_connector/unit/test_config.py -q -p no:cacheprovider 2>&1 | tail -25
}
git checkout -q cbe948fe0c && echo "### BEFORE @ $(git rev-parse --short HEAD)"
run_all > /kaggle/working/pr_before.txt 2>&1
git fetch -q https://github.com/jjzz888/vllm fix/scheduler-tests-pp-gpu-count && git checkout -q FETCH_HEAD && echo "### AFTER @ $(git rev-parse --short HEAD)"
run_all > /kaggle/working/pr_after.txt 2>&1
echo "### DONE"
EOF
nohup bash /kaggle/working/pr_validate.sh > /kaggle/working/pr_validate.log 2>&1 &
echo "started, pid $!"
```

**Check progress** (repeat until the log shows `### DONE`; keep the notebook page open or the session may stop):
```bash
!cat /kaggle/working/pr_validate.log; echo ---; tail -5 /kaggle/working/pr_before.txt /kaggle/working/pr_after.txt 2>/dev/null
```

**Print the results:**
```bash
!cat /kaggle/working/pr_validate.log; echo; echo "########## BEFORE"; cat /kaggle/working/pr_before.txt; echo; echo "########## AFTER"; cat /kaggle/working/pr_after.txt
```

- The third test command uses a single GPU and runs both whole files, which covers the two PP=2 scheduler tests.
- `%%bash` runs in the foreground and blocks the notebook. That's why Cell 2 uses `nohup … &`.
- A Kaggle environment only lasts as long as its session. Reopening an old notebook gives a fresh one, so Cell 1 is needed every time.
- The kv_connector failure is predicted from reading the code; the "before" run confirms it.
- Expected: "before" fails with `World size (N) is larger than the number of available GPUs`; "after" has no such failure. Any other failures must match exactly between before and after.

### 6. Open the PR
```bash
gh pr create
```
It targets upstream by default (`gh repo set-default vllm-project/vllm` was set by `mac_init.sh`). Fill in the PR template as in [[#PR Description Outline]].

## PR Description Outline
- **Overview.** `create_scheduler` and `test_kv_connector` build `ParallelConfig` with world size > 1 and no executor backend, so on CUDA hosts with fewer GPUs than the world size they fail in config validation before testing anything.
- **Claims.** Fixes 3 scheduler tests (the PP=3 diffusion test and two PP=2 tests) and the two TP=2/PP=2 `test_kv_connector` cases on such hosts. A sweep of every `ParallelConfig(` in `tests/v1` found no other affected sites. No behavior change on other hosts.
- **Validation.**
  - Kaggle 2× T4 before/after for the `-k "diffusion or pipeline"` run.
  - Kaggle 2× T4 before/after for `kv_connector/unit/test_config.py`.
  - Single-GPU (`CUDA_VISIBLE_DEVICES=0`) before/after for both full files.
  - The simulated table from [[#Reproduction]].
- **Details.**
  - Root cause: `ParallelConfig` only runs the device-count check when no backend is set.
  - Why there is no behavior change: `mp` is what vLLM selects anyway when GPUs suffice.
  - Precedent: `tests/v1/engine/test_kv_cache_group_metadata.py` already uses `distributed_executor_backend="mp"`.
- **Checklist.** Tick "AI assistance was used".

## Status
- [x] Root cause found and reproduced (simulated on the Mac, real on Kaggle)
- [x] Checked for existing issues and PRs (none, 2026-11-08)
- [x] Branch created from upstream `main` (`cbe948fe0c`, 2026-11-08)
- [x] Scope sweep of `tests/v1`: added `kv_connector/unit/test_config.py`
- [x] Fix written (`core/utils.py`, `kv_connector/unit/test_config.py`). On the Mac, before and after both give 67 failed / 257 passed with identical failure lists; the failures are pre-existing CPU-only limits such as no spec decode without Triton
- [x] Committed `cef9a453b2` (signed off, Co-Authored-By) and pushed to `fork/fix/scheduler-tests-pp-gpu-count`
- [x] Validated on Kaggle 2× T4, 2026-11-08 (before = `cbe948fe0c`, after = the pushed branch):

  | Run | Before | After |
  |---|---|---|
  | 2 GPUs, `test_scheduler.py -k "diffusion or pipeline"` | 1 failed, 10 passed (diffusion, `World size (3) … GPUs (2)`) | 11 passed |
  | 2 GPUs, `kv_connector/unit/test_config.py` | 2 failed, 19 passed (`[native-8.0-2-2-…]`, `[lmcache-8.0-2-2-…]`, `World size (4) … GPUs (2)`) | 21 passed |
  | 1 GPU, both files | 5 failed, 319 passed (the 3 PP scheduler tests + the 2 kv_connector cases) | 324 passed |

  Every "before" failure is the world-size check, and no other failures exist, so the fix is exactly scoped. The kv_connector failure predicted from the code is confirmed.
- [x] PR opened 2026-11-08: [vllm-project/vllm#66355](https://github.com/vllm-project/vllm/pull/66355) (description in [pr_body.md](pr_body.md))
- [ ] Review feedback addressed
