# Second SGLang PR Plan

Plan for a second SGLang PR (pull request), found in the E6 suite run for the first one ([sglang_first_pr_plan.md](sglang_first_pr_plan.md), PR [#49164](https://github.com/sgl-project/sglang/pull/49164)).

Short version: two PRs merged on 2026-11-08 crossed each other. [#49144](https://github.com/sgl-project/sglang/pull/49144) moved the scheduler's pause flag into a `PauseState` object and updated every test that used the old flag. Three hours later [#48008](https://github.com/sgl-project/sglang/pull/48008), whose branch predates #49144, added a new test that still sets the old flag. That test now fails on every machine, GPU or not. The fix is the same 2-line change #49144 made in the other tests.

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
- [[#Status]]

## Candidates
Checked on 2026-11-08 against upstream `main` `3825fcd397`, searching issues and PRs on `sgl-project/sglang`.

| Finding | Existing work | Decision |
|---|---|---|
| `test_unified_scheduler_hooks.py::TestOpportunisticCompactionErrors::test_an_error_after_processing_a_result_early_propagates` raises `AttributeError: 'Scheduler' object has no attribute 'pause_state'` | None found (searched the file name, the class name, `pause_state` and `_engine_paused`); no open PR touches the file | **Chosen** |
| `test_page_interleave_shard.py` failed in Week 1, passes now | Fixed upstream between `cbbc6b67f4` and `c24285bde9` | Nothing to do |

**Time matters.** The test is in `base-a-test-cpu` and fails without a GPU too, so it will likely show up red in the CPU CI of any PR whose run includes it, our [#49164](https://github.com/sgl-project/sglang/pull/49164) included. Someone else may fix it within days. Open this PR quickly and keep it minimal.

## Problem
On any host, with or without a GPU, at `main` `c24285bde9` or later:

```
File ".../test_unified_scheduler_hooks.py", line 232, in test_an_error_after_processing_a_result_early_propagates
    Scheduler.event_loop_overlap(scheduler)
File ".../python/sglang/srt/managers/scheduler.py", line 2178, in event_loop_overlap
    if self.pause_state.paused:
       ^^^^^^^^^^^^^^^^
AttributeError: 'Scheduler' object has no attribute 'pause_state'. Did you mean: 'init_pause_state'?
```

Seen on Kaggle 2× T4 at `61918f024d` (the first PR's branch, which differs from `c24285bde9` only in an unrelated test file); log: `results/week02/logs/e6_mem_cache_tests.log`. The other two tests in the file pass.

## Findings
- **Where.** `test/registered/unit/mem_cache/test_unified_scheduler_hooks.py`, class `TestOpportunisticCompactionErrors`, line 219. It's registered with `register_cpu_ci(est_time=5, suite="base-a-test-cpu")`.
- **The test builds a bare scheduler** with `Scheduler.__new__(Scheduler)`, which skips `__init__`, and sets by hand only the attributes `event_loop_overlap` reads. One of them is the old flag:
  ```python
  scheduler.ingest_requests = mock.Mock()
  scheduler._engine_paused = False   # line 219: stale
  scheduler.last_batch = None
  ```
- **What #49144 changed** (`b070051032`, merged 2026-11-08 08:42 UTC, author hnyls2002). It replaced `self._engine_paused` with `self.pause_state = PauseState()`, which `__init__` sets up through `init_pause_state()`. `event_loop_overlap` now reads `self.pause_state.paused`. The PR updated 13 tests that build bare schedulers, all the same way:
  ```python
  from sglang.srt.managers.scheduler_components.pause_state import PauseState
  -    scheduler._engine_paused = False
  +    scheduler.pause_state = PauseState()
  ```
- **What #48008 added** (`c24285bde9`, merged 2026-11-08 11:35 UTC, author ch-wan, one commit created before #49144 merged). It added the whole `TestOpportunisticCompactionErrors` class, including line 219. Its branch didn't include #49144 and GitHub saw no conflict, so it merged as is. This is a semantic merge race, not a mistake in either PR on its own.
- **It's the only stale use.** `git grep _engine_paused origin/main -- python test` finds only line 219.
- **`PauseState()` starts unpaused** (`self._paused = False`), so it's an exact replacement for `_engine_paused = False`.

## Analysis
- **Test bug, not a product bug.** The real `Scheduler.__init__` always calls `init_pause_state()`. Only bare schedulers built with `__new__` in tests can miss it.
- **What the test checks is still valid.** It checks that an error from `flush_opportunistic()` between steps propagates instead of being swallowed. Once the bare scheduler has the attribute, the test reaches that code again. E3 checks that the test still detects a swallowed error.
- **Fix style.** Use `PauseState()` directly, not `scheduler.init_pause_state()`, to match the 13 places #49144 changed. A reviewer who knows #49144 will recognise it at once.
- **Size.** 1 import plus 1 changed line. That's small but justified: it fixes a test that fails for everyone on `main`, and the evidence shows exactly which merge broke it.
- **Hardware.** None. It fails and passes the same way on CPU-only and GPU hosts (E1 and E2 run both).

## Reproduction
Any host with the SGLang venv. On Kaggle (from `/root/lab/sglang`):

| Case | Command | Expected before fix |
|---|---|---|
| GPU host | `.venv/bin/python test/registered/unit/mem_cache/test_unified_scheduler_hooks.py -v` | `FAILED (errors=1)`, `AttributeError ... pause_state` |
| No GPU | Same, with `CUDA_VISIBLE_DEVICES=""` | Same error |
| CI-style | `PATH=$PWD/.venv/bin:$PATH .venv/bin/python test/run_suite.py --hw cpu --suite base-a-test-cpu --only-file-list <list>` | `✗ FAILED ... exit code 1` |

The CI-style run **needs `.venv/bin` on `PATH`**: `run_suite.py` launches each file with `python3` from `PATH`. The first PR's Cell 2 missed this, and its single-file CI logs were invalid.

Precondition for opening the PR: the failure reproduces at the PR's base commit. If upstream has meanwhile fixed it, stop.

## Fix
In `test/registered/unit/mem_cache/test_unified_scheduler_hooks.py`:

```python
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.pause_state import PauseState
...
    def test_an_error_after_processing_a_result_early_propagates(self):
        ...
        scheduler.ingest_requests = mock.Mock()
        scheduler.pause_state = PauseState()
        scheduler.last_batch = None
```

- Put the import with the other `sglang.srt.managers` imports (they come after the `sys.path`/stub setup at the top of the file). Let isort place it.
- Change only line 219. Don't add `pause_state` to the shared `_scheduler()` helper: `test_an_error_when_idle_propagates` doesn't need it, and a smaller diff is easier to review. If a reviewer prefers the helper, move it.

## Experiment Plan
All on Kaggle 2× T4 after `kaggle_setup.sh sglang` (the SGLang venv is proven there; the Mac has none). Base = `main` at branch time; PR = the branch. Every result goes into the PR description.

| # | Experiment | Expected |
|---|---|---|
| E1 | Whole file at base, on GPU and with `CUDA_VISIBLE_DEVICES=""` | Both `FAILED (errors=1)` with the `pause_state` `AttributeError`; the other 2 tests `ok` |
| E2 | Whole file on the PR branch, GPU and no-GPU | `OK`, 3 tests |
| E3 | The test still checks something: on the PR branch, change `side_effect=RuntimeError("a move failed")` to `return_value=None` in a scratch copy of the working tree, so the flush no longer fails | The fixed test **fails** (`RuntimeError` not raised; `_Stop` instead). This shows it really checks error propagation. Then `git checkout -- <file>` |
| E4 | CI-style `run_suite.py` on base and PR, with `.venv/bin` on `PATH` | Base `✗ FAILED`, PR `✓ passed` |
| E5 | Sweep for the same race: `git grep -n "_engine_paused"` across `python/` and `test/`; list `Scheduler.__new__` tests that call `event_loop_*` / `on_idle` without `pause_state` | Only this file. Fold any further hits into the PR, with E1/E2 evidence |
| E6 | `sglang_tests.sh` default scope (`unit/managers` + `unit/mem_cache`) on the PR branch, to catch other tests that use the old flag | This file passes; failures otherwise match the first PR's E6b list (Rust-only files) |
| E7 | CI on the PR (a maintainer adds `run-ci`) | Green, or failures unrelated to this file |

### Kaggle cells
Settings: **Accelerator = GPU T4 x2**, **Internet = On**. Run as **Save & Run All (Commit)**. Replace `<BASE>` with the `main` commit the branch was cut from (Step 2 prints it).

**Cell 1: setup at the base**
```bash
%%bash
set -o pipefail
rm -rf /kaggle/working/inference-lab
git clone -q https://github.com/jjzz888/inference-lab.git /kaggle/working/inference-lab
SGLANG_REF=<BASE> RUN_TESTS=0 bash /kaggle/working/inference-lab/week01/kaggle_setup.sh sglang 2>&1 | tee /kaggle/working/setup_sglang.log | grep -E "==>|created:|WARN|ERROR|installed at|in venv|sglang"
```

**Cell 2: E1–E6**
```bash
%%bash
cd /root/lab/sglang
export PATH=/root/lab/sglang/.venv/bin:$PATH      # run_suite.py launches python3 from PATH
R=/kaggle/working/results/week02/pr2; mkdir -p $R
T=test/registered/unit/mem_cache/test_unified_scheduler_hooks.py
echo "$T" > /tmp/hooks_list.txt
{ nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
  echo "base: $(git rev-parse HEAD)"; } 2>&1 | tee $R/env.txt

echo "==> E1 base"
python $T -v > $R/e1_base_gpu.log 2>&1; tail -3 $R/e1_base_gpu.log
CUDA_VISIBLE_DEVICES="" python $T -v > $R/e1_base_nogpu.log 2>&1; tail -3 $R/e1_base_nogpu.log
grep -m1 "no attribute 'pause_state'" $R/e1_base_gpu.log || echo "!! E1 did NOT reproduce: stop and re-check"
echo "==> E4 base (CI-style)"
python test/run_suite.py --hw cpu --suite base-a-test-cpu --only-file-list /tmp/hooks_list.txt > $R/e4_ci_base.log 2>&1; tail -6 $R/e4_ci_base.log

git fetch -q https://github.com/jjzz888/sglang.git test/scheduler-hooks-pause-state && git checkout -q FETCH_HEAD
echo "pr: $(git rev-parse HEAD)" | tee -a $R/env.txt; git diff HEAD~1 --stat | tee -a $R/env.txt

echo "==> E2 PR"
python $T -v > $R/e2_pr_gpu.log 2>&1; tail -3 $R/e2_pr_gpu.log
CUDA_VISIBLE_DEVICES="" python $T -v > $R/e2_pr_nogpu.log 2>&1; tail -3 $R/e2_pr_nogpu.log
echo "==> E4 PR (CI-style)"
python test/run_suite.py --hw cpu --suite base-a-test-cpu --only-file-list /tmp/hooks_list.txt > $R/e4_ci_pr.log 2>&1; tail -6 $R/e4_ci_pr.log

echo "==> E3 mutation: flush no longer fails -> fixed test must fail"
sed -i 's/side_effect=RuntimeError("a move failed")/return_value=None/' $T
python $T TestOpportunisticCompactionErrors.test_an_error_after_processing_a_result_early_propagates -v > $R/e3_mutation.log 2>&1; tail -4 $R/e3_mutation.log
git checkout -q -- $T && git status --short

echo "==> E5 sweep"
{ echo "-- _engine_paused:"; git grep -n "_engine_paused" -- python test
  echo "-- Scheduler.__new__ tests that call an event loop / on_idle and never set pause_state:"
  for f in $(git grep -l "Scheduler.__new__" -- test); do
    if grep -qE "event_loop_|on_idle" "$f" && ! grep -q "pause_state" "$f"; then echo "$f"; fi
  done; } 2>&1 | tee $R/e5_sweep.txt

echo "==> E6 managers + mem_cache suite on PR branch"
bash /kaggle/working/inference-lab/week01/sglang_tests.sh > $R/e6_console.log 2>&1
cp /root/lab/logs/sglang_tests.log $R/e6_tests.log 2>/dev/null
grep '"passed": false' $R/e6_tests.log
```

**Cell 3: list the outputs**
```bash
%%bash
cp /kaggle/working/setup_sglang.log /kaggle/working/results/week02/pr2/
find /kaggle/working/results/week02/pr2 -type f | sort | xargs ls -la
```

Download with `kaggle kernels output <notebook> -p <dir>` and save to `results/week02/pr2/`.

## Steps
1. **Re-check for duplicates** right before starting: search issues and PRs for `test_unified_scheduler_hooks`, `TestOpportunisticCompactionErrors`, `pause_state`; run `git log origin/main -- test/registered/unit/mem_cache/test_unified_scheduler_hooks.py` and `git grep _engine_paused origin/main`. If it's fixed, stop.
2. **Branch from upstream** on the Mac (switch away from the first PR's branch; don't stack on it):
   ```bash
   cd ~/code/sglang && git fetch origin && git switch -c test/scheduler-hooks-pause-state origin/main && git rev-parse --short HEAD
   ```
3. **Make the edit** ([[#Fix]]) and run `pre-commit run --files test/registered/unit/mem_cache/test_unified_scheduler_hooks.py`.
4. **Commit and push to the fork:**
   ```bash
   git commit -s -m "[Test] mem_cache: Set pause_state in the opportunistic compaction error test"
   git push -u fork test/scheduler-hooks-pause-state
   ```
   Body: the merge-race explanation from [[#Findings]], plus the `Co-authored-by` trailer as in the first PR.
5. **Validate on Kaggle**: Cells 1–3 with `<BASE>` from Step 2. Check E1–E6 against the table.
6. **Open the PR** with `gh pr create --repo sgl-project/sglang --base main --head jjzz888:test/scheduler-hooks-pause-state --body-file <file>` (use `--body-file`: the first PR's inline `sed` broke on macOS). Ask for `run-ci` in the PR body.
7. **Link the two PRs.** If [#49164](https://github.com/sgl-project/sglang/pull/49164)'s CPU CI fails on this test, comment there that the failure is unrelated and fixed by this PR.
8. **Follow up**: answer reviews, rebase if `main` moves, and log the outcome in `LOG.md`.

## PR Outline
Title: `[Test] mem_cache: Set pause_state in the opportunistic compaction error test`

```markdown
## Motivation
`TestOpportunisticCompactionErrors.test_an_error_after_processing_a_result_early_propagates`
(`test/registered/unit/mem_cache/test_unified_scheduler_hooks.py`, in `base-a-test-cpu`) fails on
`main` on every host, with or without a GPU:

    AttributeError: 'Scheduler' object has no attribute 'pause_state'. Did you mean: 'init_pause_state'?

It's a merge race between two PRs merged on 2026-11-08:
- #49144 (08:42 UTC) moved `Scheduler._engine_paused` into `self.pause_state = PauseState()` and
  updated every test that builds a bare `Scheduler.__new__(Scheduler)`.
- #48008 (11:35 UTC, branch created before #49144) added this test, which still sets
  `scheduler._engine_paused = False`. `event_loop_overlap` now reads `self.pause_state.paused`.

## Modifications
Set `scheduler.pause_state = PauseState()` instead of `scheduler._engine_paused = False`, the same
change #49144 made in the other bare-scheduler tests. `PauseState()` starts unpaused, so the test's
setup means the same as before. Test-only change.

## Accuracy Tests / Speed Tests and Profiling
N/A (test-only change).

## Validation (Kaggle 2x T4; also with CUDA_VISIBLE_DEVICES="")
| Check | Base | PR |
|---|---|---|
| Whole file (3 tests), GPU and no GPU | FAILED (errors=1), AttributeError | OK |
| `run_suite.py --hw cpu --suite base-a-test-cpu`, this file | FAILED | passed |
| Flush made to succeed (mutation) | | the fixed test fails, so it still checks error propagation |
| `git grep _engine_paused` | 1 hit (this line) | 0 hits |

AI assistance: drafted with Claude (Claude Code); I reproduced and validated every result above.

As a new contributor I can't trigger CI. Could a maintainer add `run-ci`? Thanks!
```

When pasting into GitHub, write the PR references as `#49144` / `#48008` (GitHub links them). Optionally cc the two authors so they see the race. Only do that deliberately: GitHub notifies anyone `@`-mentioned.

## Status
- [x] Found in the first PR's E6b run (2026-11-08, Kaggle 2× T4, `61918f024d` on `c24285bde9`)
- [x] Root cause: #49144 × #48008 merge race; only stale use; no issue/PR (2026-11-08, `main` `3825fcd397`)
- [x] **Closed: fixed upstream.** 2026-11-13 re-check: [#49151](https://github.com/sgl-project/sglang/pull/49151) (`620413be1a`) made exactly this change on `main`. [#49178](https://github.com/sgl-project/sglang/pull/49178), opened the same day as the bug, was closed as superseded. `git grep _engine_paused` on `4f8b686753` finds nothing. The "time matters" risk came true within hours; next time, open a minimal PR the same day
- [ ] ~~Step 1 re-check right before branching~~ (not needed)
- [ ] Branch, edit, pre-commit, commit, push to fork
- [ ] E1–E6 on Kaggle; logs in `results/week02/pr2/`
- [ ] PR opened; CI triggered; linked from #49164 if its CPU CI hits this
