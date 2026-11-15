# Issue Shortlist
Week 1 community task: shortlist 5 issues in vLLM and SGLang to work on or learn from. Checked on 2026-11-15.

## Table of Contents
- [[#What the Search Found]]
- [[#Shortlist]]
- [[#Where the Next PR Should Come From]]

## What the Search Found
Two searches:
- Every open `good first issue`: 14 in vLLM, 49 in SGLang.
- Open issues from the last two months about the scheduler, prefix caching and the radix cache.

**Nearly every issue already has a PR.** Many PRs were opened by the issue's own author together with the issue. Claiming an unclaimed `good first issue` is no longer a realistic path. Popular ones have several competing PRs:
- SGLang [#6357](https://github.com/sgl-project/sglang/issues/6357), the queue-latency metric: 5 PRs.
- SGLang [#5979](https://github.com/sgl-project/sglang/issues/5979), the KV cache usage metric: 4 PRs.
- vLLM [#21948](https://github.com/vllm-project/vllm/issues/21948), parallel sampling tests: 6 PRs.

The issues below were chosen because they touch the code traced in [request_lifecycle.md](../week01/request_lifecycle.md) and their PRs are not merged yet. They leave room to review, reproduce on a T4, or take over an abandoned fix.

## Shortlist
| # | Issue | Related to | Status (2026-11-15) | What to do |
|---|---|---|---|---|
| 1 | [sgl-project/sglang#48623](https://github.com/sgl-project/sglang/issues/48623) Each radix cache eviction rebuilds a heap over all evictable leaves | RadixAttention eviction. The vLLM counterpart is the least-recently-used order of `free_block_queue` | Two competing PRs, [#48939](https://github.com/sgl-project/sglang/pull/48939) (11/06) and [#49974](https://github.com/sgl-project/sglang/pull/49974) (11/12), 1 review each | Read both PRs, write an eviction microbenchmark on the Mac with `RadixCache`, post the numbers on the PRs. Doubles as SGLang source reading |
| 2 | [sgl-project/sglang#45510](https://github.com/sgl-project/sglang/issues/45510) `--max-queued-requests 0` with priority scheduling crashes the scheduler on an empty queue | Admission of waiting requests (question 3) | PR [#45827](https://github.com/sgl-project/sglang/pull/45827), no review since 10/22 | Reproduce, review the PR or add a regression test. The smallest and most beginner-friendly of the five |
| 3 | [sgl-project/sglang#45812](https://github.com/sgl-project/sglang/issues/45812) Priority preemption requeues victims in set order, which can differ across ranks | Preemption and victim choice (questions 4 and 5) | PR [#45813](https://github.com/sgl-project/sglang/pull/45813), opened with the issue; a maintainer confirmed the root cause | Review it and compare with vLLM's victim choice (`scheduler.py:948`). A good case study |
| 4 | [vllm-project/vllm#57650](https://github.com/vllm-project/vllm/issues/57650) When scheduler initialization fails, the worker process does not exit | EngineCore and worker process lifecycle (question 2) | PR [#57657](https://github.com/vllm-project/vllm/pull/57657) stalled since 10/12; someone asked for updates on the issue | Ask on the PR whether the author is still on it. If there is no answer, offer on the issue to take it over. The one most likely to become yours |
| 5 | [sgl-project/sglang#41124](https://github.com/sgl-project/sglang/issues/41124) Penalties use one-step-stale history under the overlap scheduler | SGLang's `event_loop_overlap`, the counterpart of vLLM async scheduling | PR [#41290](https://github.com/sgl-project/sglang/pull/41290) is active (7 reviews, 24 comments) | Read only: watch how maintainers review a correctness fix |

## Where the Next PR Should Come From
Both open PRs came from bugs found by running the test suites on a Kaggle T4:
- [vllm-project/vllm#66355](https://github.com/vllm-project/vllm/pull/66355): a pipeline-parallel GPU-count check.
- [sgl-project/sglang#49164](https://github.com/sgl-project/sglang/pull/49164): an FP8 capability gate.

The T4 (SM 7.5) is old, and most contributors test on H100 or RTX 4090, so portability bugs on older GPUs go unnoticed. That is an advantage. Running more test directories on the T4 is a better source for the second PR than competing for labelled issues. Candidates:
- vLLM `tests/v1/` beyond `core`
- SGLang `test/registered/unit/` beyond `managers` and `mem_cache`

A bug found this way also gives the PR description a concrete reproduction.
