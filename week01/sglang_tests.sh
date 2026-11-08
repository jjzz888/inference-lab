#!/usr/bin/env bash
# Week 1: SGLang CPU unit tests through SGLang's own CI runner (test/run_suite.py).
# Run AFTER kaggle_setup.sh in the same session (needs ~/lab/sglang/.venv).
#
# Not part of kaggle_setup.sh on purpose: setup already runs vLLM's tests (~34 min) plus the
# JIT warm-up, and these are an occasional check (first run, new SGLang commit, before an
# SGLang PR), not something every experiment session needs.
#
# Default scope mirrors vLLM's tests/v1/core: the scheduler (unit/managers, 102 files)
# and KV/radix cache (unit/mem_cache, 194 files); 255 of them are registered in base-a-test-cpu. SGLang's GPU pre-flight suite
# (base-a-test-1-gpu-small) is skipped: it launches gated meta-llama/Llama-3.1-8B-Instruct,
# whose FP16 weights alone (~16 GB) don't fit a 15 GB T4.
#
# Usage:
#   bash sglang_tests.sh                     # managers + mem_cache (255 registered files, ~60 min)
#   SCOPE=full bash sglang_tests.sh          # whole base-a-test-cpu (~1160 files, ~4.8 h est)
#   DIRS="test/registered/unit/managers" bash sglang_tests.sh   # custom dirs (repo-relative)
#
# Env overrides:
#   WORKDIR     where kaggle_setup.sh put the repos (default: $HOME/lab)
#   LOG_DIR     (default: $WORKDIR/logs)
#   SCOPE       core | full (default: core)
#   DIRS        space-separated test dirs for SCOPE=core (default: unit/managers unit/mem_cache)
#   FORK_BATCH  files per forked worker batch (default: 1 for core, 8 for full; >1 is faster)
#   EXTRA_DEPS  extra pip packages to install if a test reports ModuleNotFoundError
set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/lab}"
LOG_DIR="${LOG_DIR:-$WORKDIR/logs}"
SCOPE="${SCOPE:-core}"
DIRS="${DIRS:-test/registered/unit/managers test/registered/unit/mem_cache}"
REPO="$WORKDIR/sglang"
PY="$REPO/.venv/bin/python"
mkdir -p "$LOG_DIR"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

if [ ! -x "$PY" ]; then
  echo "ERROR: $PY not found; run kaggle_setup.sh (or 'kaggle_setup.sh sglang') in this session first"
  exit 1
fi
cd "$REPO"
echo "sglang @ $(git rev-parse --short HEAD) ($(git log -1 --format=%ci))"

say "Test deps"
# --python pins the venv: the Kaggle image presets UV_SYSTEM_PYTHON=true.
# SGLang's full [test] extra pulls heavy packages (lm-eval, ...); install only what's needed.
# shellcheck disable=SC2086
UV_SYSTEM_PYTHON=false uv pip install --python "$PY" -q pytest expecttest parameterized ${EXTRA_DEPS:-}

args=(--hw cpu --suite base-a-test-cpu --continue-on-error)
if [ "$SCOPE" = "full" ]; then
  say "Scope: full base-a-test-cpu"
  batch="${FORK_BATCH:-8}"
else
  list="$LOG_DIR/sglang_unit_files.txt"
  # shellcheck disable=SC2086
  find $DIRS -name 'test_*.py' | sort > "$list"
  say "Scope: $(wc -l < "$list" | tr -d ' ') files from: $DIRS"
  args+=(--only-file-list "$list")
  batch="${FORK_BATCH:-1}"
fi
if [ "$batch" -gt 1 ]; then args+=(--fork-worker-batch-size "$batch"); fi

say "Running test/run_suite.py ${args[*]}"
log="$LOG_DIR/sglang_tests.log"
t0=$SECONDS
# The venv's bin on PATH: tests may spawn subprocesses that need the venv's python/CLIs.
# Failures shouldn't abort the script; the summary below is what matters.
if PATH="$REPO/.venv/bin:$PATH" "$PY" test/run_suite.py "${args[@]}" > "$log" 2>&1; then
  status="all passed"
else
  status="some tests failed"
fi

say "Done in $(( (SECONDS - t0) / 60 )) min: $status. Full log: $log"
tail -n 40 "$log"
