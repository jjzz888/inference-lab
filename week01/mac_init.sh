#!/usr/bin/env bash
# Week 1 (Mac, Apple silicon), ONE-TIME setup. Rerunning it is safe: finished steps are skipped.
#   1. clone vLLM + SGLang into $CODE_DIR (origin = upstream)
#   2. fork both on GitHub and add the fork as remote "fork" (needs `gh auth login`; FORK=0 skips)
#   3. vLLM CPU build from source in vllm/.venv, then a smoke test
# SGLang is for reading only on the Mac (no usable Mac path).
#
# Prereqs: Xcode 15.4+ with Command Line Tools (`xcode-select --install`), uv, gh.
# Usage: bash mac_init.sh        (CODE_DIR defaults to ~/code)
# Per session afterwards: source mac_session.sh
set -euo pipefail

CODE_DIR="${CODE_DIR:-$HOME/code}"
FORK="${FORK:-1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$CODE_DIR"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- 0. proxy
# git, uv and Hugging Face need GitHub/PyPI/HF access. If no proxy is set yet and a local proxy is
# listening (default http://127.0.0.1:7890, override with PROXY=...), use it. PROXY=none disables.
setup_proxy() {
  local p hp host port
  if [ -n "${https_proxy:-}${HTTPS_PROXY:-}" ]; then echo "proxy: already set (${https_proxy:-$HTTPS_PROXY})"; return 0; fi
  [ "${PROXY:-}" = "none" ] && { echo "proxy: disabled (PROXY=none)"; return 0; }
  p="${PROXY:-http://127.0.0.1:7890}"; hp="${p#*://}"; host="${hp%%:*}"; port="${hp##*:}"; port="${port%%/*}"
  if nc -z -G 2 "$host" "$port" 2>/dev/null; then
    export https_proxy="$p" http_proxy="$p" HTTPS_PROXY="$p" HTTP_PROXY="$p"
    # Keep local servers (vLLM/SGLang on localhost) off the proxy.
    export no_proxy="${no_proxy:-localhost,127.0.0.1,::1}" NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1}"
    echo "proxy: using $p"
  else
    echo "proxy: none ($host:$port not listening; set PROXY=http://host:port if you need one)"
  fi
}
say "Proxy"
setup_proxy

# ---------------------------------------------------------------- 1. clone
say "Clone"
[ -d "$CODE_DIR/vllm/.git" ]   || git clone https://github.com/vllm-project/vllm.git "$CODE_DIR/vllm"
[ -d "$CODE_DIR/sglang/.git" ] || git clone https://github.com/sgl-project/sglang.git "$CODE_DIR/sglang"

# ---------------------------------------------------------------- 2. fork
add_fork() {  # $1=repo dir  $2=upstream owner/name
  cd "$1"
  if git remote get-url fork >/dev/null 2>&1; then
    echo "$2: remote 'fork' already set -> $(git remote get-url fork)"
  else
    # Creates the fork on your GitHub account if it doesn't exist yet; only the default branch.
    gh repo fork --remote --remote-name fork --default-branch-only
  fi
  # Make `gh pr create` / `gh issue list` target upstream, not the fork.
  gh repo set-default "$2" >/dev/null
  git remote -v
}
if [ "$FORK" = "1" ]; then
  say "Fork + remotes"
  if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    add_fork "$CODE_DIR/vllm"   vllm-project/vllm
    add_fork "$CODE_DIR/sglang" sgl-project/sglang
  else
    echo "WARN: gh missing or not logged in; skipping forks. Run 'gh auth login', then rerun this script."
  fi
fi

# ---------------------------------------------------------------- 3. vLLM CPU build
say "vLLM CPU build"
cd "$CODE_DIR/vllm"
[ -d .venv ] || uv venv --python 3.12 --seed --managed-python

# An old uv can't parse vLLM's [tool.uv] table and silently ignores ALL of it
# (no-build-isolation for torch, excluding openai-harmony), giving a subtly broken env.
uv_too_old() {
  local out
  out="$(uv pip list --python .venv/bin/python 2>&1 || true)"
  case "$out" in *"Failed to parse"*) return 0 ;; *) return 1 ;; esac
}
if uv_too_old; then
  echo "$(uv --version) can't read vLLM's pyproject.toml [tool.uv]; running 'uv self update'"
  uv self update || true
  if uv_too_old; then
    echo "ERROR: uv is still too old ($(uv --version)). Update it (e.g. 'brew upgrade uv' or 'pip install -U uv'), then rerun."
    exit 1
  fi
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# VLLM_TARGET_DEVICE=cpu is set automatically on macOS.
uv pip install -r requirements/cpu.txt
uv pip install -e . 2>&1 | tee "$CODE_DIR/vllm_mac_build.log"
# If the build fails on missing headers (<map>, <cstddef>): reinstall Command Line Tools.
git rev-parse HEAD > .venv/.last_build_commit   # mac_session.sh uses this to decide on rebuilds

say "Smoke test"
# Run the engine core in-process so debugger breakpoints in the scheduler hit.
export VLLM_ENABLE_V1_MULTIPROCESSING=0
python "$HERE/smoke_offline.py"

say "Done"
echo "Open $CODE_DIR/vllm in your IDE with $CODE_DIR/vllm/.venv/bin/python as the interpreter."
echo "Each session from now on:  source $HERE/mac_session.sh"
