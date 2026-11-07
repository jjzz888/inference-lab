# Week 1 (Mac), run at the START OF EVERY SESSION. Must be sourced (works in zsh and bash):
#     source week01/mac_session.sh
#   0. use the local proxy (default http://127.0.0.1:7890) if none is set and it's listening;
#      PROXY=http://host:port to pick another, PROXY=none to skip. Exports stay in your shell.
#   1. fetch upstream for vLLM + SGLang; fast-forward the default branch if you're on it and it's clean
#      (on a feature branch it only reports how far behind you are; rebase yourself)
#   2. activate vllm/.venv and set VLLM_ENABLE_V1_MULTIPROCESSING=0 (debugger-friendly)
#   3. rebuild vLLM only if compiled parts changed since the last build (csrc/, cmake, build config)
# Env: CODE_DIR (default ~/code), SYNC=0 skip git sync, REBUILD=1 force a rebuild.
# One-time setup is mac_init.sh.

if [ -n "${BASH_VERSION:-}" ] && [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "Source this file instead of running it:  source $0"
  exit 1
fi

_lab_proxy() {  # same logic as mac_init.sh; exports stay in your shell because this file is sourced
  local p hp host port
  if [ -n "${https_proxy:-}${HTTPS_PROXY:-}" ]; then echo "proxy: already set (${https_proxy:-$HTTPS_PROXY})"; return 0; fi
  [ "${PROXY:-}" = "none" ] && { echo "proxy: disabled (PROXY=none)"; return 0; }
  p="${PROXY:-http://127.0.0.1:7890}"; hp="${p#*://}"; host="${hp%%:*}"; port="${hp##*:}"; port="${port%%/*}"
  if nc -z -G 2 "$host" "$port" 2>/dev/null; then
    export https_proxy="$p" http_proxy="$p" HTTPS_PROXY="$p" HTTP_PROXY="$p"
    export no_proxy="${no_proxy:-localhost,127.0.0.1,::1}" NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1}"
    echo "proxy: using $p"
  else
    echo "proxy: none ($host:$port not listening; set PROXY=http://host:port if you need one)"
  fi
}

_lab_sync() {  # $1=repo dir
  local repo="$1" name branch default behind try ok
  name="$(basename "$repo")"
  [ -d "$repo/.git" ] || { echo "$name: not cloned (run mac_init.sh)"; return 0; }
  # Retry: proxied connections sometimes drop mid-transfer (curl 56 / "bad pack header").
  ok=0
  for try in 1 2 3; do
    if git -C "$repo" fetch --quiet origin; then ok=1; break; fi
    [ "$try" -lt 3 ] && { echo "$name: fetch attempt $try failed, retrying..."; sleep 3; }
  done
  if [ "$ok" != "1" ]; then
    echo "$name: fetch failed 3 times (offline? proxy down?), skipping sync"; return 0
  fi
  branch="$(git -C "$repo" symbolic-ref --short -q HEAD)"
  default="$(git -C "$repo" symbolic-ref --short -q refs/remotes/origin/HEAD | sed 's#^origin/##')"
  [ -n "$default" ] || default=main
  behind="$(git -C "$repo" rev-list --count "HEAD..origin/$default")"
  if [ "$branch" = "$default" ] && [ -z "$(git -C "$repo" status --porcelain --untracked-files=no)" ]; then
    if [ "$behind" -gt 0 ]; then
      git -C "$repo" merge --ff-only --quiet "origin/$default" \
        && echo "$name: fast-forwarded $behind commits -> $(git -C "$repo" rev-parse --short HEAD)" \
        || echo "$name: fast-forward failed (local commits on $default?)"
    else
      echo "$name: up to date ($(git -C "$repo" rev-parse --short HEAD))"
    fi
  else
    echo "$name: on '${branch:-detached}', $behind commits behind origin/$default (left as is)"
  fi
}

_lab_needs_rebuild() {  # $1=vllm dir; prints the reason and returns 0 if a rebuild is needed
  local repo="$1" last changed
  if [ "${REBUILD:-0}" = "1" ]; then echo "REBUILD=1"; return 0; fi
  last="$(cat "$repo/.venv/.last_build_commit" 2>/dev/null)"
  if [ -z "$last" ]; then echo "no record of the last build"; return 0; fi
  # Committed changes since the last build + uncommitted edits, both limited to compiled/build files.
  # (Paths written out literally: zsh doesn't word-split unquoted variables.)
  changed="$( { git -C "$repo" diff --name-only "$last" HEAD -- csrc cmake CMakeLists.txt setup.py \
                  pyproject.toml requirements/cpu.txt requirements/common.txt 2>/dev/null
               git -C "$repo" diff --name-only HEAD -- csrc cmake CMakeLists.txt setup.py \
                  pyproject.toml requirements/cpu.txt requirements/common.txt; } | sort -u | head -5)"
  if [ -n "$changed" ]; then printf 'compiled/build files changed:\n%s\n' "$changed"; return 0; fi
  return 1
}

_lab_main() {
  local code_dir="${CODE_DIR:-$HOME/code}" vllm reason
  vllm="$code_dir/vllm"
  if [ ! -d "$vllm/.venv" ]; then echo "No $vllm/.venv; run mac_init.sh first."; return 1; fi

  echo "==> proxy"
  _lab_proxy

  if [ "${SYNC:-1}" = "1" ]; then
    echo "==> git sync"
    _lab_sync "$vllm"
    _lab_sync "$code_dir/sglang"
  fi

  echo "==> venv"
  # shellcheck disable=SC1091
  . "$vllm/.venv/bin/activate"
  export VLLM_ENABLE_V1_MULTIPROCESSING=0
  echo "python: $(command -v python)"

  if reason="$(_lab_needs_rebuild "$vllm")"; then
    echo "==> rebuilding vLLM ($reason)"
    # Subshell so a failed build doesn't kill your shell.
    if ( cd "$vllm" && uv pip install -r requirements/cpu.txt \
           && uv pip install -e . > "$code_dir/vllm_mac_build.log" 2>&1 ); then
      git -C "$vllm" rev-parse HEAD > "$vllm/.venv/.last_build_commit"
      echo "rebuild OK"
    else
      echo "rebuild FAILED; see $code_dir/vllm_mac_build.log"
    fi
  else
    echo "==> no rebuild needed (Python-only changes take effect via the editable install)"
  fi
  echo "==> ready: cd $vllm"
}

_lab_main
unset -f _lab_proxy _lab_sync _lab_needs_rebuild _lab_main
