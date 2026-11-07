# 第 1 周（Mac）：每次开始实验时运行。必须用 source 运行（zsh 和 bash 都可以）：
#     source week01/mac_session_zh.sh
#   0. 如果还没设代理、且本机代理端口在监听（默认 http://127.0.0.1:7890），就自动使用它；
#      PROXY=http://host:port 指定其他地址，PROXY=none 跳过。设置会留在你的终端里。
#   1. 拉取 vLLM 和 SGLang 的 upstream 最新代码；如果你在默认分支上且没有未提交改动，就快进到最新
#      （如果在功能分支上，只报告落后多少个 commit，不改动分支；需要时自己 rebase）
#   2. 激活 vllm/.venv，并设置 VLLM_ENABLE_V1_MULTIPROCESSING=0（方便调试器打断点）
#   3. 只有在需要编译的部分（csrc/、cmake、构建配置）自上次构建后有变化时，才重新构建 vLLM
# 环境变量：CODE_DIR（默认 ~/code），SYNC=0 跳过 git 同步，REBUILD=1 强制重新构建。
# 一次性初始化脚本是 mac_init_zh.sh。
#
# 注意：这个文件是被 source 到你当前的 shell 里执行的，所以这里不用 `set -e`，
# 否则任何一条命令失败都会直接关掉你的终端。

# 如果是用 bash 直接运行（而不是 source），激活的虚拟环境不会留在你的终端里，所以提示并退出。
# （zsh 下 BASH_SOURCE 不存在，这个检查会被跳过。）
if [ -n "${BASH_VERSION:-}" ] && [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "Source this file instead of running it:  source $0"
  exit 1
fi

# 设置代理，逻辑和 mac_init_zh.sh 相同。因为本文件是被 source 的，这里 export 的变量会留在你的终端里，
# 之后在同一个终端里运行的 python / git / uv / HF 下载都会自动走代理。
_lab_proxy() {  # same logic as mac_init.sh; exports stay in your shell because this file is sourced
  local p hp host port
  # 已经设过代理就沿用，不覆盖。
  if [ -n "${https_proxy:-}${HTTPS_PROXY:-}" ]; then echo "proxy: already set (${https_proxy:-$HTTPS_PROXY})"; return 0; fi
  # 显式设置 PROXY=none 时不使用代理。
  [ "${PROXY:-}" = "none" ] && { echo "proxy: disabled (PROXY=none)"; return 0; }
  # 从代理地址里拆出 host 和 port。
  p="${PROXY:-http://127.0.0.1:7890}"; hp="${p#*://}"; host="${hp%%:*}"; port="${hp##*:}"; port="${port%%/*}"
  # 端口在监听（代理软件开着）才设置。
  if nc -z -G 2 "$host" "$port" 2>/dev/null; then
    export https_proxy="$p" http_proxy="$p" HTTPS_PROXY="$p" HTTP_PROXY="$p"
    # 本地服务（localhost 上的 vLLM / SGLang server）不走代理。
    export no_proxy="${no_proxy:-localhost,127.0.0.1,::1}" NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1}"
    echo "proxy: using $p"
  else
    echo "proxy: none ($host:$port not listening; set PROXY=http://host:port if you need one)"
  fi
}

# 同步一个仓库。参数：$1=仓库目录。
_lab_sync() {  # $1=repo dir
  # local 让这些变量只在函数内有效，不会污染你的 shell。
  local repo="$1" name branch default behind try ok
  # 仓库名（vllm / sglang），用于打印。
  name="$(basename "$repo")"
  # 仓库不存在时提示先运行初始化脚本，然后跳过。
  [ -d "$repo/.git" ] || { echo "$name: not cloned (run mac_init.sh)"; return 0; }
  # 从 origin（upstream）拉取最新提交，最多尝试 3 次，每次间隔 3 秒：
  # 走代理时连接偶尔会在传输中途断开（curl 56 / "bad pack header"），重试通常就能成功。
  ok=0
  for try in 1 2 3; do
    if git -C "$repo" fetch --quiet origin; then ok=1; break; fi
    [ "$try" -lt 3 ] && { echo "$name: fetch attempt $try failed, retrying..."; sleep 3; }
  done
  # 3 次都失败（比如离线或代理没开）时跳过同步，不影响后续步骤。
  if [ "$ok" != "1" ]; then
    echo "$name: fetch failed 3 times (offline? proxy down?), skipping sync"; return 0
  fi
  # 当前所在分支名；处于 detached HEAD 时为空。
  branch="$(git -C "$repo" symbolic-ref --short -q HEAD)"
  # upstream 的默认分支名（通常是 main），从 origin/HEAD 推断。
  default="$(git -C "$repo" symbolic-ref --short -q refs/remotes/origin/HEAD | sed 's#^origin/##')"
  # 推断不出来时退回 main。
  [ -n "$default" ] || default=main
  # 当前 HEAD 比 origin 的默认分支落后多少个 commit。
  behind="$(git -C "$repo" rev-list --count "HEAD..origin/$default")"
  # 只有在默认分支上、且没有未提交的已跟踪文件改动时，才自动更新。
  if [ "$branch" = "$default" ] && [ -z "$(git -C "$repo" status --porcelain --untracked-files=no)" ]; then
    if [ "$behind" -gt 0 ]; then
      # 只做快进合并（--ff-only）：如果本地 main 上有自己的提交，就不会产生合并提交，而是报错提示。
      git -C "$repo" merge --ff-only --quiet "origin/$default" \
        && echo "$name: fast-forwarded $behind commits -> $(git -C "$repo" rev-parse --short HEAD)" \
        || echo "$name: fast-forward failed (local commits on $default?)"
    else
      echo "$name: up to date ($(git -C "$repo" rev-parse --short HEAD))"
    fi
  else
    # 在功能分支上（或有未提交改动）时不动它，只报告落后情况。
    echo "$name: on '${branch:-detached}', $behind commits behind origin/$default (left as is)"
  fi
}

# 判断 vLLM 是否需要重新构建。参数：$1=vllm 目录。需要时打印原因并返回 0，不需要时返回 1。
_lab_needs_rebuild() {  # $1=vllm dir; prints the reason and returns 0 if a rebuild is needed
  local repo="$1" last changed
  # 用户设了 REBUILD=1，强制重新构建。
  if [ "${REBUILD:-0}" = "1" ]; then echo "REBUILD=1"; return 0; fi
  # 读取上次构建时记录的 commit。
  last="$(cat "$repo/.venv/.last_build_commit" 2>/dev/null)"
  # 没有记录（比如初始化时构建失败），保险起见重新构建。
  if [ -z "$last" ]; then echo "no record of the last build"; return 0; fi
  # 找出两类变化，只看需要编译的文件和构建配置：
  #   1) 上次构建之后新提交里的改动；2) 工作区里尚未提交的改动。
  # 路径直接写在命令里，因为 zsh 默认不会按空格拆分未加引号的变量。
  # 结果去重，最多显示 5 个文件。
  changed="$( { git -C "$repo" diff --name-only "$last" HEAD -- csrc cmake CMakeLists.txt setup.py \
                  pyproject.toml requirements/cpu.txt requirements/common.txt 2>/dev/null
               git -C "$repo" diff --name-only HEAD -- csrc cmake CMakeLists.txt setup.py \
                  pyproject.toml requirements/cpu.txt requirements/common.txt; } | sort -u | head -5)"
  # 有变化就打印变化的文件并返回 0（需要重新构建）。
  if [ -n "$changed" ]; then printf 'compiled/build files changed:\n%s\n' "$changed"; return 0; fi
  # 只有 Python 改动：editable 安装会直接生效，不需要重新构建。
  return 1
}

# 主流程。
_lab_main() {
  local code_dir="${CODE_DIR:-$HOME/code}" vllm reason
  vllm="$code_dir/vllm"
  # 还没有虚拟环境，说明没跑过初始化脚本。
  if [ ! -d "$vllm/.venv" ]; then echo "No $vllm/.venv; run mac_init.sh first."; return 1; fi

  # 第 0 步：设置代理（必须在 git fetch 之前）。
  echo "==> proxy"
  _lab_proxy

  # 第 1 步：同步两个仓库（SYNC=0 时跳过）。
  if [ "${SYNC:-1}" = "1" ]; then
    echo "==> git sync"
    _lab_sync "$vllm"
    _lab_sync "$code_dir/sglang"
  fi

  # 第 2 步：激活虚拟环境，并让引擎核心在当前进程运行（调试器断点能命中）。
  echo "==> venv"
  # shellcheck disable=SC1091
  . "$vllm/.venv/bin/activate"
  export VLLM_ENABLE_V1_MULTIPROCESSING=0
  # 打印当前 python 路径，确认用的是 vllm/.venv 里的解释器。
  echo "python: $(command -v python)"

  # 第 3 步：按需重新构建。
  if reason="$(_lab_needs_rebuild "$vllm")"; then
    echo "==> rebuilding vLLM ($reason)"
    # 在子 shell（括号）里构建：即使构建失败，也不会影响或关闭你当前的终端。
    # 构建日志写入 vllm_mac_build.log，避免刷屏。
    if ( cd "$vllm" && uv pip install -r requirements/cpu.txt \
           && uv pip install -e . > "$code_dir/vllm_mac_build.log" 2>&1 ); then
      # 构建成功才更新记录；失败时不更新，下次 source 会自动再试一次。
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

# 执行主流程。
_lab_main
# 删除这几个辅助函数，避免它们留在你的 shell 里。
unset -f _lab_proxy _lab_sync _lab_needs_rebuild _lab_main
