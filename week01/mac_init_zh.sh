#!/usr/bin/env bash
# 第 1 周（Mac，Apple 芯片）：一次性初始化。重复运行是安全的，已完成的步骤会被跳过。
#   1. 把 vLLM 和 SGLang 克隆到 $CODE_DIR（origin 指向官方 upstream 仓库）
#   2. 在 GitHub 上 fork 两个仓库，并把 fork 添加为名为 "fork" 的 remote（需要先 `gh auth login`；FORK=0 可跳过）
#   3. 在 vllm/.venv 中从源码构建 vLLM CPU 版本，然后运行 smoke test
# SGLang 在 Mac 上只用于阅读源码；Mac 上没有值得使用的执行路径。
#
# 前置条件：Xcode 15.4 或更高版本及其命令行工具（`xcode-select --install`）、uv、gh。
# 使用方法：bash mac_init_zh.sh        （未设置 CODE_DIR 时默认为 ~/code）
# 之后每次开始实验：source mac_session_zh.sh

# 严格模式：
# -e：任何命令失败就退出；-u：使用未定义变量时报错；pipefail：管道中任何命令失败都会使整个管道失败。
set -euo pipefail

# 代码存放目录：优先使用外部设置的 CODE_DIR，否则默认为用户主目录下的 code 文件夹。
CODE_DIR="${CODE_DIR:-$HOME/code}"
# 是否执行 fork 步骤：默认 1（执行），设为 0 则跳过。
FORK="${FORK:-1}"
# 获取当前脚本所在目录的绝对路径，后面用它定位 smoke_offline.py。
HERE="$(cd "$(dirname "$0")" && pwd)"
# 创建代码存放目录；如果目录已存在则不报错。
mkdir -p "$CODE_DIR"

# 打印带颜色的步骤标题，方便在长输出中找到每个阶段。
say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- 0. 代理
# git、uv 和 Hugging Face 都需要访问 GitHub / PyPI / HF。如果当前还没设代理，并且本机有代理在监听
# （默认 http://127.0.0.1:7890，可用 PROXY=... 指定其他地址），就自动使用它。PROXY=none 可关闭。
setup_proxy() {
  local p hp host port
  # 已经设过代理（比如你在终端里手动 export 过），就沿用，不覆盖。
  if [ -n "${https_proxy:-}${HTTPS_PROXY:-}" ]; then echo "proxy: already set (${https_proxy:-$HTTPS_PROXY})"; return 0; fi
  # 显式设置 PROXY=none 时不使用代理。
  [ "${PROXY:-}" = "none" ] && { echo "proxy: disabled (PROXY=none)"; return 0; }
  # 从代理地址里拆出 host 和 port，例如 http://127.0.0.1:7890 -> 127.0.0.1 和 7890。
  p="${PROXY:-http://127.0.0.1:7890}"; hp="${p#*://}"; host="${hp%%:*}"; port="${hp##*:}"; port="${port%%/*}"
  # 用 nc 测试端口是否在监听（-G 2：最多等 2 秒）；代理软件没开时就不设，避免所有请求都连不上。
  if nc -z -G 2 "$host" "$port" 2>/dev/null; then
    # 大小写两套都设：不同工具读取的变量名不一样。
    export https_proxy="$p" http_proxy="$p" HTTPS_PROXY="$p" HTTP_PROXY="$p"
    # 本地服务（localhost 上的 vLLM / SGLang server）不走代理。
    export no_proxy="${no_proxy:-localhost,127.0.0.1,::1}" NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1}"
    echo "proxy: using $p"
  else
    echo "proxy: none ($host:$port not listening; set PROXY=http://host:port if you need one)"
  fi
}
say "Proxy"
setup_proxy

# ---------------------------------------------------------------- 1. 克隆
say "Clone"
# 如果 vLLM 尚未克隆到目标目录，就从 GitHub 克隆；已有 Git 仓库时跳过。
[ -d "$CODE_DIR/vllm/.git" ]   || git clone https://github.com/vllm-project/vllm.git "$CODE_DIR/vllm"
# 如果 SGLang 尚未克隆到目标目录，就从 GitHub 克隆；已有 Git 仓库时跳过。
[ -d "$CODE_DIR/sglang/.git" ] || git clone https://github.com/sgl-project/sglang.git "$CODE_DIR/sglang"

# ---------------------------------------------------------------- 2. fork
# 为一个仓库添加 fork remote。参数：$1=本地仓库目录，$2=upstream 的 owner/name。
add_fork() {  # $1=repo dir  $2=upstream owner/name
  # 进入仓库目录，gh 会根据当前目录的 origin 判断要 fork 哪个仓库。
  cd "$1"
  # 如果已经有名为 fork 的 remote，说明之前做过，直接打印它的地址。
  if git remote get-url fork >/dev/null 2>&1; then
    echo "$2: remote 'fork' already set -> $(git remote get-url fork)"
  else
    # 在你的 GitHub 账号下创建 fork（已存在则复用），并添加为 remote "fork"；
    # --default-branch-only 只 fork 默认分支，速度更快、体积更小。origin 保持指向 upstream。
    gh repo fork --remote --remote-name fork --default-branch-only
  fi
  # 把 gh 的默认仓库设为 upstream，这样 `gh pr create` / `gh issue list` 针对的是官方仓库而不是 fork。
  gh repo set-default "$2" >/dev/null
  # 显示所有 remote，确认 origin=upstream、fork=你的 fork。
  git remote -v
}
# 只有 FORK=1 时才执行 fork 步骤。
if [ "$FORK" = "1" ]; then
  say "Fork + remotes"
  # 检查 gh 已安装并且已登录；否则给出提示并跳过，不让整个脚本失败。
  if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    add_fork "$CODE_DIR/vllm"   vllm-project/vllm
    add_fork "$CODE_DIR/sglang" sgl-project/sglang
  else
    echo "WARN: gh missing or not logged in; skipping forks. Run 'gh auth login', then rerun this script."
  fi
fi

# ---------------------------------------------------------------- 3. vLLM CPU 构建
say "vLLM CPU build"
# 进入 vLLM 源码目录，后续依赖安装与构建都在这里执行。
cd "$CODE_DIR/vllm"
# 虚拟环境不存在时创建它：使用 Python 3.12、预装基础工具（--seed），并由 uv 管理 Python 解释器。
[ -d .venv ] || uv venv --python 3.12 --seed --managed-python

# 旧版 uv 无法解析 vLLM pyproject.toml 中的 [tool.uv]，会悄悄忽略整个配置表
# （包括 torch 的 no-build-isolation、排除 openai-harmony），装出一个有隐患的环境。
# 判断方法：在 vLLM 目录下运行一个轻量的 uv 命令，看输出里有没有 "Failed to parse"。
uv_too_old() {
  local out
  # 把输出存进变量再判断，避免 `| grep -q` 提前关闭管道导致 pipefail 误判；|| true 防止 set -e 退出。
  out="$(uv pip list --python .venv/bin/python 2>&1 || true)"
  case "$out" in *"Failed to parse"*) return 0 ;; *) return 1 ;; esac
}
# 版本太旧时先尝试自动升级。
if uv_too_old; then
  echo "$(uv --version) can't read vLLM's pyproject.toml [tool.uv]; running 'uv self update'"
  # 用 brew 或 pip 安装的 uv 不支持 self update，失败也继续往下走，交给下面的二次检查。
  uv self update || true
  # 升级后仍然太旧，就提示手动升级并退出，不继续构建。
  if uv_too_old; then
    echo "ERROR: uv is still too old ($(uv --version)). Update it (e.g. 'brew upgrade uv' or 'pip install -U uv'), then rerun."
    exit 1
  fi
fi

# shellcheck 的 SC1091 提示表示它无法静态找到下面要加载的虚拟环境脚本；此处是有意加载。
# shellcheck disable=SC1091
# 激活虚拟环境，使后续的 Python 和包安装命令使用项目自己的环境。
source .venv/bin/activate

# macOS 会自动将 VLLM_TARGET_DEVICE 设置为 cpu。
# 安装 vLLM CPU 构建所需的依赖。
uv pip install -r requirements/cpu.txt
# 以 editable（可编辑）模式安装当前源码；2>&1 将错误输出合并到标准输出，
# tee 会同时在终端显示构建日志并将其保存到指定文件。
uv pip install -e . 2>&1 | tee "$CODE_DIR/vllm_mac_build.log"
# 如果构建因缺少 <map>、<cstddef> 等头文件而失败，重新安装 Xcode 命令行工具。
# 记录本次构建对应的 commit；mac_session_zh.sh 用它判断之后是否需要重新构建。
git rev-parse HEAD > .venv/.last_build_commit

say "Smoke test"
# 让引擎核心在当前进程中运行，这样调试器在调度器里打的断点才能命中。
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# 运行离线 smoke test，快速检查刚安装的环境是否能正常工作。
python "$HERE/smoke_offline.py"

say "Done"
# 提示 IDE 中应使用的 Python 解释器路径。
echo "Open $CODE_DIR/vllm in your IDE with $CODE_DIR/vllm/.venv/bin/python as the interpreter."
# 提示之后每次开始实验时要 source 的脚本。
echo "Each session from now on:  source $HERE/mac_session_zh.sh"
