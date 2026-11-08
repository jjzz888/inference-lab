#!/usr/bin/env bash
# 第 1 周：用 SGLang 自带的 CI 运行器（test/run_suite.py）跑 SGLang 的 CPU 单元测试。
# 必须在同一个 session 里先跑完 kaggle_setup.sh（需要 ~/lab/sglang/.venv）。
#
# 故意不放进 kaggle_setup.sh：setup 已经包含 vLLM 测试（约 34 分钟）和 JIT 预热，
# 而 SGLang 测试只是偶尔做一次的验证（第一次跑、换了 SGLang 版本、准备给 SGLang 提 PR 前），
# 不是每次实验都需要。
#
# 默认范围对应 vLLM 的 tests/v1/core：调度器（unit/managers，95 个文件，约 18 分钟）
# 和 KV/radix cache（unit/mem_cache，158 个文件，约 26 分钟）。
# 不跑 SGLang 的 GPU 快速检查（base-a-test-1-gpu-small）：它会启动需要申请权限的
# meta-llama/Llama-3.1-8B-Instruct，FP16 权重本身就约 16 GB，放不进 15 GB 的 T4。
#
# 使用方法：
#   bash sglang_tests_zh.sh                     # managers + mem_cache（约 253 个文件，约 45 分钟）
#   SCOPE=full bash sglang_tests_zh.sh          # 完整的 base-a-test-cpu（约 1160 个文件，预计约 4.8 小时）
#   DIRS="test/registered/unit/managers" bash sglang_tests_zh.sh   # 自定义目录（相对仓库根目录）
#
# 环境变量：
#   WORKDIR     kaggle_setup.sh 放仓库的目录（默认 $HOME/lab）
#   LOG_DIR     日志目录（默认 $WORKDIR/logs）
#   SCOPE       core | full（默认 core）
#   DIRS        SCOPE=core 时要跑的测试目录，空格分隔（默认 unit/managers 和 unit/mem_cache）
#   FORK_BATCH  每个 fork 子进程批量跑的文件数（core 默认 1，full 默认 8；大于 1 会更快）
#   EXTRA_DEPS  某个测试报 ModuleNotFoundError 时，额外要安装的包
# 严格模式：命令失败、使用未定义变量或管道中任一命令失败时退出。
set -euo pipefail

# 读取环境变量，未设置时使用默认值。
WORKDIR="${WORKDIR:-$HOME/lab}"
LOG_DIR="${LOG_DIR:-$WORKDIR/logs}"
SCOPE="${SCOPE:-core}"
DIRS="${DIRS:-test/registered/unit/managers test/registered/unit/mem_cache}"
REPO="$WORKDIR/sglang"
PY="$REPO/.venv/bin/python"
mkdir -p "$LOG_DIR"

# 打印带颜色的步骤标题。
say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# SGLang 的 venv 不存在，说明这个 session 还没跑过 setup。
if [ ! -x "$PY" ]; then
  echo "ERROR: $PY not found; run kaggle_setup.sh (or 'kaggle_setup.sh sglang') in this session first"
  exit 1
fi
cd "$REPO"
# 打印当前 SGLang 的 commit，记录结果时要用。
echo "sglang @ $(git rev-parse --short HEAD) ($(git log -1 --format=%ci))"

say "Test deps"
# --python 显式指向 venv：Kaggle 镜像预设了 UV_SYSTEM_PYTHON=true，不指定会装进系统 Python。
# SGLang 官方的 [test] 依赖组包含 lm-eval 等很重的包，这里只装必需的；缺什么再用 EXTRA_DEPS 补。
# shellcheck disable=SC2086
UV_SYSTEM_PYTHON=false uv pip install --python "$PY" -q pytest expecttest ${EXTRA_DEPS:-}

# run_suite.py 的公共参数：CPU 硬件、base-a-test-cpu 这个 suite、某个文件失败后继续跑。
args=(--hw cpu --suite base-a-test-cpu --continue-on-error)
if [ "$SCOPE" = "full" ]; then
  say "Scope: full base-a-test-cpu"
  # 全量跑时默认每个 fork 子进程批量跑 8 个文件，先预加载公共模块，明显更快。
  batch="${FORK_BATCH:-8}"
else
  # 把要跑的测试文件列进一个清单，run_suite.py 只会跑清单里、且注册在这个 suite 下的文件。
  list="$LOG_DIR/sglang_unit_files.txt"
  # shellcheck disable=SC2086
  find $DIRS -name 'test_*.py' | sort > "$list"
  say "Scope: $(wc -l < "$list" | tr -d ' ') files from: $DIRS"
  args+=(--only-file-list "$list")
  batch="${FORK_BATCH:-1}"
fi
# 批量大于 1 时才加这个参数；默认 1 表示每个文件单独启动一个进程，和 CI 的行为一致。
if [ "$batch" -gt 1 ]; then args+=(--fork-worker-batch-size "$batch"); fi

say "Running test/run_suite.py ${args[*]}"
log="$LOG_DIR/sglang_tests.log"
# 用 bash 内置的 $SECONDS 计时。
t0=$SECONDS
# 把 venv 的 bin 加进 PATH：测试可能会启动子进程，子进程需要找到 venv 里的 python 和命令。
# 测试失败不应让脚本中途退出（用 if 包住），最后的汇总才是重点。
if PATH="$REPO/.venv/bin:$PATH" "$PY" test/run_suite.py "${args[@]}" > "$log" 2>&1; then
  status="all passed"
else
  status="some tests failed"
fi

# 打印总耗时、结果，以及日志末尾 40 行（run_suite.py 的汇总在最后）。
say "Done in $(( (SECONDS - t0) / 60 )) min: $status. Full log: $log"
tail -n 40 "$log"
