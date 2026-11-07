#!/usr/bin/env bash
# 第 1 周：在 Kaggle GPU（2 张 T4）会话中从源码构建 vLLM 和 SGLang。
# 每个 Kaggle 会话都会从干净环境开始，因此每次会话启动时都要运行一次。
#
# 用法：
#   bash kaggle_setup_zh.sh            # 安装 vLLM 和 SGLang
#   bash kaggle_setup_zh.sh vllm       # 只安装 vLLM
#   bash kaggle_setup_zh.sh sglang     # 只安装 SGLang
#
# 可通过环境变量覆盖的设置：
#   WORKDIR          源码仓库和虚拟环境的目录（默认：$HOME/lab，避开 /kaggle/working 的 20 GB 输出配额）
#   VLLM_REF         vLLM 使用的 Git ref（默认：main）
#   SGLANG_REF       SGLang 使用的 Git ref（默认：驱动支持 CUDA 13 时为 main，否则为 v0.5.19）
#   VLLM_FULL_BUILD=1 完整编译 C++/CUDA，而不是复用预编译内核（Kaggle CPU 上可能耗时数小时，通常不建议）
#   RUN_TESTS=0      跳过快速测试子集
#   WARMUP_HYBRID=0  跳过测试前的 JIT 缓存预热（省 2 到 8 分钟，但 hybrid 前缀缓存测试
#                    很可能会撞上它 480 秒的 server 启动时限）
#   SGLANG_BUILD_RUST_EXTS  要编译哪些 SGLang Rust 扩展（默认 none；Kaggle 上没有 cargo）。
#                    不编译也能正常 serving：前缀缓存会退回 Python 版 TreeCore。
#                    设为 rust_tree_core 时会安装 rustup，并只编译 Rust 版 radix tree。

# 启用严格模式：命令失败、使用未定义变量或管道命令失败时退出。
set -euo pipefail

# 第一个参数为安装目标；未提供时默认为 all。
TARGET="${1:-all}"
# 工作目录；可用 WORKDIR 环境变量覆盖，默认位于用户主目录下的 lab。
WORKDIR="${WORKDIR:-$HOME/lab}"
# vLLM 的 Git ref；可用 VLLM_REF 覆盖，默认使用 main。
VLLM_REF="${VLLM_REF:-main}"
# 是否运行测试；可用 RUN_TESTS 覆盖，默认运行。
RUN_TESTS="${RUN_TESTS:-1}"
# 日志目录；可用 LOG_DIR 覆盖，默认在工作目录下的 logs。
LOG_DIR="${LOG_DIR:-$WORKDIR/logs}"
# 创建工作目录和日志目录；目录已存在时不会报错。
mkdir -p "$WORKDIR" "$LOG_DIR"

# 定义进度输出函数，用青色粗体打印带箭头的阶段标题。
say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- 环境检查
# 显示 GPU/驱动阶段标题。
say "GPU / driver"
# 显示 NVIDIA GPU 和驱动的概要信息。
nvidia-smi
# 查询每张 GPU 的名称、计算能力、驱动版本及显存总量，并以 CSV 格式输出。
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
# 从 nvidia-smi 输出中提取驱动支持的 CUDA 主版本号；提取失败时设为 0。
DRIVER_CUDA_MAJOR="$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+' || echo 0)"
# 打印驱动支持的最高 CUDA 主版本号。
echo "Driver supports CUDA up to major version: $DRIVER_CUDA_MAJOR"

# FlashInfer 没有为 sm_75（T4）预编译 kernel，第一次用到时会现场编译（JIT）。
# 链接这一步会带上 `-L/usr/local/cuda/lib64/stubs -lcuda`，但 Kaggle 镜像的 CUDA 工具包
# 在那里没有 libcuda.so（报错 "/usr/bin/ld: cannot find -lcuda"），导致 vLLM/SGLang 引擎启动时崩溃。
# 修法：在链接器会查找的目录里放一个 libcuda.so 软链接，指向宿主机驱动挂载进来的真正的 libcuda.so.1。
# Kaggle 上 /usr/local/cuda 是只读挂载，stubs 目录通常写不进去，所以会改用 /usr/local/lib
# （ld 默认查找的目录之一）。每个候选目录建好软链接后，都用一次真实链接来验证。
say "libcuda.so link stub for FlashInfer JIT"
# 用和 FlashInfer JIT 相同的参数（-L stubs -lcuda）链接一个空程序，判断 -lcuda 能否被找到。
libcuda_links() {  # can the linker resolve -lcuda the way FlashInfer's JIT asks for it?
  printf 'int main(){return 0;}\n' | c++ -x c++ - -L/usr/local/cuda/lib64/stubs -lcuda \
    -o /tmp/libcuda_link_test >/dev/null 2>&1
}
fix_libcuda_link() {
  local real dir
  # 没有 c++ 编译器就没法验证（FlashInfer JIT 本身也会失败），给警告后跳过。
  if ! command -v c++ >/dev/null 2>&1; then echo "WARN: no c++ compiler; skipping"; return 0; fi
  # 已经能链接就不用处理（比如镜像以后补上了，或者本 session 已经修过）。
  if libcuda_links; then echo "ok: -lcuda already links"; return 0; fi
  # 容器里的驱动库常常不在 ldconfig 的缓存里，而是靠 LD_LIBRARY_PATH 找到，
  # 所以先查 LD_LIBRARY_PATH 里的目录和容器常用的驱动目录。
  # LD_LIBRARY_PATH 未设置时用空字符串代替，否则在 set -u 下展开未定义变量会让脚本直接退出。
  local d llp="${LD_LIBRARY_PATH:-}"
  real=""
  for d in ${llp//:/ } /usr/local/nvidia/lib64 /usr/local/nvidia/lib \
           /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib64-nvidia; do
    if [ -e "$d/libcuda.so.1" ]; then real="$d/libcuda.so.1"; break; fi
  done
  # 上面没找到，再从 ldconfig 的库缓存里找。
  # awk 读完全部输出再打印（不提前 exit），避免 ldconfig 收到 SIGPIPE 后被 pipefail 判为失败。
  [ -n "$real" ] || real="$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1 /{p=$NF} END{print p}' || true)"
  # ldconfig 里找不到时，退而在文件系统里搜索（跳过 /proc）。
  # 最后的兜底，全盘搜索最多 60 秒（完整搜一遍 / 可能要好几分钟）。
  [ -n "$real" ] || real="$(timeout 60 find / -name 'libcuda.so.1' -not -path '/proc/*' 2>/dev/null | head -1 || true)"
  # 实在找不到就只给警告，不中断 setup。
  if [ -z "$real" ]; then
    echo "WARN: driver libcuda.so.1 not found; FlashInfer JIT linking may fail"; return 0
  fi
  # 依次尝试各个候选目录。目录只读时 mkdir/ln 会失败，用 if 包住，避免 set -e 让整个脚本退出。
  # 建好软链接后立即验证；验证不通过就删掉软链接，换下一个目录。
  for dir in /usr/local/cuda/lib64/stubs /usr/local/lib /usr/lib/x86_64-linux-gnu; do
    if mkdir -p "$dir" 2>/dev/null && ln -sf "$real" "$dir/libcuda.so" 2>/dev/null; then
      if libcuda_links; then echo "created: $dir/libcuda.so -> $real (verified: -lcuda links)"; return 0; fi
      rm -f "$dir/libcuda.so"
    fi
  done
  echo "WARN: could not make -lcuda linkable; FlashInfer JIT on sm_75 may fail"
}
fix_libcuda_link

# 每次都安装/升级 uv：Kaggle 镜像可能自带旧版 uv，旧版无法解析 vLLM 的 [tool.uv] 配置，
# 会悄悄忽略整个配置表（torch 的 no-build-isolation、排除 openai-harmony）。
say "Installing / upgrading uv"
# 安静地使用 pip 安装/升级 uv。
pip install -q -U uv
# 打印 uv 的版本，确认命令可用。
uv --version

# Kaggle 镜像可能让 `uv pip install` 装进**系统** Python（3.13，/usr/local/lib/...），
# 即使已经激活了 venv 也一样：vLLM 会和 Kaggle 自带的 torch/torchaudio 装在一起（CUDA 版本冲突），
# 而 venv 里什么都没有（"No module named 'torch'"）。
# 先打印镜像预设的 uv 设置方便排查，再把本脚本里所有 uv 调用显式固定到 venv（见下面的 use_venv）。
say "uv settings preset by the image (if any)"
# 列出所有 UV_ 开头的环境变量；没有就打印提示（|| 防止 grep 无匹配时触发 set -e 退出）。
env | grep '^UV_' || echo "(no UV_* env vars)"
# 打印系统级和用户级的 uv 配置文件（如果存在）。
for f in /etc/uv/uv.toml "$HOME/.config/uv/uv.toml"; do [ -f "$f" ] && { echo "--- $f"; cat "$f"; }; done

# 激活当前目录下的 .venv，并强制之后所有 uv pip 命令都装进这个 venv：
# UV_PYTHON 指定目标解释器，UV_SYSTEM_PYTHON=false 覆盖镜像可能设置的"装进系统 Python"。
use_venv() {  # activate ./.venv and force all uv pip calls into it
  # shellcheck disable=SC1091
  source .venv/bin/activate
  export UV_PYTHON="$VIRTUAL_ENV/bin/python" UV_SYSTEM_PYTHON=false
}
# 退出 venv，并清除上面设置的变量，避免影响下一个仓库的安装。
leave_venv() { deactivate; unset UV_PYTHON UV_SYSTEM_PYTHON; }

# 定义克隆或更新仓库的函数：参数依次为 $1=仓库 URL、$2=本地目录、$3=Git ref。
clone_or_update() {  # $1=url $2=dir $3=ref
  # 如果目标目录不是一个 Git 仓库，就克隆仓库。
  if [ ! -d "$2/.git" ]; then
    git clone "$1" "$2"
  fi
  # 从 origin 获取标签信息，静默运行。
  git -C "$2" fetch --tags --quiet origin
  # 切换到指定的分支、标签或其他 Git ref。
  git -C "$2" checkout --quiet "$3"
  # 若当前位于分支上（而不是标签或分离 HEAD），则只允许快进更新。
  if git -C "$2" symbolic-ref -q HEAD >/dev/null; then git -C "$2" pull --ff-only --quiet; fi
  # 输出仓库目录名、当前提交短哈希和指定的 ref。
  echo "$(basename "$2") @ $(git -C "$2" rev-parse --short HEAD) ($3)"
}

# ---------------------------------------------------------------- vLLM 安装
# 定义 vLLM 的克隆、安装、检查和测试流程。
setup_vllm() {
  # 显示即将使用的 vLLM ref。
  say "vLLM: clone ($VLLM_REF)"
  # 克隆或更新 vLLM 仓库。
  clone_or_update https://github.com/vllm-project/vllm.git "$WORKDIR/vllm" "$VLLM_REF"
  # 进入 vLLM 源码目录。
  cd "$WORKDIR/vllm"

  # 显示正在准备 vLLM 虚拟环境。
  say "vLLM: venv"
  # 如果虚拟环境不存在，就创建 Python 3.12 环境，预装基础工具并使用 uv 管理 Python。
  [ -d .venv ] || uv venv --python 3.12 --seed --managed-python
  # 激活 vLLM 的虚拟环境，并把 uv pip 固定到这个 venv。
  use_venv

  # 若设置 VLLM_FULL_BUILD=1，则执行完整 C++/CUDA 编译。
  if [ "${VLLM_FULL_BUILD:-0}" = "1" ]; then
    # 显示正在进行完整构建。
    say "vLLM: FULL build (C++/CUDA compile)"
    # 设置构建并发数（默认 4），以 editable 模式安装并自动选择 PyTorch 后端；输出同时显示并保存到日志。
    MAX_JOBS="${MAX_JOBS:-4}" uv pip install -e . --torch-backend=auto 2>&1 | tee "$LOG_DIR/vllm_build.log"
  # 否则使用预编译内核进行仅 Python 部分的可编辑安装。
  else
    # 显示正在使用预编译内核安装。
    say "vLLM: Python-only editable build with precompiled kernels"
    # 预编译模式必须使用 uv pip；普通的 pip install -e . 在此模式下无法工作。
    # 设置预编译模式，以 editable 方式安装，并自动选择 PyTorch 后端；输出同时显示并保存到日志。
    VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto 2>&1 | tee "$LOG_DIR/vllm_build.log"
  fi
  # 尝试下载 vLLM 内核；若命令失败或当前 ref 不支持该命令，则显示警告并继续。
  vllm download-kernels || echo "WARN: 'vllm download-kernels' failed or not available on this ref"

  # 显示正在检查 vLLM 安装是否正常。
  say "vLLM: sanity check"
  # 运行内嵌 Python 程序，打印已安装的 vLLM、PyTorch、CUDA 和 GPU 信息。
  python - <<'PY'
# 导入 PyTorch 和 vLLM，以检查安装状态及设备信息。
import sys, torch, vllm
# 打印实际使用的 Python 解释器，以及它是否在 venv 里（应为 True；False 说明又装进了系统 Python）。
print("python", sys.executable, "| in venv:", sys.prefix != sys.base_prefix)
# 打印 vLLM 版本和模块文件位置。
print("vllm", vllm.__version__, "| file", vllm.__file__)
# 打印 PyTorch 版本、PyTorch 使用的 CUDA 版本及可见 GPU 数量。
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| gpus", torch.cuda.device_count())
# 打印第 0 张 GPU 的计算能力。
print("capability", torch.cuda.get_device_capability(0))
PY

  # 当 RUN_TESTS 的值为 "1" 时运行快速测试。
  if [ "$RUN_TESTS" = "1" ]; then
    # 显示将测试 vLLM v1 核心中的调度器和 KV cache 管理器。
    say "vLLM: quick tests (tests/v1/core: scheduler + KV cache manager)"
    # 安装跑 tests/v1/core 所需的最少依赖：pytest、异步测试插件，以及 tblib
    # （tests/conftest.py 在文件顶层 import 了 tblib，缺了它所有测试都无法加载）。
    # 不装官方的 requirements/test/cuda.txt：近 7900 行，且固定 torch +cu130，会替换掉已装好的 torch。
    uv pip install -q pytest pytest-asyncio tblib

    # 为 hybrid 前缀缓存测试预热 JIT 编译缓存。这个测试的 server 必须在 480 秒内就绪
    # （写死在 tests/utils.py 里），但在 T4 + 2 个 CPU 线程上冷启动时，显存 profiling 阶段
    # 光 Triton 编译就要约 6.5 分钟。先不限时地启动一次引擎，把 ~/.triton/cache（以及
    # FlashInfer 的缓存）填好，测试里的 server 就能在约 2 分钟内起来。WARMUP_HYBRID=0 可跳过。
    # 模型名直接从测试文件里的 MODEL = "..." 读取，这样上游换了模型也会自动跟上。
    local hybrid_test=tests/v1/core/prefix_cache/test_hybrid_prefix_cache_hit_rate.py warm_model
    warm_model="$(sed -n 's/^MODEL = "\(.*\)"$/\1/p' "$hybrid_test" 2>/dev/null | head -1 || true)"
    if [ "${WARMUP_HYBRID:-1}" = "1" ] && [ -n "$warm_model" ]; then
      say "vLLM: warming JIT caches with $warm_model (cold start can take ~8 min)"
      local t0=$SECONDS
      # 用默认参数创建一次 LLM（和测试 server 的配置一致），输出写进日志，不刷屏。
      # 预热失败只给警告，不中断 setup。
      if python - "$warm_model" > "$LOG_DIR/vllm_warmup.log" 2>&1 <<'PY'
import sys
from vllm import LLM
LLM(sys.argv[1])
print("ENGINE READY")
PY
      then echo "warm-up done in $((SECONDS - t0)) s"
      else echo "WARN: warm-up failed after $((SECONDS - t0)) s; see $LOG_DIR/vllm_warmup.log (the hybrid test may time out)"
      fi
    fi

    # 运行 v1 核心测试：安静模式，遇到首个失败即停止，最多报告 5 个失败；保存测试日志。
    # 测试失败不应中止整个安装流程；失败信息会写入日志供后续排查。
    pytest tests/v1/core -q -x --maxfail=5 2>&1 | tee "$LOG_DIR/vllm_tests.log" || \
      echo "WARN: some vLLM tests failed; see $LOG_DIR/vllm_tests.log"
  fi
  # 退出 vLLM 虚拟环境，并清除 uv 的目标设置。
  leave_venv
}

# ---------------------------------------------------------------- SGLang 安装
# 定义 SGLang 的克隆、安装和检查流程。
setup_sglang() {
  # 获取 SGLANG_REF；未设置时先置为空字符串，随后根据驱动版本选择默认值。
  local ref="${SGLANG_REF:-}"
  # 如果没有指定 SGLang ref，就根据驱动支持的 CUDA 版本选择。
  if [ -z "$ref" ]; then
    # CUDA 13 或更高版本使用 main；较旧版本使用 v0.5.19。
    if [ "$DRIVER_CUDA_MAJOR" -ge 13 ]; then ref=main; else ref=v0.5.19; fi
  fi
  # 若驱动支持的 CUDA 版本低于 13，提示将使用兼容 CUDA 12 的版本。
  if [ "$DRIVER_CUDA_MAJOR" -lt 13 ]; then
    echo "NOTE: driver CUDA < 13; latest SGLang is CUDA 13 only, so using $ref (last CUDA 12 lane)."
  fi

  # 显示即将使用的 SGLang ref。
  say "SGLang: clone ($ref)"
  # 克隆或更新 SGLang 仓库。
  clone_or_update https://github.com/sgl-project/sglang.git "$WORKDIR/sglang" "$ref"
  # 进入 SGLang 源码目录。
  cd "$WORKDIR/sglang"

  # 显示正在准备 SGLang 虚拟环境并进行可编辑安装。
  say "SGLang: venv + editable install"
  # 如果虚拟环境不存在，就创建 Python 3.12 环境，预装基础工具并使用 uv 管理 Python。
  [ -d .venv ] || uv venv --python 3.12 --seed --managed-python
  # 激活 SGLang 的虚拟环境，并把 uv pip 固定到这个 venv。
  use_venv
  # SGLang 的 setup.py 会编译 Rust（PyO3）扩展，需要 cargo，而 Kaggle 镜像里没有。
  # 这些扩展只用于 gRPC、Rust 版 server、少数多模态预处理，以及 Rust 版 radix tree
  # （缺失时自动退回 Python 版 TreeCore）。所以默认跳过；用户已设置时沿用用户的值。
  export SGLANG_BUILD_RUST_EXTS="${SGLANG_BUILD_RUST_EXTS:-none}"
  # 打印实际取值，方便在日志里确认。
  echo "SGLANG_BUILD_RUST_EXTS=$SGLANG_BUILD_RUST_EXTS"
  # 需要编译 Rust 扩展、但系统里没有 cargo 时，用官方 rustup 安装最小化的 Rust 工具链。
  if [ "$SGLANG_BUILD_RUST_EXTS" != "none" ] && ! command -v cargo >/dev/null 2>&1; then
    say "SGLang: installing Rust toolchain (rustup, minimal profile)"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
    # 把 ~/.cargo/bin 加入 PATH，让后面的构建能找到 cargo 和 rustc。
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
  fi
  # 从仓库的 python 子目录以 editable 模式安装 SGLang；输出同时显示并保存到日志。
  uv pip install -e "python" 2>&1 | tee "$LOG_DIR/sglang_build.log"

  # 显示正在检查 SGLang 安装是否正常。
  say "SGLang: sanity check"
  # 运行内嵌 Python 程序，打印 SGLang、PyTorch、CUDA、GPU 信息，并尝试检查 FlashInfer。
  python - <<'PY'
# 导入 PyTorch 和 SGLang，以检查安装状态及设备信息。
import sys, torch, sglang
# 打印实际使用的 Python 解释器，以及它是否在 venv 里（应为 True）。
print("python", sys.executable, "| in venv:", sys.prefix != sys.base_prefix)
# 打印 SGLang 版本和模块文件位置。
print("sglang", sglang.__version__, "| file", sglang.__file__)
# 打印 PyTorch 版本、PyTorch 使用的 CUDA 版本及可见 GPU 数量。
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| gpus", torch.cuda.device_count())
# 尝试导入 FlashInfer 并打印其版本。
try:
    import flashinfer; print("flashinfer", flashinfer.__version__)
# 若导入失败，打印错误并提示改用 Triton attention backend。
except Exception as e:
    print("flashinfer import failed:", e, "-> use --attention-backend triton")
PY
  # SGLang 的测试目录结构经常变化，因此建议安装完成后手动挑选一两个测试文件运行，例如：
  #   ls test/ ; python -m pytest <some unit test> -q
  # 退出 SGLang 虚拟环境，并清除 uv 的目标设置。
  leave_venv
}

# 根据第一个命令行参数选择安装目标。
case "$TARGET" in
  # 只安装 vLLM。
  vllm)   setup_vllm ;;
  # 只安装 SGLang。
  sglang) setup_sglang ;;
  # 依次安装 vLLM 和 SGLang。
  all)    setup_vllm; setup_sglang ;;
  # 参数不合法时显示可用选项，并以状态码 2 退出。
  *) echo "unknown target: $TARGET (use vllm | sglang | all)"; exit 2 ;;
esac

# 显示安装完成及日志目录位置。
say "Done. Logs in $LOG_DIR"
