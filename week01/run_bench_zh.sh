#!/usr/bin/env bash
# 第 1 周：为每个推理引擎运行一次服务基准测试，使用相同模型和负载，并连同元数据一起保存结果。
# 运行本脚本前，请先运行 kaggle_setup.sh。
#
# 用法：
#   bash run_bench_zh.sh vllm    # 测试 vLLM
#   bash run_bench_zh.sh sglang  # 测试 SGLang
#   bash run_bench_zh.sh both    # 依次测试两者
#
# 可通过环境变量覆盖的参数：
#   MODEL        模型（默认：Qwen/Qwen2.5-0.5B-Instruct，可用 16 GB 显存的 T4 以 FP16 运行）
#   INPUT_LEN / OUTPUT_LEN / NUM_PROMPTS / REQUEST_RATE（默认请求速率 inf，表示尽可能同时发送）
#   GPU          使用哪张 GPU（默认：0；双 T4 环境通常编号为 0 和 1）
#   SGLANG_ATTN  SGLang attention 后端（默认不设置；若 T4 上 FlashInfer 失败，可设为 "triton"）
#   WORKDIR      与 kaggle_setup.sh 相同（默认：$HOME/lab）
#   OUT_ROOT     结果保存目录（默认：若 /kaggle/working 存在则用 /kaggle/working/results/week01，否则用 ./results/week01）
#   SERVER_TIMEOUT  等待 server 就绪的最长秒数（默认 1800）。T4 上冷启动要在 2 个 CPU 线程上
#                现场编译 FlashInfer（nvcc）和 Triton kernel，一个 vLLM 测试 server 就用了约 8.5 分钟；
#                之后的启动会复用编译缓存，快得多。

# 启用严格模式：命令失败、使用未定义变量或管道命令失败时退出。
set -euo pipefail

# 读取第一个命令行参数作为要测试的引擎；缺少参数时显示用法并报错。
ENGINE="${1:?usage: run_bench.sh vllm|sglang|both}"
# 工作目录；可用 WORKDIR 覆盖，默认是用户主目录下的 lab。
WORKDIR="${WORKDIR:-$HOME/lab}"
# 模型名称；可用 MODEL 覆盖，默认使用 Qwen 0.5B 指令模型。
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
# 每个随机请求的输入长度，默认 512。
INPUT_LEN="${INPUT_LEN:-512}"
# 每个随机请求的输出长度，默认 128。
OUTPUT_LEN="${OUTPUT_LEN:-128}"
# 生成的提示词/请求数量，默认 200。
NUM_PROMPTS="${NUM_PROMPTS:-200}"
# 请求发送速率，默认 inf，表示尽可能快地发送。
REQUEST_RATE="${REQUEST_RATE:-inf}"
# 使用的 GPU 编号，默认是 0。
GPU="${GPU:-0}"
# 如果没有显式指定结果根目录，则根据当前环境选择默认位置。
if [ -z "${OUT_ROOT:-}" ]; then
  # Kaggle 环境下将结果保存到 /kaggle/working；其他环境保存到当前目录下。
  if [ -d /kaggle/working ]; then OUT_ROOT=/kaggle/working/results/week01; else OUT_ROOT="$PWD/results/week01"; fi
fi
# 限制 CUDA 程序只使用指定的 GPU。
export CUDA_VISIBLE_DEVICES="$GPU"

# 定义进度提示函数，以青色粗体打印阶段信息。
say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# 保存后台服务进程号；为空表示当前没有记录服务器进程。
SERVER_PID=""
# 定义停止服务器并回收进程的清理函数。
cleanup() {
  # 仅当已记录进程号且进程仍然存在时执行清理。
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    # 发送终止信号；忽略终止命令本身的错误。
    kill "$SERVER_PID" 2>/dev/null || true
    # 等待后台进程结束；忽略等待过程中的错误。
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  # 清空记录的进程号。
  SERVER_PID=""
}
# 无论脚本正常结束还是因错误退出，都调用清理函数。
trap cleanup EXIT

# 定义健康检查函数：参数为 $1=端口、$2=服务器日志文件。
wait_healthy() {  # $1=端口 $2=服务器日志
  # 按秒计时（bash 内置的 $SECONDS），最长等 SERVER_TIMEOUT 秒，默认 30 分钟。
  local start=$SECONDS timeout="${SERVER_TIMEOUT:-1800}"
  while [ $((SECONDS - start)) -lt "$timeout" ]; do
    # 请求本机服务的健康检查接口；成功后报告用了多少秒，并返回成功状态。
    if curl -sf "http://127.0.0.1:$1/health" >/dev/null; then
      echo "server ready after $((SECONDS - start)) s"; return 0
    fi
    # 如果服务器进程已经退出，打印日志末尾 50 行并返回失败。
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "server exited early; last log lines:"; tail -n 50 "$2"; return 1
    fi
    # 等待 5 秒后再次检查。
    sleep 5
  done
  # 超过时限仍未就绪时，报告超时、打印日志末尾并返回失败。
  echo "timeout after ${timeout}s waiting for server"; tail -n 50 "$2"; return 1
}

# T4 上 FlashInfer 现场编译需要能链接到 libcuda.so，这个软链接由 kaggle_setup.sh 创建。
# 新 session 里如果没先跑 setup，SGLang（以及 vLLM 的部分路径）会在启动时崩溃，所以这里先提醒。
check_libcuda() {
  if [ -e /usr/local/lib/libcuda.so ] || [ -e /usr/local/cuda/lib64/stubs/libcuda.so ]; then return 0; fi
  echo "WARN: no linkable libcuda.so; run kaggle_setup.sh in this session first (FlashInfer JIT will fail)"
}

# 从 server 日志里摘出它实际选用的后端（attention 后端、SGLang 的 TreeCore），追加到 meta.txt，
# 方便对比结果时知道两边各自跑的是什么配置。|| true 防止没匹配到时 set -e 退出。
record_backends() {  # $1=out dir
  {
    echo "--- backends picked (from server.log)"
    grep -i -E "attention backend|Using .*(Backend|backend)|TreeCore" "$1/server.log" | head -8 | cut -c1-200 || true
  } >> "$1/meta.txt"
}

# 定义元数据写入函数：参数为 $1=结果目录、$2=引擎源码目录。
write_meta() {  # $1=输出目录 $2=引擎目录
  # 将下面的信息组合后写入结果目录中的 meta.txt。
  {
    # 记录 UTC 时间，格式为 ISO 8601。
    echo "date: $(date -u +%FT%TZ)"
    # 记录引擎仓库当前的完整 Git 提交哈希。
    echo "engine_commit: $(git -C "$2" rev-parse HEAD)"
    # 记录引擎的标签或最近可识别的 Git ref。
    echo "engine_ref: $(git -C "$2" describe --tags --always)"
    # 记录使用的模型。
    echo "model: $MODEL"
    # 记录随机负载的输入长度、输出长度、提示词数量和请求速率。
    echo "workload: random in=$INPUT_LEN out=$OUTPUT_LEN prompts=$NUM_PROMPTS rate=$REQUEST_RATE"
    # 记录指定 GPU 的型号、驱动版本和显存容量。
    echo "gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader -i "$GPU")"
    # 打印并记录当前 Python 环境中的 PyTorch 和 CUDA 版本。
    python -c 'import torch; print("torch:", torch.__version__, "cuda", torch.version.cuda)'
  } > "$1/meta.txt"
  # 保存已安装 Python 包及版本；若 pip freeze 失败，则改用 uv pip freeze。
  pip freeze 2>/dev/null > "$1/pip_freeze.txt" || uv pip freeze > "$1/pip_freeze.txt"
}

# 定义 vLLM 的启动服务、运行基准测试和保存结果流程。
bench_vllm() {
  # 创建带日期时间戳的 vLLM 结果目录。
  local out="$OUT_ROOT/vllm-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$out"
  # 告知 ShellCheck：虚拟环境激活脚本是运行时生成的，无法静态解析。
  # shellcheck disable=SC1091
  # 激活 vLLM 对应的 Python 虚拟环境。
  source "$WORKDIR/vllm/.venv/bin/activate"
  # 在结果目录写入运行环境和源码版本等元数据。
  write_meta "$out" "$WORKDIR/vllm"

  # 显示即将启动的模型、精度和 GPU。
  say "vLLM: starting server ($MODEL, FP16) on GPU $GPU"
  # 在后台以 FP16 启动 vLLM 服务，监听 8000 端口并将输出写入服务器日志。
  vllm serve "$MODEL" --dtype half --port 8000 > "$out/server.log" 2>&1 &
  # 保存刚启动的后台服务进程号，供健康检查和清理函数使用。
  SERVER_PID=$!
  # 等待 vLLM 的健康检查接口就绪。
  wait_healthy 8000 "$out/server.log"
  # 记录 vLLM 实际选用的后端。
  record_backends "$out"

  # 显示即将运行 vLLM 基准测试。
  say "vLLM: benchmark"
  # 使用 vLLM 自带的 serve 基准工具，以随机输入和输出长度发送请求。
  vllm bench serve --backend vllm --model "$MODEL" --port 8000 \
    --dataset-name random --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" --request-rate "$REQUEST_RATE" \
    --save-result --result-dir "$out" --result-filename result.json \
    2>&1 | tee "$out/bench.txt"

  # 测试完成后停止服务器并退出虚拟环境。
  cleanup; deactivate
  # 打印结果目录的位置。
  echo "results: $out"
}

# 定义 SGLang 的启动服务、运行基准测试和保存结果流程。
bench_sglang() {
  # 创建带日期时间戳的 SGLang 结果目录。
  local out="$OUT_ROOT/sglang-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$out"
  # 告知 ShellCheck：虚拟环境激活脚本是运行时生成的，无法静态解析。
  # shellcheck disable=SC1091
  # 激活 SGLang 对应的 Python 虚拟环境。
  source "$WORKDIR/sglang/.venv/bin/activate"
  # 在结果目录写入运行环境和源码版本等元数据。
  write_meta "$out" "$WORKDIR/sglang"

  # 创建可选参数数组；默认不向 SGLang 命令添加额外参数。
  local extra=()
  # 如果指定了 SGLang attention 后端，同时设置 attention backend 和 PyTorch sampling backend。
  if [ -n "${SGLANG_ATTN:-}" ]; then extra+=(--attention-backend "$SGLANG_ATTN" --sampling-backend pytorch); fi

  # 显示即将启动的模型、精度、GPU 和可选参数。
  say "SGLang: starting server ($MODEL, FP16) on GPU $GPU ${extra[*]:-}"
  # 在后台以 FP16 启动 SGLang 服务，监听 30000 端口并将输出写入服务器日志。
  python -m sglang.launch_server --model-path "$MODEL" --dtype half --port 30000 "${extra[@]}" \
    > "$out/server.log" 2>&1 &
  # 保存刚启动的后台服务进程号，供健康检查和清理函数使用。
  SERVER_PID=$!
  # 等待 SGLang 的健康检查接口就绪。
  wait_healthy 30000 "$out/server.log"
  # 记录 SGLang 实际选用的后端，以及是否手动指定了 attention 后端。
  record_backends "$out"
  echo "sglang_attention_backend_override: ${SGLANG_ATTN:-none}" >> "$out/meta.txt"

  # SGLang 的基准工具已经从 sglang.bench_serving 挪到了 sglang.benchmark.serving；
  # 旧版本（比如 CUDA 12 的退路 v0.5.19）只有旧模块。先探测新模块是否存在，不存在就用旧的。
  local bench_mod=sglang.benchmark.serving
  python -c "import importlib.util as u, sys; sys.exit(0 if u.find_spec('$bench_mod') else 1)" \
    || bench_mod=sglang.bench_serving
  echo "sglang_bench_module: $bench_mod" >> "$out/meta.txt"

  # 显示即将运行 SGLang 基准测试，以及用的是哪个模块。
  say "SGLang: benchmark ($bench_mod)"
  # 使用 SGLang 的服务基准工具，向本机服务发送随机负载请求。
  python -m "$bench_mod" --backend sglang --port 30000 --model "$MODEL" \
    --dataset-name random --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
    --random-range-ratio 1.0 `# 固定长度，以匹配 vLLM 的默认行为（vLLM 中 ratio 0 表示固定长度）` \
    --num-prompts "$NUM_PROMPTS" --request-rate "$REQUEST_RATE" \
    --output-file "$out/result.jsonl" \
    2>&1 | tee "$out/bench.txt"

  # 测试完成后停止服务器并退出虚拟环境。
  cleanup; deactivate
  # 打印结果目录的位置。
  echo "results: $out"
}

# 先检查 libcuda.so 链接是否已就位（没有的话只警告，不中断）。
check_libcuda
# 根据第一个命令行参数选择要运行的引擎测试。
case "$ENGINE" in
  # 只运行 vLLM 测试。
  vllm)   bench_vllm ;;
  # 只运行 SGLang 测试。
  sglang) bench_sglang ;;
  # 先运行 vLLM，再运行 SGLang。
  both)   bench_vllm; bench_sglang ;;
  # 参数不合法时显示错误并以状态码 2 退出。
  *) echo "unknown engine: $ENGINE"; exit 2 ;;
esac
