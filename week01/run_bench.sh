#!/usr/bin/env bash
# Week 1: one serving benchmark per engine, same model and workload, results saved with metadata.
# Run kaggle_setup.sh first.
#
# Usage:
#   bash run_bench.sh vllm
#   bash run_bench.sh sglang
#   bash run_bench.sh both
#
# Env overrides:
#   MODEL        (default: Qwen/Qwen2.5-0.5B-Instruct, fits a 16 GB T4 in FP16)
#   INPUT_LEN / OUTPUT_LEN / NUM_PROMPTS / REQUEST_RATE (default inf = send all at once)
#   GPU          which GPU to use (default: 0; T4 x2 gives 0 and 1)
#   SGLANG_ATTN  SGLang attention backend (default: unset; set "triton" if FlashInfer fails on T4)
#   WORKDIR      same as kaggle_setup.sh (default: $HOME/lab)
#   OUT_ROOT     where results go (default: /kaggle/working/results/week01 if it exists, else ./results/week01)
set -euo pipefail

ENGINE="${1:?usage: run_bench.sh vllm|sglang|both}"
WORKDIR="${WORKDIR:-$HOME/lab}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
GPU="${GPU:-0}"
if [ -z "${OUT_ROOT:-}" ]; then
  if [ -d /kaggle/working ]; then OUT_ROOT=/kaggle/working/results/week01; else OUT_ROOT="$PWD/results/week01"; fi
fi
export CUDA_VISIBLE_DEVICES="$GPU"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

SERVER_PID=""
cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup EXIT

wait_healthy() {  # $1=port $2=server log
  for _ in $(seq 1 180); do  # up to ~15 min (first run downloads the model)
    if curl -sf "http://127.0.0.1:$1/health" >/dev/null; then echo "server ready"; return 0; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "server exited early; last log lines:"; tail -n 50 "$2"; return 1
    fi
    sleep 5
  done
  echo "timeout waiting for server"; tail -n 50 "$2"; return 1
}

write_meta() {  # $1=out dir $2=engine dir
  {
    echo "date: $(date -u +%FT%TZ)"
    echo "engine_commit: $(git -C "$2" rev-parse HEAD)"
    echo "engine_ref: $(git -C "$2" describe --tags --always)"
    echo "model: $MODEL"
    echo "workload: random in=$INPUT_LEN out=$OUTPUT_LEN prompts=$NUM_PROMPTS rate=$REQUEST_RATE"
    echo "gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader -i "$GPU")"
    python -c 'import torch; print("torch:", torch.__version__, "cuda", torch.version.cuda)'
  } > "$1/meta.txt"
  pip freeze 2>/dev/null > "$1/pip_freeze.txt" || uv pip freeze > "$1/pip_freeze.txt"
}

bench_vllm() {
  local out="$OUT_ROOT/vllm-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$out"
  # shellcheck disable=SC1091
  source "$WORKDIR/vllm/.venv/bin/activate"
  write_meta "$out" "$WORKDIR/vllm"

  say "vLLM: starting server ($MODEL, FP16) on GPU $GPU"
  vllm serve "$MODEL" --dtype half --port 8000 > "$out/server.log" 2>&1 &
  SERVER_PID=$!
  wait_healthy 8000 "$out/server.log"

  say "vLLM: benchmark"
  vllm bench serve --backend vllm --model "$MODEL" --port 8000 \
    --dataset-name random --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" --request-rate "$REQUEST_RATE" \
    --save-result --result-dir "$out" --result-filename result.json \
    2>&1 | tee "$out/bench.txt"

  cleanup; deactivate
  echo "results: $out"
}

bench_sglang() {
  local out="$OUT_ROOT/sglang-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$out"
  # shellcheck disable=SC1091
  source "$WORKDIR/sglang/.venv/bin/activate"
  write_meta "$out" "$WORKDIR/sglang"

  local extra=()
  if [ -n "${SGLANG_ATTN:-}" ]; then extra+=(--attention-backend "$SGLANG_ATTN" --sampling-backend pytorch); fi

  say "SGLang: starting server ($MODEL, FP16) on GPU $GPU ${extra[*]:-}"
  python -m sglang.launch_server --model-path "$MODEL" --dtype half --port 30000 "${extra[@]}" \
    > "$out/server.log" 2>&1 &
  SERVER_PID=$!
  wait_healthy 30000 "$out/server.log"

  say "SGLang: benchmark"
  python -m sglang.bench_serving --backend sglang --port 30000 --model "$MODEL" \
    --dataset-name random --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
    --random-range-ratio 1.0 `# fixed lengths, to match vLLM's default (ratio 0 = fixed in vLLM)` \
    --num-prompts "$NUM_PROMPTS" --request-rate "$REQUEST_RATE" \
    --output-file "$out/result.jsonl" \
    2>&1 | tee "$out/bench.txt"

  cleanup; deactivate
  echo "results: $out"
}

case "$ENGINE" in
  vllm)   bench_vllm ;;
  sglang) bench_sglang ;;
  both)   bench_vllm; bench_sglang ;;
  *) echo "unknown engine: $ENGINE"; exit 2 ;;
esac
