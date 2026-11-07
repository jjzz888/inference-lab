#!/usr/bin/env bash
# Week 1: build vLLM and SGLang from source on a Kaggle GPU (T4 x2) session.
# Each Kaggle session starts clean, so run this once at the start of every session.
#
# Usage:
#   bash kaggle_setup.sh            # vLLM + SGLang
#   bash kaggle_setup.sh vllm       # only vLLM
#   bash kaggle_setup.sh sglang     # only SGLang
#
# Env overrides:
#   WORKDIR      where repos and venvs live          (default: $HOME/lab, outside /kaggle/working's 20 GB output quota)
#   VLLM_REF     git ref for vLLM                     (default: main)
#   SGLANG_REF   git ref for SGLang                   (default: main if driver supports CUDA 13, else v0.5.19)
#   VLLM_FULL_BUILD=1   compile C++/CUDA instead of reusing precompiled kernels (hours on Kaggle CPUs; avoid)
#   RUN_TESTS=0  skip the quick test subset
set -euo pipefail

TARGET="${1:-all}"
WORKDIR="${WORKDIR:-$HOME/lab}"
VLLM_REF="${VLLM_REF:-main}"
RUN_TESTS="${RUN_TESTS:-1}"
LOG_DIR="${LOG_DIR:-$WORKDIR/logs}"
mkdir -p "$WORKDIR" "$LOG_DIR"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- environment
say "GPU / driver"
nvidia-smi
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
DRIVER_CUDA_MAJOR="$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+' || echo 0)"
echo "Driver supports CUDA up to major version: $DRIVER_CUDA_MAJOR"

# Always upgrade: an old uv (Kaggle images may ship one) silently ignores vLLM's [tool.uv] settings.
say "Installing / upgrading uv"
pip install -q -U uv
uv --version

clone_or_update() {  # $1=url $2=dir $3=ref
  if [ ! -d "$2/.git" ]; then
    git clone "$1" "$2"
  fi
  git -C "$2" fetch --tags --quiet origin
  git -C "$2" checkout --quiet "$3"
  # Fast-forward only when on a branch (not a tag / detached HEAD).
  if git -C "$2" symbolic-ref -q HEAD >/dev/null; then git -C "$2" pull --ff-only --quiet; fi
  echo "$(basename "$2") @ $(git -C "$2" rev-parse --short HEAD) ($3)"
}

# ---------------------------------------------------------------- vLLM
setup_vllm() {
  say "vLLM: clone ($VLLM_REF)"
  clone_or_update https://github.com/vllm-project/vllm.git "$WORKDIR/vllm" "$VLLM_REF"
  cd "$WORKDIR/vllm"

  say "vLLM: venv"
  [ -d .venv ] || uv venv --python 3.12 --seed --managed-python
  # shellcheck disable=SC1091
  source .venv/bin/activate

  if [ "${VLLM_FULL_BUILD:-0}" = "1" ]; then
    say "vLLM: FULL build (C++/CUDA compile)"
    MAX_JOBS="${MAX_JOBS:-4}" uv pip install -e . --torch-backend=auto 2>&1 | tee "$LOG_DIR/vllm_build.log"
  else
    say "vLLM: Python-only editable build with precompiled kernels"
    # Must be `uv pip`: plain `pip install -e .` does not work in precompiled mode.
    VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto 2>&1 | tee "$LOG_DIR/vllm_build.log"
  fi
  vllm download-kernels || echo "WARN: 'vllm download-kernels' failed or not available on this ref"

  say "vLLM: sanity check"
  python - <<'PY'
import torch, vllm
print("vllm", vllm.__version__, "| file", vllm.__file__)
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| gpus", torch.cuda.device_count())
print("capability", torch.cuda.get_device_capability(0))
PY

  if [ "$RUN_TESTS" = "1" ]; then
    say "vLLM: quick tests (tests/v1/core: scheduler + KV cache manager)"
    uv pip install -q pytest pytest-asyncio
    # Don't abort setup on test failures; failures are worth logging (possible first issues).
    pytest tests/v1/core -q -x --maxfail=5 2>&1 | tee "$LOG_DIR/vllm_tests.log" || \
      echo "WARN: some vLLM tests failed; see $LOG_DIR/vllm_tests.log"
  fi
  deactivate
}

# ---------------------------------------------------------------- SGLang
setup_sglang() {
  local ref="${SGLANG_REF:-}"
  if [ -z "$ref" ]; then
    if [ "$DRIVER_CUDA_MAJOR" -ge 13 ]; then ref=main; else ref=v0.5.19; fi
  fi
  if [ "$DRIVER_CUDA_MAJOR" -lt 13 ]; then
    echo "NOTE: driver CUDA < 13; latest SGLang is CUDA 13 only, so using $ref (last CUDA 12 lane)."
  fi

  say "SGLang: clone ($ref)"
  clone_or_update https://github.com/sgl-project/sglang.git "$WORKDIR/sglang" "$ref"
  cd "$WORKDIR/sglang"

  say "SGLang: venv + editable install"
  [ -d .venv ] || uv venv --python 3.12 --seed --managed-python
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uv pip install -e "python" 2>&1 | tee "$LOG_DIR/sglang_build.log"

  say "SGLang: sanity check"
  python - <<'PY'
import torch, sglang
print("sglang", sglang.__version__, "| file", sglang.__file__)
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| gpus", torch.cuda.device_count())
try:
    import flashinfer; print("flashinfer", flashinfer.__version__)
except Exception as e:
    print("flashinfer import failed:", e, "-> use --attention-backend triton")
PY
  # SGLang's test layout changes often; run one or two files by hand after setup, e.g.
  #   ls test/ ; python -m pytest <some unit test> -q
  deactivate
}

case "$TARGET" in
  vllm)   setup_vllm ;;
  sglang) setup_sglang ;;
  all)    setup_vllm; setup_sglang ;;
  *) echo "unknown target: $TARGET (use vllm | sglang | all)"; exit 2 ;;
esac

say "Done. Logs in $LOG_DIR"
