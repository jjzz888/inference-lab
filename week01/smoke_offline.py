"""Smallest end-to-end vLLM v1 run. Works on Mac CPU and on GPU.

Good breakpoints for drawing the request lifecycle (paths may move between versions; grep if missing):
  vllm/v1/engine/llm_engine.py        add_request / step
  vllm/v1/core/sched/scheduler.py     Scheduler.schedule        <- who runs this step, token budget
  vllm/v1/core/kv_cache_manager.py    allocate_slots            <- PagedAttention block allocation
  vllm/v1/worker/cpu/standalone_model_runner.py  execute_model  (GPU: vllm/v1/worker/gpu_model_runner.py)
  vllm/v1/worker/cpu/sampler.py       CPUSampler.__call__       (GPU: vllm/v1/sample/sampler.py Sampler.forward)
Full ordered list: week01/README.md, "Debugging the request lifecycle".

On Mac, VLLM_ENABLE_V1_MULTIPROCESSING defaults to 0 here so the engine core runs in this
process and breakpoints in the scheduler actually hit (also when launched from an IDE).
Set it explicitly to override.
"""
import os
import sys

# Must be set before vllm is imported.
if sys.platform == "darwin":
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
from vllm import LLM, SamplingParams

model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
on_gpu = torch.cuda.is_available()

llm = LLM(
    model=model,
    dtype="half" if on_gpu else "float32",  # T4: no BF16; Mac CPU: FP32/FP16 only
    max_model_len=2048,
    enforce_eager=not on_gpu,
    # CPU backend otherwise reserves gpu_memory_utilization (0.9) x total RAM, ~19 GiB of KV cache
    # on a 24 GB Mac, and fails when that much isn't free. 2 GiB is plenty for 0.5B @ 2048 tokens.
    # (Same as VLLM_CPU_KVCACHE_SPACE=2 or --kv-cache-memory-bytes.)
    kv_cache_memory_bytes=None if on_gpu else 2 * 1024**3,
)
prompts = [
    "Explain PagedAttention in one sentence.",
    "What does a KV cache store?",
]
outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=48))
for o in outs:
    print("PROMPT:", o.prompt)
    print("OUTPUT:", o.outputs[0].text.strip(), "\n")
print("device:", "cuda" if on_gpu else "cpu", "| VLLM_ENABLE_V1_MULTIPROCESSING =",
      os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "(default)"))
