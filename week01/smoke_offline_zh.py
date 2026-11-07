"""最精简的 vLLM v1 端到端运行示例，可在 Mac CPU 和 GPU 上运行。

为了跟踪请求的完整生命周期，建议在以下位置设置断点（不同版本中的路径可能变化；找不到时可搜索对应符号）：
  vllm/v1/engine/llm_engine.py        add_request / step
  vllm/v1/core/sched/scheduler.py     Scheduler.schedule        <- 谁来执行这一步，以及 token 预算
  vllm/v1/core/kv_cache_manager.py    allocate_slots            <- PagedAttention 的块分配
  vllm/v1/worker/cpu/standalone_model_runner.py  execute_model  （GPU 上是 vllm/v1/worker/gpu_model_runner.py）
  vllm/v1/worker/cpu/sampler.py       CPUSampler.__call__       （GPU 上是 vllm/v1/sample/sampler.py 的 Sampler.forward）
完整的按顺序断点列表见 week01/README.md 的 "Debugging the request lifecycle" 一节。

在 Mac 上，本脚本默认将 VLLM_ENABLE_V1_MULTIPROCESSING 设为 0，
使引擎核心在当前进程中运行，从而能命中调度器中的断点（从 IDE 启动时也适用）。
可显式设置该环境变量以覆盖默认值。
"""
import os
import sys

# 必须在导入 vllm 之前设置该环境变量。
if sys.platform == "darwin":
    # setdefault 仅在该变量尚未设置时设为 "0"，不会覆盖用户显式设置的值。
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

# 导入 PyTorch，用于检查 CUDA 是否可用。
import torch
# 导入 vLLM 的模型接口和采样参数配置类。
from vllm import LLM, SamplingParams

# 从第一个命令行参数读取模型名称；未提供时使用默认的 Qwen 指令微调模型。
model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
# 检查 PyTorch 是否检测到 CUDA GPU。
on_gpu = torch.cuda.is_available()

# 创建 vLLM 模型实例。
llm = LLM(
    model=model,  # 要加载的模型名称或路径。
    dtype="half" if on_gpu else "float32",  # GPU 使用半精度；CPU 使用单精度。T4 不支持 BF16。
    max_model_len=2048,  # 设置模型可处理的最大序列长度。
    enforce_eager=not on_gpu,  # CPU 上强制使用 eager 执行；GPU 上不强制。
    # CPU 后端默认按 gpu_memory_utilization（0.9）× 整机内存预留空间，在 24 GB 的 Mac 上会给
    # KV cache 申请约 19 GiB，空闲内存不够就会报错。0.5B 模型、2048 token 上下文用 2 GiB 足够。
    # （等价于 VLLM_CPU_KVCACHE_SPACE=2 或 --kv-cache-memory-bytes。）GPU 上为 None，保持默认行为。
    kv_cache_memory_bytes=None if on_gpu else 2 * 1024**3,
)
# 准备两条待生成回答的提示词。
prompts = [
    "Explain PagedAttention in one sentence.",
    "What does a KV cache store?",
]
# 批量生成回答：temperature=0.0 使采样更确定；每条回答最多生成 48 个 token。
outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=48))
# 逐条遍历生成结果。
for o in outs:
    # 打印当前结果对应的提示词。
    print("PROMPT:", o.prompt)
    # 打印第一条候选回答，并去除回答首尾的空白字符。
    print("OUTPUT:", o.outputs[0].text.strip(), "\n")
# 打印实际使用的设备，以及 v1 多进程环境变量的当前值；未设置时显示 "(default)"。
print("device:", "cuda" if on_gpu else "cpu", "| VLLM_ENABLE_V1_MULTIPROCESSING =",
      os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "(default)"))
