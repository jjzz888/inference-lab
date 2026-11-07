# Week 1: Build vLLM and SGLang from source

| Where | What | Script |
|---|---|---|
| Mac (M3) | vLLM CPU build, read the code and step through it with a debugger | `mac_init.sh` (once), `mac_session.sh` (every session), `smoke_offline.py` |
| Kaggle T4 x2 | vLLM + SGLang GPU editable install, quick tests, one serving benchmark each | `kaggle_setup.sh`, `run_bench.sh` |

Lab GPUs are not used this week. Don't build vLLM from source there.

## Mac
```bash
# Once: clone, fork on GitHub (remote "fork"; origin stays upstream), CPU build, smoke test
bash week01/mac_init.sh

# Every session: sync upstream, activate venv, rebuild only if C++/build files changed
source week01/mac_session.sh
python week01/smoke_offline.py   # sets VLLM_ENABLE_V1_MULTIPROCESSING=0 on macOS
```
PR flow (Week 2): branch off `main`, `git push fork <branch>`, then `gh pr create` (targets upstream).

## Debugging the request lifecycle in VS Code (Mac)

### 1. Use `~/code/vllm/.venv/bin/python` as the interpreter
1. Open the **repo root** so VS Code finds `.venv`: `code ~/code/vllm`.
2. `Cmd+Shift+P` → **Python: Select Interpreter** → pick `.venv (3.12.x)` (`./.venv/bin/python`). If it isn't listed, choose **Enter interpreter path…** and type `~/code/vllm/.venv/bin/python`.
3. Check: open any `.py` file. The status bar shows `{ } Python` and the `.venv` interpreter. While a Markdown file is active, the interpreter is not shown. `Cmd+click` on `import torch` should jump into `.venv/lib/python3.12/site-packages/torch`.

Extensions needed: Python, Pylance, Python Debugger (debugpy). vLLM's `.gitignore` already ignores `.vscode/`.

### 2. Debug config: `~/code/vllm/.vscode/launch.json`
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "vLLM smoke (CPU, in-process)",
      "type": "debugpy",
      "request": "launch",
      "program": "/Users/jing/Documents/Obsidian Vault/nvidia/inference-lab/week01/smoke_offline.py",
      "cwd": "${workspaceFolder}",
      "python": "${workspaceFolder}/.venv/bin/python",
      "justMyCode": false,
      "console": "integratedTerminal",
      "env": {
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "https_proxy": "http://127.0.0.1:7890",
        "http_proxy": "http://127.0.0.1:7890",
        "no_proxy": "localhost,127.0.0.1,::1"
      }
    }
  ]
}
```
- `VLLM_ENABLE_V1_MULTIPROCESSING=0`: EngineCore runs in the same process, so breakpoints in the scheduler hit. (`smoke_offline.py` also defaults it on macOS.)
- `justMyCode: false`: lets you step into torch and transformers.
- Proxy vars: a program launched from VS Code doesn't see the proxy you `export`ed in a terminal. Only needed when a model isn't cached yet.

### 3. Breakpoints, in request order
Jump to a symbol with `Cmd+T` (e.g. type `allocate_slots`). Line numbers change with every upstream commit, so look functions up by name.

| # | Stage | File | Function | What to look at |
|---|---|---|---|---|
| 1 | API entry | `vllm/entrypoints/llm.py` | `LLM.generate` | prompts and `SamplingParams` going in |
| 2 | Tokenize, wrap | `vllm/v1/engine/llm_engine.py` | `LLMEngine.add_request` | prompt → token ids → `EngineCoreRequest` |
| 3 | Enqueue | `vllm/v1/engine/core.py` | `EngineCore.add_request` | `Request` lands in `scheduler.waiting` |
| 4 | One engine iteration | `vllm/v1/engine/core.py` | `EngineCore.step` | the loop: schedule → execute → update |
| 5 | **Schedule** | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` | `token_budget`, `self.waiting`, `self.running`; returned `SchedulerOutput` (`num_scheduled_tokens`, `scheduled_new_reqs`) |
| 6 | **KV blocks (PagedAttention)** | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager.allocate_slots` (and `get_computed_blocks` for prefix-cache hits) | block ids handed to each request; a new block only every `block_size` tokens |
| 7 | Forward pass | `vllm/v1/worker/cpu/standalone_model_runner.py` | `StandaloneCPUModelRunner.execute_model` | input batch built from `SchedulerOutput`; logits out |
| 8 | Sample | `vllm/v1/worker/cpu/sampler.py` (via `sample_tokens`) | `CPUSampler.__call__` | logits → next token id (greedy here: `temperature=0`) |
| 9 | Update state | `vllm/v1/core/sched/scheduler.py` | `Scheduler.update_from_output` | token appended, stop checks; finished requests → `KVCacheManager.free` |
| 10 | Detokenize, return | `vllm/v1/engine/output_processor.py` | `OutputProcessor.process_outputs` | token ids → text → `RequestOutput` |

On GPU (Kaggle), steps 7–8 are `GPUModelRunner.execute_model` / `sample_tokens` in `vllm/v1/worker/gpu_model_runner.py` and `Sampler.forward` in `vllm/v1/sample/sampler.py`. Steps 1–6 and 9–10 are the same code.

### 4. Walkthrough
1. Set the breakpoints in the table (at least 5, 6, 9), select **vLLM smoke (CPU, in-process)** in Run and Debug, and press `F5`.
2. **Startup noise:** model loading, profiling and warm-up call the model runner (7–8) before any real request. Press `F5` (Continue) until you are in `Scheduler.schedule` with a non-empty `self.waiting`.
3. **First step = prefill.** Both prompts move from `waiting` to `running`. `num_scheduled_tokens` per request equals its prompt length. `allocate_slots` hands out enough blocks for the whole prompt.
4. **Later steps = decode.** Each running request gets 1 scheduled token per step. Watch `allocate_slots`: it usually returns no new blocks, and only grabs one when a request crosses a `block_size` boundary.
5. **Finish.** When a request hits `max_tokens=48` (or EOS), `update_from_output` marks it finished and its blocks are freed. The output then flows through `process_outputs` back to `LLM.generate`.
6. Useful debugger moves: `F10` step over, `F11` step into, `Shift+F11` step out. Right-click a breakpoint → **Edit Condition** (e.g. `len(self.waiting) > 0`) to skip empty iterations. The **Call Stack** panel shows how steps 4 → 5 → 6 nest.
7. Turn what you saw into the Week 1 diagram (API server → scheduler → KV cache manager → model runner → sampler) and save it in `week01/`.

## Push to GitHub (before every Kaggle run)
Kaggle only sees what is on GitHub (`jjzz888/inference-lab`, public), not your local files. If you skip this, `git clone` on Kaggle gets an old or empty repo and fails with `kaggle_setup.sh: No such file or directory`.

The local `inference-lab/` folder is a git repo tracking `origin/main`. After changing any script:
```bash
cd "/Users/jing/Documents/Obsidian Vault/nvidia/inference-lab"
git status                      # check what changed
git add -A && git commit -m "week01: <what changed>"
git push
```
Check: `gh api repos/jjzz888/inference-lab/contents/week01 --jq '.[].name'` lists the scripts. The repo is public, so never commit tokens or keys.

## Kaggle
Notebook settings: **Accelerator = GPU T4 x2**, **Internet = On**. Avoid P100.

```python
# Cell 1: setup (~10–20 min with precompiled vLLM kernels)
# rm -rf first: a clone left over from an earlier run in this session won't update itself.
!rm -rf /kaggle/working/inference-lab
!git clone https://github.com/jjzz888/inference-lab.git /kaggle/working/inference-lab
!bash /kaggle/working/inference-lab/week01/kaggle_setup.sh
```
```python
# Cell 2: quick GPU smoke test with vLLM
!~/lab/vllm/.venv/bin/python /kaggle/working/inference-lab/week01/smoke_offline.py
```
```python
# Cell 3: benchmarks (results -> /kaggle/working/results/week01/<engine>-<time>/)
!bash /kaggle/working/inference-lab/week01/run_bench.sh both
# If SGLang fails in FlashInfer on T4:
# !SGLANG_ATTN=triton bash /kaggle/working/inference-lab/week01/run_bench.sh sglang
```
Copy `results/` and `~/lab/logs/` back into this repo and commit them. **Shut the session down afterwards**, or it keeps using quota.

## Known pitfalls
✅ = the scripts handle it automatically; the note is for when it shows up anyway, or in your own commands.

### Mac
- **Run the scripts from `inference-lab/`, or quote the absolute path.** The commands above use relative paths, and `Obsidian Vault` contains a space. `mac_session.sh` must be `source`d, not run with `bash`.
- ✅ **No proxy → `git clone` times out** (`curl 28 ... Operation timed out`). git, uv and Hugging Face all need GitHub/PyPI/HF access through the local proxy. If no proxy is set and `127.0.0.1:7890` is listening, both scripts export `https_proxy`/`http_proxy` and set `no_proxy=localhost,127.0.0.1,::1` so local vLLM/SGLang servers bypass it. Use `PROXY=http://host:port` for another port, `PROXY=none` to skip.
- ✅ **Flaky proxy drops a fetch mid-transfer** (`curl 56 ... SSL_ERROR_SYSCALL`, then `bad pack header`). This is a transient network error, not a repo problem. `mac_session.sh` retries `git fetch` 3 times. If it keeps happening, switch to a more stable proxy node.
- ✅ **Old uv silently breaks the vLLM install.** uv 0.7.x can't parse `exclude-dependencies` in vLLM's `pyproject.toml` and prints `Failed to parse pyproject.toml ... unknown field`. It then ignores the **whole** `[tool.uv]` table: torch's `no-build-isolation` and the `openai-harmony` exclusion. Fix: `uv self update`. `mac_init.sh` checks this before installing (and `kaggle_setup.sh` always upgrades uv).
- ✅ **CPU backend asks for ~19 GiB of KV cache** (`ValueError: Available memory on node 0 (6.76/24.0 GiB) ... less than requested memory for kv`). On CPU, vLLM reserves `gpu_memory_utilization` (0.9) × total RAM. That is impossible on a 24 GB laptop, so freeing memory won't fix it. `smoke_offline.py` sets `kv_cache_memory_bytes` to 2 GiB on CPU. In your own scripts, use `VLLM_CPU_KVCACHE_SPACE=2` or `--kv-cache-memory-bytes`. (The low "available" number is normal on macOS: file cache counts as used. Check real pressure with `memory_pressure`.)
- ✅ **Breakpoints in the scheduler never hit.** The v1 engine core runs in a separate process by default. `VLLM_ENABLE_V1_MULTIPROCESSING=0` keeps it in-process. `smoke_offline.py` and `mac_session.sh` set it on macOS.
- **Where the build record lives:** `~/code/vllm/.venv/.last_build_commit` (hidden file) holds the commit of the last successful build. `mac_session.sh` compares against it to decide whether C++/build files changed. Force a rebuild with `REBUILD=1 source week01/mac_session.sh`. The build log is `~/code/vllm_mac_build.log`.

### Kaggle
- ✅ **`uv pip install` lands in Kaggle's system Python instead of the venv.** Symptoms: tracebacks through `/usr/local/bin/vllm` and `/usr/local/lib/python3.13/dist-packages/...`, `PyTorch and TorchAudio were compiled with different CUDA versions` (vLLM's deps upgraded the image's torch but not its torchaudio), then `No module named 'torch'` in the venv sanity check. Cause (confirmed): the Kaggle image sets `UV_SYSTEM_PYTHON=true`, which overrides venv activation. (It also sets empty `UV_CONSTRAINT`/`UV_BUILD_CONSTRAINT`, `UV_INSTALL_DIR=/usr/local/bin` and `UV_BREAK_SYSTEM_PACKAGES=false`, all harmless.) `kaggle_setup.sh` now prints any `UV_*` vars or `uv.toml`, and `use_venv` pins every uv call with `UV_PYTHON=$VIRTUAL_ENV/bin/python UV_SYSTEM_PYTHON=false`. The sanity check prints `in venv: True`. **If it already happened in a session, use Run → Factory reset**: the system Python's torch was changed, so uninstalling vLLM alone doesn't restore it.
- ✅ **vLLM tests fail to load: `No module named 'tblib'`** (from `tests/conftest.py`). The tests need more than `pytest`. `kaggle_setup.sh` installs only what `tests/v1/core` needs (`pytest pytest-asyncio tblib`). Don't install the full `requirements/test/cuda.txt`: it has ~7.9k lines and pins `torch==2.14.0+cu130`, which would replace the venv's torch.
- ✅ **SGLang build fails: `cargo is required to discover the Rust extension modules`.** SGLang's `setup.py` builds Rust (PyO3) extensions, and Kaggle has no Rust toolchain. `kaggle_setup.sh` sets `SGLANG_BUILD_RUST_EXTS=none` by default. That is fine for HTTP serving: those extensions only serve gRPC, the Rust server, a few multimodal processors, and the Rust radix tree, which falls back to the Python TreeCore (server log: `Using the Python TreeCore`). **Note it next to SGLang benchmark numbers.** To use the Rust radix tree, run with `SGLANG_BUILD_RUST_EXTS=rust_tree_core`: the script installs rustup and builds only that crate.
- **SGLang needs CUDA 13 on `main`.** `kaggle_setup.sh` reads the driver's CUDA version and falls back to `v0.5.19` (the last CUDA 12 release). Override with `SGLANG_REF=...`.
- **No BF16 on T4.** Both scripts force `--dtype half`.
- **vLLM precompiled mode needs `uv pip`.** Plain `pip install -e .` does not work with `VLLM_USE_PRECOMPILED=1`.
- **Changing C++/CUDA code in vLLM needs a full build** (`VLLM_FULL_BUILD=1`). On Kaggle's CPUs this takes hours, so rent a box for it instead.
- **Every Kaggle session starts clean.** Rerun `kaggle_setup.sh` each time.

## Done when
- [ ] `smoke_offline.py` runs on the Mac and stops at a breakpoint in `Scheduler.schedule`
- [ ] Both engines install on Kaggle, and `tests/v1/core` has run (failures noted in `LOG.md`)
- [ ] One `run_bench.sh both` result committed, with TTFT, TPOT and throughput copied into `LOG.md`
