#!/usr/bin/env bash
# Experiment 2: Qwen3.8-27B INT8 W8A16 + MTP on 4x RTX 3090 (PCIe, no NVLink).
#
# This is the model card's validated command, adapted for this box:
#   - TP2 -> TP4 (4 cards available; the card only validated TP2)
#   - max-model-len / max-num-seqs raised to the values under test
#
# Usage:  bash experiments/run-int8.sh
# Log:    /tmp/vllm-int8.log
set -euo pipefail

cd "$(dirname "$0")/.."

# The venv's bin directory must be on PATH.
#
# Running `.venv/bin/vllm` directly works for Python imports, but it does NOT
# put .venv/bin on PATH the way `source .venv/bin/activate` does. FlashInfer's
# JIT shells out to the bare command `ninja` when it compiles prefill kernels
# at startup, so without this the workers die with:
#     FileNotFoundError: [Errno 2] No such file or directory: 'ninja'
export PATH="$PWD/.venv/bin:$PATH"
export VIRTUAL_ENV="$PWD/.venv"

# --- this machine's quirks: consumer Ampere, no P2P / NVLink ---
export NCCL_P2P_DISABLE=1        # nvidia-smi topo -m shows PHB/NODE only
export NCCL_CUMEM_ENABLE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

# --- which cards / how many ---
export CUDA_VISIBLE_DEVICES=${GPU_LIST:-0,1,2,3}
TP=${TP_SIZE:-4}

MODEL=lued/Qwen3.8-27B-INT8-W8A16-MTP

args=(
  # identity
  --served-model-name Qwen3.8-27B
  --host 127.0.0.1
  --port 8000
  --trust-remote-code

  # the four knobs that matter
  --tensor-parallel-size "$TP"
  --dtype bfloat16
  --gpu-memory-utilization 0.92
  --max-model-len 262144
  --max-num-seqs 4

  # memory shaping
  --max-num-batched-tokens 8192
  --kv-cache-dtype fp8_e4m3
  --performance-mode balanced

  # model wiring - required by this architecture, not preferences
  --mamba-cache-mode align
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder
  --enable-auto-tool-choice
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}'
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"repetition_penalty":1.0,"presence_penalty":0.0}'

  # local workarounds + safety
  --disable-custom-all-reduce
  # Safe only while the vllm#48375 patch is applied to the installed vllm --
  # see experiments/vllm-48375-mamba-drop-eagle-block.patch. Without it, an
  # MTP prefix-cache hit can resume from a rejected-draft recurrent state and
  # serve wrong output silently. Re-apply the patch after `uv sync`/upgrade.
  --enable-prefix-caching
  --enable-chunked-prefill
  --prefix-match-unit 16
  --enable-prompt-tokens-details
  --enable-per-request-metrics
)

exec .venv/bin/vllm serve "$MODEL" "${args[@]}"
