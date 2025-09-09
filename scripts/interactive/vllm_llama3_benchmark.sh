#!/usr/bin/env bash
set -euo pipefail

# Usage:
# srun --account=bbka-dtai-gh --partition=ghx4 --nodes=1 --gpus-per-node=1 \
#      --tasks=1 --tasks-per-node=1 --cpus-per-task=8 --mem=48g --pty bash
# export HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# MODEL=meta-llama/Meta-Llama-3-8B-Instruct NUM_PROMPTS=64 MAX_NEW_TOKENS=128 TP=1 \
#   ./scripts/interactive/vllm_llama3_benchmark.sh

SIF="/work/nvme/bbka/shirui/NCSA-DeltaAI-Benchmark/containers/vllm/gh200_llm.sif"
SCRIPT="/work/nvme/bbka/shirui/NCSA-DeltaAI-Benchmark/benchmarks/inference/vllm_benchmark.py"

export HF_HOME="${HF_HOME:-${TMPDIR:-$HOME/.cache}/huggingface}"
mkdir -p "$HF_HOME"

if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TP="${TP:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

echo "[vLLM] Llama-3 benchmark (interactive)"
echo "Model=${MODEL} Prompts=${NUM_PROMPTS} MaxNew=${MAX_NEW_TOKENS} TP=${TP} GPUUtil=${GPU_MEM_UTIL}"
echo "HF_HOME=${HF_HOME}"

apptainer exec --nv \
  --env HF_HOME="$HF_HOME" \
  --env HF_TOKEN="${HF_TOKEN:-}" \
  --env HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-}" \
  "$SIF" bash -lc "\
python \"$SCRIPT\" \
  --model \"$MODEL\" \
  --num-prompts $NUM_PROMPTS \
  --max-new-tokens $MAX_NEW_TOKENS \
  --tensor-parallel-size $TP \
  --gpu-memory-utilization $GPU_MEM_UTIL \
  --seed 1234 \
  --trust-remote-code\
"


