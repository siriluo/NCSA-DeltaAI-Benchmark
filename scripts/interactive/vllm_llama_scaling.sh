#!/usr/bin/env bash
set -euo pipefail

# Usage:
# 1) Allocate GPUs interactively (choose G matching your sweep upper bound):
#    srun --account=bbka-dtai-gh --partition=ghx4 --nodes=1 --gpus-per-node=4 \
#         --tasks=1 --cpus-per-task=32 --mem=192g --pty bash
# 2) Export HF token if model is gated and license accepted:
#    export HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# 3) Run the script (override MODEL/TP_LIST/NUM_PROMPTS/MAX_NEW_TOKENS as needed):
#    MODEL=meta-llama/Meta-Llama-3-70B-Instruct TP_LIST="1 2 4" NUM_PROMPTS=128 MAX_NEW_TOKENS=128 \
#      ./scripts/interactive/vllm_llama_scaling.sh

SIF="/work/nvme/bbka/shirui/NCSA-DeltaAI-Benchmark/containers/vllm/gh200_llm.sif"
SCRIPT="/work/nvme/bbka/shirui/NCSA-DeltaAI-Benchmark/benchmarks/inference/vllm_benchmark.py"

# Select a writable cache root: prefer SLURM_TMPDIR/TMPDIR, else project-local .hf_cache
CACHE_ROOT_CANDIDATES=(
  "${SLURM_TMPDIR:-}"
  "${TMPDIR:-}"
  "$PWD/.hf_cache"
)

for cand in "${CACHE_ROOT_CANDIDATES[@]}"; do
  if [[ -n "$cand" ]]; then
    mkdir -p "$cand" 2>/dev/null || true
    if [[ -d "$cand" && -w "$cand" ]]; then
      HF_CACHE_ROOT="$cand"
      break
    fi
  fi
done
HF_CACHE_ROOT="${HF_CACHE_ROOT:-$PWD/.hf_cache}"
mkdir -p "$HF_CACHE_ROOT/hf_home" "$HF_CACHE_ROOT/hub"
export HF_HOME="$HF_CACHE_ROOT/hf_home"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_ROOT/hub"
export TRANSFORMERS_CACHE="$HF_CACHE_ROOT/hub"

if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

MODEL="${MODEL:-meta-llama/Meta-Llama-3-70B-Instruct}"
TP_LIST=${TP_LIST:-"2 4"}
NUM_PROMPTS="${NUM_PROMPTS:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
# Optional: cap max model length to reduce KV cache usage during gen
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
ALLOW_LONG="${ALLOW_LONG:-}"  # set to 1 to allow overriding model max length
ENFORCE_EAGER="${ENFORCE_EAGER:-}" # set to 1 to disable CUDA graphs

# Reduce fragmentation for large models
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "[vLLM] Scaling sweep for ${MODEL}"
echo "TP_LIST=${TP_LIST}  NUM_PROMPTS=${NUM_PROMPTS}  MAX_NEW_TOKENS=${MAX_NEW_TOKENS}  ENFORCE_EAGER=${ENFORCE_EAGER:-0}"
echo "HF_HOME=${HF_HOME}"

for TP in ${TP_LIST}; do
  echo "--- Running TP=${TP} ---"
  apptainer exec --nv \
    --env HF_HOME="$HF_HOME" \
    --env HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE" \
    --env TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
    --env HF_TOKEN="${HF_TOKEN:-}" \
    --env HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-}" \
    --env PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    --bind "$HF_HOME:$HF_HOME" \
    --bind "$HUGGINGFACE_HUB_CACHE:$HUGGINGFACE_HUB_CACHE" \
    --bind "$TRANSFORMERS_CACHE:$TRANSFORMERS_CACHE" \
    "$SIF" bash -lc "\
python \"$SCRIPT\" \
  --model \"$MODEL\" \
  --num-prompts $NUM_PROMPTS \
  --max-new-tokens $MAX_NEW_TOKENS \
  --tensor-parallel-size $TP \
  --gpu-memory-utilization $GPU_MEM_UTIL \
  --seed 1234 \
  --trust-remote-code \\
  ${MAX_MODEL_LEN:+--max-model-len $MAX_MODEL_LEN} \\
  ${ALLOW_LONG:+--allow-long-max-model-len} \\
  ${ENFORCE_EAGER:+--enforce-eager} \
"
done

echo "Scaling sweep complete. See JSON outputs in benchmarks/results/"


