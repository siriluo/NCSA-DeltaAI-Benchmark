#!/bin/bash
#SBATCH -A bbka-dtai-gh
#SBATCH -J vllm-llama3-bench
#SBATCH -p ghx4
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH -t 01:00:00
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err

set -euo pipefail

module purge
# module load apptainer

SIF="/work/nvme/bbka/shirui/NCSA-DeltaAI-Benchmark/containers/vllm/gh200_llm.sif"
SCRIPT="/work/nvme/bbka/shirui/NCSA-DeltaAI-Benchmark/benchmarks/inference/vllm_benchmark.py"

# Writable HF cache inside the job sandbox (prefer TMPDIR; fallback to home cache)
export HF_HOME="${HF_HOME:-${TMPDIR:-$HOME/.cache}/huggingface}"
mkdir -p "$HF_HOME"

# Pass HF token if provided. Do NOT hardcode secrets in this script.
if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

# Defaults (override via environment at submission time)
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TP="${TP:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

echo "Benchmarking Llama-3 with vLLM"
echo "Model=${MODEL} Prompts=${NUM_PROMPTS} MaxNew=${MAX_NEW_TOKENS} TP=${TP} GPUUtil=${GPU_MEM_UTIL}"
echo "HF_HOME=${HF_HOME}"

# Note: Ensure you have accepted the Llama-3 gated license on Hugging Face for the token in use

srun apptainer exec --nv \
  --env HF_HOME="$HF_HOME" \
  --env HF_TOKEN="${HF_TOKEN:-}" \
  --env HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-}" \
  "${SIF}" bash -lc "\
python \"${SCRIPT}\" \
  --model \"${MODEL}\" \
  --num-prompts ${NUM_PROMPTS} \
  --max-new-tokens ${MAX_NEW_TOKENS} \
  --tensor-parallel-size ${TP} \
  --gpu-memory-utilization ${GPU_MEM_UTIL} \
  --seed 1234 \
  --trust-remote-code\
"

# Submit with env overrides, e.g.:
# HF_TOKEN=hf_xxx NUM_PROMPTS=128 MAX_NEW_TOKENS=256 sbatch scripts/sbatch/vllm_llama3_benchmark.slurm


