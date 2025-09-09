#!/bin/bash
#SBATCH -A bbka-dtai-gh
#SBATCH -J vllm-infer-bench
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

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TP="${TP:-1}"
MAX_LEN="${MAX_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

echo "Running vLLM benchmark"
echo "Model=${MODEL} Prompts=${NUM_PROMPTS} MaxNew=${MAX_NEW_TOKENS} TP=${TP} MaxLen=${MAX_LEN} GPUUtil=${GPU_MEM_UTIL}"

srun apptainer exec --nv "${SIF}" bash -lc "\
python \"${SCRIPT}\" \
  --model \"${MODEL}\" \
  --num-prompts ${NUM_PROMPTS} \
  --max-new-tokens ${MAX_NEW_TOKENS} \
  --tensor-parallel-size ${TP} \
  --max-model-len ${MAX_LEN} \
  --gpu-memory-utilization ${GPU_MEM_UTIL} \
  --seed 1234 \
  --trust-remote-code\
"


