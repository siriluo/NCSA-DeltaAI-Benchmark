#!/bin/bash
#SBATCH -A bbka-dtai-gh
#SBATCH -J vllm-server
#SBATCH -p ghx4
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH -t 02:00:00
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err

set -euo pipefail

# Optional: module load apptainer if required on DeltaAI
module purge
# module load apptainer

# Path to the vLLM SIF image pulled from GHCR
SIF="/work/nvme/bbka/shirui/NCSA-DeltaAI-Benchmark/containers/vllm/gh200_llm.sif"

# Configurable parameters via environment variables or defaults
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
TP="${TP:-1}"
MAX_LEN="${MAX_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

echo "Starting vLLM OpenAI-compatible server"
echo "Model: ${MODEL}"
echo "Host: ${HOST}  Port: ${PORT}  TP: ${TP}  MaxLen: ${MAX_LEN}  GPUUtil: ${GPU_MEM_UTIL}"

# Launch server (runs until job end)
srun apptainer exec --nv "${SIF}" bash -lc "\
python -m vllm.entrypoints.openai.api_server \
  --model \"${MODEL}\" \
  --host ${HOST} \
  --port ${PORT} \
  --tensor-parallel-size ${TP} \
  --max-model-len ${MAX_LEN} \
  --gpu-memory-utilization ${GPU_MEM_UTIL} \
  --disable-log-requests\
"

# Usage examples:
# 1) Submit batch job:
#    sbatch scripts/sbatch/vllm_server.slurm
#    (Override defaults) MODEL=meta-llama/Llama-3-8B-Instruct PORT=8080 sbatch scripts/sbatch/vllm_server.slurm
#
# 2) Interactive allocation (1xGPU), then run server manually:
#    srun --account=bbka-dtai-gh --partition=ghx4 --nodes=1 --gpus-per-node=1 --tasks=1 \
#         --tasks-per-node=1 --cpus-per-task=8 --mem=48g --pty bash
#    apptainer exec --nv /work/nvme/bbka/shirui/NCSA-DeltaAI-Benchmark/containers/vllm/gh200_llm.sif \
#      python -m vllm.entrypoints.openai.api_server --model ${MODEL} --host 0.0.0.0 --port 8000


