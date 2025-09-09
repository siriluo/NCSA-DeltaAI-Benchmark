## NCSA-DeltaAI-Benchmark

Benchmarking NVIDIA GH200 (Grace Hopper) on NCSA DeltaAI for popular large language models (LLMs) across inference and training workloads.

### Goals
- Establish clear, reproducible baselines for LLM inference and training on DeltaAI GH200 nodes
- Evaluate scaling with different GPU counts and nodes
- Compare implementation stacks (e.g., vLLM for inference, VERL for RL-style training)
- Capture throughput, latency, memory footprint, and utilization metrics

### System: DeltaAI (GH200)
DeltaAI is powered by NVIDIA GH200 Grace Hopper Superchips. For account setup, job submission, software environment, containers, and general usage, see the official DeltaAI User Guide.

- DeltaAI Documentation: https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/index.html

### Project Status
This repository starts with documentation and planned structure. Container recipes and scripts will be filled in incrementally as we validate builds and runs on DeltaAI.

Current progress:
- Pulled GH200-optimized vLLM SIF from GHCR (`gh200_llm.sif`) using the image from [abacusai/gh200-llm](https://github.com/abacusai/gh200-llm/tree/main)
- Added SLURM jobs for vLLM server and inference benchmark
- Added interactive script to run vLLM benchmark against HF models
- Verified TinyLlama and Qwen models run; gated models (Meta Llama, some Mistral) require accepting licenses on Hugging Face

### Repository Structure (planned)
- `containers/`
  - `vllm.def` – Apptainer definition for vLLM (inference)
  - `verl.def` – Apptainer definition for VERL (training)
  - Built images (`*.sif`) are large and should not be committed
- `scripts/sbatch/`
  - `vllm_infer.slurm` – SLURM job for vLLM server or batch inference
  - `verl_train.slurm` – SLURM job for VERL training
- `benchmarks/`
  - `inference/` – harness and prompts for inference
  - `training/` – configs and runners for training
  - `results/` – CSV/JSON logs; large logs may be excluded from git
- `configs/` – model lists, prompt sets, run configs
- `data/` – datasets/prompts (excluded from git)

### Prerequisites
- DeltaAI account with SLURM access
- Apptainer available on DeltaAI compute/login nodes
- Sufficient project storage quota for container images, models, and logs

### Containers

We will use Apptainer to encapsulate runtime environments for inference (vLLM) and training (VERL).

#### vLLM (inference)
- Upstream: https://github.com/vllm-project/vllm
- Approach: start from an upstream Docker image (if suitable) or build via a definition file

Planned options:
1) Build from Docker image (fast path):
   - Example image candidates: `docker://vllm/vllm` or `docker://vllm/vllm-openai`
   - On DeltaAI:
     - `apptainer build vllm.sif docker://vllm/vllm-openai:latest`
2) Build from definition (`containers/vllm.def`):
   - `apptainer build vllm.sif containers/vllm.def`

Runtime flags:
- Always include `--nv` to expose NVIDIA GPUs: e.g., `apptainer exec --nv vllm.sif python -c "..."`

#### VERL (training)
- Upstream: https://github.com/volcengine/verl
- Approach: definition file that installs dependencies (PyTorch + CUDA toolchain compatible with GH200), then installs VERL from source

Planned build:
- `apptainer build verl.sif containers/verl.def`

### Running on DeltaAI (SLURM)

We will provide SLURM scripts under `scripts/sbatch/`. Below are minimal templates to be refined after first runs.

#### Example: vLLM Inference (single node)
```
#!/bin/bash
#SBATCH -A <your_allocation>
#SBATCH -J vllm-infer
#SBATCH -p <gpu_partition>
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -t 00:30:00
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err

set -euo pipefail

module purge
# module load apptainer  # if needed on DeltaAI

SIF=/path/to/vllm.sif
MODEL="meta-llama/Llama-3-8B-Instruct"  # example; adjust to your access

srun apptainer exec --nv "$SIF" python - <<'PY'
from vllm import LLM, SamplingParams
prompts = ["What is the capital of France?", "Explain GH200 in 2 sentences."]
llm = LLM(model="${MODEL}")
outs = llm.generate(prompts, SamplingParams(max_tokens=64))
for o in outs:
    print(o.outputs[0].text)
PY
```

Metrics to capture during/after run:
- Throughput (tokens/s), latency (p50/p95), GPU memory usage
- `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1`

#### Example: VERL Training (single node prototype)
### Interactive Inference Benchmark (vLLM)

Request interactive GPU allocation on DeltaAI (example: 1x GH200 GPU):
```
srun --account=bbka-dtai-gh --partition=ghx4 --nodes=1 --gpus-per-node=1 --tasks=1 --tasks-per-node=1 --cpus-per-task=8 --mem=48g --pty bash
```

Run TinyLlama or Qwen (public) to sanity check (downloads cached in your job environment):
```
MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
./scripts/interactive/vllm_llama3_benchmark.sh

MODEL=Qwen/Qwen2-7B-Instruct \
./scripts/interactive/vllm_llama3_benchmark.sh
```

To run gated models like Meta Llama 3 or Mistral Instruct, first accept licenses with your HF account, then export your token in the interactive shell:
```
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
MODEL=meta-llama/Meta-Llama-3-8B-Instruct \
./scripts/interactive/vllm_llama3_benchmark.sh
```

```
#!/bin/bash
#SBATCH -A <your_allocation>
#SBATCH -J verl-train
#SBATCH -p <gpu_partition>
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -t 02:00:00
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err

set -euo pipefail

module purge
# module load apptainer  # if needed on DeltaAI

SIF=/path/to/verl.sif

srun apptainer exec --nv "$SIF" bash -lc '
python -m verl.trainer \
  --help  # replace with a concrete recipe from VERL examples
'
```

### Benchmark Matrix (initial draft)
- Models (inference): e.g., Llama-3-8B/70B variants, Mistral 7B, Qwen 7B/14B (subject to access)
- Models (training): small/medium open models (e.g., 1.3B–7B) and VERL examples/recipes
- GPU scale: {1, 2, 4, 8} per node; then multi-node if applicable
- Batch sizes and sequence lengths appropriate to model sizes and memory

### Output and Reproducibility
- Save run configs and metrics to `benchmarks/results/` (CSV/JSON). Large artifacts may be excluded from git.
- Record software versions (driver, CUDA, framework, vLLM/VERL git SHAs) and container digests.
- For comparable runs: pin seeds and document tokenization/backends.

### Git Hygiene
- Do not commit large model weights or datasets.
- Keep `data/` excluded from git. Consider excluding large logs and `.sif` images.

### References
- DeltaAI User Guide: https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/index.html
- vLLM: https://github.com/vllm-project/vllm
- VERL: https://github.com/volcengine/verl

### License
TBD


