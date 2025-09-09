#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from datetime import datetime

import torch
from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run vLLM inference and measure throughput (tokens/sec)."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Hugging Face model id or local path (e.g., meta-llama/Llama-3-8B-Instruct)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=64,
        help="Number of prompts to generate in one batch",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain the NVIDIA GH200 Grace Hopper architecture briefly.",
        help="Prompt text to replicate across the batch",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p nucleus sampling",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Max model context length. If not set, use model default.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="GPU memory utilization target for vLLM",
    )
    parser.add_argument(
        "--allow-long-max-model-len",
        action="store_true",
        help="Set VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 to bypass model max length checks.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graphs by running in eager mode.",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="",
        help="Explicit path to write JSON results. If empty, auto-generate under benchmarks/results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code for HF models",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    prompts = [args.prompt for _ in range(args.num_prompts)]

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    if args.allow_long_max_model_len:
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    llm = LLM(**llm_kwargs)

    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.perf_counter()

    elapsed_sec = max(end_time - start_time, 1e-9)

    total_output_tokens = 0
    total_prompt_tokens = 0

    for request_output in outputs:
        # prompt_token_ids is available on RequestOutput
        if hasattr(request_output, "prompt_token_ids") and request_output.prompt_token_ids is not None:
            total_prompt_tokens += len(request_output.prompt_token_ids)
        # count only the first completion per prompt for simplicity
        if request_output.outputs:
            total_output_tokens += len(request_output.outputs[0].token_ids)

    tokens_per_second = total_output_tokens / elapsed_sec

    gpu_count = torch.cuda.device_count()
    device_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)] if gpu_count > 0 else []
    cuda_version = torch.version.cuda

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    default_results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "benchmarks",
        "results",
    )
    os.makedirs(default_results_dir, exist_ok=True)

    results_path = (
        args.results_path
        if args.results_path
        else os.path.join(default_results_dir, f"vllm_infer_{timestamp}.json")
    )

    results = {
        "timestamp_utc": timestamp,
        "model": args.model,
        "num_prompts": args.num_prompts,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "elapsed_sec": elapsed_sec,
        "total_output_tokens": total_output_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "tokens_per_second": tokens_per_second,
        "env": {
            "torch_version": torch.__version__,
            "cuda_version": cuda_version,
            "vllm_version": __import__("vllm").__version__,
            "gpu_count": gpu_count,
            "device_names": device_names,
        },
    }

    print(json.dumps(results, indent=2))
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to: {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


