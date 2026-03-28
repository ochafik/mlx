#!/usr/bin/env python3
"""Benchmark mlx-lm throughput in CPU and GPU modes."""

import argparse
import time
import mlx.core as mx

def benchmark_mlx(model_name: str, prompt: str, max_tokens: int, use_cpu: bool = False):
    """Run mlx-lm benchmark and return tokens/second."""

    # Set device BEFORE importing mlx_lm
    if use_cpu:
        mx.set_default_device(mx.cpu)
        print(f"Device set to: {mx.default_device()}")
    else:
        mx.set_default_device(mx.gpu)
        print(f"Device set to: {mx.default_device()}")

    from mlx_lm import load, generate

    print(f"Loading model: {model_name}")
    model, tokenizer = load(model_name)

    # Warmup run
    print("Warmup run...")
    _ = generate(model, tokenizer, prompt=prompt, max_tokens=10, verbose=False)

    # Benchmark run
    print(f"Generating {max_tokens} tokens...")
    start = time.perf_counter()
    response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    elapsed = time.perf_counter() - start

    # Count output tokens
    output_tokens = len(tokenizer.encode(response)) - len(tokenizer.encode(prompt))
    tokens_per_sec = output_tokens / elapsed

    print(f"\n{'='*50}")
    print(f"MLX-LM Benchmark Results ({'CPU' if use_cpu else 'GPU/Metal'})")
    print(f"{'='*50}")
    print(f"Model: {model_name}")
    print(f"Output tokens: {output_tokens}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Throughput: {tokens_per_sec:.2f} tokens/sec")
    print(f"{'='*50}")

    return tokens_per_sec

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark mlx-lm throughput")
    parser.add_argument("--model", default="mlx-community/Qwen3-4B-Thinking-2507-4bit",
                        help="Model name/path")
    parser.add_argument("--prompt", default="Explain the theory of relativity in simple terms.",
                        help="Prompt to generate from")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Maximum tokens to generate")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU mode")

    args = parser.parse_args()
    benchmark_mlx(args.model, args.prompt, args.max_tokens, args.cpu)
