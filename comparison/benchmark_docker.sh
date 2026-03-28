#!/bin/bash
# CPU-only benchmark for MLX vs llama.cpp in Docker

set -e

PROMPT="Explain the theory of relativity in simple terms."
MAX_TOKENS=256

echo "=============================================="
echo "CPU-Only Benchmark: MLX vs llama.cpp"
echo "=============================================="
echo ""

# Get CPU info
echo "CPU Info:"
lscpu | grep "Model name" || cat /proc/cpuinfo | grep "model name" | head -1
nproc_count=$(nproc)
echo "CPU cores: $nproc_count"
echo ""

# Benchmark MLX CPU
echo "=============================================="
echo "1. MLX-LM (CPU)"
echo "=============================================="
python /app/benchmark_mlx.py \
    --model mlx-community/Qwen3-4B-Thinking-2507-4bit \
    --prompt "$PROMPT" \
    --max-tokens $MAX_TOKENS \
    --cpu

echo ""

# Benchmark llama.cpp Q4_0
echo "=============================================="
echo "2. llama.cpp Q4_0 (CPU)"
echo "=============================================="
echo "Model: /app/models/Qwen3-4B-Thinking-Q4_0.gguf"
echo "Max tokens: $MAX_TOKENS"
echo ""

/usr/local/bin/llama-cli \
    --model /app/models/Qwen3-4B-Thinking-Q4_0.gguf \
    --prompt "$PROMPT" \
    --n-predict $MAX_TOKENS \
    --n-gpu-layers 0 \
    --ctx-size 2048 \
    --temp 0.7 \
    --threads $nproc_count \
    --no-display-prompt \
    2>&1 | tee /tmp/llama_q4_0.log

echo ""
echo "Q4_0 Timing:"
grep -E "(eval time|total time|tokens per second)" /tmp/llama_q4_0.log || true

echo ""

# Benchmark llama.cpp Q4_1
echo "=============================================="
echo "3. llama.cpp Q4_1 (CPU)"
echo "=============================================="
echo "Model: /app/models/Qwen3-4B-Thinking-Q4_1.gguf"
echo "Max tokens: $MAX_TOKENS"
echo ""

/usr/local/bin/llama-cli \
    --model /app/models/Qwen3-4B-Thinking-Q4_1.gguf \
    --prompt "$PROMPT" \
    --n-predict $MAX_TOKENS \
    --n-gpu-layers 0 \
    --ctx-size 2048 \
    --temp 0.7 \
    --threads $nproc_count \
    --no-display-prompt \
    2>&1 | tee /tmp/llama_q4_1.log

echo ""
echo "Q4_1 Timing:"
grep -E "(eval time|total time|tokens per second)" /tmp/llama_q4_1.log || true

echo ""
echo "=============================================="
echo "BENCHMARK COMPLETE"
echo "=============================================="
