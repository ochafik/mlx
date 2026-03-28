#!/bin/bash
# Benchmark llama.cpp throughput

MODEL_PATH="${1:-models/Qwen3-4B-Thinking-Q4_0.gguf}"
PROMPT="${2:-Explain the theory of relativity in simple terms.}"
MAX_TOKENS="${3:-256}"
USE_GPU="${4:-1}"  # 1 for GPU (Metal), 0 for CPU

if [ "$USE_GPU" = "1" ]; then
    NGL=99  # Offload all layers to GPU
    MODE="GPU/Metal"
else
    NGL=0   # CPU only
    MODE="CPU"
fi

echo "=============================================="
echo "llama.cpp Benchmark ($MODE)"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Max tokens: $MAX_TOKENS"
echo "GPU layers: $NGL"
echo ""

# Run llama-cli with timing
llama-cli \
    --model "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --n-predict "$MAX_TOKENS" \
    --n-gpu-layers $NGL \
    --ctx-size 2048 \
    --temp 0.7 \
    --no-display-prompt \
    2>&1 | tee /tmp/llama_bench.log

echo ""
echo "=============================================="
# Extract timing info from stderr
grep -E "(eval time|total time|tokens per second)" /tmp/llama_bench.log || true
echo "=============================================="
