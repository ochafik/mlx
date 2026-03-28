FROM python:3.11-slim

# Install system dependencies for MLX CPU and llama.cpp
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libblas-dev \
    liblapack-dev \
    liblapacke-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install MLX with CPU support and mlx-lm
RUN pip install --upgrade pip && \
    pip install "mlx[cpu]" mlx-lm huggingface_hub

# Build llama.cpp from source (CPU only)
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git && \
    cd llama.cpp && \
    cmake -B build -DGGML_METAL=OFF -DGGML_CUDA=OFF && \
    cmake --build build --config Release -j$(nproc) && \
    cp build/bin/llama-cli /usr/local/bin/ && \
    cp build/bin/llama-bench /usr/local/bin/ || true

# Copy benchmark scripts
COPY benchmark_mlx.py /app/
COPY benchmark_docker.sh /app/

# Create models directory
RUN mkdir -p /app/models

ENTRYPOINT ["/bin/bash"]
