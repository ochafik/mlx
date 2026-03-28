# MLX Project Overview for GGML Port

## Project Summary

MLX is an array framework for machine learning on Apple silicon (and now with CUDA support for NVIDIA GPUs). It provides a NumPy-like API with PyTorch-inspired features, designed specifically for Apple's hardware ecosystem while maintaining extensibility to other platforms.

**Version:** 0.30.4
**Repository:** /Users/ochafik/github/mlx
**License:** MIT

## Directory Structure

```
mlx/
├── mlx/                    # Core C++ library
│   ├── backend/           # Backend implementations (CPU, Metal, CUDA, GPU)
│   ├── distributed/       # Distributed computing (NCCL, Ring, JACC, MPI)
│   ├── io/               # I/O operations (GGUF, safetensors)
│   ├── types/            # Data type definitions
│   └── 3rdparty/         # Third-party dependencies (pocketfft)
├── python/               # Python bindings
│   ├── mlx/              # Python package
│   │   ├── nn/           # Neural network modules
│   │   └── optimizers/   # Optimizers
│   └── src/              # C++ source for Python bindings
├── tests/                # C++ test suite
├── examples/             # Example applications
├── benchmarks/          # Performance benchmarks
├── docs/                # Documentation
├── cmake/               # CMake configuration
└── build/               # Build artifacts
```

## Key Architectural Principles

### 1. Unified Memory Model
- Arrays live in shared memory accessible by any device (CPU/GPU)
- No explicit data transfers between devices
- Simplifies programming model compared to frameworks requiring explicit device transfers

### 2. Lazy Computation
- Operations are executed lazily, with arrays materialized only when needed
- Via `eval()` or `async_eval()` functions
- Enables optimization opportunities through graph analysis

### 3. Dynamic Computation Graphs
- Graphs are built dynamically during execution
- Supports shape changes without recompilation
- No need for static graph definition

### 4. Multi-Device Support
- Seamless operation across CPU and GPU devices
- Device abstraction allows easy addition of new backends
- Stream-based execution model

## Core Components

### Array (mlx/array.h)
Primary data structure representing tensors as nodes in a computation graph:
- Manages shape, strides, data types, and memory buffers
- Supports automatic differentiation through reference counting
- Uses shared_ptr for efficient copying and graph management

### Primitives (mlx/primitives.h)
Abstract base class for all operations:
- Implements CPU/GPU evaluation methods
- Supports automatic differentiation (JVP, VJP)
- Vectorization capabilities (vmap)

### Backend Architecture
- **CPU Backend**: SIMD optimizations (NEON, SSE, AVX), BLAS integration
- **Metal Backend**: Apple GPU with JIT kernel compilation
- **CUDA Backend**: NVIDIA GPU with cuBLAS/cuDNN integration
- **GPU Abstraction**: Generic GPU interface for multiple backends

### Transforms (mlx/transforms.h)
Graph transformation capabilities:
- `grad()`, `value_and_grad()`: Automatic differentiation
- `vmap()`: Vectorization
- `compile()`: Function compilation and optimization
- `checkpoint()`: Gradient checkpointing

## Build System

### CMake Configuration
- Uses CMake 3.25+ with C++20 standard
- Optional build components (tests, examples, benchmarks)
- Backend selection flags (Metal, CPU, CUDA)

### Python Package
- Uses nanobind for Python-C++ bindings
- Two-stage build process for wheels
- Support for multiple platforms (macOS, Linux)

## Dependencies

### Core Dependencies
- **nanobind**: Python C++ binding library
- **BLAS**: Linear algebra (OpenBLAS, Accelerate)
- **FFT**: FFT operations (pocketfft)

### Optional Dependencies
- **NCCL**: Distributed computing for multi-GPU
- **MPI**: High-performance computing environments

## Porting Considerations for GGML

### Strengths for Porting
1. **Clean Backend Separation**: CPU/GPU backends are well-separated
2. **Primitive System**: Easy to replace backend implementations
3. **Memory Management**: Buffer donation/reuse system
4. **Type System**: Comprehensive dtype support

### Key Challenges
1. **Lazy vs Eager Execution**: MLX uses lazy evaluation; GGML uses eager
2. **Graph vs Immediate**: MLX builds computation graphs; GGML executes immediately
3. **Memory Model**: MLX has unified memory; GGML has explicit memory management
4. **Device Abstraction**: MLX has multi-device support; GGML is CPU-focused

### Recommended Approach
1. Start with GGML as CPU backend for MLX operations
2. Implement MLX primitives using GGML functions
3. Migrate high-performance operations first
4. Gradually replace MLX backend entirely with GGML

## API Surface

### Core Operations (mlx/ops.h)
- Elementwise: add, subtract, multiply, sin, cos, exp, etc.
- Reductions: sum, mean, max, argmax, etc.
- Linear algebra: matmul, dot, transpose, etc.
- Array manipulation: reshape, concat, split, etc.

### Neural Network (python/mlx/nn/)
- Layers: Linear, Convolution, Transformer, etc.
- Activations: ReLU, GELU, Swish, etc.
- Loss functions: CrossEntropy, MSE, etc.
- Normalization: LayerNorm, BatchNorm, etc.

### Transforms (mlx/transforms.h)
- Automatic differentiation
- Vectorization
- Compilation and optimization
- Custom gradients

## Performance Features

### Compiler Optimizations
- Operation fusion
- Common subexpression elimination
- Scalar merging
- No-op elimination

### Backend Optimizations
- SIMD vectorization
- Kernel caching
- Buffer pooling
- Memory donation

## Next Steps for Porting

1. Understand GGML's tensor and operation interfaces
2. Map MLX primitives to GGML operations
3. Implement memory management integration
4. Adapt Python bindings for GGML backend
5. Port autodiff system
6. Port compiler optimization passes
7. Performance testing and optimization
