# MLX Operation Implementations for GGML Port

## Overview

MLX provides a comprehensive set of operations that form the foundation for machine learning computations. Understanding these operations is critical for porting to GGML, as each operation needs to be mapped or re-implemented.

## Operation Categories

### 1. Elementwise Operations

#### Mathematical Operations

**Basic Arithmetic:**
- `add`, `subtract`, `multiply`, `divide`, `floor_divide`, `remainder`
- `negative`, `abs`, `sign`, `reciprocal`
- `power`, `sqrt`, `rsqrt`, `square`, `exp`, `expm1`

**Trigonometric Functions:**
- `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `arctan2`
- `sinh`, `cosh`, `tanh`, `arcsinh`, `arccosh`, `arctanh`
- `degrees`, `radians`

**Logarithmic and Exponential:**
- `log`, `log2`, `log10`, `log1p`, `logaddexp`
- `logsumexp`, `sigmoid`, `erf`, `erfinv`

**Comparison and Logical:**
- `equal`, `not_equal`, `greater`, `greater_equal`, `less`, `less_equal`
- `logical_and`, `logical_or`, `logical_not`
- `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_invert`
- `left_shift`, `right_shift`

**Rounding and Special Functions:**
- `round`, `floor`, `ceil`, `isnan`, `isinf`, `isfinite`, `isposinf`, `isneginf`

#### Implementation Details
- **Dispatch**: Operations are dispatched through template-based SIMD implementations
- **Optimization**: Uses specialized kernels for different data layouts (contiguous, strided, broadcasted)
- **Type Support**: Full support for float32, float16, bfloat16, int32, uint32, bool_, complex64

### 2. Reduction Operations

#### Core Reductions
- `sum`, `mean`, `prod`, `max`, `min`, `all`, `any`
- `argmax`, `argmin` (with support for multiple axes)
- `var`, `std` (with optional delta degrees of freedom)

#### Advanced Reductions
- `logsumexp` (numerically stable)
- `topk`, `sort`, `argsort`
- `partition`, `argpartition`

#### Reduction Strategy
The system categorizes reductions into several types:
- `ContiguousAllReduce`: All axes reduced, contiguous input
- `ContiguousReduce`: Last axis reduced, contiguous input
- `ContiguousStridedReduce`: Last axis not reduced, contiguous input
- `GeneralContiguousReduce`: Non-contiguous but last axis reduced
- `GeneralStridedReduce`: Non-contiguous, last axis not reduced
- `GeneralReduce`: Fully general case

### 3. Linear Algebra Operations

#### Matrix Operations
- **Matrix Multiplication**: `matmul`, `addmm`, `block_masked_mm`, `gather_mm`, `segmented_mm`
- **Specialized**: `quantized_matmul`, `qqmm`, `gather_qmm` for quantized matrices
- **Tensor Operations**: `tensordot`, `outer`, `inner`

#### Matrix Utilities
- `transpose`, `moveaxis`, `swapaxes`
- `diagonal`, `trace`, `identity`, `eye`
- `kron` (Kronecker product)

#### Convolution Operations
- `conv_general` with full parameter support:
  - `stride`, `padding_lo`, `padding_hi`
  - `kernel_dilation`, `input_dilation`
  - `groups`, `flip`
- Specialized: `conv1d`, `conv2d`, `conv3d`
- Transposed: `conv_transpose1d`, `conv_transpose2d`, `conv_transpose3d`

### 4. Array Manipulation Operations

#### Shape Operations
- `reshape`, `flatten`, `unflatten`
- `squeeze`, `expand_dims`
- `broadcast_to`, `broadcast_arrays`

#### Indexing and Slicing
- **Basic**: `slice`, `slice_update`
- **Advanced**: Dynamic slicing with `slice` (array indices)
- **Gather Operations**: `gather`, `take`, `take_along_axis`
- **Scatter Operations**: `scatter`, `scatter_add`, `scatter_prod`, `scatter_max`, `scatter_min`
- **Masked**: `masked_scatter`

#### Array Construction
- `arange`, `linspace`, `full`, `zeros`, `ones`, `eye`, `identity`
- `meshgrid`, `concatenate`, `stack`, `repeat`, `tile`
- `pad` with multiple padding modes

#### Special Operations
- `hadamard_transform` for signal processing
- `roll` for circular shifts
- `contiguous` for memory layout optimization

## Backend Dispatch Architecture

### Device Abstraction
```cpp
struct Device {
    enum class DeviceType { cpu, gpu };
    DeviceType type;
    int index;
};
```

### Primitive System
Each operation is implemented as a `Primitive` class with:
- **eval_cpu()**: CPU implementation
- **eval_gpu()**: GPU implementation
- **vmap()**: Vectorization support
- **jvp()/vjp()**: Automatic differentiation support

### Stream-based Execution
- **Streams**: `Stream(int index, Device device)`
- **Scheduler**: Thread-per-stream execution model
- **GPU**: Command buffer-based execution (Metal/CUDA)
- **CPU**: Direct execution with memory management

## Operation Template

```cpp
// ops.h - API declaration
MLX_API array add(const array& a, const array& b, StreamOrDevice s = {});

// ops.cpp - Implementation
array add(const array& a, const array& b, StreamOrDevice s) {
    // Create primitive and return array
    return array(shape, dtype,
                 std::make_shared<Add>(to_stream(s)),
                 {a, b});
}

// primitives.h - Primitive definition
class Add : public UnaryPrimitive {
    void eval_cpu(const vector<array>& inputs, array& out) override;
    void eval_gpu(const vector<array>& inputs, array& out) override;
    // ...
};
```

## Backend Implementation

### CPU Implementation
```cpp
void Add::eval_cpu(const vector<array>& inputs, array& out) {
    // Use SIMD-optimized kernels
    // Handle different input layouts
}
```

### GPU Implementation
```cpp
void Add::eval_gpu(const vector<array>& inputs, array& out) {
    // Dispatch to Metal/CUDA kernels
    // Handle memory transfers
}
```

## Porting Considerations for GGML

### Operation Mapping

| MLX Operation | GGML Equivalent | Notes |
|---------------|-----------------|-------|
| add | ggml_add | Direct mapping |
| multiply | ggml_mul | Direct mapping |
| matmul | ggml_mul_mat | Matrix multiplication |
| transpose | ggml_transpose | Direct mapping |
| reshape | ggml_reshape | Direct mapping |
| concatenate | ggml_concat | Direct mapping |
| conv_general | ggml_conv_* | May need custom implementation |
| quantized_matmul | ggml_mul_mat_q | Quantized matrix mul |

### Key Areas to Adapt

1. **Backend Layer**: Replace Metal/CUDA with GGML's compute abstractions
2. **Scheduler**: Adapt GGML's execution model
3. **Memory**: Integrate GGML's memory management
4. **Data Types**: Map GGML data types to MLX system
5. **Kernels**: Implement MLX operations using GGML primitives

### API Mapping Strategy

1. **Keep MLX's high-level API**: Maintain ops.h interface for compatibility
2. **Replace primitive implementations**: Use GGML calls underneath
3. **Adapt device/stream handling**: Map to GGML's execution context
4. **Integrate memory management**: Combine buffer donation with GGML's allocator

### Challenges

1. **Lazy vs Eager**: MLX uses lazy evaluation; GGML uses eager execution
2. **Graph Execution**: MLX builds graphs; GGML executes immediately
3. **Type System**: MLX has richer type support (bfloat16, complex64)
4. **Quantization**: MLX has extensive quantization support

### Recommendations

1. **Start with core operations**: Implement basic arithmetic and linear algebra first
2. **Implement autodiff**: Add gradient computation for operations
3. **Add optimizations**: Implement fusion and other optimizations
4. **Expand coverage**: Add remaining operations incrementally
5. **Performance testing**: Benchmark and optimize critical paths

The operation surface in MLX is comprehensive and well-structured. The clean separation between API and implementation makes it feasible to port to GGML while maintaining compatibility with existing MLX code.
