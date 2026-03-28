# MLX Compiler and Optimization Passes for GGML Port

## Overview

The MLX compiler is responsible for optimizing computation graphs through multiple passes including graph simplification, operation fusion, and code generation. Understanding this system is critical for implementing similar optimizations in GGML.

## The compile() Function

Located in `mlx/compile.h` and `mlx/compile.cpp`:

```cpp
std::function<std::vector<array>(const std::vector<array>&)> compile(
    std::function<std::vector<array>(const std::vector<array>&)> fun,
    bool shapeless = false);
```

### Key Features

- **Compilation Modes** (CompileMode enum):
  - `disabled`: No compilation
  - `no_simplify`: Skip simplification but allow fusion
  - `no_fuse`: Skip fusion but allow simplification
  - `enabled`: Full optimization pipeline

- **Shapeless Compilation**: With `shapeless=True`, compiled function can handle different input shapes without recompilation

## Optimization Passes

### A. Graph Tracing and Building

**compile_trace**: Executes function with placeholder inputs to build graph
```cpp
std::tie(entry.inputs, entry.outputs, entry.extra) =
    compile_trace(fun, inputs, shapeless);
```

**compile_dfs**: Depth-first search to build tape and parents map
```cpp
std::tie(entry.tape, parents_map) =
    compile_dfs(entry.inputs, entry.outputs, inputs);
```

### B. Simplification Pass (compile_simplify)

Performs several optimizations:

**1. Scalar Merging**: Identical scalar constants merged to reduce memory
**2. No-op Elimination**: Removes unnecessary operations like `Copy` and `StopGradient`
**3. Common Subexpression Elimination**: Merges identical operations

Key simplification strategies:
```cpp
// Merge identical scalars
if (is_scalar(arr)) {
    auto scalar = scalars.find(get_scalar_rep(arr));
    if (scalar->second.id() != arr.id()) {
        merge(scalar->second, arr, parents_map);
        continue;
    }
}

// Remove no-ops
if (!arr.has_primitive() || !is_noop(arr.primitive())) {
    new_tape.push_back(std::move(arr));
    continue;
}
```

### C. Fusion Pass (compile_fuse)

Most important optimization - combines multiple operations into single kernel.

**Fusion Criteria:**

1. **Operation Types**: Only fusable operations considered:
   ```cpp
   bool is_fusable(const Primitive& p) {
       return is_unary(p) || is_binary(p) || is_ternary(p) || is_broadcast(p);
   }
   ```

2. **Depth Limit**: Maximum fusion depth of 11 operations (`max_compile_depth = 11`)
3. **Input Limit**: Maximum of 24 inputs/outputs (`max_compile_arrays = 24`)
4. **Stream Consistency**: All operations must be on same stream
5. **Shape Compatibility**: Output shapes must be compatible

**Fusion Process:**
1. **Backward Traversal**: Goes through tape in reverse to find fusable sub-graphs
2. **Recursive Collection**: For each operation, recursively collects fusable parents
3. **Broadcast Handling**: Special handling for broadcast operations
4. **Compiled Primitive Creation**: Creates `Compiled` primitive representing fused operations

## Graph Transformations

### Graph Representation

- **Tape**: Linearized list of operations in execution order
- **Parents Map**: Tracks which operations consume each array
- **Node Naming**: Uses `NodeNamer` for readable names (A, B, C, ...)

### Array Merging and Splitting
```cpp
void merge(array& dst, array& src, ParentsMap& parents_map);
array split_one(const array& x, ParentsMap& parents_map,
                const std::unordered_set<uintptr_t>& divider);
```

### Constant Handling

Distinguishes between:
- **Inputs**: Variables provided by caller
- **Constants**: Scalar values embedded in computation
- **Scalars**: Zero-dimensional arrays

## Cost Model

MLX uses simple but effective heuristics:

**Fusion Cost Factors:**
1. **Depth**: Limits to 11 operations to prevent overly large kernels
2. **Input Count**: Limits to 24 inputs to manage kernel complexity
3. **Operation Type**: Only fuses unary, binary, ternary, and broadcast
4. **Memory Patterns**: Prefers contiguous operations

**Simplification Cost Factors:**
1. **Scalar Value**: Identical scalars merged
2. **Operation Equivalence**: Same type and inputs merged
3. **Output Usage**: Unused operations eliminated

## Operation Fusion Mechanics

1. **Finding Fusable Regions**: Traverses backward from each operation
2. **Checking Compatibility**: Ensures all operations can be fused
3. **Creating Compiled Primitives**: Generates `Compiled` primitive
4. **Updating Dependencies**: Updates parents map

Example kernel generation in CPU backend:
```cpp
build_kernel(kernel, kernel_name, inputs, outputs, tape,
             is_constant_, contiguous, ndim);

auto fn_ptr = compile(kernel_name, [&]() {
    std::ostringstream kernel;
    kernel << get_kernel_preamble() << std::endl;
    build_kernel(...);
    return kernel.str();
});
```

## Overall Compilation Pipeline

1. **Input Validation**: Check if compilation enabled and supported
2. **Cache Lookup**: Check if compiled version exists
3. **Graph Tracing**: Execute with tracers to build graph
4. **Graph Processing**: DFS traversal for tape and parents map
5. **Simplification**: Apply simplification passes (3 passes default)
6. **Fusion**: Apply fusion optimization
7. **Code Generation**: Generate backend-specific kernel code
8. **JIT Compilation**: Compile to machine code
9. **Cache Storage**: Store for future reuse

## Backend-Specific Implementations

### CPU Backend (mlx/backend/cpu/compiled.cpp)
- Uses g++ to compile generated C++ code
- Creates shared libraries (.so files)
- Supports contiguous and strided kernels
- Implements buffer donation

### CUDA Backend (mlx/backend/cuda/compiled.cpp)
- Generates CUDA kernels
- Uses vectorization for memory access
- Multi-dimensional indexing for non-contiguous data

### Metal Backend (mlx/backend/metal/compiled.cpp)
- Generates Metal shader code
- Specialized fused kernels like GEMM
- Optimized for Apple GPUs

## Adaptation Strategy for GGML

### Key Differences

| Aspect | MLX | GGML |
|--------|-----|------|
| Execution Model | Lazy with compilation | Eager execution |
| Optimization | Multi-pass compiler | Runtime optimization |
| Fusion | Automatic graph fusion | Manual kernel fusion |
| Code Generation | JIT kernel generation | Pre-built kernels |

### Recommended Approaches

1. **Simplified Compilation**
   - Implement operation fusion at runtime
   - Use pre-built fused kernels for common patterns
   - Avoid full JIT compilation for simplicity

2. **Pattern-Based Fusion**
   - Detect common patterns (e.g., activation after linear)
   - Use specialized fused kernels
   - Extend pattern library incrementally

3. **Memory Optimization**
   - Implement buffer reuse/reclamation
   - Use in-place operations where safe
   - Plan memory allocation for graphs

4. **Gradient Optimization**
   - Fuse backward operations
   - Reuse forward pass allocations
   - Implement gradient checkpointing

### Implementation Steps

1. **Basic Optimization**
   ```cpp
   // Detect fusable patterns
   bool is_fusable_pattern(ggml_tensor *t);

   // Apply fused kernel
   ggml_tensor * apply_fused(ggml_context *ctx, ggml_tensor *t);
   ```

2. **Graph Analysis**
   ```cpp
   // Analyze computation graph
   void analyze_graph(ggml_tensor *output);

   // Find optimization opportunities
   std::vector<ggml_tensor*> find_optimizations(ggml_tensor *output);
   ```

3. **Memory Planning**
   ```cpp
   // Plan memory allocation
   void plan_memory(ggml_context *ctx, ggml_tensor *output);

   // Reuse buffers
   void* reuse_buffer(ggml_context *ctx, size_t size);
   ```

### Fusion Patterns to Implement

1. **Linear + Activation**: `matmul -> add -> activation`
2. **Elementwise Chains**: Multiple elementwise operations
3. **Reduction + Reshape**: Common post-processing patterns
4. **Normalization + Activation**: LayerNorm followed by activation

### Performance Considerations

1. **Kernel Overhead**: Balance fusion benefit vs kernel launch cost
2. **Memory Usage**: Fused operations may use more intermediate memory
3. **Code Size**: Limit fusion to manageable kernel sizes
4. **Compilation Time**: Consider caching compiled kernels

The MLX compiler demonstrates practical graph optimization that balances performance improvements with compilation complexity. Its design principles can inform GGML optimization strategies while adapting to GGML's eager execution model.
