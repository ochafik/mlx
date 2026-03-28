# MLX Automatic Differentiation System for GGML Port

## Overview

MLX implements a sophisticated automatic differentiation (autodiff) system that supports both forward-mode and reverse-mode differentiation. The system is built into the core primitive architecture and is essential for training machine learning models.

## Gradient Computation Modes

### Reverse Mode Differentiation (VJP)

MLX primarily uses **reverse mode automatic differentiation** (backpropagation) through its `vjp` (Vector-Jacobian Product) implementation. This is the standard approach for most machine learning applications where the number of outputs is small relative to inputs.

### Forward Mode Differentiation (JVP)

MLX also supports **forward mode** through `jvp` (Jacobian-vector Product) for cases where forward mode is more efficient (e.g., when the number of inputs is small relative to outputs).

## JVP (Jacobian-Vector Product) Implementation

Located in `mlx/transforms.cpp` lines 526-639:

```cpp
std::pair<std::vector<array>, std::vector<array>> jvp(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    const std::vector<array>& primals,
    const std::vector<array>& tangents)
```

### Process

1. **Mark tracers**: Input arrays are marked as "tracers"
2. **Execute function**: Builds computation graph with tracers
3. **Forward traversal**: Computes tangent values by traversing graph forward
4. **Primitive-specific computation**: Each primitive implements its own `jvp` method

### Example: Add Primitive

```cpp
std::vector<array> Add::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
    return {
        tangents.size() > 1 ? add(tangents[0], tangents[1], stream())
                            : tangents[0]
    };
}
```

## VJP (Vector-Jacobian Product) Implementation

Located in `mlx/transforms.cpp` lines 327-504:

```cpp
std::pair<std::vector<array>, std::vector<array>> vjp(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums)
```

### Process

1. **Create tracers**: Creates tracers from primal inputs
2. **Execute function**: Builds computation graph
3. **Identify dependencies**: Determines which parts need gradient computation
4. **Backward traversal**: Accumulates cotangents by traversing backward
5. **Primitive-specific computation**: Each primitive implements its own `vjp` method

### Example: Add Primitive

```cpp
std::vector<array> Add::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
    if (argnums.size() == 1) {
        return cotangents;
    } else {
        return {cotangents[0], cotangents[0]};
    }
}
```

## User-Facing Gradient Functions

### grad() Function

Returns a function that computes gradients:

```cpp
std::function<std::vector<array>(const std::vector<array>&)> grad(
    const std::function<array(const std::vector<array>&)>& fun,
    const std::vector<int>& argnums)
```

- Internally calls `value_and_grad()` and discards the function value
- `argnums` parameter specifies which arguments to compute gradients for

### value_and_grad() Function

Returns a function that computes both values and gradients:

```cpp
ValueAndGradFn value_and_grad(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    const std::vector<int>& argnums)
```

- Wraps function to apply `stop_gradient` to non-scalar outputs
- Uses `vjp` with unit cotangent (1.0f) for scalar outputs
- Handles multiple outputs by selecting first for gradient computation

## Computation Graph Tracking

### Tracer Arrays

Arrays created during autodiff operations are marked as "tracers":
- `set_tracer(true)`: Marks array as tracer
- Tracers preserve computation graph structure
- `detail::InTracing` class tracks autodiff context

### Graph Construction

When function is called with tracer inputs:
1. MLX builds directed acyclic graph (DAG)
2. Each operation creates `Primitive` object
3. Primitives know forward and backward passes
4. Graph is built topologically during execution

### Key Classes

- **`Primitive`**: Base class for all operations
  - Defines `eval_cpu`, `eval_gpu`, `jvp`, `vjp`, `vmap`
- **`UnaryPrimitive`**: Base for single-output operations
- **`Array`**: Main data structure with tracer status

## Graph Transformations

### vmap (Vectorization)

```cpp
std::function<std::vector<array>(const std::vector<array>&)> vmap(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    const std::vector<int>& in_axes = {},
    const std::vector<int>& out_axes = {})
```

**Process:**
1. **Trace**: Creates placeholder graph with axis dimensions removed
2. **Replace**: Transforms primitives to support vectorized operations
3. Each primitive implements `vmap` to handle vectorization

### custom_function and custom_vjp

Allows users to provide custom gradient implementations:
- Creates `CustomTransforms` primitive
- Stores user-defined autodiff functions
- Essential for complex custom operations with optimal gradients

### checkpoint

Implements gradient checkpointing to trade memory for computation:
- Uses `Depends` primitive to mark intermediate values
- Recomputes intermediate values during backward pass
- Reduces memory usage at cost of extra computation

## stop_gradient Implementation

Prevents gradients from flowing through specific parts of computation graph:

```cpp
array stop_gradient(const array& a, StreamOrDevice s = {}) {
    return array(
        a.shape(), a.dtype(),
        std::make_shared<StopGradient>(to_stream(s)),
        {a});
}
```

### StopGradient Primitive

- Inherits from `UnaryPrimitive`
- Passes input through during forward pass
- `vjp` implementations check for `StopGradient` and skip gradient computation

## Primitive Gradient Implementation Pattern

Each primitive must implement gradient functions:

```cpp
class MyPrimitive : public UnaryPrimitive {
    // Forward pass
    void eval_cpu(const vector<array>& inputs, array& out) override;

    // Backward pass (reverse mode)
    vector<array> vjp(
        const vector<array>& primals,
        const vector<array>& cotangents,
        const vector<int>& argnums,
        const vector<array>& outputs) override;

    // Forward mode
    vector<array> jvp(
        const vector<array>& primals,
        const vector<array>& tangents,
        const vector<int>& argnums) override;

    // Vectorization
    pair<vector<array>, vector<int>> vmap(
        const vector<array>& inputs,
        const vector<int>& axes) override;
};
```

## Adaptation Strategy for GGML

### Key Differences

| Aspect | MLX | GGML |
|--------|-----|------|
| Execution Model | Lazy graph building | Eager execution |
| Gradient Storage | Tracer arrays in graph | Explicit gradient tensors |
| Memory Management | Automatic graph cleanup | Manual gradient management |
| Graph Traversal | Topological sort | Explicit backward pass |

### Implementation Approach

1. **Gradient Accumulation**
   - Implement gradient tensor storage
   - Add gradient accumulation functions
   - Handle multiple uses of same tensor

2. **Operation Gradients**
   - Implement gradients for each operation
   - Use chain rule for composition
   - Handle special cases (in-place ops, views)

3. **Computational Graph**
   - Track operations during forward pass
   - Build graph structure for backward pass
   - Implement topological sort for gradient order

4. **Memory Management**
   - Decide what to store from forward pass
   - Implement gradient checkpointing option
   - Handle intermediate value cleanup

### Recommended Patterns

```cpp
// Gradient computation function
void ggml_compute_grad(
    struct ggml_context * ctx,
    struct ggml_tensor * tensor,
    struct ggml_tensor * grad);

// Backward pass
struct ggml_tensor * ggml_backward(
    struct ggml_context * ctx,
    struct ggml_tensor * loss);
```

### Challenges

1. **Lazy vs Eager**: MLX builds graph then evaluates; GGML evaluates immediately
2. **Gradient Storage**: Need explicit gradient tensor management
3. **Graph Traversal**: Need to track and traverse computation dependencies
4. **Memory Efficiency**: MLX has sophisticated memory planning

### Solutions

1. **Operation Tracking**: Track operations during forward pass
2. **Gradient Tensors**: Store gradients alongside tensors
3. **Topological Sort**: Implement for correct gradient order
4. **Memory Planning**: Implement checkpointing for memory efficiency

The MLX autodiff system is sophisticated but follows clean principles that can be adapted to GGML. The key is implementing the primitive-based approach where each operation knows how to compute its derivatives.
