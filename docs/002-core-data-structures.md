# MLX Core Data Structures for GGML Port

## Array Class (mlx/array.h)

The `array` class is the fundamental tensor data structure in MLX. It represents a tensor as a node in a computation graph rather than just a data container.

### ArrayDesc Structure

```cpp
struct ArrayDesc {
    Shape shape;                    // Array dimensions (int32_t)
    Strides strides;                // Memory strides (int64_t)
    size_t size;                   // Total elements
    Dtype dtype;                   // Data type
    std::shared_ptr<Primitive> primitive;  // Operation definition
    Status status;                 // Evaluation status
    Event event;                   // Synchronization event
    bool is_tracer{false};         // Graph transformation flag
    std::shared_ptr<Data> data;   // Memory buffer reference
    int64_t offset{0};            // Offset in buffer
    size_t data_size;             // Size of accessible data
    Flags flags;                   // Contiguity flags
    std::vector<array> inputs;     // Input arrays for computation
    std::vector<array> siblings;  // Multi-output primitives
    uint32_t position{0};          // Position in output list
};
```

### Critical Design Patterns

#### 1. Shared Ownership Pattern
Uses `shared_ptr<ArrayDesc>` for efficient copying and graph management:
- Arrays can be copied cheaply (reference counting)
- Graph nodes are automatically managed
- Supports multi-output primitives via sibling system

#### 2. Lazy Evaluation
Arrays remain unevaluated until explicitly requested:
- Operations create graph nodes
- Data is materialized on-demand via `eval()`
- Enables optimization opportunities

#### 3. Memory Layout Flags
```cpp
struct Flags {
    bool contiguous : 1;      // No gaps in underlying data
    bool row_contiguous : 1;  // C-style row-major contiguous
    bool col_contiguous : 1;  // Fortran-style column-major contiguous
};
```

These flags help optimize operations but add overhead.

## Data Structure (Memory Buffer)

```cpp
struct Data {
    allocator::Buffer buffer;    // Memory buffer
    Deleter d;                  // Custom deleter
    Data(allocator::Buffer buffer, Deleter d = allocator::free)
        : buffer(buffer), d(d) {}
};
```

### Buffer Abstraction
```cpp
class Buffer {
    void* ptr_;  // Raw pointer with metadata
    void* raw_ptr();  // Get raw pointer
};
```

## Memory Management

### Allocator Interface (mlx/allocator.h)

```cpp
class Allocator {
    virtual Buffer malloc(size_t size) = 0;
    virtual void free(Buffer buffer) = 0;
    virtual size_t size(Buffer buffer) const = 0;
    virtual Buffer make_buffer(void* ptr, size_t size);
};
```

### Memory Management Features

1. **Cache System**: MLX implements memory caching for performance
2. **Memory Limits**: Configurable memory limits for preventing OOM
3. **Peak Tracking**: Monitors peak memory usage
4. **Donation Mechanism**: `is_donatable()` flag indicates when buffers can be reused

## Primitive System (mlx/primitives.h)

### Base Primitive Class

```cpp
class Primitive {
    Stream stream_;  // Execution stream/device

    virtual void eval_cpu(const std::vector<array>& inputs,
                          std::vector<array>& outputs) = 0;
    virtual void eval_gpu(const std::vector<array>& inputs,
                          std::vector<array>& outputs) = 0;
    virtual std::vector<array> jvp(const std::vector<array>& primals,
                                   const std::vector<array>& tangents,
                                   const std::vector<int>& argnums);
    virtual std::vector<array> vjp(const std::vector<array>& primals,
                                   const std::vector<array>& cotangents,
                                   const std::vector<int>& argnums,
                                   const std::vector<array>& outputs);
    virtual std::pair<std::vector<array>, std::vector<int>> vmap(
        const std::vector<array>& inputs,
        const std::vector<int>& axes);
};
```

### Unary Primitive Specialization

```cpp
class UnaryPrimitive : public Primitive {
    virtual void eval_cpu(const std::vector<array>& inputs, array& output) = 0;
    virtual void eval_gpu(const std::vector<array>& inputs, array& output) = 0;
};
```

### Key Patterns

1. **Macros for Boilerplate**: `DEFINE_NAME`, `DEFINE_GRADS` reduce template code
2. **Stream Awareness**: Each primitive knows its execution device/stream
3. **Automatic Differentiation**: Built-in JVP/VJP support for autograd
4. **Vectorization**: Built-in vmap support for operation vectorization

## Stream and Device Management

### Stream Structure
```cpp
struct Stream {
    int index;           // Stream index
    Device device;        // Target device (CPU/GPU)
    explicit Stream(int index, Device device)
        : index(index), device(device) {}
};
```

### Device Management
```cpp
struct Device {
    enum class DeviceType { cpu, gpu };
    DeviceType type;
    int index;  // Device index (for multi-GPU)
};
```

### Scheduler Implementation
```cpp
class Scheduler {
    std::vector<StreamThread*> threads_;     // CPU threads
    std::vector<Stream> streams_;           // All streams
    std::unordered_map<Device::DeviceType, Stream> default_streams_;
};
```

## Computation Graph Execution

### Evaluation System
```cpp
void eval(std::vector<array> outputs);
void async_eval(std::vector<array> outputs);
```

### Graph Execution Flow
1. **Graph Construction**: Operations create primitive nodes with inputs
2. **Lazy Evaluation**: Arrays remain unscheduled until `.eval()` is called
3. **Topological Sort**: System implicitly schedules based on dependencies
4. **Parallel Execution**: Independent operations run concurrently via scheduler

## Adaptation Strategy for GGML

### 1. Memory System Integration
- Replace MLX's allocator with GGML's pool-based system
- Keep buffer abstraction for type safety
- Adapt contiguity flags for GGML's layout requirements

### 2. Operation Abstraction Layer
- Create GGML primitive wrapper
- Implement MLX-style autograd on top of GGML operations
- Preserve stream abstraction for potential future GPU support

### 3. Array Interface Compatibility
- Keep MLX's array API for compatibility
- Implement GGML backend underneath
- Gradually migrate operations to GGML implementations

### 4. Performance Considerations
- MLX's compile-time optimization could complement GGML's runtime optimization
- The stream scheduler could enhance GGML's threading
- Memory pooling from both systems could be combined intelligently

### 5. Gradual Migration Path
1. Start with GGML as CPU backend for MLX operations
2. Implement MLX primitives using GGML functions
3. Migrate specific high-performance operations first
4. Eventually replace MLX backend entirely with GGML

## Key Data Structures Summary

| Structure | Purpose | GGML Equivalent |
|-----------|---------|-----------------|
| `array` | Tensor node in computation graph | `ggml_tensor` |
| `ArrayDesc` | Array metadata and data | Internal tensor struct |
| `Data` | Memory buffer reference | Memory buffer pointers |
| `Primitive` | Operation abstraction | GGML operation functions |
| `Stream` | Execution context | GGML context/work queue |
| `Device` | Device abstraction | GGML device type (CPU/GPU) |
| `Allocator` | Memory management | GGML allocator functions |

The MLX architecture provides clean abstractions that can be adapted to GGML's more immediate execution model while maintaining compatibility with the existing MLX API surface.
