# MLX Python Bindings and Neural Network Modules for GGML Port

## Python Bindings Architecture

### Nanobind Integration

MLX uses **nanobind** as the primary Python-C++ binding library. The binding system is organized in `python/src/`:

**Key Files:**
- `mlx.cpp`: Main entry point, initializes all Python modules
- `array.cpp`: Core array class bindings and data types
- `ops.cpp`: Mathematical operations bindings
- `convert.cpp`: Array conversion between C++ and Python
- `device.cpp`: Device management (CPU, Metal, CUDA)
- `transforms.cpp`: Array transformations
- `memory.cpp`: Memory management

### Module Structure

The build system uses `nanobind_add_module()` in `CMakeLists.txt` to compile all C++ files into a single Python extension module called `core`.

```cmake
nanobind_add_module(core PRIVATE
    mlx.cpp
    array.cpp
    ops.cpp
    convert.cpp
    # ... other source files
)
```

## API Exposed to Python

### Core Module (mlx.core)

Automatically imported as `mlx.core`, provides:

- Array operations and data types
- Mathematical functions
- Device management
- Memory allocation
- All core MLX functionality

```python
import mlx.core as mx
x = mx.array([1, 2, 3])
y = mx.add(x, x)
```

### Neural Network Module (mlx.nn)

Located in `python/mlx/nn/`:

- Layer implementations
- Loss functions
- Initialization utilities

### Optimizers Module (mlx.optimizers)

Optimization algorithms:
- SGD, Adam, AdamW, etc.

### Short Alias

```python
import mlx.core as mx  # Standard
import mx              # Short alias
```

## Array Conversion

### Python → C++ Conversion

Located in `python/src/convert.cpp`:

**Features:**
1. Uses nanobind's `ndarray_traits` for type mapping
2. Supports all major data types:
   - bool, int8/16/32/64
   - float16/32/64
   - bfloat16
   - complex64
3. Handles shape validation and type conversion
4. Creates MLX arrays with proper memory management

### C++ → Python Conversion

**Features:**
1. MLX arrays implement Python's buffer protocol
2. Support for NumPy-like indexing
3. Custom representation for pretty printing
4. Pickle serialization via `__getstate__` and `__setstate__`

```python
# NumPy interoperability
import mlx.core as mx
import numpy as np

# Convert to NumPy
x = mx.array([1, 2, 3])
np_x = np.array(x)

# Convert from NumPy
np_arr = np.array([1, 2, 3])
mx_arr = mx.array(np_arr)
```

## Neural Network Primitives (mlx.nn)

### Base Module Class

Located in `python/mlx/nn/layers/base.py`:

**Key Features:**
- **Dict-based inheritance**: All modules inherit from `dict` for parameter management
- **Parameter tracking**: Automatic discovery of trainable parameters
- **Training modes**: Support for `train()` and `eval()` methods
- **Freezing/unfreezing**: Ability to freeze specific parameters
- **Weight loading/saving**: Support for `.npz` and `.safetensors` formats

```python
class Module(dict):
    def __init__(self):
        super().__init__()
        self._modules = {}
        self._training = True

    def parameters(self):
        """Return all trainable parameters"""
        # ...

    def train(self, mode=True):
        """Set training mode"""
        self._training = mode

    def eval(self):
        """Set evaluation mode"""
        self.train(False)
```

### Layer Implementations

#### Linear Layers (`layers/linear.py`)

- `Linear`: Standard affine transformation (y = xW^T + b)
- `Bilinear`: Bilinear transformation for two inputs
- `Identity`: No-op layer for debugging
- `QuantizedLinear`: Quantized linear layers for inference

```python
class Linear(Module):
    def __init__(self, input_dims, output_dims, bias=True):
        super().__init__()
        self.weight = mx.random.uniform((input_dims, output_dims))
        if bias:
            self.bias = mx.zeros((output_dims,))

    def __call__(self, x):
        return mx.addmm(self.bias, x, self.weight)
```

#### Convolutional Layers (`layers/convolution.py`)

- `Conv1d`, `Conv2d`, `Conv3d`: Standard convolution layers
- Support for stride, padding, dilation, grouped convolutions
- Proper weight initialization using Kaiming uniform

```python
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        # ... initialization

    def __call__(self, x):
        return mx.conv_general(x, self.weight,
                              stride=self.stride,
                              padding=self.padding,
                              dilation=self.dilation,
                              groups=self.groups)
```

#### Transformer Layers (`layers/transformer.py`)

- `MultiHeadAttention`: Multi-head attention with optional masking
- `TransformerEncoderLayer`, `TransformerDecoderLayer`: Transformer building blocks
- Support for pre/post-normalization
- Dropout integration

```python
class MultiHeadAttention(Module):
    def __init__(self, d_model, num_heads, bias=True, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # ... QKV projections, output projection

    def __call__(self, x, mask=None):
        # ... attention computation
```

#### Normalization Layers (`layers/normalization.py`)

- `LayerNorm`: Layer normalization
- `BatchNorm`: Batch normalization
- `RMSNorm`: Root Mean Square normalization
- `InstanceNorm`, `GroupNorm`: Instance and group normalization

```python
class LayerNorm(Module):
    def __init__(self, dims, eps=1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.bias = mx.zeros((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.layer_norm(x, self.weight, self.bias, eps=self.eps)
```

#### Activation Functions (`layers/activations.py`)

Common activations:
- ReLU, Sigmoid, Tanh, Softmax
- Advanced: GELU, Swish, Mish, ELU, SELU

All activations are compiled with `@partial(mx.compile, shapeless=True)` for optimization.

```python
@partial(mx.compile, shapeless=True)
def relu(x):
    return mx.maximum(x, 0)

@partial(mx.compile, shapeless=True)
def gelu(x):
    return x * 0.5 * (1.0 + mx.erf(x / mx.sqrt(2.0)))
```

#### Other Key Layers

- **Pooling**: Max pooling, average pooling (1D, 2D, 3D)
- **Dropout**: Standard dropout layers
- **Embedding**: Lookup table for discrete inputs
- **Recurrent**: GRU, LSTM, RNN layers

### Loss Functions (`losses.py`)

- **Cross Entropy**: With label smoothing support
- **Binary Cross Entropy**: With logits option for numerical stability
- **Mean Squared Error**: Standard regression loss
- Support for custom weights and reduction strategies

```python
def cross_entropy(logits, targets, weight=None, reduction='mean',
                 label_smoothing=0.0):
    # ... cross-entropy computation
```

### Initialization (`init.py`)

Comprehensive weight initialization schemes:

- **Glorot (Xavier)**: Normal and uniform variants
- **He (Kaiming)**: Normal and uniform variants for ReLU networks
- **Orthogonal**: For recurrent networks
- **Sparse**: For sparse connectivity
- **Uniform/Normal**: Basic distributions

```python
def glorot_uniform(inputs):
    scale = mx.sqrt(6.0 / (inputs.shape[-2] + inputs.shape[-1]))
    return mx.random.uniform(-scale, scale, inputs.shape)

def kaiming_normal(inputs):
    std = mx.sqrt(2.0 / inputs.shape[-2])
    return mx.random.normal(0.0, std, inputs.shape)
```

### Model Utilities

- **Sequential**: Container for layer stacks
- **Quantization**: Support for quantized linear layers
- **Positional Encoding**: ALiBi, RoPE, sinusoidal for transformers
- **Upsample**: Interpolation-based upsampling

## Porting Considerations for GGML

### Python API Compatibility

**Goal**: Maintain MLX's Python API while using GGML backend

**Approach**:
1. Keep MLX's Python interface unchanged
2. Implement GGML backend underneath
3. Expose GGML-specific features through extensions

### Conversion Between MLX and GGML

```python
# Conversion layer
def mlx_to_ggml(mlx_array):
    """Convert MLX array to GGML tensor"""
    # ... implementation

def ggml_to_mlx(ggml_tensor):
    """Convert GGML tensor to MLX array"""
    # ... implementation
```

### Module Adaptation

```python
class GGMLModule(mx.Module):
    """Base class for GGML-backed modules"""

    def __init__(self):
        super().__init__()
        self._ggml_params = {}

    def to_ggml(self):
        """Convert parameters to GGML format"""
        # ... implementation
```

### Key Challenges

1. **Lazy vs Eager**: MLX uses lazy evaluation; GGML uses eager
2. **Parameter Storage**: Different parameter management approaches
3. **Gradient Computation**: Different autodiff implementations
4. **Model Serialization**: Different serialization formats

### Recommendations

1. **Maintain API Compatibility**: Keep MLX's Python interface
2. **Transparent Backend**: Use GGML underneath without changing API
3. **Gradual Migration**: Start with core operations, expand coverage
4. **Performance Testing**: Benchmark against pure MLX implementation

### Implementation Steps

1. **Create GGML Backend Module**
   ```python
   # mlx_ggml/backend.py
   class GGMLBackend:
       def __init__(self):
           self.ctx = ggml_init()

       def execute(self, graph):
           # ... execution logic
   ```

2. **Implement Operation Mapping**
   ```python
   # mlx_ggml/ops.py
   def add(a, b):
       ggml_a = mlx_to_ggml(a)
       ggml_b = mlx_to_ggml(b)
       ggml_result = ggml_add(ggml_a, ggml_b)
       return ggml_to_mlx(ggml_result)
   ```

3. **Create Compatible Module Base**
   ```python
   # mlx_ggml/nn.py
   class GGMLModule(mx.Module):
       def __call__(self, x):
           # Convert input to GGML
           # Execute computation
           # Convert result back to MLX
   ```

4. **Handle Autodiff**
   ```python
   # mlx_ggml/autodiff.py
   class GGMLGradFunction:
       def __call__(self, *inputs):
           # Forward pass
           # Backward pass
           # Return gradients
   ```

The MLX Python bindings and neural network modules provide a clean, NumPy-like API while maintaining efficient C++ performance. Porting to GGML requires maintaining this API surface while implementing the backend with GGML's primitives.
