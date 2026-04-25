# Framework Integration

This document outlines integration patterns for exposing PTO-based kernels to framework runtimes. The exact registration APIs and backend names depend on the framework version, `torch_npu` integration model, and product release, so treat the snippets below as implementation patterns rather than drop-in code.

---

## 1. Integration Overview

### 1.1 Integration Architecture

```
Application layer
    ↓
Framework runtime and operator registration
    ↓
PTO-based kernel implementation
    ↓
Target Ascend AI Core execution
```

### 1.2 Integration Methods

| Method | Pros | Cons | Use Case |
|--------|------|------|----------|
| **Python Extension** | Fast development | Python/C++ boundary overhead | Prototyping |
| **C++ Extension** | High performance | More complex build and registration | Production |
| **Framework plugin / custom backend path** | Closer to deployment path | Higher maintenance cost | Stable product integration |

---

## 2. PyTorch Integration

### 2.1 Integration via torch_npu

#### Step 1: Define Operator Schema

```cpp
// my_ops.cpp
#include <torch/extension.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>

// Define operator schema
TORCH_LIBRARY_FRAGMENT(npu, m) {
  // Basic operator
  m.def("my_add(Tensor x, Tensor y) -> Tensor");
  
  // With scalar parameter
  m.def("my_mul(Tensor x, Scalar alpha) -> Tensor");
  
  // Multiple outputs
  m.def("my_split(Tensor x, int dim) -> (Tensor, Tensor)");
  
  // Inplace operator
  m.def("my_relu_(Tensor(a!) self) -> Tensor(a!)");
}
```

#### Step 2: Implement Operator

```cpp
#include <pto/pto-inst.hpp>

// PTO Kernel implementation
__global__ __aicore__ void MyAddKernel(
    __gm__ float* out,
    __gm__ const float* x,
    __gm__ const float* y,
    uint32_t length) {
  
  int block_idx = get_block_idx();
  int block_num = get_block_num();
  
  int elements_per_block = (length + block_num - 1) / block_num;
  int start = block_idx * elements_per_block;
  int end = min(start + elements_per_block, length);
  
  using TileT = Tile<TileType::Vec, float, 16, 256>;
  
  for (int i = start; i < end; i += 16 * 256) {
    TileT tile_x, tile_y, tile_out;
    
    TLOAD(tile_x, GlobalTensor(x + i));
    TLOAD(tile_y, GlobalTensor(y + i));
    TADD(tile_out, tile_x, tile_y);
    TSTORE(GlobalTensor(out + i), tile_out);
  }
}

// PyTorch operator implementation
at::Tensor my_add_impl(const at::Tensor& x, const at::Tensor& y) {
  // Check inputs
  TORCH_CHECK(x.device() == y.device(), "Inputs must be on same device");
  TORCH_CHECK(x.sizes() == y.sizes(), "Inputs must have same shape");
  
  // Allocate output
  at::Tensor out = at::empty_like(x);
  
  // Get data pointers
  float* out_ptr = out.data_ptr<float>();
  const float* x_ptr = x.data_ptr<float>();
  const float* y_ptr = y.data_ptr<float>();
  uint32_t length = x.numel();
  
  // Launch kernel
  int block_num = 24;  // A3 core count
  EXEC_KERNEL_CMD(MyAddKernel, block_num, out_ptr, x_ptr, y_ptr, length);
  
  return out;
}
```

#### Step 3: Register Implementation

```cpp
// Register to NPU backend
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
  m.impl("my_add", TORCH_FN(my_add_impl));
}
```

#### Step 4: Build Python Extension

**setup.py**:
```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='my_pto_ops',
    ext_modules=[
        CppExtension(
            name='my_pto_ops',
            sources=['my_ops.cpp'],
            include_dirs=[
                '/path/to/pto-isa/include',
                '/path/to/torch_npu/include',
            ],
            library_dirs=['/path/to/pto-isa/lib'],
            libraries=['pto'],
            extra_compile_args=['-std=c++20', '-O3'],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

**Build**:
```bash
python setup.py install
```

#### Step 5: Python Usage

```python
import torch
import torch_npu
import my_pto_ops

# Create inputs
x = torch.randn(1024, 1024).npu()
y = torch.randn(1024, 1024).npu()

# Call custom operator
z = torch.ops.npu.my_add(x, y)

# Verify result
expected = x + y
assert torch.allclose(z, expected, rtol=1e-5)

print("✓ Custom op works correctly!")
```

---

## 3. TensorFlow Integration

### 3.1 Custom Op

#### Step 1: Define Op

```cpp
// my_ops.cc
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

REGISTER_OP("MyAdd")
    .Input("x: float")
    .Input("y: float")
    .Output("z: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
Custom add operator

Args:
  x: First input tensor
  y: Second input tensor

Returns:
  z: x + y
)doc");
```

#### Step 2: Implement Kernel

```cpp
#include "tensorflow/core/framework/op_kernel.h"
#include <pto/pto-inst.hpp>

class MyAddOp : public tensorflow::OpKernel {
 public:
  explicit MyAddOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    // Get inputs
    const tensorflow::Tensor& x = context->input(0);
    const tensorflow::Tensor& y = context->input(1);
    
    // Check shapes
    OP_REQUIRES(context, x.shape() == y.shape(),
                tensorflow::errors::InvalidArgument("Inputs must have same shape"));
    
    // Allocate output
    tensorflow::Tensor* z = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, x.shape(), &z));
    
    // Call PTO kernel
    const float* x_ptr = x.flat<float>().data();
    const float* y_ptr = y.flat<float>().data();
    float* z_ptr = z->flat<float>().data();
    uint32_t length = x.NumElements();
    
    EXEC_KERNEL_CMD(MyAddKernel, 24, z_ptr, x_ptr, y_ptr, length);
  }
};

// Register kernel
REGISTER_KERNEL_BUILDER(
    Name("MyAdd").Device(tensorflow::DEVICE_NPU),
    MyAddOp);
```

#### Step 3: Build

```bash
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

g++ -std=c++17 -shared my_ops.cc -o my_ops.so \
    ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} \
    -I/path/to/pto-isa/include \
    -L/path/to/pto-isa/lib -lpto \
    -fPIC -O3
```

#### Step 4: Python Usage

```python
import tensorflow as tf

# Load custom op
my_ops = tf.load_op_library('./my_ops.so')

# Use
x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
y = tf.constant([[5.0, 6.0], [7.0, 8.0]])
z = my_ops.my_add(x, y)

print(z.numpy())
# [[6. 8.]
#  [10. 12.]]
```

---

## 4. ONNX Runtime Integration

### 4.1 Custom Execution Provider

#### Step 1: Define Kernel

```cpp
// my_onnx_ops.cc
#include "onnxruntime/core/framework/op_kernel.h"

class MyAddKernel : public onnxruntime::OpKernel {
 public:
  MyAddKernel(const onnxruntime::OpKernelInfo& info) : OpKernel(info) {}

  onnxruntime::Status Compute(onnxruntime::OpKernelContext* context) const override {
    // Get inputs
    const onnxruntime::Tensor* X = context->Input<onnxruntime::Tensor>(0);
    const onnxruntime::Tensor* Y = context->Input<onnxruntime::Tensor>(1);
    
    // Allocate output
    onnxruntime::Tensor* Z = context->Output(0, X->Shape());
    
    // Call PTO kernel
    const float* x_data = X->Data<float>();
    const float* y_data = Y->Data<float>();
    float* z_data = Z->MutableData<float>();
    size_t length = X->Shape().Size();
    
    EXEC_KERNEL_CMD(MyAddKernel, 24, z_data, x_data, y_data, length);
    
    return onnxruntime::Status::OK();
  }
};
```

#### Step 2: Register Kernel

```cpp
ONNX_OPERATOR_KERNEL_EX(
    Add,
    kOnnxDomain,
    7,  // opset version
    kNpuExecutionProvider,
    MyAddKernel);
```

#### Step 3: Python Usage

```python
import onnxruntime as ort

# Register custom EP
session_options = ort.SessionOptions()
session_options.register_custom_ops_library('my_onnx_ops.so')

# Create session
session = ort.InferenceSession(
    'model.onnx',
    session_options,
    providers=['NpuExecutionProvider', 'CPUExecutionProvider']
)

# Inference
outputs = session.run(None, {'input': input_data})
```

---

## 5. Performance Optimization

### 5.1 Operator Fusion

```python
# PyTorch example: Fuse Add + ReLU
@torch.jit.script
def fused_add_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.relu(x + y)

# Use custom fused operator
torch.ops.npu.fused_add_relu(x, y)
```

### 5.2 Memory Optimization

```cpp
// Inplace operator
at::Tensor& my_add_inplace(at::Tensor& x, const at::Tensor& y) {
  // Modify x directly, avoid allocating new memory
  float* x_ptr = x.data_ptr<float>();
  const float* y_ptr = y.data_ptr<float>();
  uint32_t length = x.numel();
  
  EXEC_KERNEL_CMD(MyAddInplaceKernel, 24, x_ptr, y_ptr, length);
  
  return x;
}
```

### 5.3 Asynchronous Execution

```cpp
// Use CUDA Stream (or NPU Stream)
at::Tensor my_add_async(const at::Tensor& x, const at::Tensor& y) {
  at::Tensor out = at::empty_like(x);
  
  // Get current stream
  auto stream = at::cuda::getCurrentCUDAStream();
  
  // Launch kernel asynchronously
  EXEC_KERNEL_ASYNC(MyAddKernel, 24, stream, 
                    out.data_ptr<float>(),
                    x.data_ptr<float>(),
                    y.data_ptr<float>(),
                    x.numel());
  
  return out;
}
```

---

## References

- [PyTorch Custom Operators](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [TensorFlow Custom Ops](https://www.tensorflow.org/guide/create_op)
- [ONNX Runtime Custom Ops](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html)
- [Add Operator Example](../../demos/baseline/add/README.md)
- [Debugging Guide](debug.md)
- [Performance Optimization](opt.md)

