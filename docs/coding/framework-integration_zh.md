# 算子集成到推理框架

本文档详细介绍如何将 PTO 算子集成到主流推理框架（PyTorch、TensorFlow、ONNX Runtime 等），实现端到端的模型部署。

## 目录

- [1. 集成概述](#1-集成概述)
- [2. PyTorch 集成](#2-pytorch-集成)
- [3. TensorFlow 集成](#3-tensorflow-集成)
- [4. ONNX Runtime 集成](#4-onnx-runtime-集成)
- [5. 推理框架集成](#5-推理框架集成)
- [6. 性能优化](#6-性能优化)
- [7. 调试与测试](#7-调试与测试)
- [8. 最佳实践](#8-最佳实践)

---

## 1. 集成概述

### 1.1 集成架构

```
┌─────────────────────────────────────────────────────────────┐
│                    应用层 (Python/C++)                       │
│                  模型定义、训练、推理                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              框架层 (PyTorch/TensorFlow/ONNX)                │
│              算子注册、图优化、内存管理                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  PTO 算子层 (C++/CUDA)                       │
│                  自定义算子实现、内核启动                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  硬件层 (NPU/GPU/CPU)                       │
│                  指令执行、数据传输                         │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 集成方式对比

| 集成方式 | 优点 | 缺点 | 适用场景 |
|---------|------|------|---------|
| **Python 扩展** | 开发快速、易调试 | 性能开销较大 | 原型开发、快速验证 |
| **C++ 扩展** | 性能好、类型安全 | 开发复杂、编译慢 | 生产环境、性能关键 |
| **JIT 编译** | 灵活、动态优化 | 首次运行慢 | 动态图、研究实验 |
| **AOT 编译** | 启动快、可优化 | 灵活性差 | 静态图、部署环境 |

### 1.3 集成流程

```
1. 定义算子接口
   ├─ 输入/输出 Tensor 规格
   ├─ 参数类型和默认值
   └─ 算子属性（inplace、deterministic）

2. 实现算子逻辑
   ├─ 前向计算
   ├─ 反向传播（训练）
   └─ 形状推导

3. 注册算子
   ├─ 框架算子注册
   ├─ 后端绑定
   └─ 类型推导

4. 测试验证
   ├─ 单元测试
   ├─ 数值正确性
   └─ 性能基准测试

5. 文档和示例
   ├─ API 文档
   ├─ 使用示例
   └─ 性能报告
```

---

## 2. PyTorch 集成

### 2.1 通过 torch_npu 集成

#### 步骤1：定义算子 Schema

```cpp
// my_ops.cpp
#include <torch/extension.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>

// 定义算子 schema
TORCH_LIBRARY_FRAGMENT(npu, m) {
  // 基本算子
  m.def("my_add(Tensor x, Tensor y) -> Tensor");

  // 带标量参数
  m.def("my_mul(Tensor x, Scalar alpha) -> Tensor");

  // 多输出
  m.def("my_split(Tensor x, int dim) -> (Tensor, Tensor)");

  // inplace 算子
  m.def("my_relu_(Tensor(a!) self) -> Tensor(a!)");

  // 可选参数
  m.def("my_conv(Tensor input, Tensor weight, Tensor? bias=None, "
        "int stride=1, int padding=0) -> Tensor");
}
```

#### 步骤2：实现算子

**简单算子实现**：
```cpp
#include <pto/pto-inst.hpp>

// PTO Kernel 实现
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
    int size = min(16 * 256, end - i);

    TileT tile_x, tile_y, tile_out;

    TLOAD(tile_x, GlobalTensor(x + i));
    TLOAD(tile_y, GlobalTensor(y + i));
    TADD(tile_out, tile_x, tile_y);
    TSTORE(GlobalTensor(out + i), tile_out);
  }
}

// PyTorch 算子实现
at::Tensor my_add_impl(const at::Tensor& x, const at::Tensor& y) {
  // 检查输入
  TORCH_CHECK(x.device() == y.device(), "Inputs must be on same device");
  TORCH_CHECK(x.sizes() == y.sizes(), "Inputs must have same shape");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 supported");

  // 分配输出
  at::Tensor out = at::empty_like(x);

  // 获取数据指针
  float* out_ptr = out.data_ptr<float>();
  const float* x_ptr = x.data_ptr<float>();
  const float* y_ptr = y.data_ptr<float>();
  uint32_t length = x.numel();

  // 启动 kernel
  int block_num = 24;  // A3 核心数
  EXEC_KERNEL_CMD(MyAddKernel, block_num, out_ptr, x_ptr, y_ptr, length);

  return out;
}
```

**复杂算子实现（带反向传播）**：
```cpp
// 前向
class MyConvFunction : public torch::autograd::Function<MyConvFunction> {
 public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& input,
      const at::Tensor& weight,
      const at::Tensor& bias,
      int stride,
      int padding) {

    // 保存用于反向传播的张量
    ctx->save_for_backward({input, weight, bias});
    ctx->saved_data["stride"] = stride;
    ctx->saved_data["padding"] = padding;

    // 调用 PTO kernel
    at::Tensor output = run_conv_forward(input, weight, bias, stride, padding);

    return output;
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      std::vector<at::Tensor> grad_outputs) {

    // 恢复保存的张量
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto weight = saved[1];
    auto bias = saved[2];

    int stride = ctx->saved_data["stride"].toInt();
    int padding = ctx->saved_data["padding"].toInt();

    auto grad_output = grad_outputs[0];

    // 计算梯度
    at::Tensor grad_input = run_conv_backward_input(
        grad_output, weight, stride, padding);
    at::Tensor grad_weight = run_conv_backward_weight(
        grad_output, input, stride, padding);
    at::Tensor grad_bias = run_conv_backward_bias(grad_output);

    return {grad_input, grad_weight, grad_bias,
            at::Tensor(), at::Tensor()};  // stride, padding 无梯度
  }
};

// 包装函数
at::Tensor my_conv(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    int stride,
    int padding) {
  return MyConvFunction::apply(input, weight, bias, stride, padding);
}
```

#### 步骤3：注册实现

```cpp
// 注册到 NPU 后端
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
  m.impl("my_add", TORCH_FN(my_add_impl));
  m.impl("my_mul", TORCH_FN(my_mul_impl));
  m.impl("my_conv", TORCH_FN(my_conv));
}

// 注册 autograd
TORCH_LIBRARY_IMPL(npu, Autograd, m) {
  m.impl("my_conv", TORCH_FN(my_conv));
}
```

#### 步骤4：编译为 Python 扩展

**setup.py**：
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
            library_dirs=[
                '/path/to/pto-isa/lib',
            ],
            libraries=['pto'],
            extra_compile_args=['-std=c++20', '-O3'],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

**编译**：
```bash
python setup.py install
```

#### 步骤5：Python 使用

```python
import torch
import torch_npu
import my_pto_ops

# 创建输入
x = torch.randn(1024, 1024).npu()
y = torch.randn(1024, 1024).npu()

# 调用自定义算子
z = torch.ops.npu.my_add(x, y)

# 验证结果
expected = x + y
assert torch.allclose(z, expected, rtol=1e-5)

print("✓ Custom op works correctly!")
```

### 2.2 通过 torch.library 集成（PyTorch 2.0+）

**更简洁的注册方式**：
```python
import torch
from torch.library import custom_op

@custom_op("mylib::my_add", mutates_args=())
def my_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """自定义加法算子"""
    return torch.ops.mylib.my_add_impl(x, y)

@my_add.register_fake
def _(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """形状推导"""
    assert x.shape == y.shape
    return torch.empty_like(x)

# 使用
x = torch.randn(10, 10)
y = torch.randn(10, 10)
z = torch.ops.mylib.my_add(x, y)
```

### 2.3 完整示例：Add 算子

详细教程参考：[demos/baseline/add/README_zh.md](../../demos/baseline/add/README_zh.md)

---

## 3. TensorFlow 集成

### 3.1 自定义 Op

#### 步骤1：定义 Op

```cpp
// my_ops.cc
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

REGISTER_OP("MyAdd")
    .Input("x: float")
    .Input("y: float")
    .Output("z: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      // 形状推导
      c->set_output(0, c->input(0));
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
自定义加法算子

Args:
  x: 第一个输入张量
  y: 第二个输入张量

Returns:
  z: x + y
)doc");
```

#### 步骤2：实现 Kernel

```cpp
#include "tensorflow/core/framework/op_kernel.h"
#include <pto/pto-inst.hpp>

class MyAddOp : public tensorflow::OpKernel {
 public:
  explicit MyAddOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    // 获取输入
    const tensorflow::Tensor& x = context->input(0);
    const tensorflow::Tensor& y = context->input(1);

    // 检查形状
    OP_REQUIRES(context, x.shape() == y.shape(),
                tensorflow::errors::InvalidArgument(
                    "Inputs must have same shape"));

    // 分配输出
    tensorflow::Tensor* z = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, x.shape(), &z));

    // 调用 PTO kernel
    const float* x_ptr = x.flat<float>().data();
    const float* y_ptr = y.flat<float>().data();
    float* z_ptr = z->flat<float>().data();
    uint32_t length = x.NumElements();

    EXEC_KERNEL_CMD(MyAddKernel, 24, z_ptr, x_ptr, y_ptr, length);
  }
};

// 注册 kernel
REGISTER_KERNEL_BUILDER(
    Name("MyAdd").Device(tensorflow::DEVICE_NPU),
    MyAddOp);
```

#### 步骤3：编译

```bash
# 使用 TensorFlow 的编译工具
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

g++ -std=c++17 -shared my_ops.cc -o my_ops.so \
    ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} \
    -I/path/to/pto-isa/include \
    -L/path/to/pto-isa/lib -lpto \
    -fPIC -O3
```

#### 步骤4：Python 使用

```python
import tensorflow as tf

# 加载自定义 op
my_ops = tf.load_op_library('./my_ops.so')

# 使用
x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
y = tf.constant([[5.0, 6.0], [7.0, 8.0]])
z = my_ops.my_add(x, y)

print(z.numpy())
# [[6. 8.]
#  [10. 12.]]
```

### 3.2 注册梯度

```python
@tf.RegisterGradient("MyAdd")
def _my_add_grad(op, grad):
    """MyAdd 的梯度"""
    return grad, grad  # ∂z/∂x = 1, ∂z/∂y = 1
```

---

## 4. ONNX Runtime 集成

### 4.1 自定义 Execution Provider

#### 步骤1：定义 Kernel

```cpp
// my_onnx_ops.cc
#include "onnxruntime/core/framework/op_kernel.h"

class MyAddKernel : public onnxruntime::OpKernel {
 public:
  MyAddKernel(const onnxruntime::OpKernelInfo& info) : OpKernel(info) {}

  onnxruntime::Status Compute(onnxruntime::OpKernelContext* context) const override {
    // 获取输入
    const onnxruntime::Tensor* X = context->Input<onnxruntime::Tensor>(0);
    const onnxruntime::Tensor* Y = context->Input<onnxruntime::Tensor>(1);

    // 分配输出
    onnxruntime::Tensor* Z = context->Output(0, X->Shape());

    // 调用 PTO kernel
    const float* x_data = X->Data<float>();
    const float* y_data = Y->Data<float>();
    float* z_data = Z->MutableData<float>();
    size_t length = X->Shape().Size();

    EXEC_KERNEL_CMD(MyAddKernel, 24, z_data, x_data, y_data, length);

    return onnxruntime::Status::OK();
  }
};
```

#### 步骤2：注册 Kernel

```cpp
ONNX_OPERATOR_KERNEL_EX(
    Add,
    kOnnxDomain,
    7,  // opset version
    kNpuExecutionProvider,
    MyAddKernel);
```

#### 步骤3：创建 Execution Provider

```cpp
class NpuExecutionProvider : public onnxruntime::IExecutionProvider {
 public:
  NpuExecutionProvider() : IExecutionProvider(kNpuExecutionProvider) {}

  std::vector<std::unique_ptr<onnxruntime::ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const std::vector<const onnxruntime::KernelRegistry*>& registries) const override {
    // 返回支持的算子
    // ...
  }
};
```

#### 步骤4：Python 使用

```python
import onnxruntime as ort

# 注册自定义 EP
session_options = ort.SessionOptions()
session_options.register_custom_ops_library('my_onnx_ops.so')

# 创建会话
session = ort.InferenceSession(
    'model.onnx',
    session_options,
    providers=['NpuExecutionProvider', 'CPUExecutionProvider']
)

# 推理
outputs = session.run(None, {'input': input_data})
```

---

## 5. 推理框架集成

### 5.1 MindSpore Lite 集成

```cpp
// 注册自定义算子
#include "include/registry/register_kernel.h"

class MyAddKernel : public mindspore::kernel::Kernel {
 public:
  int Prepare() override { return RET_OK; }

  int Execute() override {
    auto input0 = in_tensors_[0];
    auto input1 = in_tensors_[1];
    auto output = out_tensors_[0];

    // 调用 PTO kernel
    // ...

    return RET_OK;
  }
};

// 注册
REGISTER_CUSTOM_KERNEL(NPU, MyProvider, kNumberTypeFloat32, Add, MyAddKernel)
```

### 5.2 TensorRT 集成

```cpp
// 自定义 Plugin
class MyAddPlugin : public nvinfer1::IPluginV2DynamicExt {
 public:
  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs,
              void* const* outputs,
              void* workspace,
              cudaStream_t stream) noexcept override {

    // 调用 PTO kernel
    // ...

    return 0;
  }
};

// 注册
REGISTER_TENSORRT_PLUGIN(MyAddPluginCreator);
```

---

## 6. 性能优化

### 6.1 算子融合

```python
# PyTorch 示例：融合 Add + ReLU
@torch.jit.script
def fused_add_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.relu(x + y)

# 使用自定义融合算子替换
torch.ops.npu.fused_add_relu(x, y)
```

### 6.2 内存优化

```cpp
// Inplace 算子
at::Tensor& my_add_inplace(at::Tensor& x, const at::Tensor& y) {
  // 直接修改 x，避免分配新内存
  float* x_ptr = x.data_ptr<float>();
  const float* y_ptr = y.data_ptr<float>();
  uint32_t length = x.numel();

  EXEC_KERNEL_CMD(MyAddInplaceKernel, 24, x_ptr, y_ptr, length);

  return x;
}
```

### 6.3 异步执行

```cpp
// 使用 CUDA Stream（或 NPU Stream）
at::Tensor my_add_async(const at::Tensor& x, const at::Tensor& y) {
  at::Tensor out = at::empty_like(x);

  // 获取当前 stream
  auto stream = at::cuda::getCurrentCUDAStream();

  // 异步启动 kernel
  EXEC_KERNEL_ASYNC(MyAddKernel, 24, stream,
                    out.data_ptr<float>(),
                    x.data_ptr<float>(),
                    y.data_ptr<float>(),
                    x.numel());

  return out;
}
```

---

## 7. 调试与测试

### 7.1 单元测试

```python
import unittest
import torch
import my_pto_ops

class TestMyOps(unittest.TestCase):
    def test_my_add(self):
        x = torch.randn(100, 100).npu()
        y = torch.randn(100, 100).npu()

        # 自定义算子
        z_custom = torch.ops.npu.my_add(x, y)

        # 参考实现
        z_ref = x + y

        # 验证
        self.assertTrue(torch.allclose(z_custom, z_ref, rtol=1e-5))

    def test_my_add_backward(self):
        x = torch.randn(100, 100, requires_grad=True).npu()
        y = torch.randn(100, 100, requires_grad=True).npu()

        z = torch.ops.npu.my_add(x, y)
        loss = z.sum()
        loss.backward()

        # 验证梯度
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)
        self.assertTrue(torch.allclose(x.grad, torch.ones_like(x)))

if __name__ == '__main__':
    unittest.main()
```

### 7.2 性能基准测试

```python
import torch
import time

def benchmark(func, *args, warmup=10, iterations=100):
    # 预热
    for _ in range(warmup):
        func(*args)

    # 同步
    torch.npu.synchronize()

    # 测量
    start = time.time()
    for _ in range(iterations):
        func(*args)
    torch.npu.synchronize()
    end = time.time()

    avg_time = (end - start) / iterations * 1000  # ms
    return avg_time

# 对比性能
x = torch.randn(1024, 1024).npu()
y = torch.randn(1024, 1024).npu()

time_custom = benchmark(lambda: torch.ops.npu.my_add(x, y))
time_builtin = benchmark(lambda: x + y)

print(f"Custom op: {time_custom:.3f} ms")
print(f"Built-in op: {time_builtin:.3f} ms")
print(f"Speedup: {time_builtin / time_custom:.2f}x")
```

---

## 8. 最佳实践

### 8.1 设计原则

✅ **DO**：
- 保持算子接口简单清晰
- 提供完整的类型支持（float32, float16, int32 等）
- 实现形状推导和类型推导
- 提供详细的文档和示例
- 编写完整的单元测试

❌ **DON'T**：
- 不要在算子内部分配大量临时内存
- 不要假设输入总是连续的（使用 contiguous()）
- 不要忽略边界情况（空张量、单元素张量）
- 不要在算子内部使用全局状态

### 8.2 性能检查清单

- [ ] 算子是否支持 inplace 操作
- [ ] 是否实现了算子融合
- [ ] 是否使用了异步执行
- [ ] 是否避免了不必要的内存拷贝
- [ ] 是否支持多种数据类型
- [ ] 是否进行了性能基准测试

### 8.3 兼容性检查清单

- [ ] 是否支持动态形状
- [ ] 是否支持广播语义
- [ ] 是否支持梯度计算（训练）
- [ ] 是否支持 JIT 编译
- [ ] 是否支持导出为 ONNX
- [ ] 是否提供 CPU fallback

---

## 参考资源

- [PyTorch 自定义算子教程](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [TensorFlow 自定义 Op 指南](https://www.tensorflow.org/guide/create_op)
- [ONNX Runtime 自定义算子](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html)
- [PTO Add 算子示例](../../demos/baseline/add/README_zh.md)
- [算子调试指南](debug_zh.md)
- [性能优化指南](opt_zh.md)
