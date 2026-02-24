# PTO 演示示例

本目录包含演示示例，展示如何在不同场景中使用 PTO Tile Library。

## 目录结构

```
demos/
├── baseline/         # 生产级 PyTorch 算子示例（NPU）
│   ├── add/          # 基础逐元素加法
│   ├── gemm_basic/   # 带流水线优化的 GEMM
│   └── flash_atten/  # 带动态分块的 Flash Attention
├── cpu/              # CPU 模拟演示（跨平台）
│   ├── gemm_demo/
│   ├── flash_attention_demo/
│   └── mla_attention_demo/
└── torch_jit/        # PyTorch JIT 编译示例
    ├── add/
    ├── gemm/
    └── flash_atten/
```

## 演示类别

### 1. Baseline (`baseline/`)

生产级示例，展示如何实现自定义 PTO 内核并通过 `torch_npu` 将其作为 PyTorch 算子公开。包含从内核实现到 Python 集成的完整工作流程，带 CMake 构建系统和 wheel 打包。

**支持平台**：A2/A3/A5

**示例**：逐元素加法、带双缓冲流水线的 GEMM、带自动 tile 大小选择的 Flash Attention。

### 2. CPU 模拟 (`cpu/`)

在 CPU（x86_64/AArch64）上运行的跨平台示例，无需 Ascend 硬件。适用于算法原型设计、学习 PTO 编程模型和 CI/CD 测试。

**示例**：基础 GEMM、Flash Attention、多潜在注意力。

### 3. PyTorch JIT (`torch_jit/`)

展示即时 C++ 编译和与 PyTorch 张量直接集成的示例。适用于快速原型设计，无需预先构建 wheel。

**示例**：JIT 加法、JIT GEMM、带基准测试套件的 JIT Flash Attention。

## 快速开始

### CPU 模拟（推荐第一步）

```bash
python3 tests/run_cpu.py --demo gemm --verbose
python3 tests/run_cpu.py --demo flash_attn --verbose
```

### NPU Baseline 示例

```bash
cd demos/baseline/add
python -m venv virEnv && source virEnv/bin/activate
pip install -r requirements.txt
export PTO_LIB_PATH=[YOUR_PATH]/pto-isa
python3 setup.py bdist_wheel
pip install dist/*.whl
cd test && python3 test.py
```

### JIT 示例

```bash
export PTO_LIB_PATH=[YOUR_PATH]/pto-isa
cd demos/torch_jit/add
python add_compile_and_run.py
```

## 前置要求

**Baseline 和 JIT（NPU）**：
- Ascend AI 处理器（910B/910C/950）
- CANN Toolkit 8.5.0+
- 带 `torch_npu` 的 PyTorch
- Python 3.8+、CMake 3.16+

**CPU 演示**：
- 支持 C++23 的 C++ 编译器
- CMake 3.16+
- Python 3.8+（可选）

## 文档

- 入门指南：[docs/getting-started.md](../docs/getting-started.md)
- 编程教程：[docs/coding/tutorial.md](../docs/coding/tutorial.md)
- ISA 参考：[docs/isa/README.md](../docs/isa/README.md)

## 相关

- 手工内核：[kernels/manual/README.md](../kernels/manual/README.md)
- 自定义算子：[kernels/custom/README.md](../kernels/custom/README.md)
- 测试用例：[tests/README.md](../tests/README.md)
