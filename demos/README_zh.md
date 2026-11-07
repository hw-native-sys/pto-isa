# PTO 演示示例

本目录包含演示示例，展示如何在不同场景中使用 PTO Tile Library。

## 目录结构

```
demos/
├── baseline/         # 生产级 PyTorch 算子示例（NPU）
│   ├── add/          # 基础逐元素加法
│   ├── allgather_async/  # 与流水线重叠的异步 all-gather
│   ├── gemm_basic/   # 带流水线优化的 GEMM
│   └── flash_atten/  # 带动态分块的 Flash Attention
├── cpu/              # CPU 模拟演示（跨平台）
│   ├── gemm_demo/
│   ├── flash_attention_demo/
│   └── mla_attention_demo/
├── auto_mode/        # Auto 模式编排示例
│   ├── baseline/     # Auto 模式 baseline 算子示例
│   └── torch_jit/    # Auto 模式 JIT 示例
└── torch_jit/        # PyTorch JIT 编译示例
    ├── add/
    ├── gemm/
    └── flash_atten/
```

## 演示类别

### 1. Baseline (`baseline/`)

生产级示例，展示如何实现自定义 PTO 内核并通过 `torch_npu` 将其作为 PyTorch 算子公开。包含从内核实现到 Python 集成的完整工作流程，带 CMake 构建系统和 wheel 打包。

**支持平台**：A2/A3/A5

**示例**：逐元素加法、带双缓冲流水线的 GEMM、带自动 tile 大小选择的 Flash Attention、与流水线重叠的异步 all-gather。

### 2. CPU 模拟 (`cpu/`)

在 CPU（x86_64/AArch64）上运行的跨平台示例，无需 Ascend 硬件。适用于算法原型设计、学习 PTO 编程模型和 CI/CD 测试。

**示例**：基础 GEMM、Flash Attention、多潜在注意力。

### 3. PyTorch JIT (`torch_jit/`)

展示即时 C++ 编译和与 PyTorch 张量直接集成的示例。适用于快速原型设计，无需预先构建 wheel。

**示例**：JIT 加法、JIT GEMM、带基准测试套件的 JIT Flash Attention。

### 4. Auto 模式 (`auto_mode/`)

Auto 模式编排示例，由 PTO 自动推断 AIC/AIV 切分与事件同步，无需手写跨核握手。目前 `baseline/`、`torch_jit/` 下各有一个 `add` 示例，走 Auto 模式构建路径。

**示例**：Auto 模式 `add`（baseline 形式）、Auto 模式 `add`（JIT 形式）。

## 快速开始

### CPU 模拟（推荐第一步）

```bash
python3 tests/run_cpu.py --demo gemm --verbose
python3 tests/run_cpu.py --demo flash_attn --verbose
```

### A5 仿真器 ST（单用例）

`demos/` 面向 NPU 与 CPU；A5 仿真器（即 Ascend 950PR&950DT 系列产品）入口在 `tests/script/run_st.py`，在该系列仿真器上运行单个 ST 用例：

```bash
python3 tests/script/run_st.py -r sim -v a5 -t tadd -g TADDTest.case_float_64x64_64x64_64x64_64x64
```

完整 ST 工作流（编译选项、GoogleTest ABI 匹配、多 rank 通信测试）见 [tests/README.md](../tests/README_zh.md)。

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
- Ascend AI 处理器 A2/A3/A5（910B/910C/950）
- CANN Toolkit 8.5.0+
- 带 `torch_npu` 的 PyTorch
- Python >= 3.9.x、CMake 3.16+

**CPU 演示**：
- 支持 C++23 的 C++ 编译器。演示所依赖的 CPU 仿真头文件使用 `_Float16`，主机编译器需通过 `tests/run_cpu.py` 的探测门槛：
  - GCC >= 13，或
  - Clang >= 15（Apple Clang 同用 15 门槛）。
  如需原生 `std::bfloat16_t`，工具链还需提供 C++23 `<stdfloat>`（GCC >= 14 / Clang >= 15，见 `include/pto/common/type.hpp` 说明）。更早的工具链（如 GCC 11）在编译前就会报 `Could not find a suitable compiler`。
- CMake 3.16+
- Python >= 3.9.x（可选）

> **备注：** Python 宣布 3.7.x/3.8.x 已经 EOL，CANN 即将停止对该版本的支持，请升级到 Python >= 3.9.x 的版本。

## 文档

- 入门指南：[docs/getting-started.md](../docs/getting-started_zh.md)
- 编程教程：[docs/coding/tutorial.md](../docs/coding/tutorial_zh.md)
- ISA 参考：[docs/isa/README.md](../docs/isa/README_zh.md)

## 相关

- 手工内核：[kernels/manual/README.md](../kernels/manual/README_zh.md)
- 自定义算子：[kernels/custom/README.md](../kernels/custom/README_zh.md)
- 测试用例：[tests/README.md](../tests/README_zh.md)
