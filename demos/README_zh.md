# PTO Demos

本目录包含 PTO Tile Library 在不同场景下的演示示例。

## 按任务选择

| 你的目标 | 从这里开始 |
|----------|----------|
| 快速验证算法（无需硬件） | CPU 模拟 demo — `tests/run_cpu.py --demo` |
| 学习 PTO tile 编程 | CPU demo — `flash_attn` 或 `gemm` |
| 生产级 NPU 算子 | `baseline/` — 带 PyTorch 集成的完整示例 |
| 即时编译与调试 | `torch_jit/` — JIT 编译示例 |
| Auto Mode | `auto_mode/baseline/add/` — Auto Mode 示例 |

## 目录结构

```
demos/
├── baseline/                     # 生产级 PyTorch 算子示例（NPU）
│   ├── add/                   # 逐元素加法
│   ├── gemm_basic/           # GEMM（含流水线优化）
│   ├── flash_atten/          # Flash Attention（含动态分块）
│   └── allgather_async/      # 异步 AllGather
│
├── auto_mode/                   # Auto Mode 示例（CPU / NPU 均可）
│   └── baseline/add/          # Auto Mode 逐元素加法
│
├── cpu/                        # CPU 模拟 demo（跨平台，无需 Ascend 硬件）
│   ├── gemm_demo/
│   ├── flash_attention_demo/
│   └── mla_attention_demo/
│
└── torch_jit/                 # PyTorch JIT 编译示例
    ├── add/
    ├── gemm/
    └── flash_atten/
```

## 示例类别

### 1. Baseline（`baseline/`）

生产级示例，展示如何实现自定义 PTO kernel 并通过 `torch_npu` 将其作为 PyTorch 算子公开。包含从 kernel 实现到 Python 集成的完整工作流程，带 CMake 构建系统和 wheel 打包。

**支持平台**：A2/A3/A5

**示例**：
- 逐元素加法 — 最基础的 PTO 算子示例
- GEMM — 带双缓冲流水线的矩阵乘法
- Flash Attention — 带自动 tile 大小选择的 Flash Attention
- AllGather-Async — 异步 AllGather 通信

### 2. CPU 模拟（`cpu/`）

在 CPU（x86_64/AArch64）上运行的跨平台示例，无需 Ascend 硬件。适用于算法原型设计、学习 PTO 编程模型和 CI/CD 测试。

**示例**：基础 GEMM、Flash Attention、多潜在注意力（MLA）

### 3. Auto Mode（`auto_mode/`）

展示 PTO AUTO 模式的代码。Auto 模式下编译器自动管理 tile buffer 地址分配与流水线同步，无需手动 `TASSIGN` 和 `set_flag`/`wait_flag`。

**示例**：Auto Mode 逐元素加法

### 4. PyTorch JIT（`torch_jit/`）

展示即时 C++ 编译和与 PyTorch 张量直接集成的示例。适用于快速原型设计，无需预先构建 wheel。

**示例**：JIT 加法、JIT GEMM、带基准测试套件的 JIT Flash Attention

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

### Auto Mode 示例

```bash
cd demos/auto_mode/baseline/add
# See the README_zh.md inside for build and run instructions
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
- Python 3.8+、CMake 3.16+

**CPU 演示**：
- 支持 C++20 的 C++ 编译器
- CMake 3.16+
- Python 3.8+（可选）

## 相关文档

| 文档 | 内容 |
|------|------|
| [demos/README_zh.md](./README_zh.md) | 中文版入口 |
| [docs/getting-started_zh.md](../docs/getting-started_zh.md) | 入门指南 |
| [docs/coding/tutorial_zh.md](../docs/coding/tutorial_zh.md) | 编程教程 |
| [docs/isa/README_zh.md](../docs/isa/README_zh.md) | ISA 参考 |
