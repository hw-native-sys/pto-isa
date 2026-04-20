# PTO Demos

This directory contains demonstration examples showing how to use PTO Tile Library in different scenarios.

## Choose by Task

| Your goal | Start here |
|-----------|-----------|
| Verify algorithms quickly (no hardware needed) | CPU simulation demos — `tests/run_cpu.py --demo` |
| Learn PTO tile programming | CPU demos — `flash_attn` or `gemm` |
| Production NPU operators | `baseline/` — full examples with PyTorch integration |
| Just-in-time compilation and debugging | `torch_jit/` — JIT compilation examples |
| Auto Mode | `auto_mode/baseline/add/` — Auto Mode example |

## Directory Structure

```
demos/
├── baseline/                     # Production-grade PyTorch operator examples (NPU)
│   ├── add/                   # Element-wise addition
│   ├── gemm_basic/           # GEMM with pipeline optimization
│   ├── flash_atten/          # Flash Attention with dynamic tiling
│   └── allgather_async/      # Asynchronous AllGather
│
├── auto_mode/                   # Auto Mode examples (CPU / NPU compatible)
│   └── baseline/add/          # Auto Mode element-wise addition
│
├── cpu/                        # CPU simulation demos (cross-platform, no Ascend hardware)
│   ├── gemm_demo/
│   ├── flash_attention_demo/
│   └── mla_attention_demo/
│
└── torch_jit/                 # PyTorch JIT compilation examples
    ├── add/
    ├── gemm/
    └── flash_atten/
```

## Demo Categories

### 1. Baseline (`baseline/`)

Production-ready examples showing how to implement custom PTO kernels and expose them as PyTorch operators via `torch_npu`. Includes complete workflow from kernel implementation to Python integration with CMake build system and wheel packaging.

**Supported Platforms**: A2/A3/A5

**Examples**:
- Element-wise addition — the most basic PTO operator example
- GEMM — matrix multiplication with double-buffering pipeline
- Flash Attention — with automatic tile size selection
- AllGather-Async — asynchronous AllGather communication

### 2. CPU Simulation (`cpu/`)

Cross-platform examples that run on CPU (x86_64/AArch64) without requiring Ascend hardware. Ideal for algorithm prototyping, learning PTO programming model, and CI/CD testing.

**Examples**: Basic GEMM, Flash Attention, Multi-Latent Attention (MLA)

### 3. Auto Mode (`auto_mode/`)

Examples showcasing PTO AUTO mode. In Auto mode, the compiler automatically manages tile buffer address allocation and pipeline synchronization — no manual `TASSIGN` or `set_flag`/`wait_flag` needed.

**Examples**: Auto Mode element-wise addition

### 4. PyTorch JIT (`torch_jit/`)

Examples showing on-the-fly C++ compilation and direct integration with PyTorch tensors. Useful for rapid prototyping without pre-building wheels.

**Examples**: JIT addition, JIT GEMM, JIT Flash Attention with benchmark suite.

## Quick Start

### CPU Simulation (Recommended First Step)

```bash
python3 tests/run_cpu.py --demo gemm --verbose
python3 tests/run_cpu.py --demo flash_attn --verbose
```

### NPU Baseline Example

```bash
cd demos/baseline/add
python -m venv virEnv && source virEnv/bin/activate
pip install -r requirements.txt
export PTO_LIB_PATH=[YOUR_PATH]/pto-isa
python3 setup.py bdist_wheel
pip install dist/*.whl
cd test && python3 test.py
```

### Auto Mode Example

```bash
cd demos/auto_mode/baseline/add
# See the README inside for build and run instructions
```

### JIT Example

```bash
export PTO_LIB_PATH=[YOUR_PATH]/pto-isa
cd demos/torch_jit/add
python add_compile_and_run.py
```

## Prerequisites

**For Baseline and JIT (NPU)**:
- Ascend AI Processor A2/A3/A5 (910B/910C/950)
- CANN Toolkit 8.5.0+
- PyTorch with `torch_npu`
- Python 3.8+, CMake 3.16+

**For CPU Demos**:
- C++ compiler with C++20 support
- CMake 3.16+
- Python 3.8+ (optional)

## Related Documents

| Document | Content |
|----------|---------|
| [demos/README_zh.md](./README_zh.md) | 中文版入口 |
| [docs/getting-started.md](../docs/getting-started.md) | Getting started guide |
| [docs/coding/tutorial.md](../docs/coding/tutorial.md) | Programming tutorial |
| [docs/isa/README.md](../docs/isa/README.md) | ISA reference |
