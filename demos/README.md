# PTO Demos

This directory contains demonstration examples showing how to use PTO Tile Library in different scenarios.

## Directory Structure

```
demos/
├── baseline/         # Production PyTorch operator examples (NPU)
│   ├── add/          # Basic element-wise addition
│   ├── gemm_basic/   # GEMM with pipeline optimization
│   └── flash_atten/  # Flash Attention with dynamic tiling
├── cpu/              # CPU simulation demos (cross-platform)
│   ├── gemm_demo/
│   ├── flash_attention_demo/
│   └── mla_attention_demo/
└── torch_jit/        # PyTorch JIT compilation examples
    ├── add/
    ├── gemm/
    └── flash_atten/
```

## Demo Categories

### 1. Baseline (`baseline/`)

Production-ready examples showing how to implement custom PTO kernels and expose them as PyTorch operators via `torch_npu`. Includes complete workflow from kernel implementation to Python integration with CMake build system and wheel packaging.

**Supported Platforms**: A2/A3/A5

**Examples**: Element-wise addition, GEMM with double-buffering pipeline, Flash Attention with automatic tile size selection.

### 2. CPU Simulation (`cpu/`)

Cross-platform examples that run on CPU (x86_64/AArch64) without requiring Ascend hardware. Ideal for algorithm prototyping, learning PTO programming model, and CI/CD testing.

**Examples**: Basic GEMM, Flash Attention, Multi-Latent Attention.

### 3. PyTorch JIT (`torch_jit/`)

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

### JIT Example

```bash
export PTO_LIB_PATH=[YOUR_PATH]/pto-isa
cd demos/torch_jit/add
python add_compile_and_run.py
```

## Prerequisites

**For Baseline and JIT (NPU)**:
- Ascend AI Processor A2/A3/A5(910B/910C/950)
- CANN Toolkit 8.5.0+
- PyTorch with `torch_npu`
- Python 3.8+, CMake 3.16+

**For CPU Demos**:
- C++ compiler with C++23 support
- CMake 3.16+
- Python 3.8+ (optional)

## Documentation

- Getting Started: [docs/getting-started.md](../docs/getting-started.md)
- Programming Tutorial: [docs/coding/tutorial.md](../docs/coding/tutorial.md)
- ISA Reference: [docs/isa/README.md](../docs/isa/README.md)

## Related

- Manual Kernels: [kernels/manual/README.md](../kernels/manual/README.md)
- Custom Operators: [kernels/custom/README.md](../kernels/custom/README.md)
- Test Cases: [tests/README.md](../tests/README.md)
