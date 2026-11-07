# PTO Demos

This directory contains demonstration examples showing how to use PTO Tile Library in different scenarios.

## Directory Structure

```
demos/
├── baseline/         # Production PyTorch operator examples (NPU)
│   ├── add/          # Basic element-wise addition
│   ├── allgather_async/  # Async all-gather with pipeline overlap
│   ├── gemm_basic/   # GEMM with pipeline optimization
│   └── flash_atten/  # Flash Attention with dynamic tiling
├── cpu/              # CPU simulation demos (cross-platform)
│   ├── gemm_demo/
│   ├── flash_attention_demo/
│   └── mla_attention_demo/
├── auto_mode/        # Auto-mode orchestration demos
│   ├── baseline/     # Auto-mode baseline operator examples
│   └── torch_jit/    # Auto-mode JIT examples
└── torch_jit/        # PyTorch JIT compilation examples
    ├── add/
    ├── gemm/
    └── flash_atten/
```

## Demo Categories

### 1. Baseline (`baseline/`)

Production-ready examples showing how to implement custom PTO kernels and expose them as PyTorch operators via `torch_npu`. Includes complete workflow from kernel implementation to Python integration with CMake build system and wheel packaging.

**Supported Platforms**: A2/A3/A5

**Examples**: Element-wise addition, GEMM with double-buffering pipeline, Flash Attention with automatic tile size selection, async all-gather with pipeline overlap.

### 2. CPU Simulation (`cpu/`)

Cross-platform examples that run on CPU (x86_64/AArch64) without requiring Ascend hardware. Ideal for algorithm prototyping, learning PTO programming model, and CI/CD testing.

**Examples**: Basic GEMM, Flash Attention, Multi-Latent Attention.

### 3. PyTorch JIT (`torch_jit/`)

Examples showing on-the-fly C++ compilation and direct integration with PyTorch tensors. Useful for rapid prototyping without pre-building wheels.

**Examples**: JIT addition, JIT GEMM, JIT Flash Attention with benchmark suite.

### 4. Auto Mode (`auto_mode/`)

Auto-mode orchestration examples that let PTO infer the AIC/AIV split and event synchronisation automatically, instead of writing the cross-core handshakes by hand. Currently provides one `add` example in each of the `baseline/` and `torch_jit/` forms, under the auto-mode build path.

**Examples**: Auto-mode `add` in the baseline form, auto-mode `add` in the JIT form.

## Quick Start

### CPU Simulation (Recommended First Step)

```bash
python3 tests/run_cpu.py --demo gemm --verbose
python3 tests/run_cpu.py --demo flash_attn --verbose
```

### A5 Simulator ST (Single-Testcase)

The `demos/` tree targets NPU / CPU; the A5 simulator entry point lives under `tests/script/run_st.py` and runs a single ST testcase on the Ascend 950PR simulator:

```bash
python3 tests/script/run_st.py -r sim -v a5 -t tadd -g TADDTest.case_float_64x64_64x64_64x64_64x64
```

See [tests/README.md](../tests/README.md) for the full ST workflow (build flags, GoogleTest ABI matching, multi-rank comm tests).

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
- Python >= 3.9.x, CMake 3.16+

**For CPU Demos**:
- C++ compiler with C++23 support. The CPU simulator headers the demos build against use `_Float16`, so the host compiler must pass the probe in `tests/run_cpu.py`:
  - GCC >= 13, or
  - Clang >= 15 (Apple Clang uses the same 15 threshold).
  Native `std::bfloat16_t` additionally requires a C++23 `<stdfloat>` toolchain (GCC >= 14 / Clang >= 15, see the note in `include/pto/common/type.hpp`). Earlier toolchains (e.g. GCC 11) fail with `Could not find a suitable compiler` before compilation starts.
- CMake 3.16+
- Python >= 3.9.x (optional)

> **Note:** Python has announced that 3.7.x/3.8.x are EOL. CANN will soon stop supporting these versions. Please upgrade to Python >= 3.9.x.

## Documentation

- Getting Started: [docs/getting-started.md](../docs/getting-started.md)
- Programming Tutorial: [docs/coding/tutorial.md](../docs/coding/tutorial.md)
- ISA Reference: [docs/isa/README.md](../docs/isa/README.md)

## Related

- Manual Kernels: [kernels/manual/README.md](../kernels/manual/README.md)
- Custom Operators: [kernels/custom/README.md](../kernels/custom/README.md)
- Test Cases: [tests/README.md](../tests/README.md)
