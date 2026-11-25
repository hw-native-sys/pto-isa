# FA PTO PyTorch Porting Example

## Overview

This example demonstrates how to implement a Flash Attention kernel using PTO (Parallel Template Operators) and expose it as a custom PyTorch operator via `torch_npu`. It showcases high-performance custom kernel integration on Ascend AI Processors with automatic tiling adaptation.

## Supported AI Processors

- A2/A3/A5

## 1. Prepare the Environment

Create a virtual environment and install dependencies:

```bash
python -m venv virEnv
source virEnv/bin/activate
python3 -m pip install -r requirements.txt
```

Ensure the Ascend Toolkit and PTO library are configured:

```bash
export ASCEND_HOME_PATH=[YOUR_ASCEND_PATH/SYSTEM_ASCEND_PATH]
source [YOUR_ASCEND_PATH/SYSTEM_ASCEND_PATH]/latest/bin/setenv.bash
export PTO_LIB_PATH=[YOUR_PATH]/pto-isa
```

## 2. Build the Wheel

The project supports building for different SOC versions via the `SOC_VERSION` environment variable. The build system automatically configures the correct optimization macros (e.g., `PTO_NPU_ARCH_A2A3` vs `PTO_NPU_ARCH_A5`) based on the target SOC.

**Default Build (A2 / A3):**
```bash
python3 setup.py bdist_wheel
```

**Build for Specific SOC (e.g., A5):**
```bash
# Example for A5
SOC_VERSION=ascend910_9599 python3 setup.py bdist_wheel
```

## 3. Install the Wheel

```bash
pip install dist/*.whl --force-reinstall
```

## 4. Run Tests

Run the verification script to compare kernel results against the golden reference. The test covers various sequence lengths (1k to 32k) and validates the dynamic tiling logic.

```bash
cd test
python3 test.py
```

## Features

- **Dynamic Tiling**: Automatically selects the optimal tile size (128 or 256) based on input sequence length.
- **Cross-Architectural Support**: Unified codebase supporting both A2/A3 and A5 architectures via build-time configuration.
