# Flash Attention Performance Kernel (A5)

## Overview

This example demonstrates how to implement a mixed-precision Flash Attention (FA) operator using PTO on the Ascend A5 platform, including project setup, build, and execution.

For detailed operator description, optimization strategies, and pipeline orchestration of Flash Attention, please refer to the common version documentation: [../../common/flash_atten/README.md](../../common/flash_atten/README.md)

## Supported AI Processors

- Ascend A5

## Directory Layout

```
kernels/manual/a5/flash_atten/
├── scripts/
│   ├── gen_data.py                  # Generates input and golden output
│   ├── generate_cases.py            # Generates test cases
│   ├── pipeline_log_analysis.py     # Pipeline log analysis
│   ├── pipeline_schedule_gen.py     # Pipeline schedule generation
│   └── run_timeline.sh              # Timeline analysis script
├── CMakeLists.txt                   # Build configuration
├── fa_performance_kernel.cpp        # Kernel implementation
├── fa_performance_kernel.h          # Kernel header
├── main.cpp                         # Host-side entry point
├── pto_macro_fa_gu.hpp              # FA GU macro definitions
├── pto_macro_fa_softmax.hpp         # FA Softmax macro definitions
├── pto_macro_matmul.hpp             # Matmul macro definitions
└── run.sh                           # Convenience script
```

## Build and Run

1. Configure your Ascend CANN environment (example path):

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

2. Run the example:

```bash
cd ${git_clone_path}/kernels/manual/a5/flash_atten

# Run default cases (same set baked into generated_cases.*)
bash run.sh -r npu -v Ascend910_9599

# Run a single case from the generated set
bash run.sh -r npu -v Ascend910_9599 -c case_float_H_128_S0_128_S1_1024

# Provide custom cases (semicolon separated: HEAD_SIZE,S0,S1,CUBE_S0,TILE_S1)
# TILE_S1: supports 128 (=CUBE_S1), 256, 512
bash run.sh -r npu -v Ascend910_9599 --cases "128,128,1024,128,128;128,2048,2048,128,512"

# Provide custom cases and run just one of them
bash run.sh -r npu -v Ascend910_9599 --cases "128,128,1024,128,128;128,512,2048,128,128" \
  -c case_float_H_128_S0_128_S1_1024
```

If the run succeeds, the output prints:

```text
test success
```

## Performance

A5 reference performance numbers are not yet published for this kernel.

For the current 910B2 (A2/A3 generation) Flash Attention performance reference, see [../../common/flash_atten/README.md](../../common/flash_atten/README.md).

## Operator Description

For detailed implementation notes, mathematical formulas, tiling strategies, and pipeline orchestration of the Flash Attention operator, please refer to:

- [Common Flash Attention Documentation](../../common/flash_atten/README.md)

The documentation includes:
1. Computation Flow (FlashAttention 2.0)
2. Tensor shape progression (per stage)
3. Per-stage implementation & optimizations (`compute_qk`, `compute_p`, `compute_pv`, `compute_gu`)
4. Pipeline orchestration & cube/vector parallelism
5. Multicore task split, tiling and load balancing

## A5 Platform-Specific Optimizations

Compared to the A2/A3 platform, the A5 version of the Flash Attention kernel has been optimized for A5 architecture characteristics:

- Adjusted buffer allocation strategy for A5's memory hierarchy
- Optimized synchronization mechanisms between Cube and Vector cores
- Tuned pipeline depth parameters to match A5 hardware characteristics
