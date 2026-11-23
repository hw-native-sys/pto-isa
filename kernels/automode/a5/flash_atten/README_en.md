# Flash Attention Performance Kernel (A5, Auto Mode)

## Overview

This example demonstrates how to implement a mixed-precision Flash Attention (FA) operator using PTO auto mode on the Ascend A5 platform, including project setup, build, and execution. Its CMake configuration enables auto mode with `--cce-pto-auto-enable`.

For algorithm background and a manual A2/A3 implementation reference, see the [common Flash Attention documentation](../../../manual/common/flash_atten/README_en.md). Build commands, platform-specific configuration, and scheduling details for this A5 auto-mode example are defined in this directory.

## Supported AI Processors

- Ascend A5

## Directory Layout

```
kernels/automode/a5/flash_atten/
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
cd ${git_clone_path}/kernels/automode/a5/flash_atten

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

This section covers performance measurements for the A5 auto-mode Flash Attention kernel in this directory.

Definitions:
- `S0`: query sequence length (rows of Q/O).
- `S1`: key/value sequence length (rows of K/V).
- `Total task time (us)`: end-to-end kernel time per task (microseconds).
- `GOps`: total operations counted for the task.
- `TFLOPS`: `GOps / time`.

### Performance Measurements (Pending)

Timing and throughput for the following workloads have not yet been recorded on Ascend A5:

| Cores | S0 | S1 | Total task time (us) | GOps | TFLOPS |
| --- | --- | --- | --- | --- | --- |
| 1 | 128 | 1024 | TBD | 67.11 | TBD |
| 1 | 128 | 2048 | TBD | 134.22 | TBD |
| 1 | 128 | 4096 | TBD | 268.44 | TBD |

*Note: TBD indicates data to be measured.*

## Operator Description

For mathematical formulas, tiling strategies, and pipeline orchestration in the manual A2/A3 reference implementation, see:

- [Common Flash Attention Documentation](../../../manual/common/flash_atten/README_en.md)

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

Specific tuning parameters and performance data will be continuously updated in future versions.
