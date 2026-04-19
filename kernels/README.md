# Kernels

This directory contains kernel/operator implementations that complement PTO Tile Library.

Most subdirectories are **self-contained mini-projects** (kernel + host + scripts) with their own `README.md`, `CMakeLists.txt`, and `run.sh` for independent discovery and execution.

## Choose by Task

| Your goal | Start here |
|-----------|-----------|
| Learn PTO programming | [docs/coding/tutorial.md](../docs/coding/tutorial.md) |
| High-performance GEMM | [manual/a2a3/gemm_performance/README.md](manual/a2a3/gemm_performance/README.md) |
| Flash Attention | [manual/common/flash_atten/README.md](manual/common/flash_atten/README.md) |
| Conv2D forward | [manual/a2a3/conv2d_forward/README.md](manual/a2a3/conv2d_forward/README.md) |
| MXFP8 / MXFP4 Matmul | [manual/a5/matmul_mxfp8_performance/README.md](manual/a5/matmul_mxfp8_performance/README.md) |
| Custom operator scaffolding | [custom/README.md](custom/README.md) |

## Directory Layout

```
kernels/
├── manual/                    # Hand-tuned (hand-written, performance-oriented) NPU kernels
│   ├── a2a3/                 # Ascend A2/A3 platforms
│   │   ├── gemm_performance/ # High-performance GEMM — pipeline optimization, double-buffering, address planning
│   │   ├── conv2d_forward/  # Conv2D forward kernel — img2col + GEMM
│   │   ├── topk/            # TopK kernel — sorting and selection
│   │   └── allgather_gemm/  # Multi-NPU GEMM — AllGather communication fused with GEMM
│   │
│   ├── a5/                  # Ascend A5 platforms
│   │   ├── flash_atten/     # Flash Attention — A5-specific optimization
│   │   ├── matmul_mxfp8_performance/  # MXFP8 matrix multiplication
│   │   ├── matmul_mxfp4_performance/  # MXFP4 matrix multiplication
│   │   ├── allgather_gemm/  # Multi-NPU GEMM (A5)
│   │   └── engram_simt/     # Engram SIMT example
│   │
│   └── common/               # Cross-platform kernels (work on A2/A3/A5)
│       └── flash_atten/      # Flash Attention — cross-platform generic version
│
└── custom/                   # Custom operator scaffolding and examples
    └── fused_add_relu_mul/  # Fused operator example: Add + ReLU + Mul
```

## What Makes Manual Kernels Different

Unlike examples in `demos/`, kernels under `manual/` target **production-grade performance tuning**:

- **Explicit management** of tile buffer address allocation (`TASSIGN`)
- **Explicit management** of pipeline synchronization (`set_flag`/`wait_flag`)
- **Double/multi-buffering** to overlap data movement and compute
- **Address alignment and planning** to maximize UB / L0 utilization
- Microarchitectural tuning specific to each SoC (A2/A3 vs A5)

## Related Docs

| Document | Content |
|----------|---------|
| [kernels/README_zh.md](./README_zh.md) | 中文版入口 |
| [demos/](../demos/README.md) | End-to-end demos (including CPU versions) |
| [docs/coding/opt.md](../docs/coding/opt.md) | Performance optimization and bottleneck analysis |
| [docs/isa/README.md](../docs/isa/README.md) | PTO ISA instruction reference |

## Notes for Adding New Kernels

When adding a new kernel project, please include:

1. A short `README_zh.md` (Chinese) and `README.md` (English) explaining platform requirements and how to run
2. A `run.sh` or equivalent script for consistent discovery and execution
3. Use `pto_add_kernel(<target_name>)` in `CMakeLists.txt`
