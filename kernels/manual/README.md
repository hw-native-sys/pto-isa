# Manual Kernels

This folder contains **manual (hand-tuned) kernel examples** that use explicit buffering, synchronization, and pipeline control for maximum performance on Ascend NPUs.

If you are new to PTO programming, start from the ISA and tutorials first:

- Programming tutorials: [docs/coding/tutorial.md](../../docs/coding/tutorial.md)
- Optimization notes: [docs/coding/opt.md](../../docs/coding/opt.md)
- PTO ISA reference: [docs/isa/README.md](../../docs/isa/README.md)

## Platforms

| Platform | Directory | Typical Kernels |
|----------|----------|----------------|
| Ascend A2/A3 | `a2a3/` | GEMM, Conv2D, TopK, AllGather-GEMM |
| Ascend A5 | `a5/` | Flash Attention, MXFP4/8 Matmul, AllGather-GEMM |
| Cross-platform | `common/` | Flash Attention (shared across A2/A3/A5) |

## Catalog

### A2/A3 Kernels (`a2a3/`)

| Kernel | Description | Key Techniques |
|--------|-------------|----------------|
| [GEMM Performance](a2a3/gemm_performance/) | High-performance matrix multiplication | Double-buffering, L0A/L0B/L0C tiling, UB staging |
| [Conv2D Forward](a2a3/conv2d_forward/) | Conv2D forward pass via img2col | img2col + GEMM fusion, fractal layout |
| [TopK](a2a3/topk/) | Top-K element selection | Sorting-based selection, tile-level reduction |
| [AllGather-GEMM](a2a3/allgather_gemm/) | Multi-NPU GEMM with AllGather | Collective communication fused with GEMM |

### A5 Kernels (`a5/`)

| Kernel | Description | Key Techniques |
|--------|-------------|----------------|
| [Flash Attention](a5/flash_atten/) | Flash Attention algorithm | Dynamic tiling, online softmax, A5-specific optimizations |
| [MXFP8 Matmul](a5/matmul_mxfp8_performance/) | MXFP8 precision matrix multiplication | MXFP8 dequantization, FP8 compute, A5 hardware support |
| [MXFP4 Matmul](a5/matmul_mxfp4_performance/) | MXFP4 precision matrix multiplication | MXFP4 dequantization, FP4 compute |
| [AllGather-GEMM](a5/allgather_gemm/) | Multi-NPU GEMM (A5) | Collective communication fused with GEMM |
| [Engram SIMT](a5/engram_simt/) | Engram SIMT example | SIMT-style programming on A5 |

### Cross-Platform Kernels (`common/`)

| Kernel | Description | Platforms |
|--------|-------------|-----------|
| [Flash Attention](common/flash_atten/) | Flash Attention algorithm | A2/A3/A5 |

## How to Run

Each subdirectory is a standalone example with its own build/run instructions. See the `README.md` in each folder.

## See Also

- [kernels/README.md](../README.md) — Parent directory entry
- [docs/coding/opt.md](../../docs/coding/opt.md) — Performance bottleneck analysis
- [docs/isa/README.md](../../docs/isa/README.md) — PTO ISA reference
