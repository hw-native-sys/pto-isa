# Kernels

This directory contains kernel/operator implementations that complement PTO Tile Lib.

Most kernel subdirectories are **self-contained mini-projects** (kernel + host + scripts) with their own `README.md`, `CMakeLists.txt`, and `run.sh`.

## Where to start

- Manual (hand-tuned) NPU kernels: [manual](manual/README_en.md)
- Custom operator scaffolding: [custom](custom/README_en.md)
- End-to-end demos (including CPU): [demos](../demos/README_en.md)

## Directory layout

- `manual/`: Hand-tuned kernels with explicit buffering/synchronization (NPU-focused)
  - `manual/a2a3/`: Kernels for A2/A3 platforms
    - `manual/a2a3/gemm_performance/`: High-performance GEMM example
    - `manual/a2a3/conv2d_forward/`: Conv2D forward kernel example
    - `manual/a2a3/topk/`: TopK kernel example
  - `manual/a5/`: Kernels for A5 platforms
    - `manual/a5/flash_atten/`: Flash-Attention kernel for A5
    - `manual/a5/matmul_mxfp4_performance/`: MXFP4 matrix multiplication example
    - `manual/a5/matmul_mxfp8_performance/`: MXFP8 matrix multiplication example
  - `manual/common/`: Cross-platform kernels
    - `manual/common/flash_atten/`: Flash-Attention kernel (A2/A3/A5)
- `custom/`: Examples/scaffolding for custom kernel/operator extensions

## Notes

- Public interfaces live in `include/`; tests live in `tests/`.
- If you add a new kernel project here, prefer adding a small `README.md` and a `run.sh` so it can be discovered and executed consistently.
