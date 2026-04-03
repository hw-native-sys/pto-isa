# tests/gpu/st/

Standalone CUDA smoke / correctness tests for the PTO NVIDIA GPU backend.

## What it covers today

Current executables:

- `pto_gpu_core` — correctness / smoke checks
- `pto_gpu_perf` — lightweight GB10 matmul microbench

Current checks:

- `TLOAD` ND row-major path
- `TLOAD` DN col-major path
- `TSTORE` ND row-major path
- `TSTORE` DN col-major path
- `TADD` correctness against a host reference
- GPU swizzle physical-layout smoke test
- GPU swizzle `TLOAD`/`TSTORE` round-trip smoke test
- GPU swizzle `TADD` round-trip smoke test
- `sm121` float `TMATMUL` inline-PTX FMA fallback smoke test
- `sm121` half `TMATMUL` tensor-core MMA tiled-path smoke test
- `sm121` bfloat16 `TMATMUL` tensor-core MMA tiled-path smoke test
- `sm121` half `TMATMUL` larger 64x64x64 tensor-core correctness test
- `sm121` bfloat16 `TMATMUL` larger 64x64x64 tensor-core correctness test
- `sm121` half `TMATMUL_ACC` tensor-core fast-path smoke test
- `sm121` bfloat16 `TMATMUL_BIAS` tensor-core fast-path smoke test
- `sm121` half `TMATMUL_MX` API-path smoke test
- `sm121` half `TGEMV_MX` API-path smoke test

## Build

```bash
cmake -S tests/gpu/st -B build/tests/gpu-st   -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc   -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build build/tests/gpu-st -j
```

## Run

Correctness lane:

```bash
cd build/tests/gpu-st
ctest --output-on-failure
```

Perf microbench:

```bash
./build/tests/gpu-st/testcase/pto_gpu_perf/pto_gpu_perf
```

## Notes

- This lane is intentionally lightweight and self-contained.
- It uses CTest directly instead of the repo's CPU/NPU test harnesses.
- `sm121` now has a real warp-level tensor-core MMA path for half/bfloat16 using 16x16x16 MMA tiles, composed into larger 16-aligned matrix tiles in software.
- `TMATMUL_ACC` and `TMATMUL_BIAS` now reuse the same tensor-core tiled path on supported `sm121` shapes.
- float matmul currently uses the lighter inline-PTX FMA fallback path.
