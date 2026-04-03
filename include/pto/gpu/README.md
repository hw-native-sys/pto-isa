# include/pto/gpu/

Scaffold for a future NVIDIA GPU backend for PTO Tile Lib.

## Purpose

This directory mirrors the existing backend split used by the CPU simulator and Ascend NPU implementations:

- `common/`: shared GPU helpers, tile adapters, launch abstractions, layout helpers
- `smXX/`: architecture-specific specializations and microkernel registries

## Design intent

The long-term goal is to provide:

- a unified PTO API at `#include <pto/pto-inst.hpp>`
- a GPU lowering path that keeps PTO semantics stable
- per-SM specializations for performance-critical PTO instructions
- dedicated microkernels for hot PTO ISA operations on NVIDIA GPUs

## Initial status

This backend is now past pure scaffold stage.

Implemented groundwork includes:

- backend dispatch macros
- inclusion from `pto_instr_impl.hpp`
- initial CUDA build/test plumbing
- early `sm121` matmul fast paths
- a GPU-specific swizzle layout family for row-major tiles

Current GPU swizzle support:

- layout enum: `SLayout::GpuSwizzle128B`
- aliases in `pto_tile.hpp`:
  - `TileVecGpuSwizzle`
  - `TileLeftGpuSwizzle`
  - `TileRightGpuSwizzle`
  - `TileAccGpuSwizzle`
- element offset mapping is GPU-specific and intentionally not tied to the NPU boxed layouts
- current swizzle implementation is row-major only and is intended as groundwork for future shared-memory / tensor-core friendly paths

See `docs/gpu_backend_plan.md` for the phase plan and backend bring-up checklist.
