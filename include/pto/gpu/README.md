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

This is currently a **planning scaffold** only.

Not implemented yet:

- backend dispatch macros
- inclusion from `pto_instr_impl.hpp`
- CUDA build plumbing
- real instruction implementations

See `docs/gpu_backend_plan.md` for the phase plan and backend bring-up checklist.
