# PTO Tile Intrinsics Programming Model

PTO Tile Lib provides **tile-granularity** intrinsics that map to the PTO ISA. The model is designed for:

- **Portability across device generations**: hardware may change (instruction details, storage layout, scheduling), but the programming model remains stable.
- **Near-hardware performance**: the Tile and GlobalTensor abstractions are low-level enough to express efficient data movement and compute.
- **Two user profiles**: a productive “compiler does the hard work” style, and an expert “I control placement and sync” style.

For the abstract execution model (core/device/host), see `docs/en/machine/abstract-machine.md`.

## Core concepts

- **Tile**: a fixed-capacity 2-D on-chip buffer (conceptually a tile register / SRAM block) and the unit of computation for most PTO instructions. See `docs/en/coding/Tile.md`.
- **GlobalTensor**: a lightweight view of global memory (GM) as a 5-D tensor with shape/stride/layout metadata, used by memory instructions such as `TLOAD` and `TSTORE`. See `docs/en/coding/GlobalTensor.md`.
- **Scalar**: immediate values and enumerations that parameterize instructions (rounding modes, comparison modes, atomic modes, etc.). See `docs/en/coding/Scalar.md`.
- **Event**: explicit dependency tokens between pipeline classes, used to order operations without introducing a full barrier everywhere. See `docs/en/coding/Event.md`.

## Two development styles

### PTO-Auto

PTO-Auto targets developers who prefer a simple, portable programming experience:

- The compiler/runtime chooses memory placement and address binding.
- The compiler inserts required synchronization.
- The compiler schedules operations and applies fusions when possible.

This mode is a good starting point for correctness and portability.

### PTO-Manual

PTO-Manual targets developers who need full control for performance tuning:

- The developer controls memory placement and binding (for example via `TASSIGN`).
- The developer explicitly expresses ordering with events or low-level flags.
- The developer controls the operation schedule.

This mode enables expert tuning on critical kernels while still using the shared Tile/GlobalTensor abstractions.

## Execution models: SPMD and MPMD

PTO supports both **SPMD** and **MPMD** execution models.

These execution models describe **how work is mapped onto cores**. They are orthogonal to the **Auto vs Manual**
development styles (you can write SPMD-Auto, SPMD-Manual, MPMD-Auto, or MPMD-Manual code).

### SPMD (Single Program, Multiple Data)

In SPMD, all participating cores run the same entry function, and each core selects its own data region using its
runtime identity (for example `block_idx`).

When sub-block decomposition exists, a stable “virtual id” can be constructed:

```cpp
auto cid = get_block_idx();
auto vid = get_block_idx() * get_subblockdim() + get_subblockid();
```

SPMD is a good fit for regular tensor tiling (GEMM, softmax-by-rows, elementwise ops).

### MPMD (Multiple Program, Multiple Data)

In MPMD, different cores (or groups of cores) may execute **different tile programs** as part of the same overall
tile graph. Conceptually, the **Device Machine scheduler** chooses which “program” a core runs.

One portable way to express this is to pass a scheduler-provided **task id** into the kernel entry function and
dispatch based on it:

```cpp
__global__ __aicore__ void KernelMPMD(__gm__ float* out,
                                     __gm__ const float* in,
                                     uint32_t task_id) {
  switch (task_id) {
    case 0: return ProducerStage(out, in);
    case 1: return ConsumerStage(out, in);
    default: return;
  }
}
```

Notes:

- The exact mechanism that delivers `task_id` is platform/runtime dependent; the abstract model only requires that
  the Device Machine can schedule different tile blocks onto available cores.
- If you prefer, MPMD can also be expressed as **multiple entry points** (multiple kernels) rather than a single
  kernel with a `switch`.
