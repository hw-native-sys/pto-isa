<!-- Generated from `docs/isa/instruction-families/other-families.md` -->

# Other Families

Other-family documentation covers communication and residual supporting behavior that is architecture-visible but does not fit cleanly into the main tile, vector, or scalar/control buckets.

## Overview

| Family | Description | Availability |
|--------|-------------|------------|
| [Communication and Runtime](./other/communication-and-runtime.md) | Inter-NPU collective communication | A2/A3, A5 |
| [Non-ISA Supporting Ops](./other/non-isa-and-supporting-ops.md) | Convenience operations over tile sequences | All profiles |

### Communication and Runtime

These operations span multiple NPUs in a parallel group and require a `ParallelGroup` handle:

| Category | Operations |
|----------|-----------|
| Collective broadcast | `tbroadcast`, `tscatter`, `tgather` |
| Point-to-point | `tget`, `tget_async`, `tput`, `tput_async` |
| Collective reduction | `treduce` |
| Notification | `tnotify`, `ttest`, `twait` |

**CPU simulator**: These ops are **not available** on the CPU simulator. Programs using them on CPU will produce a runtime error.

### Non-ISA Supporting Operations

These provide higher-level semantics over tile sequences or memory management. Some are convenience wrappers that expand to multiple core ISA operations:

| Category | Operations |
|----------|-----------|
| Tile sequence | `talias`, `tconcat`, `taxpy` |
| Memory management | `tfree` |
| Quantization | `tquant`, `tdequant` |
| Counting | `tpop`, `tpush` |
| A5-only | `thistogram`, `tpack`, `trandom` |

## Shared Constraints

- **Communication ops** require all participating NPUs to call the operation with matching `ParallelGroup` handles.
- **Non-root ranks** for collective ops must have destination buffers allocated and writable for the operation duration.
- **CPU simulator** does not support communication ops.

## Navigation

See the [Other ISA reference](./other/README.md) for the full per-op reference.

## See Also

- [Other instruction surface](./instruction-surfaces/other-instructions.md) — High-level surface description
- [Instruction families](./README.md) — All family groups
- [Format of instruction descriptions](../reference/format-of-instruction-descriptions.md) — Per-op page standard
