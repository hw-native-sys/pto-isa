<!-- Generated from `docs/isa/instruction-families/scalar-and-control-families.md` -->

# Scalar And Control Families

Scalar and control family documentation covers the state-setting and control-shell parts of PTO that surround tile and vector payload execution.

## Overview

| Family | Description | Examples |
|--------|-------------|----------|
| [Control and Configuration](./scalar/control-and-configuration.md) | NOP, barrier, yield, and control setup | `nop`, `barrier`, `yield` |
| [Pipeline Sync](./scalar/pipeline-sync.md) | Event and barrier synchronization between pipelines | `set_flag`, `wait_flag`, `pipe_barrier`, `mem_bar` |
| [DMA Copy](./scalar/dma-copy.md) | GM↔UB memory transfer configuration and initiation | `copy_gm_to_ubuf`, `copy_ubuf_to_gm`, `set_loop_size_outtoub` |
| [Predicate Load/Store](./scalar/predicate-load-store.md) | Mask-based scalar memory access | `pld`, `plds`, `pdi`, `pst`, `psts`, `psti`, `pstu` |
| [Predicate Generation](./scalar/predicate-generation-and-algebra.md) | Predicate construction and algebra | `pset_b8`, `pge_b8`, `plt_b8`, `pand`, `por`, `pxor`, `pnot` |
| Shared Arithmetic | Scalar arithmetic ops shared across surfaces | Scalar integer/float ops |
| Shared SCF | Scalar structured control flow | Loops, conditionals |

## Shared Constraints

All scalar/control families must state:

1. **Architectural state produced or consumed** — What state the operations create or modify.
2. **Pipe and event spaces** — Which pipe/event identifiers are supported by the target profile.
3. **Target-profile narrowing** — Where A2/A3 and A5 differ from the portable ISA contract.
4. **Cases that are not allowed** — Conditions that are illegal across the family.

## Shared Dialect Surfaces

Some scalar/control ops belong to shared dialect surfaces (e.g., `scf.if`, `scf.for`) that extend the core ISA with structured control flow. These ops are marked as part of the documented PTO source surface, not as PTO-specific mnemonics.

## Pipe Spaces by Profile

| Pipe | CPU Sim | A2/A3 | A5 |
|------|:-------:|:------:|:--:|
| `PIPE_MTE1` | Simulated | Supported | Supported |
| `PIPE_MTE2` | Simulated | Supported | Supported |
| `PIPE_MTE3` | Simulated | Supported | Supported |
| `PIPE_V` | Emulated | Emulated | Native |
| `PIPE_M` | Simulated | Supported | Supported |

## Event Ordering

Scalar/control sync ops use a matching `set_flag`/`wait_flag` protocol:

```
Producer:  set_flag(src_pipe=PIPE_MTE2, dst_pipe=PIPE_V, event_id=EID0)
Consumer:  wait_flag(src_pipe=PIPE_MTE2, dst_pipe=PIPE_V, event_id=EID0)
```

Waiting on an event that was never established by a matching producer is **illegal**.

## Navigation

See the [Scalar ISA reference](./scalar/README.md) for the full per-op reference under `scalar/ops/`.

## See Also

- [Scalar and control instruction surface](./instruction-surfaces/scalar-and-control-instructions.md) — High-level surface description
- [Instruction families](./README.md) — All family groups
- [Format of instruction descriptions](../reference/format-of-instruction-descriptions.md) — Per-op page standard
