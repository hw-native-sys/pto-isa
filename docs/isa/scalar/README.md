# Scalar and Control Reference

`pto.*` scalar and control instructions manage synchronization, DMA, predicates, control flow, and shared scalar support logic. They provide the execution shell around tile and vector payload regions.

## Organization

The scalar reference is organized by instruction family, with individual per-op pages under `scalar/ops/`.

## Instruction Families

| Family | Description | Operations |
|--------|-------------|-----------|
| [Control and Configuration](./control-and-configuration.md) | NOP, barrier, yield; tsetf32mode, tsethf32mode, tsetfmatrix | `nop`, `barrier`, `yield`, etc. |
| [PTO Micro-Instruction Reference](./ops/micro-instruction/README.md) | Scalar micro-instructions: BlockDim, pointer ops, vector scope, alignment state | `pto.get_block_idx`, `pto.castptr`, `pto.vecscope`, etc. |
| [Pipeline Sync](./pipeline-sync.md) | Event-based synchronization between pipes | `set_flag`, `wait_flag`, `wait_flag_dev`, `pipe_barrier`, `mem_bar`, `get_buf`, `rls_buf`, `set_cross_core`, `set_intra_block`, `wait_intra_core` |
| [DMA Copy](./dma-copy.md) | GM↔UB and UB↔UB data movement | `copy_gm_to_ubuf`, `copy_ubuf_to_gm`, `copy_ubuf_to_ubuf`, loop size/stride setters |
| [Predicate Load Store](./predicate-load-store.md) | Predicate-aware scalar load/store | `pld`, `plds`, `pldi`, `psts`, `pst`, `psti`, `pstu` |
| [Predicate Generation and Algebra](./predicate-generation-and-algebra.md) | Predicate construction and logic | `pset_b8/b16/b32`, `pge_b8/b16/b32`, `plt_b8/b16/b32`, `pand`, `por`, `pxor`, `pnot`, `psel`, `ppack`, `punpack`, `pdintlv_b8`, `pintlv_b16` |
| [Shared Arithmetic](./shared-arith.md) | Scalar arithmetic shared across instruction sets | Scalar arithmetic ops |
| [Shared SCF](./shared-scf.md) | Scalar structured control flow | `scf.for`, `scf.if`, `scf.while` |

## Common Constraints

- Pipe / event spaces are constrained by the target profile.
- DMA parameters must be self-consistent.
- Predicate widths and control parameters must match the target operation.
- Ordering edges must align with subsequent tile / vector payloads.

## Key Architectural Concepts

### Pipe Types

| Pipe | Role |
|------|------|
| `PIPE_V` | Vector pipeline |
| `PIPE_MTE1` | Memory transfer engine 1 (GM↔UB inbound) |
| `PIPE_MTE2` | Memory transfer engine 2 (UB↔tile buffer inbound) |
| `PIPE_MTE3` | Memory transfer engine 3 (tile buffer↔UB↔GM outbound) |
| `PIPE_CUBE` | Cube/matrix multiply unit |

### Event Synchronization

Events (`event_t`) coordinate asynchronous operations across pipes. Programs set flags (`set_flag`) from one pipe and wait on them from another (`wait_flag`).

## See Also

- [Scalar and control instruction surface](../instruction-surfaces/scalar-and-control-instructions.md) — High-level description
- [Scalar and control instruction families](../instruction-families/scalar-and-control-families.md) — Normative contracts
- [Format of instruction descriptions](../reference/format-of-instruction-descriptions.md) — Per-op page format standard
