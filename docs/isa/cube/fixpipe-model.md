# FIXPIPE Writeback Model

`FIXPIPE` is the cube core's dedicated writeback / post-processing pipeline. It moves `L0C` accumulator results out to `L1`, `UB`, or `GM` while applying the layout conversion (NZ → ND) required by the destination, plus optional dequantization / scale / clip / activation post-processing.

This page describes the FIXPIPE addressing model used by the three writeback ops:

- [`pto.mte_l0c_l1`](./ops/data-movement/mte-l0c-l1.md) — L0C → L1
- [`pto.mte_l0c_gm`](./ops/data-movement/mte-l0c-gm.md) — L0C → GM
- [`pto.mte_l0c_ub`](./ops/data-movement/mte-l0c-ub.md) — L0C → UB

## Source Layout

The L0C source tile is laid out as `N1 M1 M0 N0` (FRACTAL_NZ: col-major outer, row-major inner). FIXPIPE addresses one M-row of results at a time and emits them in the destination memory's natural order.

## NZ → ND Conversion at Writeback

For each cube fragment in L0C, FIXPIPE applies:

```text
C_nz[n1][m1][m0][n0]  -->  C_nd[m1*M0 + m0][n1*N0 + n0]
```

The conversion is **fused** with the writeback — no separate explicit transpose step is required. Destination strides are expressed in ND coordinates on the FIXPIPE op.

## Dual-Destination Broadcast (1 → 2 Cube-to-Vector)

When the FIXPIPE destination is a Vector block UB, the cube can simultaneously broadcast to both AIV0 and AIV1 UB regions via the dedicated on-chip data path, with the tile split either along the row axis (`DualModeSplitM`) or the column axis (`DualModeSplitN`):

| Split | AIV0 receives | AIV1 receives |
|-------|---------------|---------------|
| Split-M (rows) | Upper `[M/2, N]` in ND | Lower `[M/2, N]` in ND |
| Split-N (cols) | Left `[M, N/2]` in ND | Right `[M, N/2]` in ND |

This 1→2 broadcast with in-hardware tile split is the architectural basis for 1:2 Cube-to-Vector tile distribution and is selected as an attribute on `pto.mte_l0c_ub`.

## Burst / Loop Model

Like the [scalar DMA](../scalar/dma-copy.md) and [cube data-movement](./README.md#cube-data-movement-ops) ops, FIXPIPE writeback uses the grouped `nburst(...)` / `loop(...)` clause form to express row-stride and outer-stride repetition without external configuration registers.

## Post-Processing Hooks

FIXPIPE optionally applies the following post-processing along the writeback path, configured via clauses or auxiliary `FB` payload loaded by [`pto.mte_l1_fb`](./ops/data-movement/mte-l1-fb.md):

- Dequantization (per-channel scale / zero-point)
- Clip / saturate to destination element-type range
- Activation (ReLU / clipped linear, target-defined)

Per-op pages document which post-processing clauses each `pto.mte_l0c_*` variant accepts.

## Synchronization Around FIXPIPE

`FIXP` is one of the four AIC-side issue queues (alongside `MTE2`, `MTE1`, `CUBE`). The standard producer / consumer chain is:

```text
CUBE (pto.mad*)  --set_flag(CUBE -> FIXP)-->  FIXP (pto.mte_l0c_*)  -->  L1 / UB / GM
```

After a `pto.mad*` finishes a tile in L0C, the producer must `set_flag` from `PIPE_CUBE` to `PIPE_FIXP` (using one of the configured event IDs); the FIXPIPE consumer issues a matching `wait_flag` before issuing `pto.mte_l0c_*` against the same L0C tile. Failure to synchronize results in a read of in-flight L0C state and is a verifier error.

## Related Sections

- [NZ Fractal Layout](./nz-fractal-layout.md)
- [Buffer Hierarchy](./buffer-hierarchy.md)
- [Pipeline Synchronization](../scalar/ops/pipeline-sync/)
