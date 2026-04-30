# TAXPY

## Tile Operation Diagram

![TAXPY tile operation](../../../../figures/isa/TAXPY.svg)

## Summary

`pto.taxpy` is the tile-level form of the BLAS AXPY primitive. It computes `dst = src0 * scalar + src1` elementwise across the destination valid region in a single fused vector operation, avoiding the temporary tile that a separate TMULS + TADD pair would materialise.

## Mechanism

For every element index `i` in the destination valid region:

$$\mathrm{dst}_i = \alpha \cdot \mathrm{src0}_i + \mathrm{src1}_i$$

where $\alpha$ is the scalar argument. The op executes on the **vector pipe** (PIPE_V) as a fused multiply-add against a broadcast scalar, with no intermediate tile written back.

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

### IR Level 1 (SSA)

```text
%dst = pto.taxpy %src0, %scalar, %src1 : (!pto.tile<...>, <scalar>, !pto.tile<...>) -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.taxpy ins(%src0, %scalar, %src1) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename DstTileData, typename SrcTileData0, typename ScalarT, typename SrcTileData1,
          typename... WaitEvents>
PTO_INST RecordEvent TAXPY(DstTileData &dst, SrcTileData0 &src0, ScalarT scalar,
                           SrcTileData1 &src1, WaitEvents&... events);
```

## Inputs

| Operand | Description |
|---------|-------------|
| `src0` | Tile multiplied by `scalar`. |
| `scalar` | Broadcast scalar coefficient (same element type as `src0`/`src1`). |
| `src1` | Tile added to the scaled `src0`. |
| `events...` | Optional `RecordEvent` tokens to wait on before issuing. |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `RecordEvent` | token | Signals completion of the fused multiply-add. |
| `dst` | tile | Holds `src0 * scalar + src1` over the valid region; padding region is unmodified. |

## Side Effects

- Issues a single PIPE_V instruction; no DMA traffic.
- Does not implicitly fence unrelated tile traffic.
- Reads `scalar` once (broadcast); does not consume index resources.

## Constraints

!!! warning "Constraints"
    - `dst`, `src0`, `src1` must share the same shape, layout, and element type.
    - `scalar` must be an arithmetic type compatible with the tile element type (no implicit narrowing conversions).
    - `dst` may alias `src0` or `src1` (in-place fused update is supported).
    - Refer to backend-specific legality checks for data type/layout/location/shape constraints not covered above.

## Performance

### A2/A3 Cycle Count

`pto.taxpy` runs on the **vector pipe** as a fused multiply-add. Throughput matches PIPE_V for the target element type — the same as the underlying TMULS or TADD, but without the temporary tile and the intervening write-back.

**Cycle model**:

```
total ≈ startup + R × C / vfmla_throughput
```

where `R × C` is the destination valid shape and `vfmla_throughput` is the per-cycle FP/INT FMA width on the vector unit.

### Layout and Shape Impact

| Element type | Path | Notes |
|--------------|------|-------|
| FP32 | PIPE_V | Native FMA; one round of rounding |
| FP16 / BF16 | PIPE_V | FP32-accumulate FMA available; check backend |
| INT32 / INT16 | PIPE_V | Integer mul-add; saturation depends on variant |

Replacing a TMULS + TADD pair with a single `taxpy` halves the vector-pipe issues for the same work and removes one tile of intermediate UB pressure.

> Note: cycle numbers are first-order estimates; populate with measured values from `pto-isa/a2a3_benchmark.csv` and `pto-isa/a5_benchmark.csv`.

## Exceptions

!!! danger "Exceptions"
    - Mismatched element types between `src0`, `src1`, and `dst` are rejected at compile time via `static_assert`.
    - A `scalar` whose type cannot be broadcast-converted to the tile element type is rejected at compile time.
    - Programs must not rely on behavior outside the documented legal domain of this operation.

## Examples

```cpp
// dst = 0.5f * src0 + src1, fused, in-place into src1's storage when desired.
TAXPY(dst, src0, 0.5f, src1);
```

See related instruction pages in `docs/isa/` for concrete Auto/Manual usage patterns.

## See Also

- Instruction set overview: [Tile-Scalar / Immediate](../../tile-scalar-and-immediate.md)
- Previous: [pto.tadds](./tadds.md)
- Next: [pto.tsubs](./tsubs.md)
- Non-fused composition: [pto.tmuls](./tmuls.md) + [pto.tadds](./tadds.md)
