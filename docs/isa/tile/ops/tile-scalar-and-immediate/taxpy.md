# TAXPY

## Tile Operation Diagram

![TAXPY tile operation](../../../../figures/isa/TAXPY.svg)

## Summary

`pto.taxpy` is the tile-level form of the BLAS AXPY primitive. It updates `dst` in place with `dst = dst + src0 * scalar` elementwise across the destination valid region in a single fused vector operation, avoiding the temporary tile that a separate TMULS + TADD pair would materialise.

## Mechanism

For every element index `i` in the destination valid region:

$$\mathrm{dst}_i \mathrel{+}= \alpha \cdot \mathrm{src0}_i$$

where $\alpha$ is the scalar argument. The op executes on the **vector pipe** (PIPE_V) as a fused multiply-add against a broadcast scalar, accumulating into `dst` with no intermediate tile written back.

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

### IR Level 1 (SSA)

```text
%dst, %event = pto.taxpy %dst_in, %src0, %scalar : (!pto.tile<...>, !pto.tile<...>, <scalar>) -> (!pto.tile<...>, !pto.record_event)
```

### IR Level 2 (DPS)

```text
%event = pto.taxpy ins(%src0, %scalar) outs(%dst : !pto.tile_buf<...>) -> !pto.record_event
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TAXPY(TileDataDst &dst, TileDataSrc &src0,
                           typename TileDataSrc::DType scalar,
                           WaitEvents&... events);
```

## Inputs

| Operand | Description |
|---------|-------------|
| `dst` | Accumulator tile; read and written in place (`dst += src0 * scalar`). |
| `src0` | Tile multiplied by `scalar`. |
| `scalar` | Broadcast scalar coefficient with element type `TileDataSrc::DType`. |
| `events...` | Optional `RecordEvent` tokens to wait on before issuing. |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `RecordEvent` | token | Signals completion of the fused multiply-add. |
| `dst` | tile | Holds `dst + src0 * scalar` over the valid region; padding region is unmodified. |

## Side Effects

- Issues a single PIPE_V instruction; no DMA traffic.
- Does not implicitly fence unrelated tile traffic.
- Reads `scalar` once (broadcast); does not consume index resources.

## Constraints

!!! warning "Constraints"
    - `dst` and `src0` must share the same shape, layout, and element type.
    - `scalar` must match `TileDataSrc::DType`; no implicit narrowing conversions are performed.
    - `dst` is read-modify-written in place; the prior contents of `dst` are part of the result.
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
    - Mismatched element types between `src0` and `dst` are rejected at compile time via `static_assert`.
    - A `scalar` whose type cannot be broadcast-converted to the tile element type is rejected at compile time.
    - Programs must not rely on behavior outside the documented legal domain of this operation.

## Examples

```cpp
// dst += 0.5f * src0, fused multiply-accumulate in place.
TAXPY(dst, src0, 0.5f);
```

See related instruction pages in `docs/isa/` for concrete Auto/Manual usage patterns.

## See Also

- Instruction set overview: [Tile-Scalar / Immediate](../../tile-scalar-and-immediate.md)
- Previous: [pto.tadds](./tadds.md)
- Next: [pto.tsubs](./tsubs.md)
- Non-fused composition: [pto.tmuls](./tmuls.md) + [pto.tadds](./tadds.md)
