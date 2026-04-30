# TPACK

## Tile Operation Diagram

![TPACK tile operation](../../../../figures/isa/TPACK.svg)

## Summary

`pto.tpack` packs (or narrows) tile elements into a more compact destination representation — for example, packing two FP16 lanes into a single FP32 word, or compressing INT16 lanes into INT8. It is the dual of `pto.tunpack` and is commonly used to prepare quantised activations for subsequent layout-converting moves.

## Mechanism

For every destination element index `i` in the valid region, `tpack` consumes one or more contiguous source lanes and writes a packed word:

$$\mathrm{dst}_i = \mathrm{Pack}(\mathrm{src}_{k \cdot i},\, \dots,\, \mathrm{src}_{k \cdot i + k - 1})$$

where $k$ is the packing ratio set by `(SrcDType, DstDType)`. The op runs on the **vector pipe** (PIPE_V) as a structured shuffle/narrow.

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

### IR Level 1 (SSA)

```text
%dst, %event = pto.tpack %src : (!pto.tile<srcType>) -> (!pto.tile<dstType>, !pto.record_event)
```

### IR Level 2 (DPS)

```text
%event = pto.tpack ins(%src) outs(%dst : !pto.tile_buf<...>) -> !pto.record_event
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TPACK(DstTileData &dst, SrcTileData &src, WaitEvents&... events);
```

## Inputs

| Operand | Description |
|---------|-------------|
| `src` | Source tile carrying the wider/looser element type. |
| `events...` | Optional `RecordEvent` tokens to wait on before issuing. |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `RecordEvent` | token | Signals completion of the pack. |
| `dst` | tile | Holds the packed/narrowed representation over the valid region. |

## Side Effects

- Issues a single PIPE_V instruction; no DMA traffic.
- Does not implicitly fence unrelated tile traffic.

## Constraints

!!! warning "Constraints"
    - `(SrcDType, DstDType)` must be a supported packing pair on the target backend; unsupported pairs are rejected by `static_assert`.
    - The destination element count must equal `src` element count divided by the packing ratio `k`.
    - `dst` and `src` may not alias for non-1:1 packing ratios; use a separate buffer.
    - Refer to backend-specific legality checks for data type/layout/location/shape constraints not covered above.

## Performance

### A2/A3 Cycle Count

`pto.tpack` runs on the **vector pipe** as a narrow/shuffle. Cost scales with the source element count, since the pack is throughput-bound by the source read width.

**Cycle model**:

```
total ≈ startup + (Rsrc × Csrc) / vpack_throughput
```

### Layout and Shape Impact

| (Src → Dst) | Ratio `k` | Path |
|-------------|-----------|------|
| FP32 → FP16 | 2:1 | PIPE_V narrow |
| FP16 → INT8 | 2:1 | PIPE_V narrow |
| INT16 → INT8 | 2:1 | PIPE_V narrow |
| INT32 → INT16 | 2:1 | PIPE_V narrow |

A `tpack` followed by a layout-converting move is the canonical path for producing quantised activations destined for INT8 GEMM.

> Note: cycle numbers are first-order estimates; populate with measured values from `pto-isa/a2a3_benchmark.csv` and `pto-isa/a5_benchmark.csv`.

## Exceptions

!!! danger "Exceptions"
    - Unsupported `(SrcDType, DstDType)` pairs are rejected at compile time via `static_assert`.
    - Aliasing `dst` and `src` for ratios `k > 1` is undefined behavior.
    - Programs must not rely on behavior outside the documented legal domain of this operation.

## Examples

See related instruction pages in `docs/isa/` for concrete Auto/Manual usage patterns.

## See Also

- Instruction set overview: [Layout and Rearrangement](../../layout-and-rearrangement.md)
- Previous: [pto.tconcat](./tconcat.md)
- Next: [pto.textract](./textract.md)
- Related quantization op: [pto.tquant](../irregular-and-complex/tquant.md)
