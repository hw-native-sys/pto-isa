# THISTOGRAM

## Tile Operation Diagram

![THISTOGRAM tile operation](../../../../figures/isa/THISTOGRAM.svg)

## Summary

`pto.thistogram` accumulates radix-style histogram bin counts from a source tile into a destination bin tile, selecting **one byte lane** of each multi-byte source element as the bin index. The selected byte is chosen by the `HistByte byte` template parameter (`BYTE_0`/`BYTE_1` for `uint16_t`, plus `BYTE_2`/`BYTE_3` for wider types). For non-MSB passes a separate `idx` tile is consumed to mask the contribution of each row, so `pto.thistogram` chains naturally into a multi-pass radix sort. This op is **A5-only** and is exposed under the `Statistics` category of the irregular-and-complex group.

## Mechanism

For every source element `src_i` (within the valid region), let `b = extract_byte<byte>(src_i)` be the bin index extracted from the selected byte lane. The destination bin tile is updated by:

$$\mathrm{dst}_b \mathrel{+}= 1 \quad \forall\, i \in [0,\, R \cdot C)$$

When `byte` selects a non-MSB lane (e.g. `BYTE_0`), the per-row `idx` tile is used to gate which previously bucketed rows still contribute, enabling multi-pass MSB-→LSB radix passes. When `byte` selects the MSB (e.g. `BYTE_1` for `uint16_t`), `idx` is unused but must still be supplied as a valid operand.

The op executes on the irregular path: the vector pipe consumes the source tile and the FIXP / scalar-update path applies the per-bin increments. Because increments to the same bin are inherently serialised, the inner loop is throughput-limited by the bin-write port rather than by source bandwidth.

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

### IR Level 1 (SSA)

```text
%event = pto.thistogram %src, %idx, %dst_in {byte = #pto.hist_byte<byte_n>}
          : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>) -> !pto.record_event
```

### IR Level 2 (DPS)

```text
pto.thistogram {byte = #pto.hist_byte<byte_n>}
    ins(%src, %idx) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp` (gated to A5 / CPU simulator):

```cpp
template <HistByte byte,
          typename TileDataDst, typename TileDataSrc, typename TileDataIdx,
          typename... WaitEvents>
PTO_INST RecordEvent THISTOGRAM(TileDataDst &dst, TileDataSrc &src, TileDataIdx &idx,
                                WaitEvents&... events);
```

`HistByte` is defined in `include/pto/common/type.hpp` and selects which byte lane of each source element is histogrammed (`BYTE_0` is the LSB, `BYTE_3` the MSB).

## Inputs

| Operand | Description |
|---------|-------------|
| `byte` (template) | `HistByte` enum selecting the byte lane used as the bin index. Must satisfy the source-type constraint (e.g. only `BYTE_0`/`BYTE_1` for `uint16_t`). |
| `src` | Source tile whose elements supply the bin index byte. |
| `idx` | Per-row index/mask tile that gates contributions for non-MSB passes; supplied (and typically loaded) but unused on the MSB pass. Element type is `uint8_t`. |
| `events...` | Optional `RecordEvent` tokens to wait on before issuing. |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `RecordEvent` | token | Signals completion of the histogram update. |
| `dst` | tile | Holds the running per-bin counts. The op **adds** to existing values; clear `dst` first to obtain a fresh histogram. |

## Side Effects

- Reads `src`, accumulates into `dst` in place — `dst` is **not** cleared by the op.
- No DMA traffic; runs entirely within the AI core.
- Does not implicitly fence unrelated tile traffic.

## Constraints

!!! warning "Constraints"
    - Backend support: **A5 only**. A2/A3 reject this op at lowering time.
    - `dst` element type must be a counter-friendly integer type (`uint32_t` in the reference path) and must contain 256 bins per row (one per possible byte value).
    - `src` element type determines which `HistByte` values are legal: `uint16_t` accepts `BYTE_0`/`BYTE_1`; wider types extend the set up to `BYTE_3`. A `static_assert` rejects illegal `byte` selections.
    - `idx` is a per-row `uint8_t` tile aligned to `BLOCK_BYTE_SIZE`; it must be loaded for non-MSB passes and may be left uninitialised but still passed for the MSB pass.
    - `dst` must be initialised by the caller (e.g. via `TASSIGN`, `TFILL`, or `TMOV`) before the first accumulation.
    - Refer to backend-specific legality checks for data type/layout/location/shape constraints not covered above.

## Exceptions

!!! danger "Exceptions"
    - Lowering on A2/A3 backends is rejected by the verifier.
    - A `HistByte` value that is not legal for the source element type (e.g. `BYTE_2`/`BYTE_3` on `uint16_t`) is rejected at compile time via `static_assert`.
    - Programs must not rely on behavior outside the documented legal domain of this operation.

## Examples

```cpp
// MSB radix pass over a uint16_t tile: idx is unused but still required.
THISTOGRAM<HistByte::BYTE_1>(dstTile, srcTile, idxTile);

// Subsequent LSB pass: idx must be loaded with the per-row mask first.
TLOAD(idxTile, idxGlobal);
THISTOGRAM<HistByte::BYTE_0>(dstTile, srcTile, idxTile);
```

## See Also

- Instruction set overview: [Irregular / Complex](../../irregular-and-complex.md)
- Previous: [pto.trandom](./trandom.md)
- Related stats / sort ops: [pto.tsort32](./tsort32.md), [pto.tmrgsort](./tmrgsort.md)
