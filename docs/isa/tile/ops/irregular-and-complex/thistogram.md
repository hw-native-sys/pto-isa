# THISTOGRAM

## Tile Operation Diagram

![THISTOGRAM tile operation](../../../../figures/isa/THISTOGRAM.svg)

## Summary

`pto.thistogram` accumulates histogram bin counts from a source tile into a destination bin tile. Each source element is mapped to a bin index (either directly, when the source is integer indices, or via a quantisation step on the backend) and the corresponding bin counter is incremented. This op is **A5-only** and is exposed under the `Statistics` category of the irregular-and-complex group.

## Mechanism

For every source element `src_i` (within the valid region), let `b = bin(src_i)` be its bin index. The destination bin tile is updated by:

$$\mathrm{dst}_b \mathrel{+}= 1 \quad \forall\, i \in [0,\, R \cdot C)$$

The op executes on the irregular path: the vector pipe consumes the source tile and the FIXP / scalar-update path applies the per-bin increments. Because increments to the same bin are inherently serialised, the inner loop is throughput-limited by the bin-write port rather than by source bandwidth.

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

### IR Level 1 (SSA)

```text
%event = pto.thistogram %src, %dst_in : (!pto.tile<...>, !pto.tile<...>) -> !pto.record_event
```

### IR Level 2 (DPS)

```text
pto.thistogram ins(%src, %dst_in) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent THISTOGRAM(DstTileData &dst, SrcTileData &src, WaitEvents&... events);
```

## Inputs

| Operand | Description |
|---------|-------------|
| `src` | Source tile whose elements are mapped to bin indices and accumulated. |
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
    - `dst` element type must be a counter-friendly integer type (typically INT32).
    - The bin count (size of `dst`) must match the bin range produced by the source mapping; out-of-range bin indices are saturated or rejected per backend.
    - `dst` must be initialised by the caller (e.g. via `TFILL` or `TMOV`) before the first accumulation.
    - Refer to backend-specific legality checks for data type/layout/location/shape constraints not covered above.

## Performance

### A2/A3 Cycle Count

Not applicable — `pto.thistogram` is **A5-only**.

### A5 Cycle Count

The cost is dominated by per-element bin lookup and the serialised increment write-back. Throughput is approximately one source element per inner-loop cycle, with a startup cost for setting up the bin map.

**Cycle model**:

```
total ≈ startup + R × C / hist_throughput
```

The shape of `dst` (number of bins) does not change throughput, but a histogram with strong bin-locality (many increments to the same bin) can stall on the bin-write port.

> Note: cycle numbers are first-order estimates; populate with measured values from `pto-isa/a5_benchmark.csv`.

## Exceptions

!!! danger "Exceptions"
    - Lowering on A2/A3 backends is rejected by the verifier.
    - An out-of-range bin index produced by the source mapping is implementation-defined: the supported behavior is to saturate at the last bin; programs must not rely on wrap-around.
    - Programs must not rely on behavior outside the documented legal domain of this operation.

## Examples

See related instruction pages in `docs/isa/` for concrete Auto/Manual usage patterns.

## See Also

- Instruction set overview: [Irregular / Complex](../../irregular-and-complex.md)
- Previous: [pto.trandom](./trandom.md)
- Related stats / sort ops: [pto.tsort32](./tsort32.md), [pto.tmrgsort](./tmrgsort.md)
