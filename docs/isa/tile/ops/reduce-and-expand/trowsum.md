# pto.trowsum

`pto.trowsum` is part of the [Reduce And Expand](../../reduce-and-expand.md) instruction set.

## Summary

Reduce each row of a source tile by summing all elements in that row, producing a column vector of row sums.

## Mechanism

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. For each row `i` from `0` to `R-1`:

$$ \mathrm{dst}_{i,0} = \sum_{j=0}^{C-1} \mathrm{src}_{i,j} $$

The result tile has the same number of rows as the source and one column. The `tmp` tile provides scratch storage for the reduction tree; its shape and layout are constrained by the implementation.

## Syntax

### PTO Assembly Form

```text
%dst = trowsum %src : !pto.tile<...> -> !pto.tile<...>
```

Note: Lowering may introduce internal scratch tiles. The C++ intrinsic requires an explicit `tmp` operand.

### AS Level 1 (SSA)

```text
%dst = pto.trowsum %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.trowsum ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>)
          outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWSUM(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents &... events);
```

## Inputs

| Operand | Description |
|---------|-------------|
| `src` | Source tile. Must be `TileType::Vec`. Must use standard ND layout (row-major, non-fractal). |
| `tmp` | Temporary scratch tile. Used for intermediate reduction storage. Shape and layout constraints are enforced by the implementation. |
| `dst` | Destination tile. Must be `TileType::Vec`. Must have `dst.GetValidRow() == src.GetValidRow()`. |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `RecordEvent` | `RecordEvent` | Token signaling completion of the reduction |
| `dst` | tile | Row sums: `dst[i,0]` = sum of all elements in row `i` of `src` |

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated tile traffic.

## Constraints

### Tile Types

- `src` and `dst` must both be `TileType::Vec`.

### Layout

- `src` must use standard ND layout: `BLayout::RowMajor`, `SLayout::NoneBox`.
- `dst` must use one of:
  - ND layout: `BLayout::RowMajor`, `SLayout::NoneBox`, `Cols == 1`, or
  - DN layout: `BLayout::ColMajor`, `SLayout::NoneBox`, `Cols == 1`.
- `src` and `dst` must have the same element type.

### Valid Region

- `src.GetValidRow() > 0`
- `src.GetValidCol() > 0`
- `dst.GetValidRow() == src.GetValidRow()`

### Element Types

Supported: `half`, `float`, `int32_t`, `int16_t`.

## Performance

### A2/A3 Cycle Count

`TROWSUM` compiles to a multi-phase CCE instruction sequence. The `TRowReduceOp.hpp` header determines the instruction sequence based on tile geometry.

**Cycle model**:

```
total = startup + Σ(completion_i) + Σ(repeats_i × per_repeat_i) + Σ((repeats_i - 1) × interval)
```

### Instruction Sequence by Shape (FP32)

| Valid Shape | Instruction Sequence | Estimated Cycles |
|-------------|---------------------|------------------|
| 64×128 | `vcgadd`*128 → `vadd`*8 → `vcgadd`*8 → PIPE_V | ~O(1024) |
| 32×256 | `vcgadd`*128 → `vadd`*8 → `vadd`*4 → `vcgadd`*4 → PIPE_V | ~O(2048) |
| 16×512 | `vcgadd`*128 → `vcgadd`*16 → `vcgadd`*2 → PIPE_V | ~O(2048) |
| 8×1024 | `vcgadd`*128 → `vcgadd`*16 → `vadd`*8 → `vcgadd`*8 → PIPE_V | ~O(2048) |

### General Shape Algorithm

For non-special shapes or non-FP32 types:

1. **Fill phase**: `copy_ubuf_to_ubuf` to initialize tmp (if `validCol >= 2 × 8`)
2. **Loop-fill**: For each row, apply `vadd`/`vmax`/`vmin` with per-row repeats
3. **Merge phase**: `vadd`/`vmax`/`vmin` per row again
4. **Final reduction**: `vcadd`/`vcmax`/`vcmin` with `PIPE_V` barrier

### Layout and Shape Impact

| Layout | validCol | Optimization |
|--------|----------|-------------|
| `RowMajor` | ≥ 16 (FP32) | Continuous fast path |
| `RowMajor` | < 16 | General path with tail masking |
| `ColMajor` | any | General path |
| `Zigzag` | any | General path |

Integer types (int16_t/int32_t): Use simplified path with direct `vadd`/`vmax`/`vmin` per block — no tree reduction.

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
- Programs must not rely on behavior outside the documented legal domain.

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TROWSUM(dst, src, tmp);
}
```

## See Also

- Instruction set overview: [Reduce And Expand](../../reduce-and-expand.md)
- Next op in instruction set: [pto.tcolsum](./tcolsum.md)
