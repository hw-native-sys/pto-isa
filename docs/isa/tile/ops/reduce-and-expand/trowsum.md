# pto.trowsum

`pto.trowsum` is part of the [Reduce And Expand](../../reduce-and-expand.md) instruction set.

## Summary

Reduce each row by summing across columns. For each row `i` in the valid region, all `C` elements in that row are summed and written to `dst[i, 0]`.

## Mechanism

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. For `0 <= i < R`:

$$ \mathrm{dst}_{i,0} = \sum_{j=0}^{C-1} \mathrm{src}_{i,j} $$

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = trowsum %src : !pto.tile<...> -> !pto.tile<...>
```

Lowering may introduce internal scratch tiles; the C++ intrinsic requires an explicit `tmp` operand.

### AS Level 1 (SSA)

```text
%dst = pto.trowsum %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.trowsum ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWSUM(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents &... events);
```

## Inputs

- `src` is the source tile.
- `tmp` is a temporary tile used for intermediate storage.
- `dst` names the destination tile. The operation iterates over dst's valid region.

## Expected Outputs

`dst` holds the row-wise sum: for each row `i`, `dst[i,0]` = sum of all elements in row `i` of `src`.

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

### General constraints / checks

- `dst` and `src` must both be `TileType::Vec`.

- `src` must use standard ND layout: row-major and non-fractal (`BLayout::RowMajor`, `SLayout::NoneBox`).

- `dst` must use one of the following non-fractal layouts:
  - ND layout (`BLayout::RowMajor`, `SLayout::NoneBox`), or
  - DN layout with exactly one column (`BLayout::ColMajor`, `SLayout::NoneBox`, `Cols == 1`).

- `dst` and `src` must use the same element type.

- Runtime valid-region checks:
  - `src.GetValidRow() != 0`
  - `src.GetValidCol() != 0`
  - `src.GetValidRow() == dst.GetValidRow()`

- Supported element types: `half`, `float`, `int32_t`, `int16_t`.

- The implementation accepts both ND output and DN output with `Cols == 1`; it is not limited to DN output.

- The current implementation path passes `tmp` into the backend call, but this document does not add extra `tmp` shape/layout constraints beyond what is explicitly enforced by the checked implementation.

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
- Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

- The intrinsic signature requires an explicit `tmp` operand.

## Performance

### A2/A3 Cycle Count

`TROWSUM` compiles to a multi-phase CCE instruction sequence. The `TRowReduceOp.hpp` header determines the instruction sequence based on tile geometry.

**Cycle model**:
```
total = startup + Σ(completion_i) + Σ(repeats_i × per_repeat_i) + Σ((repeats_i - 1) × interval)
```

### Instruction Sequence by Shape (FP32)

The performance model applies special optimized sequences for specific FP32 shapes:

| Valid Shape | Instruction Sequence | Total Estimated Cycles |
|-------------|---------------------|----------------------|
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

**Integer types** (int16_t/int32_t): Use simplified path with direct `vadd`/`vmax`/`vmin` per block — no tree reduction.

### Valid Region Sensitivity

- `src.GetValidRow() == 0` → undefined behavior
- `src.GetValidCol() == 0` → undefined behavior
- `src.GetValidRow() != dst.GetValidRow()` → rejected by runtime check

---

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TROWSUM(dst, src, tmp);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(tmp, 0x3000);
  TROWSUM(dst, src, tmp);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.trowsum %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.trowsum %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = trowsum %src : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.trowsum ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Reduce And Expand](../../reduce-and-expand.md)
- Next op in instruction set: [pto.tcolsum](./tcolsum.md)
