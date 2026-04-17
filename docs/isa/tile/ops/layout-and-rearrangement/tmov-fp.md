# pto.tmov_fp

`pto.tmov_fp` is part of the [Layout And Rearrangement](../../layout-and-rearrangement.md) instruction set.

## Summary

Move or convert from an accumulator tile into a destination tile through the fix-pipe path.

## Mechanism

`TMOV_FP` is a named wrapper around the `TMOV_IMPL(..., fp)` path and is part of the [`pto.tmov`](./tmov.md) instruction set. The `_fp` suffix means **fix pipe**, not floating point. The auxiliary `fp` tile is the sideband tile consumed by the backend `set_fpc(...)` path.

Conceptually the fix-pipe path applies an implementation-defined conversion configured by `fp`:

$$ \mathrm{dst}_{i,j} = \mathrm{Convert}\!\left(\mathrm{src}_{i,j};\ \mathrm{fp}\right) $$

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = tmov.fp %src, %fp : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tmov.fp %src, %fp : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tmov.fp ins(%src, %fp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

### IR Level 1 (SSA)

```text
%dst = pto.tmov.fp %src, %fp : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.tmov.fp ins(%src, %fp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp` and `include/pto/common/constants.hpp`:

```cpp
template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TMOV_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp, WaitEvents &... events);
```

## Inputs

- `src` is the source accumulator tile.
- `fp` is the auxiliary fix-pipe tile for the backend FPC path.
- `dst` names the destination tile. The operation iterates over dst's valid region.
- `reluMode` (optional): specifies ReLU mode.

## Expected Outputs

`dst` holds the values moved from `src` through the fix-pipe path configured by `fp`.

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

- Operand shape, mode, and state tuples MUST match the documented contract of this operation and its instruction set overview.

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
- Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

- **Implementation checks (A2A3)**:
    - The fp path is only supported for accumulator conversion and is validated by internal compile-time checks in `TMOV_IMPL(dst, src, fp)`.
    - `FpTileData::Loc` must be `TileType::Scaling` (`static_assert`).

- **Implementation checks (A5)**:
    - Validated by `CheckTMovAccValid(...)` and related compile-time checks in `TMOV_IMPL(dst, src, fp)`.
    - `FpTileData::Loc` must be `TileType::Scaling` (`static_assert`).
    - Destination location is target-dependent (`Vec` or `Mat` are supported in the fp path).

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using AccT = TileAcc<float, 16, 16>;
  using DstT = Tile<TileType::Vec, int8_t, 16, 16>;
  using FpT = Tile<TileType::Scaling, uint64_t, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox>;

  AccT acc;
  DstT dst;
  FpT fp;
  TMOV_FP(dst, acc, fp);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using AccT = TileAcc<float, 16, 16>;
  using DstT = Tile<TileType::Vec, int8_t, 16, 16>;
  using FpT = Tile<TileType::Scaling, uint64_t, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox>;

  AccT acc;
  DstT dst;
  FpT fp;
  TASSIGN(acc, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(fp,  0x3000);
  TMOV_FP(dst, acc, fp);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tmov.fp %src, %fp : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tmov.fp %src, %fp : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tmov.fp %src, %fp : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tmov.fp ins(%src, %fp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Layout And Rearrangement](../../layout-and-rearrangement.md)
- Previous op in instruction set: [pto.tmov](./tmov.md)
- Next op in instruction set: [pto.treshape](./treshape.md)
