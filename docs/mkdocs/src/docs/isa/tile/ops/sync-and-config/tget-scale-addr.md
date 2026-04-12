<!-- Generated from `docs/isa/tile/ops/sync-and-config/tget-scale-addr.md` -->

# pto.tget_scale_addr

Standalone reference page for `pto.tget_scale_addr`. This page belongs to the [Sync And Config](../../sync-and-config.md) family in the PTO ISA manual.

## Summary

Bind the on-chip address of output tile to a scaled factor of that of input tile.

## Mechanism

Bind the on-chip address of output tile as a scaled address of the input tile.

The scaling factor is defined by a right-shift amount `SHIFT_MX_ADDR` in `include/pto/npu/a5/utils.hpp`. It is part of the tile synchronization or configuration shell, so the visible effect is ordering or state setup rather than arithmetic payload transformation.

Address(`dst`) = Address(`src`) >> `SHIFT_MX_ADDR`

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = tget_scale_addr %src : !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tget_scale_addr %src : (!pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tget_scale_addr ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

### IR Level 1 (SSA)

```text
%dst = pto.tget_scale_addr %src : (!pto.tile<...>) -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.tget_scale_addr ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TGET_SCALE_ADDR(TileDataDst &dst, TileDataSrc &src, aitEvents&... events);
```

## Inputs

- `src` is the source tile.
- `dst` names the destination tile that holds the scaled address.

## Expected Outputs

`dst` carries the result tile or updated tile payload produced by the operation.

## Side Effects

This operation may establish a synchronization edge, bind or configure architectural tile state, or update implementation-defined configuration that later tile instructions consume.

## Constraints

Enforced by `TGET_SCALE_ADDR_IMPL`:

- **Both `src` and `dst` must be Tile instances**

- **Currently only work in auto mode** (will support manual mode in the future)

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend surface.
- Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

- `pto.tget_scale_addr` preserves PTO-visible semantics across CPU simulation, A2/A3-class targets, and A5-class targets, but concrete support subsets may differ by profile.

- Portable code must rely only on the documented type, layout, shape, and mode combinations that the selected target profile guarantees.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T, int ARows, int ACols, int BRows, int BCols>
void example() {
    using LeftTile = TileLeft<T, ARows, ACols>;
    using RightTile = TileRight<T, BRows, BCols>;

    using LeftScaleTile = TileLeftScale<T, ARows, ACols>;
    using RightScaleTile = TileRightScale<T, BRows, BCols>;

    LeftTile aTile;
    RightTile bTile;
    LeftScaleTile aScaleTile;
    RightScaleTile bScaleTile;

    TGET_SCALE_ADDR(aScaleTile, aTile);
    TGET_SCALE_ADDR(bScaleTile, bTile);
}
```

## Related Ops / Family Links

- Family overview: [Sync And Config](../../sync-and-config.md)
- Previous op in family: [pto.tsubview](./tsubview.md)
