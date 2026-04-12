<!-- Generated from `docs/isa/tile/ops/reduce-and-expand/tcolexpandmax.md` -->

# pto.tcolexpandmax

Standalone reference page for `pto.tcolexpandmax`. This page belongs to the [Reduce And Expand](../../reduce-and-expand.md) family in the PTO ISA manual.

## Summary

Column-wise broadcast max with per-column scalar vector.

## Mechanism

Column-wise broadcast max: take `max(src0, src1)` where `src1` provides one scalar per column. It operates on tile payloads rather than scalar control state, and its legality is constrained by tile shape, layout, valid-region, and target-profile support.

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. Let `s_j` be the per-column scalar taken from `src1` (one value per column).

For `0 <= i < R` and `0 <= j < C`:

$$
\mathrm{dst}_{i,j} = \max(\mathrm{src0}_{i,j}, s_j)
$$

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = tcolexpandmax %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tcolexpandmax %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tcolexpandmax ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDMAX(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

## Inputs

- `src0` is the first source tile (the tile to be modified).
- `src1` is the second source tile providing per-column scalar values.
- `dst` names the destination tile. The operation iterates over dst's valid region.

## Expected Outputs

`dst[i,j]` = max(`src0[i,j]`, `src1[0,j]`) (column-wise broadcast max of per-column scalar).

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

- `TileDataDst::DType`, `TileDataSrc1::DType` must be one of: `half`, `float`.

- Tile shape/layout constraint (compile-time): `TileDataDst::isRowMajor`.

- `src1` is expected to provide **one scalar per column** (i.e., its valid shape must cover `C` values).

- Exact layout/fractal constraints are target-specific; see backend headers under `include/pto/npu/*/TColExpand*.hpp`.

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend surface.
- Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

- `pto.tcolexpandmax` preserves PTO-visible semantics across CPU simulation, A2/A3-class targets, and A5-class targets, but concrete support subsets may differ by profile.

- Portable code must rely only on the documented type, layout, shape, and mode combinations that the selected target profile guarantees.

## Examples

See related examples in `docs/isa/` and `docs/coding/tutorials/`.

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tcolexpandmax %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tcolexpandmax %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tcolexpandmax %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tcolexpandmax ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Family Links

- Family overview: [Reduce And Expand](../../reduce-and-expand.md)
- Previous op in family: [pto.tcolexpandadd](./tcolexpandadd.md)
- Next op in family: [pto.tcolexpandmin](./tcolexpandmin.md)
