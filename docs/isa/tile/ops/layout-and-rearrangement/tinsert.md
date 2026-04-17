# pto.tinsert

`pto.tinsert` is part of the [Layout And Rearrangement](../../layout-and-rearrangement.md) instruction set.

## Summary

Insert a sub-tile into a destination tile at an (indexRow, indexCol) offset.

## Mechanism

Insert a source sub-tile into a destination tile at `(indexRow, indexCol)`. This is conceptually the inverse of `TEXTRACT` for many layouts. It belongs to the tile instructions and carries architecture-visible behavior that is not reducible to a plain elementwise compute pattern.

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. Conceptually, for `0 <= i < R` and `0 <= j < C`:

$$
\mathrm{dst}_{\mathrm{indexRow}+i,\;\mathrm{indexCol}+j} = \mathrm{src}_{i,j}
$$

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = tinsert %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tinsert %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tinsert ins(%src[%r0, %r1] : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TINSERT_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

#ifdef PTO_NPU_ARCH_A5
template <TInsertMode mode, typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src, uint32_t indexRow = 0, uint32_t indexCol = 0, WaitEvents &... events);
#endif
```

## Inputs

- `src` is the source tile.
- `indexRow` is the starting row offset in `dst`.
- `indexCol` is the starting column offset in `dst`.
- `dst` names the destination tile. The operation iterates over src's valid region.
- `fp` (optional for TINSERT_FP): auxiliary fix-pipe tile consumed by the backend FPC path.
- `reluMode` (optional): specifies ReLU mode.
- `preQuantScalar` (optional): scalar for pre-quantization.

## Expected Outputs

`dst` holds the result of inserting `src` into `dst` at position (indexRow, indexCol), with optional conversion.

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

- **A2/A3**:
    - The documented overloads map to `Acc -> Mat` insertion paths, including plain, `reluMode`, scalar pre-quant, and vector pre-quant (`TINSERT_FP`) forms.
    - Runtime bounds must satisfy `indexRow + src.Rows <= dst.Rows` and `indexCol + src.Cols <= dst.Cols`.

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
- Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

- **A5**:
    - In addition to the `Acc -> Mat` insertion paths above, A5 also exposes `template <TInsertMode mode, ...> TINSERT(...)` for `Vec -> Mat` and `Vec -> Vec` insertion variants.
    - `mode == TInsertMode::ND` requires a row-major source vector tile and inserts into a matrix tile in ND layout.
    - `mode == TInsertMode::ND_VEC` requires both source and destination to be row-major vector tiles.
    - NZ-instruction set modes (`NZ`, `NZ_PLUS_1`, `SPLIT2_NZ_PLUS_1`, `SPLIT4_NZ_PLUS_1`) require an NZ-format source vector tile and a matrix destination tile.

## Examples

See related examples in `docs/isa/` and `docs/coding/tutorials/`.

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tinsert %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tinsert %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tinsert %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tinsert ins(%src[%r0, %r1] : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Layout And Rearrangement](../../layout-and-rearrangement.md)
- Previous op in instruction set: [pto.timg2col](./timg2col.md)
- Next op in instruction set: [pto.tinsert_fp](./tinsert-fp.md)
