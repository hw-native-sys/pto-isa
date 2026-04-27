# pto.timg2col

`pto.timg2col` is part of the [Layout And Rearrangement](../../layout-and-rearrangement.md) instruction set.

## Summary

Image-to-column transform for convolution-like workloads.

## Mechanism

Transform an input feature-map tile (e.g. NC1HWC0 layout) into an im2col-style matrix tile for convolution-like workloads. Parameters are provided via `Img2colTileConfig` and `(posM, posK)` offsets. It belongs to the tile instructions and carries architecture-visible behavior that is not reducible to a plain elementwise compute pattern.

Unless otherwise specified, semantics are defined over the valid region. On A2/A3 and A5, the im2col transform operates with target-specific tile layouts (NC1HWC0 input, row-major output); on the CPU simulator, the transform follows the natural tile row-major ordering.

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

### AS Level 1 (SSA)

```text
%dst = pto.timg2col %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.timg2col ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

### IR Level 1 (SSA)

```text
%dst = pto.timg2col %src : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.timg2col ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
PTO_INST RecordEvent TIMG2COL(TileData &dst, ConvTileData &src, uint16_t posM = 0, uint16_t posK = 0,
                              WaitEvents&... events);
```

## Inputs

- `src` is the source ConvTileData (feature-map tile in NC1HWC0 layout).
- `dst` names the destination im2col matrix tile.
- `posM` is the output row offset.
- `posK` is the output column offset.

## Expected Outputs

`dst` holds the im2col transformed data from `src` according to the Img2colTileConfig parameters.

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

- This instruction is target/implementation-specific. See `include/pto/npu/*/TImg2col.hpp` for the supported tile types/layouts and config fields.

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
- Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

- `pto.timg2col` preserves PTO-visible semantics across CPU simulation, A2/A3-class targets, and A5-class targets, but concrete support subsets may differ by profile.

- Portable code must rely only on the documented type, layout, shape, and mode combinations that the selected target profile guarantees.

## Examples

See related examples in `docs/isa/` and `docs/coding/tutorials/`.

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.timg2col %src : !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.timg2col %src : !pto.tile<...> -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = pto.timg2col %src : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.timg2col ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Layout And Rearrangement](../../layout-and-rearrangement.md)
- Previous op in instruction set: [pto.textract_fp](./textract.md)
- Next op in instruction set: [pto.tinsert](./tinsert.md)
