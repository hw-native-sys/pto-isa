<!-- Generated from `docs/isa/tile/ops/irregular-and-complex/tquant.md` -->

# pto.tquant

Standalone reference page for `pto.tquant`. This page belongs to the [Irregular And Complex](../../irregular-and-complex.md) family in the PTO ISA manual.

## Summary

Quantize a tile (e.g. FP32 to FP8) producing exponent/scaling/max outputs.

## Mechanism

Quantize an FP32 tile into a lower-precision format (e.g. FP8), producing auxiliary exponent/scaling/max tiles. The quantization mode is a compile-time template parameter (`mode`). It belongs to the tile surface and carries architecture-visible behavior that is not reducible to a plain elementwise compute pattern.

Unless otherwise specified, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

### AS Level 1 (SSA)

```text
%dst = pto.tquant %src, %qp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tquant ins(%src, %qp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

### IR Level 1 (SSA)

```text
%dst = pto.tquant %src, %qp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.tquant ins(%src, %qp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <auto quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataExp, typename TileDataMax,
          typename... WaitEvents>
PTO_INST RecordEvent TQUANT(TileDataOut &dst, TileDataSrc &src, TileDataExp *exp, TileDataMax *max, TileDataSrc *scaling, WaitEvents &... events);

template <auto quant_type, auto store_mode, typename TileDataOut, typename TileDataSrc, typename TileDataExp,
          typename TileDataMax, typename TileDataIdx, typename... WaitEvents>
PTO_INST RecordEvent TQUANT(TileDataOut &dst, TileDataSrc &src, TileDataExp *exp, TileDataMax *max, TileDataSrc *scaling, TileDataExp *exp_zz, TileDataIdx *vgather_idx, WaitEvents &... events);

template <auto quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataPara, typename... WaitEvents>
PTO_INST RecordEvent TQUANT(TileDataOut &dst, TileDataSrc &src, TileDataPara &scale, TileDataPara *offset = nullptr, WaitEvents &... events);
```

## Inputs

- `src` is the source tile to quantize.
- `qp` is the quantization parameter tile.
- `exp` (output): exponent tile.
- `max` (output): max tile.
- `scaling` (output): scaling tile.
- `dst` names the destination tile. The operation iterates over src's valid region.

## Expected Outputs

`dst` holds the quantized output. `exp`, `max`, `scaling` hold quantization metadata.

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

- This instruction is currently implemented for specific targets (see `include/pto/npu/*/TQuant.hpp`).

- Input type requirements and output tile types are mode/target-dependent.

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend surface.
- Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

- `pto.tquant` preserves PTO-visible semantics across CPU simulation, A2/A3-class targets, and A5-class targets, but concrete support subsets may differ by profile.

- Portable code must rely only on the documented type, layout, shape, and mode combinations that the selected target profile guarantees.

## Examples

See related examples in `docs/isa/` and `docs/coding/tutorials/`.

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tquant %src, %qp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tquant %src, %qp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = pto.tquant %src, %qp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tquant ins(%src, %qp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Family Links

- Family overview: [Irregular And Complex](../../irregular-and-complex.md)
- Previous op in family: [pto.tscatter](./tscatter.md)
