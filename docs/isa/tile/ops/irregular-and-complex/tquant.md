# pto.tquant

`pto.tquant` is part of the [Irregular and Complex](../../irregular-and-complex.md) tile instruction family.

## Summary

Quantize a tile (e.g. FP32 to FP8) producing exponent/scaling/max outputs.

## Mechanism

Quantize an FP32 tile into a lower-precision format (e.g. FP8), producing auxiliary exponent/scaling/max tiles. The quantization mode is a compile-time template parameter (`mode`). It belongs to the tile instructions and carries architecture-visible behavior that is not reducible to a plain elementwise compute pattern.

Unless otherwise specified, semantics are defined over the valid region. On A2/A3 and A5, the quantizer operates on the full valid region with target-specific quantization modes; the CPU simulator emulates the nearest available mode.

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

!!! warning "Constraints"
    - This instruction is currently implemented for specific targets (see `include/pto/npu/*/TQuant.hpp`).

    - Input type requirements and output tile types are mode/target-dependent.

## Performance

### A2/A3 Cycle Count

`pto.tquant` is dispatched on the **fix-pipe (FIXP)** pipeline rather than the vector pipe. Cost is dominated by FIXP throughput and the L0C → UB / OUT bandwidth on the asymmetric FIXP write path.

On A5 the FIXP pipe has roughly 4× write-bandwidth asymmetry between L0C-side reads and UB/OUT writes, so wide quantized stores are usually FIXP-bound rather than vector-pipe-bound.

**Cycle model**: `total ≈ startup + R × C / FIXP_throughput + drain`.

MXFP8 quantization (A5) uses block-wise scales; cost includes one scale-fetch per block in addition to the per-element conversion.

> Note: cycle numbers below are first-order estimates; populate with measured values from `pto-isa/a2a3_benchmark.csv` and `pto-isa/a5_benchmark.csv`.

## Exceptions

!!! danger "Exceptions"
    - Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
    - Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
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

## See Also

- Instruction set overview: [Irregular And Complex](../../irregular-and-complex.md)
- Previous op in instruction set: [pto.tpartmin](./tpartmin.md)
- Next op in instruction set: [pto.tdequant](./tdequant.md)

