# pto.tfillpad_inplace

`pto.tfillpad_inplace` is part of the [Layout And Rearrangement](../../layout-and-rearrangement.md) instruction set.

## Summary

In-place fill/pad variant.

## Mechanism

In-place fill/pad variant of TFILLPAD — dst and src must have the same shape, and valid region elements from src are copied to dst while the padding region of dst is filled with PadVal. On A2/A3: the operation iterates over the destination's valid region (via GetValidRow/GetValidCol), copying from the corresponding source position and filling all other destination positions with PadVal; because src and dst share the same shape, every source position maps to exactly one destination position. On A5: behavior is identical, but the hardware enforces 32-byte alignment constraints on the major dimension which may affect the effective valid region boundaries for certain tile shapes. On the CPU simulator: the operation is emulated by iterating over the tile shape and performing elementwise copy or fill based on whether the current position is within the valid region.

Unless otherwise specified, semantics are defined over the valid region. On A2/A3 and A5 the in-place fill writes directly into the tile buffer; on the CPU simulator it writes element-by-element. The pad value must be representable in the destination element type.

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

### AS Level 1 (SSA)

```text
%dst = pto.tfillpad_inplace %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tfillpad_inplace ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

### IR Level 1 (SSA)

```text
%dst = pto.tfillpad_inplace %src : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.tfillpad_inplace ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD_INPLACE(DstTileData &dst, SrcTileData &src, WaitEvents &... events);
```

## Inputs

- `src` is the source tile.
- `dst` names the destination tile. Must have same shape as `src`.
- `PadVal` is the compile-time pad value for elements outside the valid region.

## Expected Outputs

`dst` holds the in-place copy of `src` with valid region copied and padded region filled with the specified pad value.

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

!!! warning "Constraints"
    Type/layout/location/shape legality is backend-dependent; treat implementation-specific notes as normative for that backend.

## Performance

### A2/A3 Cycle Count

`pto.tfillpad-inplace` writes the pad value directly into the tile without copying any source data; it is typically used to clear the invalid region of a tile that was just produced.

**Cycle model**: `total ≈ startup + R × per_row_pad_store`.

> Note: cycle numbers below are first-order estimates; populate with measured values from `pto-isa/a2a3_benchmark.csv` and `pto-isa/a5_benchmark.csv`.

## Exceptions

!!! danger "Exceptions"
    - Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
    - Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - `pto.tfillpad_inplace` preserves PTO-visible semantics across CPU simulation, A2/A3-class targets, and A5-class targets, but concrete support subsets may differ by profile.

    - Portable code must rely only on the documented type, layout, shape, and mode combinations that the selected target profile guarantees.

## Examples

See related examples in `docs/isa/` and `docs/coding/tutorials/`.

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tfillpad_inplace %src : !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tfillpad_inplace %src : !pto.tile<...> -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = pto.tfillpad_inplace %src : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tfillpad_inplace ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## See Also

- Instruction set overview: [Layout And Rearrangement](../../layout-and-rearrangement.md)
- Previous op in instruction set: [pto.tfillpad](./tfillpad.md)
- Next op in instruction set: [pto.tfillpad_expand](./tfillpad-expand.md)

