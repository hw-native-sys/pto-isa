# pto.tfillpad_expand

`pto.tfillpad_expand` is part of the [Layout And Rearrangement](../../layout-and-rearrangement.md) instruction set.

## Summary

Fill/pad while allowing dst to be larger than src.

## Mechanism

Expand fill/pad variant of TFILLPAD — allows dst to be larger than src, copying the valid region from src and filling the remainder with PadVal. On A2/A3: the operation iterates over dst's valid region via GetValidRow/GetValidCol, copying the src valid region into the corresponding top-left portion of dst and filling all other elements with PadVal; if src is smaller than dst in either dimension, the padded portion receives PadVal. On A5: behavior is identical — the valid region copy and pad fill follow the same semantics; the hardware enforces 32-byte alignment on the major dimension which may affect the effective valid region boundaries for some tile shapes. On the CPU simulator: the operation is emulated by iterating over the destination shape and performing elementwise copies or fills, matching the A2/A3/A5 semantics exactly.

Unless otherwise specified, semantics are defined over the valid region. On A2/A3 and A5: TFILLPAD_EXPAND iterates over the destination valid region (defined by GetValidRow/GetValidCol on the destination tile), copying elements from the corresponding source positions and filling any positions that fall outside the source's valid region with PadVal. On the CPU simulator: the operation is emulated element-by-element, following the same valid-region iteration semantics as the hardware targets.

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

### AS Level 1 (SSA)

```text
%dst = pto.tfillpad_expand %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tfillpad_expand ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

### IR Level 1 (SSA)

```text
%dst = pto.tfillpad_expand %src : !pto.tile<...> -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.tfillpad_expand ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD_EXPAND(DstTileData &dst, SrcTileData &src, WaitEvents &... events);
```

## Inputs

- `src` is the source tile.
- `dst` names the destination tile. May be larger than `src`.
- `PadVal` is the compile-time pad value for elements outside the valid region.

## Expected Outputs

`dst` holds a copy of `src` with valid region copied and padded region filled with the specified pad value.

## Side Effects

No architectural side effects beyond producing the destination tile. Does not implicitly fence unrelated traffic.

## Constraints

Type/layout/location/shape legality is backend-dependent; treat implementation-specific notes as normative for that backend.

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
- Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

- `pto.tfillpad_expand` preserves PTO-visible semantics across CPU simulation, A2/A3-class targets, and A5-class targets, but concrete support subsets may differ by profile.

- Portable code must rely only on the documented type, layout, shape, and mode combinations that the selected target profile guarantees.

## Examples

See related examples in `docs/isa/` and `docs/coding/tutorials/`.

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tfillpad_expand %src : !pto.tile<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tfillpad_expand %src : !pto.tile<...> -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = pto.tfillpad_expand %src : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tfillpad_expand ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Layout And Rearrangement](../../layout-and-rearrangement.md)
- Previous op in instruction set: [pto.tfillpad_inplace](./tfillpad-inplace.md)
- Next op in instruction set: [pto.tmov](./tmov.md)
