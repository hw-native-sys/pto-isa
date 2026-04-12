<!-- Generated from `docs/isa/tile/ops/memory-and-data-movement/tprefetch.md` -->

# pto.tprefetch

Standalone reference page for `pto.tprefetch`. This page belongs to the [Memory And Data Movement](../../memory-and-data-movement.md) family in the PTO ISA manual.

## Summary

Prefetch data from global memory into a tile-local cache/buffer (hint).

## Mechanism

Prefetch data from global memory into a tile-local cache/buffer (implementation-defined). This is typically used to reduce latency before a subsequent `TLOAD`.

Note: unlike most PTO instructions, `TPREFETCH` does **not** implicitly call `TSYNC(events...)` in the C++ wrapper. It is part of the tile memory/data-movement surface, so the visible behavior includes explicit transfer between GM-visible data and tile-visible state.

Unless otherwise specified, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Synchronous form:

```text
%dst = tprefetch %src : !pto.global<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tprefetch %src : !pto.global<...> -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tprefetch ins(%src : !pto.global<...>) outs(%dst : !pto.tile_buf<...>)
```

### IR Level 1 (SSA)

```text
%dst = pto.tprefetch %src : !pto.global<...> -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.tprefetch ins(%src : !pto.global<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename GlobalData>
PTO_INST RecordEvent TPREFETCH(TileData &dst, GlobalData &src);
```

## Inputs

- `src` is the source GlobalTensor to prefetch.
- `dst` names the destination tile (cache buffer).

## Expected Outputs

`dst` holds the prefetched data from `src`. This is a hint; behavior is implementation-defined.

## Side Effects

This operation may read from global memory. Prefetch hints may be ignored by some targets.

## Constraints

- Semantics and caching behavior are target/implementation-defined.

- Some targets may ignore prefetches or treat them as hints.

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend surface.
- Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

- `pto.tprefetch` preserves PTO-visible semantics across CPU simulation, A2/A3-class targets, and A5-class targets, but concrete support subsets may differ by profile.

- Portable code must rely only on the documented type, layout, shape, and mode combinations that the selected target profile guarantees.

## Examples

See related examples in `docs/isa/` and `docs/coding/tutorials/`.

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tprefetch %src : !pto.global<...> -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tprefetch %src : !pto.global<...> -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tprefetch %src : !pto.global<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tprefetch ins(%src : !pto.global<...>) outs(%dst : !pto.tile_buf<...>)
```

## Related Ops / Family Links

- Family overview: [Memory And Data Movement](../../memory-and-data-movement.md)
- Previous op in family: [pto.tload](./tload.md)
- Next op in family: [pto.tstore](./tstore.md)
