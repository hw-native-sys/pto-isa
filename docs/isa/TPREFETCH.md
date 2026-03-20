# TPREFETCH


## Tile Operation Diagram

![TPREFETCH tile operation](../figures/isa/TPREFETCH.svg)

## Introduction

Prefetch data from global memory into a tile-local cache/buffer (implementation-defined). This is typically used to reduce latency before a subsequent `TLOAD`.

Note: unlike most PTO instructions, `TPREFETCH` does **not** implicitly call `TSYNC(events...)` in the C++ wrapper.

## Assembly Syntax

PTO-AS form: see [PTO-AS Specification](../assembly/PTO-AS.md).

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
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename GlobalData>
PTO_INST RecordEvent TPREFETCH(TileData &dst, GlobalData &src);
```

## Constraints

- Semantics and caching behavior are target/implementation-defined.
- Some targets may ignore prefetches or treat them as hints.

## Math Interpretation

Unless otherwise specified, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

## Examples

See related examples in `docs/isa/` and `docs/coding/tutorials/`.

## ASM Form Examples

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

