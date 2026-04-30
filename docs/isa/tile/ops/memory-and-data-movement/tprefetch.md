# pto.tprefetch

`pto.tprefetch` is part of the [Memory And Data Movement](../../memory-and-data-movement.md) instruction set.

## Summary

Prefetch data from global memory into a tile-local cache/buffer (hint).

## Mechanism

Prefetch data from global memory into a tile-local cache/buffer. On A2/A3: the hardware maintains a dedicated on-chip scratchpad that TPREFETCH can populate; the prefetch issues a DMA read into the tile buffer and blocks until the data is resident before returning. On A5: the hardware uses a similar scratchpad mechanism but the cache-line fill granularity may differ (typically 32-byte aligned fills); the prefetch may also overlap with other memory traffic differently than on A2/A3. On the CPU simulator: TPREFETCH is emulated as a blocking tile load and has no effect beyond what TLOAD would do — it cannot be ignored since the simulator must produce correct data.

Note: unlike most PTO instructions, `TPREFETCH` does **not** implicitly call `TSYNC(events...)` in the C++ wrapper. It is part of the tile memory/data-movement instruction set, so the visible behavior includes explicit transfer between GM-visible data and tile-visible state.

Unless otherwise specified, semantics are defined over the valid region. On A2/A3 and A5 the prefetch transfers data from the specified GM region into the tile buffer; on the CPU simulator it copies the data directly. Whether the data is placed in a dedicated cache or the general tile buffer is target-specific.

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

`dst` holds the prefetched data from `src`. On A2/A3 and A5: the prefetch hint is advisory — the hardware may populate the tile buffer with data from the source GlobalTensor, but the returned token signals only that the operation was issued, not that the data is necessarily cached or privileged; the DMA engine may overlap the fill with subsequent operations. On the CPU simulator: the prefetch performs a blocking copy from GM into the tile buffer, so the data is guaranteed resident when the token is returned.

## Side Effects

This operation may read from global memory. Prefetch hints may be ignored by some targets.

## Constraints

!!! warning "Constraints"
    - Semantics and caching behavior vary by target: on A2/A3, the prefetch issues a DMA read into the tile buffer with no additional caching layer beyond the scratchpad; on A5, the DMA engine may use different memory-transaction scheduling and the hint may be treated as a non-blocking request that does not block subsequent tile operations; on the CPU simulator, the prefetch is a blocking copy that copies data from the source GlobalTensor into the destination tile.

    - Some targets may ignore prefetches or treat them as hints.

## Performance

### A2/A3 Cycle Count

`pto.tprefetch` issues a non-blocking prefetch hint into the L1/UB cache hierarchy; it does not produce architectural data and is effectively free in the pipeline if scheduled in parallel with compute. Its purpose is to hide MTE2 latency on subsequent `tload` issues.

**Cycle model**: `total ≈ startup` (compute-overlapped); the perceived cost is the latency saved on a later `tload`, not the issue latency itself.

> Note: cycle numbers below are first-order estimates; populate with measured values from `pto-isa/a2a3_benchmark.csv` and `pto-isa/a5_benchmark.csv`.

## Exceptions

!!! danger "Exceptions"
    - Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend instruction set.
    - Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
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

## See Also

- Instruction set overview: [Memory And Data Movement](../../memory-and-data-movement.md)
- Previous op in instruction set: [pto.tload](./tload.md)
- Next op in instruction set: [pto.tstore](./tstore.md)

