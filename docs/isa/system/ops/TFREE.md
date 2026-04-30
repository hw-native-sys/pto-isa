# TFREE

## Tile Operation Diagram

![TFREE tile operation](../../../figures/isa/TFREE.svg)

## Summary

`pto.tfree` releases a tile or pipe slot previously acquired via `pto.tpush` (or held by an in-flight tile), returning the slot to the producer side of the FIFO. It is the primary reclaim hook in the system-side three-phase tile-pipe protocol and is paired with `pto.tpop` on the consumer side.

## Mechanism

`pto.tfree` performs a single release transaction against the tile-pipe metadata:

1. The current consumer signals it is done with the slot.
2. The slot's reference is decremented; when it drops to zero, the slot is returned to the producer pool.
3. Any waiter blocked on `pto.tpush` for an empty slot is unblocked.

The op executes on the system pipe and does not move tile data; it only updates pipe-control state.

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

### IR Level 1 (SSA)

```text
pto.tfree %pipe, %tile : (!pto.pipe<...>, !pto.tile_buf<...>)
```

### IR Level 2 (DPS)

```text
pto.tfree ins(%pipe, %tile) outs()
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename PipeT, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TFREE(PipeT &pipe, TileData &tile, WaitEvents&... events);
```

## Inputs

| Operand | Description |
|---------|-------------|
| `pipe` | The tile pipe (`TMPipe` or compatible) the slot belongs to. |
| `tile` | The tile handle obtained from a prior `pto.tpop`. |
| `events...` | Optional `RecordEvent` tokens to wait on before issuing. |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `RecordEvent` | token | Signals that the slot has been released and is reusable by the producer. |

## Side Effects

- Updates pipe-control metadata in place; no tile data is moved.
- May unblock a producer waiting on `pto.tpush`.
- After `pto.tfree`, the released `tile` handle must not be read or written by the consumer.

## Constraints

!!! warning "Constraints"
    - `tile` must originate from a prior `pto.tpop` on the same `pipe`.
    - Each `pto.tpop` must be matched by exactly one `pto.tfree`; double-free is undefined.
    - `pto.tfree` must not be issued from the producer side; it is consumer-side only.
    - Refer to backend-specific legality checks for data type/layout/location/shape constraints not covered above.

## Performance

### A2/A3 Cycle Count

`pto.tfree` is a control-only update. Cost is dominated by the cross-core synchronisation flag write that releases the slot.

**Cycle model**:

```
total ≈ startup + sync_release_overhead
```

The op does not consume vector or DMA bandwidth and is independent of tile shape.

### Layout and Shape Impact

Tile shape does not affect cost; the op is metadata-only.

> Note: cycle numbers are first-order estimates; populate with measured values from `pto-isa/a2a3_benchmark.csv` and `pto-isa/a5_benchmark.csv`.

## Exceptions

!!! danger "Exceptions"
    - Calling `pto.tfree` on a `tile` not obtained from `pto.tpop` is undefined behavior.
    - Calling `pto.tfree` twice on the same handle is undefined behavior.
    - Calling `pto.tfree` from the producer side is rejected by the verifier.
    - Programs must not rely on behavior outside the documented legal domain of this operation.

## Examples

```cpp
// Consumer side of a TMPipe loop.
auto tile = TPOP(pipe);
// ... compute on tile ...
TFREE(pipe, tile);
```

See related instruction pages in `docs/isa/` for concrete Auto/Manual usage patterns.

## See Also

- Instruction set overview: [System Ops](../README.md)
- Producer side: [TPUSH](./TPUSH.md)
- Consumer-acquire side: [TPOP](./TPOP.md)
