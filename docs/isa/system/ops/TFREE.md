# TFREE

## Tile Operation Diagram

![TFREE tile operation](../../../figures/isa/TFREE.svg)

## Summary

`pto.tfree` releases the consumer-side reservation on a tile pipe previously acquired by `pto.tpop`, returning the slot to the producer side of the FIFO. It is the primary reclaim hook in the system-side three-phase tile-pipe protocol and is paired with `pto.tpop` on the consumer side.

## Mechanism

`pto.tfree` performs a single release transaction against the tile-pipe metadata:

1. The current consumer signals it is done with the slot it last popped from `pipe`.
2. The slot's reference is decremented; when it drops to zero, the slot is returned to the producer pool.
3. Any waiter blocked on `pto.tpush` for an empty slot is unblocked.

The op executes on the system pipe and does not move tile data; it only updates pipe-control state. The slot identity is implicit in `pipe` — there is no tile handle operand.

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

### IR Level 1 (SSA)

```text
%event = pto.tfree %pipe : (!pto.pipe<...>) -> !pto.record_event
```

### IR Level 2 (DPS)

```text
pto.tfree ins(%pipe) outs(%event : !pto.record_event)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename Pipe, typename... WaitEvents>
PTO_INST RecordEvent TFREE(Pipe &pipe, WaitEvents&... events);
```

## Inputs

| Operand | Description |
|---------|-------------|
| `pipe` | The tile pipe (`TMPipe` or compatible) whose most recently popped slot is being released. |
| `events...` | Optional `RecordEvent` tokens to wait on before issuing. |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `RecordEvent` | token | Signals that the slot has been released and is reusable by the producer. |

## Side Effects

- Updates pipe-control metadata in place; no tile data is moved.
- May unblock a producer waiting on `pto.tpush`.
- After `pto.tfree`, the tile previously returned by `pto.tpop` on the same pipe must not be read or written by the consumer.

## Constraints

!!! warning "Constraints"
    - Each `pto.tpop` on a pipe must be matched by exactly one `pto.tfree` on the same pipe; double-free is undefined.
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
    - Issuing `pto.tfree` on a pipe with no outstanding `pto.tpop` is undefined behavior.
    - Calling `pto.tfree` more times than `pto.tpop` on the same pipe is undefined behavior.
    - Calling `pto.tfree` from the producer side is rejected by the verifier.
    - Programs must not rely on behavior outside the documented legal domain of this operation.

## Examples

```cpp
// Consumer side of a TMPipe loop.
auto tile = TPOP(pipe);
// ... compute on tile ...
TFREE(pipe);
```

See related instruction pages in `docs/isa/` for concrete Auto/Manual usage patterns.

## See Also

- Instruction set overview: [System Ops](../README.md)
- Producer side: [TPUSH](./TPUSH.md)
- Consumer-acquire side: [TPOP](./TPOP.md)
