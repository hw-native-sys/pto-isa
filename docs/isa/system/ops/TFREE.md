# pto.tfree

## Tile Operation Diagram

![TFREE tile operation](../../../figures/isa/TFREE.svg)

## Summary

`pto.tfree` is the explicit release operation for the `GlobalData` slot-view form of `TPOP`. The simpler TileData `TPOP(pipe, tile)` flow already performs its free-space notification internally, so `TFREE(Pipe&)` is currently a no-op kept for API symmetry and synchronization spelling.

## Mechanism

The public API has two relevant shapes:

1. `TFREE(Pipe&)` waits on any supplied events and then calls the backend no-op `TFREE_IMPL(pipe)`.
2. `TFREE(Pipe&, GlobalData&)` releases a FIFO slot view previously produced by `TPOP(Pipe&, GlobalData&)`.
3. The `GlobalData` form checks the consumer free state and emits a free-space notification only when the pipe direction and split mode require one.

The op executes on the system/control side and does not move tile data. There is no public `TFREE(pipe, tile)` overload.

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

template <typename Pipe, TileSplitAxis Split, typename... WaitEvents>
PTO_INST RecordEvent TFREE(Pipe &pipe, WaitEvents&... events);

template <typename Pipe, typename GlobalData, TileSplitAxis Split, typename... WaitEvents>
PTO_INST RecordEvent TFREE(Pipe &pipe, GlobalData &gmTensor, WaitEvents&... events);
```

## Inputs

| Operand | Description |
|---------|-------------|
| `pipe` | The tile pipe (`TPipe` / `TMPipe` or compatible) whose consumer-side free state is being updated. |
| `gmTensor` | Optional `GlobalTensor` slot view returned by the `TPOP(Pipe&, GlobalData&)` form. |
| `events...` | Optional `RecordEvent` tokens to wait on before issuing. |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `RecordEvent` | token | Signals that the `TFREE` control operation has completed. |

## Side Effects

- `TFREE(Pipe&)` has no backend-visible data movement or free notification in the current TileData flow.
- `TFREE(Pipe&, GlobalData&)` may notify producer-side free space for a FIFO slot view.
- No tile payload is read or written.

## Constraints

!!! warning "Constraints"
    - Do not pass a tile operand to `TFREE`; no public tile-handle overload exists.
    - TileData `TPOP(pipe, tile)` does not require a matching `TFREE(Pipe&)`.
    - The `GlobalData` slot-view flow must call `TFREE(Pipe&, gmTensor)` after all loads from the popped slot are complete.
    - `TFREE` is consumer-side only; producer-side release is handled by `TPUSH`.
    - Refer to backend-specific legality checks for data type/layout/location/shape constraints not covered above.

## Exceptions

!!! danger "Exceptions"
    - Calling `TFREE(Pipe&, GlobalData&)` with a `gmTensor` that was not produced by the matching `TPOP` slot-view flow is outside the supported contract.
    - Calling `TFREE` from the producer side is rejected by the verifier.
    - Programs must not rely on behavior outside the documented legal domain of this operation.

## Examples

```cpp
// TileData TPOP already performs the free-space notification.
TPOP(pipe, tile);
// ... compute on tile; no TFREE is required for this tile.

// GlobalData slot-view flow: release the slot after all reads are complete.
TPOP<Pipe, SlotGlobal, TileSplitAxis::TILE_UP_DOWN>(pipe, slot);
TLOAD(tile, slot);
TFREE<Pipe, SlotGlobal, TileSplitAxis::TILE_UP_DOWN>(pipe, slot);
```

See related instruction pages in `docs/isa/` for concrete Auto/Manual usage patterns.

## See Also

- Instruction set overview: [System Ops](../README.md)
- Producer side: [TPUSH](./TPUSH.md)
- Consumer-acquire side: [TPOP](./TPOP.md)
