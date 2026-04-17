# Sync And Config Instruction Set

Sync-and-config operations manage tile-visible state: resource binding, event setup, mode control, and synchronization. These operations do not produce arithmetic payload — they change state that later tile instructions consume.

## Operations

| Operation | Description | Category | C++ Intrinsic |
|-----------|-------------|----------|---------------|
| [pto.tassign](./ops/sync-and-config/tassign.md) | Bind tile register to a UB address | Resource | `TASSIGN(tile, addr)` |
| [pto.tsync](./ops/sync-and-config/tsync.md) | Synchronize execution, wait on events, insert barrier | Sync | `TSYNC(events...)` |
| [pto.tsettf32mode](./ops/sync-and-config/tsettf32mode.md) | Set TF32 computation mode | Config | `TSETTF32MODE(mode)` |
| [pto.tset_img2col_rpt](./ops/sync-and-config/tset-img2col-rpt.md) | Set img2col repetition count | Config | `TSET_IMG2COL_RPT(rpt)` |
| [pto.tset_img2col_padding](./ops/sync-and-config/tset-img2col-padding.md) | Set img2col padding configuration | Config | `TSET_IMG2COL_PADDING(pad)` |
| [pto.tsubview](./ops/sync-and-config/tsubview.md) | Create a sub-view of a tile | View | `TSUBVIEW(tile, offsets, shape)` |
| [pto.tget_scale_addr](./ops/sync-and-config/tget-scale-addr.md) | Get scale address for quantized matmul | Config | `TGET_SCALE_ADDR(tile)` |

## Mechanism

Sync-and-config operations change tile-visible state that later tile instructions consume:

- **`TASSIGN`**: binds a physical UB address to a tile register. Without `TASSIGN`, the compiler/runtime auto-assigns addresses. `TASSIGN` enables manual placement for performance tuning.
- **`TSYNC`**: waits on event tokens (`events...`) or inserts per-op pipeline barriers (`TSYNC<Op>()`). See [Ordering and Synchronization](../machine-model/ordering-and-synchronization.md) for the full event model.
- **`TSET*`**: tile-local configuration in this instruction set is limited to tile-side controls such as TF32 or IMG2COL setup. Legacy mode-register ops such as `tsethf32mode` and `tsetfmatrix` are documented in the scalar/control lane because their architectural effect is scalar-visible configuration, not tile payload transformation.
- **`TSUBVIEW`**: creates a logical view of a tile with adjusted offsets and/or reduced shape. The underlying storage is shared with the source tile.
- **`TGET_SCALE_ADDR`**: retrieves the physical UB address of a scale tensor used in quantized matmul operations.

## Sync Model

`TSYNC` operates at two levels:

1. **Event-wait form**: `TSYNC(%e0, %e1)` blocks until the specified events have been recorded. Events are produced by preceding operations (e.g., `TLOAD` produces an event; `TSYNC` waits on it).

2. **Barrier form**: `TSYNC<Op>()` inserts a pipeline barrier for the specified operation class. All operations of class `Op` that appear before the barrier complete before any operation of class `Op` that appears after the barrier begins.

See [Producer-Consumer Ordering](../memory-model/producer-consumer-ordering.md) for the complete synchronization model.

## Constraints

- `TASSIGN` binds an address; using the same address for two non-alias tiles simultaneously results in undefined behavior.
- `TSYNC` with no operands is a no-op.
- Tile-side `TSET*` mode configurations affect subsequent operations until the next mode-setting operation of the same kind.
- `TSUBVIEW` creates a view with reduced shape; accessing elements outside the view's shape but within the underlying tile's shape is undefined behavior.

## Cases That Are Not Allowed

- **MUST NOT** use the same physical tile register for two non-alias tiles without an intervening `TSYNC`.
- **MUST NOT** wait on an event that has not been produced by a preceding operation.
- **MUST NOT** configure mode registers while dependent operations are in-flight.

## C++ Intrinsic

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

// Assign tile to UB address
template <typename TileT>
PTO_INST void TASSIGN(TileT& tile, uint64_t addr);

// Synchronize on events
template <typename... EventTs>
PTO_INST RecordEvent TSYNC(EventTs&... events);

// Pipeline barrier for op class
template <typename OpTag>
PTO_INST void TSYNC();

// Set computation modes
PTO_INST void TSETTF32MODE(TF32Mode mode);

// Subview creation
template <typename TileT>
PTO_INST TileT TSUBVIEW(TileT& src, int rowOffset, int colOffset,
                         int newRows, int newCols);
```

## See Also

- [Tile instruction set](../instruction-families/tile-families.md) — Instruction set overview
- [Ordering and Synchronization](../machine-model/ordering-and-synchronization.md) — Event model
- [Tile instruction set](../instruction-surfaces/tile-instructions.md) — Instruction Set description
- [Scalar control and configuration](../scalar/control-and-configuration.md) — canonical location for `tsethf32mode` and `tsetfmatrix`
