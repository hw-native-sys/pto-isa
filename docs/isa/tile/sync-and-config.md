# Sync And Config Instruction Set

Sync-and-config operations manage tile-visible state: resource binding, event setup, mode control, and synchronization. These operations do not produce arithmetic payload — they change state that later tile instructions consume.

## Operations

| | Operation | Description | Category | C++ Intrinsic |
|-|-----------|-------------|----------|---------------|
| | [pto.tassign](./ops/sync-and-config/tassign.md) | Bind tile register to a UB address | Resource | `TASSIGN(tile, addr)` |
| | [pto.tsync](./ops/sync-and-config/tsync.md) | Synchronize execution, wait on events, insert barrier | Sync | `TSYNC(events...)` |
| | [pto.talias](./ops/sync-and-config/talias.md) | Create an alias view that shares tile storage | View | `TALIAS(dst, src)` |
| | [pto.settf32mode](./ops/sync-and-config/settf32mode.md) | Set TF32 computation mode | Config | `SETTF32MODE(mode)` |
| | [pto.setfmatrix](./ops/sync-and-config/setfmatrix.md) | Set FMATRIX engine mode and address | Config | `SETFMATRIX(tile)` |
| | [pto.set_img2col_rpt](./ops/sync-and-config/set-img2col-rpt.md) | Set img2col repetition count | Config | `SET_IMG2COL_RPT(rpt)` |
| | [pto.set_img2col_padding](./ops/sync-and-config/set-img2col-padding.md) | Set img2col padding configuration | Config | `SET_IMG2COL_PADDING(pad)` |
| | [pto.subview](./ops/sync-and-config/subview.md) | Create a sub-view of a tile | View | `SUBVIEW(tile, offsets, shape)` |
| | [pto.get_scale_addr](./ops/sync-and-config/get-scale-addr.md) | Get scale address for quantized matmul | Config | `GET_SCALE_ADDR(tile)` |

## Mechanism

Sync-and-config operations change tile-visible state that later tile instructions consume:

- **`TASSIGN`**: binds a physical UB address to a tile register. Without `TASSIGN`, the compiler/runtime auto-assigns addresses. `TASSIGN` enables manual placement for performance tuning.
- **`TSYNC`**: waits on event tokens (`events...`) or inserts per-op pipeline barriers (`TSYNC<Op>()`). See [Ordering and Synchronization](../machine-model/ordering-and-synchronization.md) for the full event model.
- **`TALIAS`**: creates a second tile view over the same payload storage. It changes the visible tile view, not the underlying bytes.
- **`SETTF32MODE` / `SETFMATRIX` / `SET_IMG2COL_RPT` / `SET_IMG2COL_PADDING`**: tile-local configuration for TF32 computation mode, FMATRIX engine binding, and IMG2COL parameters. These program tile-side registers consumed by subsequent compute and DMA operations.
- **`SUBVIEW`**: creates a logical view of a tile with adjusted offsets and/or reduced shape. The underlying storage is shared with the source tile.
- **`GET_SCALE_ADDR`**: computes a right-shifted address of a scale tensor used in quantized matmul operations.

## Sync Model

`TSYNC` operates at two levels:

1. **Event-wait form**: `TSYNC(%e0, %e1)` blocks until the specified events have been recorded. Events are produced by preceding operations (e.g., `TLOAD` produces an event; `TSYNC` waits on it).

2. **Barrier form**: `TSYNC<Op>()` inserts a pipeline barrier for the specified operation class. All operations of class `Op` that appear before the barrier complete before any operation of class `Op` that appears after the barrier begins.

See [Producer-Consumer Ordering](../memory-model/producer-consumer-ordering.md) for the complete synchronization model.

## Constraints

- `TASSIGN` binds an address; using the same address for two non-alias tiles simultaneously results in undefined behavior.
- `TSYNC` with no operands is a no-op.
- Tile-side configuration operations affect subsequent operations until the next mode-setting operation of the same kind.
- `SUBVIEW` creates a view with reduced shape; accessing elements outside the view's shape but within the underlying tile's shape is undefined behavior.
- `TALIAS` shares storage with its source; writes through either view are visible through the other view according to the alias contract.

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
PTO_INST void SETTF32MODE(bool enable, RoundMode mode);
PTO_INST void SETFMATRIX(TileData& tile);

// Subview creation
template <typename TileT>
PTO_INST TileT SUBVIEW(TileT& src, int rowOffset, int colOffset,
                        int newRows, int newCols);

// Get scale address for quantized matmul
PTO_INST void GET_SCALE_ADDR(TileDataDst& dst, TileDataSrc& src);
```

## See Also

- [Tile instruction set](../instruction-families/tile-families.md) — Instruction set overview
- [Ordering and Synchronization](../machine-model/ordering-and-synchronization.md) — Event model
- [Tile instruction set](../instruction-families/tile-families.md) — Instruction Set description
