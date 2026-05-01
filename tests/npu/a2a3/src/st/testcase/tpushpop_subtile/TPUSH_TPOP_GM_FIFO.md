# TALLOC / TPUSH / TPOP / TFREE

## Introduction

These interfaces provide an address-based GM FIFO protocol for producer/consumer communication through `TPipe`.
Unlike the tile-based `TPUSH(pipe, tile)` and `TPOP(pipe, tile)` forms, these APIs only manage FIFO synchronization and assign the GM slot address into a `GlobalTensor` object. The caller is responsible for moving data with ordinary `TSTORE` and `TLOAD`.

This is useful when a producer or consumer needs to access a FIFO slot as one or more `GlobalTensor` views, for example when a cube core stores several subtiles into one FIFO slot and vector cores later load slices from that slot.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename Pipe, typename GlobalData, TileSplitAxis Split,
          std::enable_if_t<is_global_data_v<GlobalData>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TALLOC(Pipe &pipe, GlobalData &gmTensor, WaitEvents &... events);

template <typename Pipe, typename GlobalData, TileSplitAxis Split,
          std::enable_if_t<is_global_data_v<GlobalData>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe &pipe, GlobalData &gmTensor, WaitEvents &... events);

template <typename Pipe, typename GlobalData, TileSplitAxis Split,
          std::enable_if_t<is_global_data_v<GlobalData>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TPOP(Pipe &pipe, GlobalData &gmTensor, WaitEvents &... events);

template <typename Pipe, typename GlobalData, TileSplitAxis Split,
          std::enable_if_t<is_global_data_v<GlobalData>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TFREE(Pipe &pipe, GlobalData &gmTensor, WaitEvents &... events);
```

## Parameters

- `Pipe`: a `TPipe<FlagID, Direction, SlotSize, SlotNum, ...>` type. The pipe owns FIFO state, slot base address, producer/consumer counters, and cross-core flags.
- `GlobalData`: a `GlobalTensor` type used for address calculation. Its `RawDType` and static shape determine byte offsets for split modes.
- `gmTensor`: a runtime `GlobalTensor` object. `TALLOC` and `TPOP` assign the computed FIFO GM address into this object, while `TPUSH` and `TFREE` use it as the transaction descriptor.
- `Split`: one of `TileSplitAxis::TILE_NO_SPLIT`, `TileSplitAxis::TILE_UP_DOWN`, or `TileSplitAxis::TILE_LEFT_RIGHT`.
- `events`: optional PTO events to synchronize before the interface body executes.

## Producer Flow

### `TALLOC`

`TALLOC<Pipe, GlobalData, Split>(pipe, gmTensor)` begins a producer-side FIFO transaction and assigns the GM address for the current FIFO slot into `gmTensor`.

It performs:

1. `TSYNC(events...)`
2. Producer-side allocation/wait for a free FIFO slot
3. FIFO slot address calculation
4. Producer tile index increment
5. `gmTensor` address assignment

It does not write data and does not notify the consumer. After `TALLOC`, the caller should use `TSTORE` or other GM-writing logic to populate the slot described by `gmTensor`.

### `TPUSH`

`TPUSH<Pipe, GlobalData, Split>(pipe, gmTensor)` commits the producer transaction.

It performs:

1. `TSYNC(events...)`
2. Producer-side record/notify so the consumer can observe the data-ready flag

Call `TPUSH` only after all writes to the FIFO slot are complete.

## Consumer Flow

### `TPOP`

`TPOP<Pipe, GlobalData, Split>(pipe, gmTensor)` begins a consumer-side FIFO transaction and assigns the GM address for the current FIFO slot or subslot into `gmTensor`.

It performs:

1. `TSYNC(events...)`
2. Consumer-side wait for producer data-ready
3. FIFO slot address calculation
4. Consumer tile index increment
5. `gmTensor` address assignment

It does not load data and does not free the slot. After `TPOP`, the caller can use `gmTensor` directly or create additional `GlobalTensor` views from `gmTensor.data()` and use `TLOAD` to read data.

### `TFREE`

`TFREE<Pipe, GlobalData, Split>(pipe, gmTensor)` completes the consumer transaction.

It performs:

1. `TSYNC(events...)`
2. Consumer-side free/notify so the producer can reuse the FIFO slot

Call `TFREE` only after all reads from the FIFO slot are complete.

## Split Address Calculation

For `DIR_C2V`, vector consumers receive the same FIFO slot base address and can build their own `GlobalTensor` views from it. The `GlobalData` template argument used by `TPOP` normally matches the producer slot descriptor, so `TPOP` assigns the slot address into `gmTensor`. Consumers can then add explicit element offsets from `gmTensor.data()` before constructing smaller load views.

- `TILE_NO_SPLIT`: no sub-core offset is added.
- `TILE_UP_DOWN`: sub-core offset is `get_subblockid() * rows * cols * sizeof(dtype)`.
- `TILE_LEFT_RIGHT`: sub-core offset is `get_subblockid() * cols * sizeof(dtype)`.

For `DIR_V2C`, producer-side split offsets are computed similarly when vector cores write separate subregions for a cube consumer.

`GlobalData::staticShape[DIM_3]`, `GlobalData::staticShape[DIM_4]`, and `GlobalData::RawDType` are used for split offset calculations. When the consumer wants to manually load subtiles from a full FIFO slot, use the same full-slot `GlobalData` type for `TALLOC` and `TPOP`, then construct narrower `GlobalTensor` load views from `gmTensor.data()`.

## Example

```cpp
constexpr int M = 128;
constexpr int N = 128;
constexpr int RepeatN = 4;
constexpr int FullN = N * RepeatN;
constexpr int VecCores = 2;
constexpr int VecM = 16;
constexpr int VecLoadTimes = M / (VecCores * VecM);

using Pipe = TPipe<0, Direction::DIR_C2V, M * FullN * sizeof(float), 2>;
using SlotGlobal = GlobalTensor<float, Shape<1, 1, 1, M, FullN>, Stride<1, 1, 1, FullN, 1>>;
using StoreGlobal = GlobalTensor<float, Shape<1, 1, 1, M, N>, Stride<1, 1, 1, FullN, 1>>;

// TPOP uses this descriptor for split offset calculation, so each vector receives one [64, 512] half.
using PopGlobal = GlobalTensor<float, Shape<1, 1, 1, M / VecCores, FullN>, Stride<1, 1, 1, FullN, 1>>;

// One 3D TLOAD gathers four [16, 128] blocks into one logical [16, 512] vector tile.
using VecTileData = Tile<TileType::Vec, float, RepeatN * VecM, N, BLayout::RowMajor, RepeatN * VecM, N>;
using LoadGlobal3D = GlobalTensor<float, Shape<1, 1, RepeatN, VecM, N>, Stride<1, 1, N, FullN, 1>>;
using OutGlobal3D = GlobalTensor<float, Shape<1, 1, RepeatN, VecM, N>, Stride<1, 1, N, FullN, 1>>;

Pipe pipe(fifoMem, 0x0, 0x0);

// Cube producer: fill one [128, 512] FIFO slot with four [128, 128] stores.
SlotGlobal pushGlobal;
TALLOC<Pipe, SlotGlobal, TileSplitAxis::TILE_UP_DOWN>(pipe, pushGlobal);
for (int nTile = 0; nTile < RepeatN; ++nTile) {
    StoreGlobal storeGlobal(pushGlobal.data() + nTile * N);
    TSTORE(storeGlobal, accTile);
}
TPUSH<Pipe, SlotGlobal, TileSplitAxis::TILE_UP_DOWN>(pipe, pushGlobal);

// Vector consumer: each vector consumes four [16, 512] row slices from its [64, 512] half.
VecTileData vecTile;
VecTileData dstTile;
TASSIGN(vecTile, 0x0);
TASSIGN(dstTile, 0x10000);

PopGlobal popGlobal;
TPOP<Pipe, PopGlobal, TileSplitAxis::TILE_UP_DOWN>(pipe, popGlobal);

uint32_t subBlockIdx = get_subblockid();
for (int rowSlice = 0; rowSlice < VecLoadTimes; ++rowSlice) {
    size_t vecBaseRow = static_cast<size_t>(M / VecCores) * static_cast<size_t>(subBlockIdx);
    size_t localRowOffset = static_cast<size_t>(rowSlice * VecM);
    size_t outRowOffset = (vecBaseRow + localRowOffset) * static_cast<size_t>(FullN);

    LoadGlobal3D loadGlobal(popGlobal.data() + localRowOffset * static_cast<size_t>(FullN));
    TLOAD(vecTile, loadGlobal);
    TADDS(dstTile, vecTile, static_cast<float>(3.14));

    OutGlobal3D outGlobal(out + outRowOffset);
    TSTORE(outGlobal, dstTile);
}
TFREE<Pipe, PopGlobal, TileSplitAxis::TILE_UP_DOWN>(pipe, popGlobal);
```

## Notes

- These interfaces are address-based. They intentionally do not perform `TSTORE` or `TLOAD` internally.
- `TALLOC` must be paired with `TPUSH`.
- `TPOP` must be paired with `TFREE`.
- `TALLOC` and `TPOP` write the computed GM address into the provided `GlobalTensor`.
- The `GlobalData` template argument is used as a type-level shape/dtype descriptor, and the runtime `gmTensor` carries the computed GM address for later `TSTORE`, `TLOAD`, `TPUSH`, or `TFREE` calls.
