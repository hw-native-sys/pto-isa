# TPOP

## 简介

从 `TPipe` FIFO 中弹出消费者 tile，用于 Cube-Vector 通信。

本文同时描述 TileData 重载和 `GlobalData` 槽位视图重载。在 TileData 流程中，`TPOP` 同时执行数据就绪等待和空闲空间通知；同一个 tile 不需要再单独调用 `TFREE`。在 `GlobalData` 流程中，`TPOP` 返回 FIFO 槽位视图，调用者必须使用 `TFREE(Pipe&, GlobalData&)` 释放该槽位。

## 操作语义

对于 TileData 重载，`TPOP` 执行三个步骤：

1. 等待生产者的数据就绪同步。
2. 将当前 FIFO 槽位加载到消费者 tile 中。
3. 当 `Pipe::shouldNotifyFree(tileIndex)` 为 true 时，通知空闲空间。

消费者 tile 索引会在 FIFO 槽位地址计算完成后递增。空闲空间通知使用已经弹出的 tile 索引。

对于 `GlobalData` 重载，`TPOP` 会等待数据就绪，将 `gmTensor` 赋值为当前 FIFO 槽位地址，并递增消费者 tile 索引。它不会把数据加载到本地 tile，也不会释放槽位。

## C++ Intrinsic

声明位置：`include/pto/common/pto_instr.hpp`：

```cpp
template <typename Pipe, typename TileCons, TileSplitAxis Split,
          std::enable_if_t<is_tile_data_v<TileCons>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TPOP(Pipe &pipe, TileCons &tile, WaitEvents &... events);

template <typename Pipe, typename GlobalData, TileSplitAxis Split,
          std::enable_if_t<is_global_data_v<GlobalData>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TPOP(Pipe &pipe, GlobalData &gmTensor, WaitEvents &... events);
```

`Pipe` 通常是 `include/pto/npu/a2a3/TPush.hpp` 中声明的 A2A3 `TPipe`：

```cpp
template <uint8_t FlagID, uint8_t DirType, uint32_t SlotSize, uint32_t SlotNum,
          uint32_t LocalSlotNum = 2, bool IsNoSplit = false, bool EN_UNIT_FLAG = false>
struct TPipe;
```

## 约束

- **A2A3 TileData 消费者**：
    - `TileCons::Loc` 必须是 `TileType::Vec` 或 `TileType::Mat`。
    - `Direction::DIR_C2V`：vector 消费 cube 生产的数据。
    - `Direction::DIR_V2C`：cube 消费 vector 生产的数据。
    - `Direction::DIR_BOTH`：同一个 pipe 类型同时支持 C2V 和 V2C 消费者。
- **本地消费者缓冲区**：
    - 对于 C2V vector 消费者，`TPipe` 会将 tile 分配到 `C2V_CONSUMER_BUF`，并使用本地 FIFO 轮转。
    - 对于 V2C matrix 消费者，`TPipe` 会将 tile 分配到 `V2C_CONSUMER_BUF`，并使用本地 FIFO 轮转。
- **分裂行为**：
    - `TileSplitAxis::TILE_NO_SPLIT`：不应用子向量偏移。
    - `TileSplitAxis::TILE_UP_DOWN`：向量子块消费上下两个行半区。
    - `TileSplitAxis::TILE_LEFT_RIGHT`：向量子块消费左右两个列半区。
- **同步**：
    - 每次 `TPOP` 都会执行数据就绪等待。
    - 空闲空间通知是稀疏的，并由 `Pipe::SyncPeriod` 控制。
- **GlobalData 槽位视图**：
    - `gmTensor` 会被赋值为由 `pipe.cons.tileIndex` 选择的 FIFO 槽位基地址。
    - 对于 C2V 分裂模式，会根据 `Split` 应用向量子块偏移。
    - 从槽位视图完成所有加载后，调用者必须调用 `TFREE(Pipe&, GlobalData&)`。

## 示例

### C2V Vector Pop

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
AICORE void example_c2v(__gm__ void *fifoMem)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 128;
    constexpr uint32_t FlagID = 0;
    constexpr uint32_t FifoDepth = 2;
    constexpr uint32_t LocalBase = 0x0;

    using Pipe = TPipe<FlagID, Direction::DIR_C2V, M * N * sizeof(T), FifoDepth>;
    using VecTile = Tile<TileType::Vec, T, M / 2, N, BLayout::RowMajor, M / 2, N>;

    Pipe pipe(fifoMem, LocalBase, 0x0);
    VecTile tile;

    TPOP<Pipe, VecTile, TileSplitAxis::TILE_UP_DOWN>(pipe, tile);
}
```

### V2C Matrix Pop

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
AICORE void example_v2c(__gm__ void *fifoMem)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 128;
    constexpr uint32_t FlagID = 0;
    constexpr uint32_t FifoDepth = 2;
    constexpr uint32_t LocalBase = 0x0;

    using Pipe = TPipe<FlagID, Direction::DIR_V2C, M * N * sizeof(T), FifoDepth>;
    using MatTile = Tile<TileType::Mat, T, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;

    Pipe pipe(fifoMem, 0x0, LocalBase);
    MatTile tile;

    TPOP<Pipe, MatTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, tile);
}
```

### GlobalData 槽位弹出

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
AICORE void example_globaldata(__gm__ void *fifoMem)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 128;
    constexpr uint32_t FlagID = 0;
    constexpr uint32_t FifoDepth = 2;

    using Pipe = TPipe<FlagID, Direction::DIR_C2V, M * N * sizeof(T), FifoDepth>;
    using SlotGlobal = GlobalTensor<T, Shape<1, 1, 1, M / 2, N>, Stride<1, 1, 1, N, 1>>;
    using VecTile = Tile<TileType::Vec, T, M / 2, N, BLayout::RowMajor, M / 2, N>;

    Pipe pipe(fifoMem, 0x0, 0x0);
    SlotGlobal slot;
    VecTile tile;
    TASSIGN(tile, 0x0);

    TPOP<Pipe, SlotGlobal, TileSplitAxis::TILE_UP_DOWN>(pipe, slot);
    TLOAD(tile, slot);
    TFREE<Pipe, SlotGlobal, TileSplitAxis::TILE_UP_DOWN>(pipe, slot);
}
```

## ASM 形式示例

当前公开的汇编参考尚未为 `TPOP` 定义稳定的 PTO-AS 写法。手写 CV FIFO 程序时请使用 C++ intrinsic 形式。
