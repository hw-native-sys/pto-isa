# TPUSH

## 简介

将生产者 tile 推入FIFO中，用于 Cube-Vector之间的数据传输和核间同步。

本指令支持两类数据的推送，分别为Tile类型的数据和GlobalTensor类型的数据。所以分别设计了基于`Tile` 重载和 `GlobalTensor` 重载。

## 操作语义

对于 Tile类型接口，`TPUSH` 执行三个步骤：

1. 当 `Pipe::shouldWaitFree(pipe.prod.tileIndex)` 为 true 时，等待 FIFO 空间。
2. 将生产者 tile 存入当前 FIFO 槽位。该步骤中：
   - 如果是Cube->Vector的数据推送，会将AccTile推送到TPipe的FIFO中
   - 如果是Vector->Cube的数据推送，会将VecTile推送到TPipe的FIFO中
3. 为消费者记录数据就绪同步。

生产者 tile 索引会在 FIFO 槽位地址计算完成后递增。

对于 `GlobalData` 类型接口，`TPUSH` 只为已经由 `TALLOC` 分配的槽位记录数据就绪同步。它本身不会存储 tile 数据。

## C++ Intrinsic

声明位置：`include/pto/common/pto_instr.hpp`：

```cpp
template <typename Pipe, typename TileProd, TileSplitAxis Split,
          std::enable_if_t<is_tile_data_v<TileProd>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe &pipe, TileProd &tile, WaitEvents &... events);

template <typename Pipe, typename GlobalData, TileSplitAxis Split,
          std::enable_if_t<is_global_data_v<GlobalData>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe &pipe, GlobalData &gmTensor, WaitEvents &... events);
```

`Pipe` 通常是 `TPush.hpp` 中声明的  `TPipe`类型：

```cpp
template <uint8_t FlagID, uint8_t DirType, uint32_t SlotSize, uint32_t SlotNum,
          uint32_t LocalSlotNum = 2, bool IsNoSplit = false, bool EN_UNIT_FLAG = false>
struct TPipe;
```

## 约束

- **TileData 生产者**：
    - `TileProd::Loc` 必须是 `TileType::Acc`、`TileType::Vec` 或 `TileType::Ctrl`。
    - `Direction::DIR_C2V`：Cube 生产 accumulator tile，供 vector 消费。
    - `Direction::DIR_V2C`：Vector 生产 vector tile，供 cube 消费。
    - `Direction::DIR_BOTH`：同一个 pipe 类型同时支持 C2V 和 V2C 生产者。
- **FIFO 槽位**：
    - `SlotSize` 必须足够容纳一个逻辑 FIFO 条目。
    - `SlotNum >= 1`。
- **A2A3切分行为**：
    - `TileSplitAxis::TILE_NO_SPLIT`：不做切分。在A2A3上要使能此切分模式，需要AIV0,AIV1陪跑同步操作。
    - `TileSplitAxis::TILE_UP_DOWN`：向量子块映射到上下两个行半区。 
    - `TileSplitAxis::TILE_LEFT_RIGHT`：向量子块映射到左右两个列半区。
- **A5切分行为**：
    - `TileSplitAxis::TILE_NO_SPLIT`：不做切分。
    - `TileSplitAxis::TILE_UP_DOWN`：将数据按照上下切分。当Cube->Vector方向且L0C->UB通路时，该切分模式仅支持数据类型为b32，且srcTile的validRows必须2的整数倍；当Vector->Cube方向且UB->L1通路时，该切分模式下validCols必须是32bytes的整数倍。
    - `TileSplitAxis::TILE_LEFT_RIGHT`：将数据按照左右切分成两个列半区。当Cube->Vector方向且L0C->UB通路时，该切分模式仅支持数据类型为b32，且srcTile的validCols必须为32的整数倍。当Vector->Cube方向且UB->L1通路时，该切分模式下validCols必须是32bytes的整数倍。
- **同步**：
    - 空闲空间等待是稀疏的，并由 `Pipe::SyncPeriod` 控制。
    - 每次 `TPUSH` 都会发出数据就绪记录。
- **GlobalData 类型生产者**：
    - `gmTensor` 必须是由 `TALLOC` 返回的 FIFO 槽位视图。
    - 调用 `TPUSH(Pipe&, GlobalData&)` 之前，数据必须已经写入 `gmTensor`。
    - `TPUSH(Pipe&, GlobalData&)` 忽略 tensor 内容，只将 FIFO 槽位提交给消费者。
- **Tile 类型支持**：
    - **TPUSH/TPOP 支持的 Tile 类型**：
        - `TileType::Acc`（累加器 Tile）：Cube 核心使用，用于 C2V 方向通信。
        - `TileType::Vec`（向量 Tile）：Vector 核心使用，用于 V2C 方向通信。
        - `TileType::Ctrl`（控制 Tile）：Vector 核心使用，用于 V2C_CTRL 方向的控制信号通信。

## 示例

### C2V Accumulator Push

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

    using Pipe = TPipe<FlagID, Direction::DIR_C2V, M * N * sizeof(float), FifoDepth>;
    using AccTile = TileAcc<float, M, N, M, N>;

    Pipe pipe(fifoMem, 0x0, 0x0);
    AccTile acc;
    TASSIGN(acc, 0x0);

    // Fill acc with a cube computation before pushing.
    TPUSH<Pipe, AccTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, acc);
}
```

### V2C Vector Push

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

    using Pipe = TPipe<FlagID, Direction::DIR_V2C, M * N * sizeof(T), FifoDepth>;
    using VecTile = Tile<TileType::Vec, T, M, N, BLayout::RowMajor, M, N>;

    Pipe pipe(fifoMem, 0x0, 0x0);
    VecTile tile;
    TASSIGN(tile, 0x0);

    TPUSH<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, tile);
}
```

### GlobalData 推送示例

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
    using SlotGlobal = GlobalTensor<T, Shape<1, 1, 1, M, N>, Stride<1, 1, 1, N, 1>>;
    using VecTile = Tile<TileType::Vec, T, M, N, BLayout::RowMajor, M, N>;

    Pipe pipe(fifoMem, 0x0, 0x0);
    SlotGlobal slot;
    VecTile tile;
    TASSIGN(tile, 0x0);

    TALLOC<Pipe, SlotGlobal, TileSplitAxis::TILE_NO_SPLIT>(pipe, slot);
    TSTORE(slot, tile);
    TPUSH<Pipe, SlotGlobal, TileSplitAxis::TILE_NO_SPLIT>(pipe, slot);
}
```

## ASM 形式示例

当前公开的汇编参考尚未为 `TPUSH` 定义稳定的 PTO-AS 写法。手写 CV FIFO 程序时请使用 C++ intrinsic 形式。
