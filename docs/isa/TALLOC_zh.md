# TALLOC

## 简介

从 `TPipe` 中分配一个生产者 FIFO 槽位，并将其暴露为 `GlobalTensor` 视图。

`TALLOC` 用于 `GlobalData` 分裂接口。它允许生产者获取当前 FIFO 槽位地址，使用 `TSTORE` 等普通内存指令向该槽位写入数据，然后通过 `TPUSH(Pipe&, GlobalData&)` 提交该槽位。

## 操作语义

`TALLOC` 执行三个步骤：

1. 当 `pipe.prod.getAllocateStatus()` 和 `Pipe::shouldWaitFree(pipe.prod.tileIndex)` 同时为 true 时，等待 FIFO 空闲空间。
2. 根据 `pipe.prod.tileIndex` 计算当前 FIFO 槽位地址。
3. 将 `gmTensor` 赋值为 FIFO 槽位地址，并递增生产者 tile 索引。

`TALLOC` 不写入任何数据，也不会通知消费者。生产者必须先写入槽位内容，然后再调用 `TPUSH(Pipe&, GlobalData&)`。

## C++ Intrinsic

声明位置：`include/pto/common/pto_instr.hpp`：

```cpp
template <typename Pipe, typename GlobalData, TileSplitAxis Split,
          std::enable_if_t<is_global_data_v<GlobalData>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TALLOC(Pipe &pipe, GlobalData &gmTensor, WaitEvents &... events);
```

`Pipe` 通常是 `include/pto/npu/a2a3/TPush.hpp` 中声明的 A2A3 `TPipe`：

```cpp
template <uint8_t FlagID, uint8_t DirType, uint32_t SlotSize, uint32_t SlotNum,
          uint32_t LocalSlotNum = 2, bool IsNoSplit = false, bool EN_UNIT_FLAG = false>
struct TPipe;
```

## 约束

- **A2A3 GlobalData 生产者**：
    - `GlobalData` 必须满足 `is_global_data_v<GlobalData>`。
    - `Direction::DIR_C2V`：生产者可以看到整个 FIFO 槽位。
    - `Direction::DIR_V2C`：对于向量子块，可能会根据 `Split` 应用分裂偏移。
- **FIFO 槽位**：
    - `SlotSize` 必须足够容纳一个逻辑 FIFO 条目。
    - `SlotNum >= 1`。
    - `Pipe::SyncPeriod` 由 `SlotNum` 派生：`(SlotNum <= 2) ? SlotNum : SlotNum / 2`。
- **分裂行为**：
    - `TileSplitAxis::TILE_NO_SPLIT`：不应用子向量偏移。
    - `TileSplitAxis::TILE_UP_DOWN`：向量子块映射到上下两个行半区。
    - `TileSplitAxis::TILE_LEFT_RIGHT`：向量子块映射到左右两个列半区。
- **同步**：
    - 空闲空间等待是稀疏的，并由 `Pipe::SyncPeriod` 控制。
    - `TALLOC` 不记录数据就绪；写入槽位后使用 `TPUSH(Pipe&, GlobalData&)`。

## 示例

### 分配、存储、提交

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
AICORE void example_talloc(__gm__ void *fifoMem)
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

### V2C 分裂分配

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
AICORE void example_v2c_split(__gm__ void *fifoMem)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 128;
    constexpr uint32_t FullM = M * 2;
    constexpr uint32_t FlagID = 0;
    constexpr uint32_t FifoDepth = 2;

    using Pipe = TPipe<FlagID, Direction::DIR_V2C, FullM * N * sizeof(T), FifoDepth>;
    using SlotGlobal = GlobalTensor<T, Shape<1, 1, 1, M, N>, Stride<1, 1, 1, N, 1>>;
    using VecTile = Tile<TileType::Vec, T, M, N, BLayout::RowMajor, M, N>;

    Pipe pipe(fifoMem, 0x0, 0x0);
    SlotGlobal slot;
    VecTile tile;
    TASSIGN(tile, 0x0);

    TALLOC<Pipe, SlotGlobal, TileSplitAxis::TILE_UP_DOWN>(pipe, slot);
    TSTORE(slot, tile);
    TPUSH<Pipe, SlotGlobal, TileSplitAxis::TILE_UP_DOWN>(pipe, slot);
}
```

## ASM 形式示例

当前公开的汇编参考尚未为 `TALLOC` 定义稳定的 PTO-AS 写法。手写 CV FIFO 程序时请使用 C++ intrinsic 形式。
