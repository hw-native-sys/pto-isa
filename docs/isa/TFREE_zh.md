# TFREE

## 简介

释放 `TPipe` 条目的 FIFO 空间。

对于 TileData `TPOP` 流程，`TPOP` 已经在内部执行空闲空间通知步骤。因此，面向 TileData 的 `TFREE(Pipe &pipe)` 接口当前是空操作，只是为了与 `GlobalData` 流程保持 API 对称。

对于 `GlobalData` 流程，`TFREE(Pipe&, GlobalData&)` 会释放由 `TPOP(Pipe&, GlobalData&)` 返回的 FIFO 槽位视图。

## 操作语义

对于 TileData push/pop：

1. `TPUSH(Pipe&, TileData&)` 将 tile 存入 FIFO，并记录数据就绪。
2. `TPOP(Pipe&, TileData&)` 等待数据就绪，将 FIFO 槽位加载到 tile 中，并根据 `Pipe::SyncPeriod` 通知空闲空间。
3. `TFREE(Pipe&)` 不执行额外操作。

只有在使用 `GlobalData` 分裂接口时才需要使用 `TFREE(Pipe&, GlobalData&)`。在该接口中，`TPOP` 返回 FIFO 槽位视图，由调用者显式决定何时可以释放槽位。

对于 `GlobalData`，`TFREE` 会在发出空闲空间通知前检查 `pipe.cons.getFreeStatus()` 和 `Pipe::shouldNotifyFree(...)`。

## C++ Intrinsic

声明位置：`include/pto/common/pto_instr.hpp`：

```cpp
template <typename Pipe, TileSplitAxis Split, typename... WaitEvents>
PTO_INST RecordEvent TFREE(Pipe &pipe, WaitEvents &... events);

template <typename Pipe, typename GlobalData, TileSplitAxis Split,
          std::enable_if_t<is_global_data_v<GlobalData>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TFREE(Pipe &pipe, GlobalData &gmTensor, WaitEvents &... events);
```

`include/pto/npu/a2a3/TPop.hpp` 中对应的 A2A3 实现对该重载有意保持为空（A5 上实现位于 `include/pto/npu/a5/TPop.hpp`，执行实际的空闲空间通知）：

```cpp
template <typename Pipe, TileSplitAxis Split>
PTO_INTERNAL void TFREE_IMPL(Pipe &pipe)
{
    return;
}
```

## 约束

- **TileData 流程**：
    - 当弹出的 FIFO 槽位中的数据不再需要时，使用 `TFREE(Pipe&)`。
    - 搭配使用TPUSH/TPOP/TFREE实现核间同步和数据传输，数据传输时推入的tileshape和弹出的tileshape的大小比例关系是1:1或者1:2。

- **GlobalData 流程**：
    - 当弹出的 FIFO 槽位中的数据不再需要时，使用 `TFREE(Pipe&)`。
    - `gmTensor` 只用于选择重载；实现不会读取或写入 tensor 内容。
    - 空闲空间通知是稀疏的，并由 `Pipe::SyncPeriod` 控制。

## 示例

### TileData 流程

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
AICORE void example_tiledata(__gm__ void *fifoMem)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 128;
    constexpr uint32_t FlagID = 0;
    constexpr uint32_t FifoDepth = 2;

    using Pipe = TPipe<FlagID, Direction::DIR_C2V, M * N * sizeof(T), FifoDepth>;
    using VecTile = Tile<TileType::Vec, T, M / 2, N, BLayout::RowMajor, M / 2, N>;

    Pipe pipe(fifoMem, 0x0, 0x0);
    VecTile tile;

    TPOP<Pipe, VecTile, TileSplitAxis::TILE_UP_DOWN>(pipe, tile);
    ...  // final use of VecTile
    TFREE<Pipe, TileSplitAxis::TILE_UP_DOWN>(pipe);
}
```

### GlobalData 槽位释放

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

    Pipe pipe(fifoMem, 0x0, 0x0);
    SlotGlobal slot;

    TPOP<Pipe, SlotGlobal, TileSplitAxis::TILE_UP_DOWN>(pipe, slot); // TPOP 会重新赋值 slot
    // 在此处从 slot 加载或消费数据。
    TFREE<Pipe, SlotGlobal, TileSplitAxis::TILE_UP_DOWN>(pipe, slot);
}
```
