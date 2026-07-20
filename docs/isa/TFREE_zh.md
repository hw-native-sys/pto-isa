# TFREE

## 简介

释放 `TPipe` 条目的 FIFO 空间。

对于TileData `TPOP` 流程，Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品平台上`TPOP` 已经在内部执行空闲空间通知步骤。因此，面向TileData的 `TFREE(Pipe &pipe)` 接口当前是空操作，只是为了与 `GlobalData` 流程保持API对称。Ascend 950PR/Ascend 950DT平台上TFREE会释放TPOP使用的FIFO槽位空间。

对于 `GlobalData` 流程，`TFREE(Pipe&, GlobalData&)` 会释放由 `TPOP(Pipe&, GlobalData&)` 返回的FIFO槽位视图。

## 操作语义

对于TileData流程：

1. `TPUSH(Pipe&, TileData&, Split)` 将生产者tile存入当前FIFO槽位，并为消费者记录数据就绪同步。生产者tile索引在槽位地址计算完成后递增。
2. `TPOP(Pipe&, TileData&, Split)` 等待生产者的数据就绪同步，将当前FIFO槽位加载到消费者tile中。消费者tile索引在槽位地址计算完成后递增。
3. `TFREE(Pipe&, Split)` 释放FIFO中的槽位空间。Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品平台上此接口为空操作（`TPOP` 已在内部执行空闲空间通知），Ascend 950PR/Ascend 950DT平台上会释放 `TPOP` 使用的FIFO槽位空间。

对于GlobalData流程:

1. `TALLOC(Pipe&, GlobalData&)` 从 `TPipe` 中分配一个生产者FIFO槽位，并将其暴露为 `GlobalTensor` 视图。生产者可通过 `TSTORE` 等指令向该槽位写入数据。
2. `TPUSH(Pipe&, GlobalData&)` 为已经由 `TALLOC` 分配的槽位记录数据就绪同步，将FIFO槽位提交给消费者。它本身不会存储tile数据。
3. `TPOP(Pipe&, GlobalData&)` 等待数据就绪，将 `gmTensor` 赋值为当前FIFO槽位地址，并递增消费者tile索引。它不会将数据加载到本地tile，也不会释放槽位。消费者可通过 `TLOAD` 等指令从槽位中读取数据。
4. `TFREE(Pipe&, GlobalData&)` 释放由 `TPOP(Pipe&, GlobalData&)` 返回的FIFO槽位视图，通知生产者该槽位空间已空闲。

## C++ Intrinsic

声明位置：`include/pto/common/pto_instr.hpp`：

```cpp
template <typename Pipe, TileSplitAxis Split, typename... WaitEvents>
PTO_INST RecordEvent TFREE(Pipe &pipe, WaitEvents &... events);

template <typename Pipe, typename GlobalData, TileSplitAxis Split,
          std::enable_if_t<is_global_data_v<GlobalData>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TFREE(Pipe &pipe, GlobalData &gmTensor, WaitEvents &... events);
```

`include/pto/npu/a2a3/TPop.hpp` 中对应的Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品实现对该重载有意保持为空（Ascend 950PR/Ascend 950DT上实现位于 `include/pto/npu/a5/TPop.hpp`，执行实际的空闲空间通知）：

```cpp
template <typename Pipe, TileSplitAxis Split>
PTO_INTERNAL void TFREE_IMPL(Pipe &pipe)
{
    return;
}
```

## 约束

- **TileData流程**：
    - 当弹出的FIFO槽位中的数据不再需要时，使用 `TFREE(Pipe&)`。
    - 搭配使用TPUSH/TPOP/TFREE实现核间同步和数据传输，数据传输时推入的tileshape和弹出的tileshape的大小比例关系是1:1或者1:2。

- **GlobalData流程**：
    - 当弹出的FIFO槽位中的数据不再需要时，使用 `TFREE(Pipe&)`。
    - `gmTensor` 只用于选择重载；实现不会读取或写入tensor内容。
    - 空闲空间通知是稀疏的，并由 `Pipe::SyncPeriod` 控制。

## 示例

### TileData流程

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

### GlobalData槽位释放

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

## ASM形式示例

当前公开的汇编参考尚未为 `TFREE` 定义稳定的PTO-AS写法。手写CV FIFO程序时请使用C++ intrinsic形式。
```
