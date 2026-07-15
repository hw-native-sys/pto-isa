# TPUSH

## 简介

将生产者 tile 推入FIFO中，用于 Cube-Vector之间的数据传输和核间同步。

本指令支持多类数据的推送，包括基于 `TileSplitAxis` 的 Tile 重载、简化版 Tile 重载（参数顺序相反、无需 Split）、GlobalTensor 重载以及基于 TConfig 的重载。

## 操作语义
对于 TileData 流程：

1. `TPUSH(Pipe&, TileData&, Split)` 将生产者 tile 存入当前 FIFO 槽位，并为消费者记录数据就绪同步。生产者 tile 索引在槽位地址计算完成后递增。
2. `TPOP(Pipe&, TileData&, Split)` 等待生产者的数据就绪同步，将当前 FIFO 槽位加载到消费者 tile 中。消费者 tile 索引在槽位地址计算完成后递增。
3. `TFREE(Pipe&, Split)` 释放 FIFO 中的槽位空间。A2A3 平台上此接口为空操作（`TPOP` 已在内部执行空闲空间通知），A5 平台上会释放 `TPOP` 使用的 FIFO 槽位空间。

对于 GlobalData 流程:

1. `TALLOC(Pipe&, GlobalData&)` 从 `TPipe` 中分配一个生产者 FIFO 槽位，并将其暴露为 `GlobalTensor` 视图。生产者可通过 `TSTORE` 等指令向该槽位写入数据。
2. `TPUSH(Pipe&, GlobalData&)` 为已经由 `TALLOC` 分配的槽位记录数据就绪同步，将 FIFO 槽位提交给消费者。它本身不会存储 tile 数据。
3. `TPOP(Pipe&, GlobalData&)` 等待数据就绪，将 `gmTensor` 赋值为当前 FIFO 槽位地址，并递增消费者 tile 索引。它不会将数据加载到本地 tile，也不会释放槽位。消费者可通过 `TLOAD` 等指令从槽位中读取数据。
4. `TFREE(Pipe&, GlobalData&)` 释放由 `TPOP(Pipe&, GlobalData&)` 返回的 FIFO 槽位视图，通知生产者该槽位空间已空闲。

对于 `TConfig` 重载 `TPUSH(Pipe&, TileProd&, TConfig)`，`TConfig` 模板参数用于配置L0C->GM/UB的 fixpipe 参数。

## C++ Intrinsic

声明位置：`include/pto/common/pto_instr.hpp`：

```cpp
template <typename Pipe, typename TileProd, TileSplitAxis Split,
          std::enable_if_t<is_tile_data_v<TileProd>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe &pipe, TileProd &tile, WaitEvents &... events);

template <typename Pipe, typename GlobalData, TileSplitAxis Split,
          std::enable_if_t<is_global_data_v<GlobalData>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe &pipe, GlobalData &gmTensor, WaitEvents &... events);

template <typename Pipe, typename TileProd, typename TConfig, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe &pipe, TileProd &tile, WaitEvents &... events);
```

`Pipe` 通常是 `TPush.hpp` 中声明的  `TPipe`类型：

```cpp
template <uint8_t FlagID, uint8_t DirType, uint32_t SlotSize, uint32_t SlotNum,
          uint32_t LocalSlotNum = 2, bool IsNoSplit = false, bool EN_UNIT_FLAG = false>
struct TPipe;
```

## 约束

- **TileData 类型生产者**：
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
    - `TileSplitAxis::TILE_UP_DOWN`：将数据按照上下切分。当Cube->Vector方向且L0C->UB通路时，该切分模式仅支持数据类型为b32，且srcTile的validRows必须为2的整数倍；当Vector->Cube方向且UB->L1通路时，该切分模式下validCols必须是32字节的整数倍。
    - `TileSplitAxis::TILE_LEFT_RIGHT`：将数据按照左右切分成两个列半区。当Cube->Vector方向且L0C->UB通路时，该切分模式仅支持数据类型为b32，且srcTile的validCols必须为32的整数倍。当Vector->Cube方向且UB->L1通路时，该切分模式下validCols必须是32字节的整数倍。
- **简化版 TileData 接口**：
    - `TPUSH(TileData&, Pipe&)` 内部使用 `TileSplitAxis::TILE_NO_SPLIT` 语义。
    - `TileData::Loc` 必须为 `TileType::Acc` 或 `TileType::Vec`。
- **TConfig 接口**：
    - `TConfig` 是决定推送行为的配置类型（实现定义）。
    - `TileProd::Loc` 必须为 `TileType::Acc`、`TileType::Vec` 或 `TileType::Ctrl`。
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

## 定义 TConfig

`TPUSH(Pipe&, TileProd&, TConfig)` 重载中的 `TConfig` 模板参数是一个配置结构体，用于控制推送过程中的 fixpipe 行为。PTO 提供了 `FixpipeParams` 结构体来实现此功能。

声明于 `include/pto/common/fixpipe.hpp`：

```cpp
template <LayoutMode_t layoutMode = LayoutMode_t::NZ2ND,
          QuantMode_t quantMode = QuantMode_t::NoQuant,
          ReluPreMode reluMode = ReluPreMode::NoRelu,
          STPhase phase = STPhase::Unspecified,
          uint8_t subBlockId = 0,
          AtomicType atomicT = AtomicType::AtomicNone,
          ClipReluMode_t clipReluMode = ClipReluMode_t::NOCLIP_RELU,
          bool isChannelSplit = false>
struct FixpipeParams {
    static constexpr LayoutMode_t LayoutMode = layoutMode;
    static constexpr QuantMode_t QuantPre = quantMode;
    static constexpr ReluPreMode ReluMode = reluMode;
    static constexpr STPhase Phase = phase;
    static constexpr uint8_t SubBlockId = subBlockId;
    static constexpr AtomicType AtomicT = atomicT;
    static constexpr ClipReluMode_t ClipReluMode = clipReluMode;
    static constexpr bool IsChannelSplit = isChannelSplit;
};
```

### TConfig 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `LayoutMode` | `LayoutMode_t` | 输出数据布局：`NZ2NZ`（NZ→NZ）、`NZ2ND`（NZ→行主序）、`NZ2DN`（NZ→列主序）。默认值：`NZ2ND`。 |
| `QuantPre` | `QuantMode_t` | 量化/反量化模式（由 CANN 定义）。控制 fixpipe 推送过程中的数据类型转换。默认值：`NoQuant`。 |
| `ReluMode` | `ReluPreMode` | ReLU 激活模式：`NoRelu` 或 `NormalRelu`。默认值：`NoRelu`。 |
| `Phase` | `STPhase` | 存储阶段（用于 unit-flag 路径）：`Unspecified`、`Partial` 或 `Final`。默认值：`Unspecified`。 |
| `SubBlockId` | `uint8_t` | 子块标识符，用于累加器到向量搬运模式映射（仅 A5）。默认值：`0`。 |
| `AtomicT` | `AtomicType` | GM 写入的原子操作类型：`AtomicNone` 或 `AtomicAdd`。默认值：`AtomicNone`。 |
| `ClipReluMode` | `ClipReluMode_t` | Clip ReLU 模式：`NOCLIP_RELU` 或 `CLIP_RELU`。默认值：`NOCLIP_RELU`。 |
| `IsChannelSplit` | `bool` | 是否启用通道切分。默认值：`false`。 |

### TConfig 使用示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
AICORE void example_tconfig_push(__gm__ void *fifoMem)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 128;
    constexpr uint32_t FlagID = 0;
    constexpr uint32_t FifoDepth = 2;

    using Pipe = TPipe<FlagID, Direction::DIR_C2V, M * N * sizeof(T), FifoDepth>;
    using AccTile = TileAcc<float, M, N, M, N>;

    // 定义 TConfig：NZ→行主序布局，反量化到 half，启用 ReLU
    using MyConfig = FixpipeParams<LayoutMode_t::NZ2ND, QuantMode_t::DEQF16, ReluPreMode::NormalRelu>;

    Pipe pipe(fifoMem, 0x0, 0x0);
    AccTile acc;
    TASSIGN(acc, 0x0);

    TPUSH<Pipe, AccTile, MyConfig>(pipe, acc);
}
```

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
```
