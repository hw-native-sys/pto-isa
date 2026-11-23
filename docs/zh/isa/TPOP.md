# TPOP

## 标量控制 FIFO

A5 NPU 的 `TPipe` 控制通路支持 V2C 的 OR/AND 谓词归约与 C2V 标量广播。
`CtrlPipeConfig` 可选择参与的 AIV，并独立启用或禁用前向和反向依赖。
共享的 A6 后端仅支持 AIV0。控制 FIFO 按消息释放槽位，不使用数据 FIFO 的批量释放规则。
控制通路的模板参数、存储布局及生命周期约定见
[标量控制 FIFO](../../../tests/npu/a5/src/st/testcase/tpushpop_ctrl/README.md)；下文的数据 Tile 约束仍适用于数据 FIFO。

## 产品支持情况<a name="section73648168211"></a>

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：支持
<!-- end id3 -->


## 简介

从 `TPipe` FIFO中弹出消费者tile，用于Cube-Vector通信。

本指令支持显式指定 `TileSplitAxis`、显式指定subblock ID、参数反转且不指定 `Split` 的TileData重载，以及 `GlobalData` 重载和 A6R2 MaxTrack 多 Tile 重载。

## 操作语义

对于TileData流程：

1. `TPUSH(Pipe&, TileData&, Split)` 将生产者tile存入当前FIFO槽位，并为消费者记录数据就绪同步。生产者tile索引在槽位地址计算完成后递增。
2. `TPOP(Pipe&, TileData&, Split)` 等待生产者的数据就绪同步，将当前FIFO槽位加载到消费者tile中。消费者tile索引在槽位地址计算完成后递增。
3. `TFREE(Pipe&, Split)` 释放FIFO中的槽位空间。Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品平台上此接口为空操作（`TPOP` 已在内部执行空闲空间通知），Ascend 950PR/Ascend 950DT平台上会释放 `TPOP` 使用的FIFO槽位空间。

对于GlobalData流程：

1. `TALLOC(Pipe&, GlobalData&)` 从 `TPipe` 中分配一个生产者FIFO槽位，并将其暴露为 `GlobalTensor` 视图。生产者可通过 `TSTORE` 等指令向该槽位写入数据。
2. `TPUSH(Pipe&, GlobalData&)` 为已经由 `TALLOC` 分配的槽位记录数据就绪同步，将FIFO槽位提交给消费者。它本身不会存储tile数据。
3. `TPOP(Pipe&, GlobalData&)` 等待数据就绪，将 `gmTensor` 赋值为当前FIFO槽位地址，并递增消费者tile索引。它不会将数据加载到本地tile，也不会释放槽位。消费者可通过 `TLOAD` 等指令从槽位中读取数据。
4. `TFREE(Pipe&, GlobalData&)` 释放由 `TPOP(Pipe&, GlobalData&)` 返回的FIFO槽位视图，通知生产者该槽位空间已空闲。

### A6R2 MaxTrack（C2V Acc→UB）

在 A6R2 上，当生产者使用带 `FixpipeParams<..., MaxTrack<groupmax_en=true, ...>>` 的 `TPUSH` 时，消费者需同时弹出主 vec tile 与 MaxTrack 旁路：

1. 在 C2V FIFO 上等待数据就绪（与普通 TileData 路径相同）。
2. 将 `vecTile` 绑定到当前 `C2V_CONSUMER_BUF` 槽位。
3. 将 `gMaxTile` 绑定到对应的 `C2V_GROUPMAX_BUF` 槽位（`TPOP(pipe, vec, gMax)`）。
4. 若同时启用了 RowMax，再将 `rMaxTile` 绑定到 `C2V_ROWMAX_BUF`（`TPOP(pipe, vec, gMax, rMax)`）。
5. 消费完成后调用 `TFREE` 释放 FIFO 槽位。

上述重载仅在定义了 `PTO_NPU_ARCH_A6R2` 时编译。实现位于 `include/pto/npu/a6r2/TPop.hpp`。

## C++ Intrinsic

声明位置：`include/pto/common/pto_instr.hpp`：

```cpp
template <typename Pipe, typename TileCons, TileSplitAxis Split,
          std::enable_if_t<is_tile_data_v<TileCons>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TPOP(Pipe &pipe, TileCons &tile, WaitEvents &... events);

template <typename Pipe, typename TileCons, TileSplitAxis Split, typename... WaitEvents>
PTO_INST RecordEvent TPOP(Pipe &pipe, TileCons &tile, int32_t subBlockId,
                          WaitEvents &... events);

template <typename TileData, typename Pipe, typename... WaitEvents>
PTO_INST RecordEvent TPOP(TileData &tile, Pipe &pipe, WaitEvents &... events);

template <typename Pipe, typename GlobalData, TileSplitAxis Split,
          std::enable_if_t<is_global_data_v<GlobalData>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TPOP(Pipe &pipe, GlobalData &gmTensor, WaitEvents &... events);

#if defined(PTO_NPU_ARCH_A6R2)
// A6R2 MaxTrack: vec + gMax  /  vec + gMax + rMax
// TConfig 提供 group_num（与对应 TPUSH 使用同一套 FixpipeParams / MaxTrack 配置）。
template <typename Pipe, typename VecTile, typename GMaxTile, typename TConfig,
          TileSplitAxis Split,
          std::enable_if_t<is_tile_data_v<VecTile> && is_tile_data_v<GMaxTile>, int> = 0,
          typename... WaitEvents>
PTO_INST RecordEvent TPOP(Pipe &pipe, VecTile &vecTile, GMaxTile &gMaxTile,
                          WaitEvents &... events);

template <typename Pipe, typename VecTile, typename GMaxTile, typename RMaxTile,
          typename TConfig, TileSplitAxis Split, bool EndRowMaxCmp = false,
          std::enable_if_t<is_tile_data_v<VecTile> && is_tile_data_v<GMaxTile> &&
                           is_tile_data_v<RMaxTile>, int> = 0,
          typename... WaitEvents>
PTO_INST RecordEvent TPOP(Pipe &pipe, VecTile &vecTile, GMaxTile &gMaxTile,
                          RMaxTile &rMaxTile, WaitEvents &... events);
#endif
```

`Pipe` 通常是所选后端 `TPush.hpp` 中声明的 `TPipe`，包括 `include/pto/cpu/TPush.hpp`、
`include/pto/npu/a2a3/TPush.hpp`、`include/pto/npu/a5/TPush.hpp`，
以及 `include/pto/npu/a6r2/TPush.hpp` 中的 A6R2 MaxTrack 构造函数：

```cpp
template <uint8_t FlagID, uint8_t DirType, uint32_t SlotSize, uint32_t SlotNum,
          uint32_t LocalSlotNum = 2, bool IsNoSplit = false, bool EN_UNIT_FLAG = false>
struct TPipe;
```

## 约束

- **TileData消费者**：
    - `TileCons::Loc` 必须是 `TileType::Vec`、`TileType::Mat` 或 `TileType::Ctrl`。
    - `Direction::DIR_C2V`：vector消费cube生产的数据。
    - `Direction::DIR_V2C`：cube消费vector生产的数据。
    - `Direction::DIR_BOTH`：同一个pipe类型同时支持C2V和V2C消费者。
- **本地消费者缓冲区**：
    - 对于C2V vector消费者，`TPipe` 会将tile分配到 `C2V_CONSUMER_BUF`，并使用本地FIFO轮转。
    - 对于V2C matrix消费者，`TPipe` 会将tile分配到 `V2C_CONSUMER_BUF`，并使用本地FIFO轮转。
- **切分行为**：
    - `TileSplitAxis::TILE_NO_SPLIT`：不进行切分，Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品上需要AIV0/AIV1陪跑核间同步操作。
    - `TileSplitAxis::TILE_UP_DOWN`：消费上下两个行半区。
    - `TileSplitAxis::TILE_LEFT_RIGHT`：消费左右两个列半区。
- **A6切分行为**：
    - 仅支持 `TileSplitAxis::TILE_NO_SPLIT`，因为 A6 硬件只支持一个 AIC 与一个 AIV 通信。
- **A6R2 MaxTrack 重载**：
    - 仅在定义了 `PTO_NPU_ARCH_A6R2` 时可用；要求 C2V UB FIFO（`Pipe::is_c2v_ub` 或 `Pipe::is_both`）。
    - `TPOP(pipe, vecTile, gMaxTile)` 对应生产者 `groupmax_en=true, rowmax_en=false`。
    - `TPOP(pipe, vecTile, gMaxTile, rMaxTile)` 对应生产者 `groupmax_en=true, rowmax_en=true`。
    - 生产者 MaxTrack 要求 `LayoutMode::NZ2ND` 或 `LayoutMode::NZ2DN`；`vecTile` 须无 fractal（NZ2ND 为 RowMajor，NZ2DN 为 ColMajor）。
    - 数据类型必须是 `float`、`half` 或 `bfloat16_t`；`vec` / `gMax` / `rMax` 类型须一致。
    - 消费者 Tile 须与 MaxTrack push 形状一致：NZ2ND/NZ2DN fixpipe vec；GroupMax 的 ValidRow/ValidCol 为 `DYNAMIC`（NZ2ND `[srcValidRow, srcValidCol/group_num]`，NZ2DN 转置）；RowMax 为 `[1, ceil32B(ProdM)]`（`DYNAMIC` validCol）。`TPOP` 需传入 `TConfig`（提供 `group_num`），并根据 `popTile` 与 `TConfig::group_num` 设置 gMax/rMax 的 valid shape。
    - 弹出后按普通 TileData 路径调用 `TFREE`。
- **NPU同步**：
    - 每次 `TPOP` 都会执行数据就绪等待。
    - 空闲空间通知是稀疏的，并由 `Pipe::SyncPeriod` 控制。
- **GlobalData槽位视图**：
    - `gmTensor` 会被赋值为由 `pipe.cons.tileIndex` 选择的FIFO槽位基地址。
    - 对于C2V切分模式，会根据 `Split` 应用向量子块偏移。
    - 从槽位视图完成所有加载后，调用者必须调用 `TFREE(Pipe&, GlobalData&)`。
- **CPU_SIM FIFO模型**：
    - `TPOP` 使用主机FIFO的互斥锁和条件变量等待生产者提交槽位。
    - TileData 数据从主机 FIFO 状态拥有的存储加载；即使 `TPipe` 带有非空 NPU GM workspace，CPU_SIM TileData 流程也不会访问该 workspace。
    - 若目标 Tile 的 `data()` 返回空指针，CPU_SIM 会分配独立的线程局部后备存储；已绑定的目标继续使用其现有存储。该行为适用于切分和不切分的 TileData 流程。
    - 独立分配的存储会登记到调用线程的内存模型中。`TASSIGN` 接受其基址和内部指针，检查完整视图范围，并创建共享数据的别名。别名不拥有该分配。详见 [TPOP 后备存储与 Tile 视图](../coding/cpu_sim.md#tpop-backed-tile-views)。
    - 切分模式根据当前subblock上下文选择lane。对于启用 `IsNoSplit` 的C2V管道，`TILE_NO_SPLIT` 会根据运行时subblock数量协调一个或两个vector消费者subblock；在其他no-split vector场景中，非零的未参与lane会按需填零。
    - CPU_SIM 当前未实现显式传入 `int32_t subBlockId` 的重载；应通过模拟subblock执行上下文选择split lane。
    - 对于 `DIR_BOTH`，消费者在自身方向的独立环形队列中等待。no-split 和 V2C 消费者按提交序号取出最早的已提交槽位，在延迟释放导致队列出现空隙时仍保持 FIFO 顺序。重叠的 pop 会预留不同槽位；`TFREE` 在相应方向释放调用方消费者最早尚未释放的 pop。
    - 每次启用的 `TFREE` 都会在 CPU_SIM 中释放一个已占用的具体槽位；CPU_SIM 不模拟 NPU 的 `SyncPeriod` 通知节奏。
    - TileData 数据按消费者 Tile 的声明形状读回，因此该形状必须与生产者推送的窗口一致（`TILE_NO_SPLIT` 下即其有效形状）。
    - CPU_SIM 按逻辑坐标将 TileData 复制到目标布局。A5 本地 FIFO 的 `TPOP` 则将 Tile 绑定到 FIFO 地址，不执行布局转换。要使 V2C 行为与 NPU 一致，应使用匹配的生产者和消费者布局，例如 NZ Vec 到 NZ Mat，并在 `TPUSH` 前完成所需的布局转换。
    - CPU_SIM 当前不支持 GlobalData 重载。

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

    Pipe pipe(fifoMem, LocalBase, 0x0); // C2V 模式：第二参数为 C2V 消费者缓冲基址，第三参数为 V2C 消费者缓冲基址（此处为 0）
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

### GlobalData槽位弹出

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

### A6R2 MaxTrack C2V 弹出

```cpp
#include <pto/pto-inst.hpp>
#include <pto/common/fixpipe.hpp>

using namespace pto;

AICORE void example_a6r2_maxtrack_pop()
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 128;
    constexpr uint8_t kGroupNum = 8;
    constexpr uint16_t FLAG_ID = 0;
    constexpr uint8_t FIFO_DEPTH = 2;

    using AccTile = TileAcc<float, M, N, M, N>;
    using VecTile = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, N>;
    using GMaxTile = Tile<TileType::Vec, float, M, N / kGroupNum, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
    constexpr int kRMaxCols =
        ((M + static_cast<int>(C0_SIZE_BYTE / sizeof(float)) - 1) / static_cast<int>(C0_SIZE_BYTE / sizeof(float))) *
        static_cast<int>(C0_SIZE_BYTE / sizeof(float));
    using RMaxTile = Tile<TileType::Vec, float, 1, kRMaxCols, BLayout::RowMajor, 1, DYNAMIC>;
    using MaxTrackCfg = MaxTrack<true, true, true, false, true, kGroupNum>;
    using FixpipeConfig = FixpipeParams<
        LayoutMode_t::NZ2ND, QuantMode_t::NoQuant, ReluPreMode::NoRelu, STPhase::Unspecified, 0, AtomicType::AtomicNone,
        ClipReluMode_t::NOCLIP_RELU, false, MaxTrackCfg>;
    using Pipe = TPipe<FLAG_ID, Direction::DIR_C2V, M * N * sizeof(float), FIFO_DEPTH, 2, true>;

    constexpr uint32_t C2V_BUF = 0x0;
    constexpr uint32_t GMAX_BUF = 0x20000;
    constexpr uint32_t RMAX_BUF = 0x28000;
    Pipe pipe((__gm__ void*)(uint64_t)0x0, C2V_BUF, (uint32_t)0x0, GMAX_BUF, RMAX_BUF);

    VecTile vecTile;
    GMaxTile gMaxTile(M, N / kGroupNum);
    RMaxTile rMaxTile(M);

    // 生产者须以 MaxTrack groupmax_en && rowmax_en 推送。
    TPOP<Pipe, VecTile, GMaxTile, RMaxTile, FixpipeConfig, TileSplitAxis::TILE_NO_SPLIT>(
        pipe, vecTile, gMaxTile, rMaxTile);
    // 消费 tile...
    TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
}
```
