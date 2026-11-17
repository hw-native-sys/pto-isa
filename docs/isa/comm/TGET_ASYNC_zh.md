# TGET_ASYNC

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

`TGET_ASYNC` 是异步远程读原语。它启动一次从远端GM到本地GM的传输，并立即返回 `AsyncEvent`。

数据流：

`srcGlobalData（远端 GM）` → DMA引擎 → `dstGlobalData（本地 GM）`

## 模板参数

- `engine`：
    - `DmaEngine::SDMA`（默认）
    - `DmaEngine::URMA`（Ascend 950PR/Ascend 950DT，仅NPU_ARCH 3510）
    - `DmaEngine::RDMA`（Ascend 950PR/Ascend 950DT，仅NPU_ARCH 3510；当前网卡平台仅支持 HNS1825）

> **注意（布局约束）**
> 各引擎的 Tensor 重载目前都**仅支持扁平连续的逻辑一维tensor**，不支持非一维或非连续布局。

## C++内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`：

### 连续 Tensor 重载

```cpp
// 全架构可用。URMA/RDMA 使用 session 中绑定的源rank。
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TGET_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                               const AsyncSession &session, WaitEvents &... events);

// 仅Ascend 950和CPU仿真可用。URMA/RDMA 用 peer 选择源rank；SDMA 忽略 peer。
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TGET_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                               const AsyncSession &session, uint32_t peer,
                               WaitEvents &... events);
```

### URMA Scatter 重载

```cpp
struct ScatterSegment {
    __gm__ uint8_t *addr;        // 本地 GM 目的分段
    uint32_t length;             // 字节数
};

struct ScatterSource {
    __gm__ const uint8_t *addr;  // 连续远端 GM 源区间
    uint32_t segCount;           // 消耗的目的分段数
    uint32_t peer;               // 源 rank
};

template <DmaEngine engine = DmaEngine::URMA, typename... WaitEvents>
PTO_INST AsyncEvent TGET_ASYNC(
    __gm__ const ScatterSegment *destinationSegments, uint32_t destinationSegmentCount,
    __gm__ const ScatterSource *sources, uint32_t sourceCount,
    const AsyncSession &session, WaitEvents &... events);
```

该重载将连续远端 GM 区间读入本地 GM 分段，仅支持 Ascend 950 上的 URMA，不支持 CPU 仿真。
每个 `sources[i]` 从 `peer` 指定的 rank 读取数据，按顺序填充接下来的 `segCount` 个目的分段；
满足 `sum(sources[i].segCount) == destinationSegmentCount`，一次调用可读取多个 peer。

`AsyncSession` 是引擎无关的会话对象。使用 `BuildAsyncSession<engine>()` 构建一次后，传递给所有异步调用和事件等待。模板参数 `engine` 在编译期选择DMA后端，使代码对未来引擎（CCU等）保持前向兼容。

`events` 为可空的变参列表，每个事件需提供无参 `Wait()`，指令在发起传输前依次等待。`AsyncEvent` 使用 `Wait(session)`，不能作为 `events` 传入。

## AsyncSession构建

使用 `include/pto/comm/async_common/async_event_impl.hpp` 中的 `BuildAsyncSession`，按引擎选择重载。构建失败返回 `false`，只有构建成功的Session才能用于后续调用。

### SDMA构建（默认）

```cpp
template <DmaEngine engine = DmaEngine::SDMA, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(ScratchTile &scratchTile,
                                    __gm__ uint8_t *workspace,
                                    AsyncSession &session,
                                    uint32_t syncId = 0,
                                    const sdma::SdmaBaseConfig &baseConfig = {sdma::kDefaultSdmaBlockBytes, 0, 1},
                                    uint32_t channelGroupIdx = sdma::kAutoChannelGroupIdx);
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `scratchTile` | — | 用于SDMA控制元数据的UB scratch tile（参见 [scratchTile的作用](#scratchtile的作用)）。|
| `workspace` | — | 由调用方分配并传给主机侧 `SdmaWorkspaceManager` 初始化的GM指针。|
| `session` | — | 输出的 `AsyncSession` 对象。|
| `syncId` | `0` | MTE3/MTE2管道同步事件ID（0-7）。若kernel在相同ID上使用了其他管道屏障，则需覆盖此值。|
| `baseConfig` | `{kDefaultSdmaBlockBytes, 0, 1}` | `{block_bytes, comm_block_offset, queue_num}`。适用于大多数单队列传输场景。|
| `channelGroupIdx` | `kAutoChannelGroupIdx` | SDMA通道组索引。默认内部使用 `get_block_idx()` 映射到当前AI Core。多block、并发或自定义通道映射场景下需覆盖此值。|

### URMA构建（仅NPU_ARCH 3510）

> URMA（User-level RDMA Memory Access）是Ascend 950PR/Ascend 950DT（NPU_ARCH 3510）上的硬件加速RDMA传输引擎。
> URMA要求CANN Toolkit **>= 9.1.0**。

```cpp
#ifdef PTO_URMA_SUPPORTED
// 不绑定源rank，配合带 peer 的 Tensor 重载或 Scatter 重载使用。
template <DmaEngine engine>
PTO_INTERNAL bool BuildAsyncSession(__gm__ uint8_t *workspace,
                                    AsyncSession &session);

// 同上，并传入Scatter SIMT路径所需的UB scratch。
template <DmaEngine engine, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(__gm__ uint8_t *workspace,
                                    ScratchTile &scratchTile,
                                    AsyncSession &session);

// 绑定源rank，配合不带 peer 的指令重载使用。
template <DmaEngine engine>
PTO_INTERNAL bool BuildAsyncSession(__gm__ uint8_t *workspace,
                                    uint32_t destRankId,
                                    AsyncSession &session);
#endif
```

| 参数 | 说明 |
|---|---|
| `scratchTile` | 由kernel分配的UB scratch tile，仅Scatter SIMT路径需要。|
| `workspace` | 由调用方分配并传给主机侧 `UrmaWorkspaceManager` 初始化的GM指针。|
| `destRankId` | 此会话绑定的远端PE rank id。对于 `TGET_ASYNC`，这是数据来源的源rank。|
| `session` | 输出的 `AsyncSession` 对象。|

连续 Tensor 重载不需要 `scratchTile`。Scatter 使用不绑定 rank 的 session 构造函数。
Scatter SIMT 路径需要至少 `sizeof(urma::UrmaGatherSimtScratch)` 字节的 UB；未提供时回退到 Scalar 路径。

构建时按当前AI Core划分该核可用的URMA队列。`workspace` 无效，或当前block数超出workspace配置的队列容量时，返回 `false`。

### RDMA构建（仅NPU_ARCH 3510）

RDMA面向经典跨机场景组网，当前网卡平台仅支持 HNS1825。

```cpp
#ifdef PTO_RDMA_SUPPORTED
// 不绑定peer，配合带 peer 的指令重载使用，一个Session可访问多个远端rank。
template <DmaEngine engine, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(ScratchTile &scratchTile,
                                    __gm__ uint8_t *workspace,
                                    uint32_t myPe,
                                    AsyncSession &session,
                                    uint32_t syncId = 0,
                                    uint32_t qpIdx = 0);

// 绑定源rank，配合不带 peer 的指令重载使用。
template <DmaEngine engine, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(ScratchTile &scratchTile,
                                    __gm__ uint8_t *workspace,
                                    uint32_t destRankId,
                                    uint32_t myPe,
                                    AsyncSession &session,
                                    uint32_t syncId = 0,
                                    uint32_t qpIdx = 0);
#endif
```

| 参数 | 说明 |
|---|---|
| `scratchTile` | 用于 WQE/CQE 控制数据的 UB/Vec tile，至少 64 字节。|
| `workspace` | Host侧 RDMA 初始化流程返回的 GM 指针。|
| `destRankId` | 此会话绑定的远端rank。|
| `myPe` | 本地 rank id，用于选择已注册的本地内存区域。|
| `session` | 输出的 `AsyncSession` 对象。|
| `syncId` | MTE/Scalar 同步事件 ID，取值范围为 0-7。|
| `qpIdx` | 每个peer选用的QP编号，默认0，需小于Host配置的 [`qpNum`](TPUT_ASYNC_zh.md#rdma构建仅npu_arch-3510)。 |

访问多个远端rank时，推荐构建不绑定 peer 的Session，再通过显式 `peer` 参数选择源rank：

```cpp
comm::AsyncSession session;
if (comm::BuildAsyncSession<comm::DmaEngine::RDMA>(scratchTile, rdmaWorkspace, myPe, session, syncId)) {
    auto event = comm::TGET_ASYNC<comm::DmaEngine::RDMA>(dstG, srcG, session, peer);
    (void)event.Wait(session);
}
```

## 约束

连续 Tensor 重载：

- `GlobalSrcData::RawDType == GlobalDstData::RawDType`
- `GlobalSrcData::layout == GlobalDstData::layout`
- 源、目的tensor均须为**扁平连续的逻辑一维**，且目的容量不小于源元素数

URMA Scatter 重载：

- GM 描述符、分段长度、MR 范围、peer、Session 与生命周期要求同 [Gather PUT 约束](TPUT_ASYNC_zh.md#约束)，源/目的方向对调
- 当前 GET 每个 READ SQE 使用一个 SGE（一个目的分段），PUT 每个 WRITE SQE 最多 12 个；因此相同分段数下，SQE 数、Scalar/SIMT 路径选择及 `OneCqe` 的 CQE 数可能不同
- 目的区间在未完成操作间不得重叠
- 参数校验或资源预留失败返回失败事件，Wait/Test 返回 `false`；不要单凭 `valid()` 判断结果

通用约束：

- SDMA workspace必须是由调用方分配并传给主机侧 `SdmaWorkspaceManager` 初始化的有效GM指针
- URMA workspace必须是由调用方分配并传给主机侧 `UrmaWorkspaceManager` 初始化的有效GM指针
- Session和workspace的生命周期必须覆盖相关Event的完成阶段
- URMA仅在NPU_ARCH 3510（Ascend 950PR/Ascend 950DT）上可用
- URMA要求CANN Toolkit **>= 9.1.0**
- 传给 `UrmaWorkspaceManager::Init()` 的对称数据缓冲区必须由大页内存支撑（使用 `ACL_MEM_MALLOC_HUGE_ONLY` 分配）。底层MR注册要求大页背景；`ACL_MEM_MALLOC_HUGE_FIRST` 在小尺寸分配时可能静默回退到4KB小页，导致注册失败
- RDMA仅在Ascend 950PR/Ascend 950DT（NPU_ARCH 3510）上可用，当前网卡平台仅支持 HNS1825
- RDMA源、目的tensor必须是扁平连续的逻辑一维，且本地和远端完整传输范围必须位于Host初始化阶段注册的内存区域内
- 单次RDMA传输不能超过 `0x7fffffff` 字节

违反上述tensor约束时由 `PTO_ASSERT` 拦截。

## scratchTile的作用

`scratchTile` 不承载传输数据，仅作为SDMA控制元数据和完成标志轮询、URMA Scatter 下发及RDMA WQE/CQE控制数据的临时UB工作区。payload始终沿远端GM → DMA引擎 → 本地GM路径传输。

类型要求：必须是UB/Vec位置的 `pto::Tile`（`ScratchTile::Loc == TileType::Vec`）。SDMA至少8字节，RDMA至少64字节，推荐 `Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>`（256Byte）。

## 批量提交

连续 Tensor `TGET_ASYNC` 不加入 `AsyncSubmitMode` 批次，但在发起本次读取前，会先提交同一Session上由 `TPUT_ASYNC` 暂存的批次。URMA下该批次的目标rank须与本次调用的peer一致，否则返回失败Event。

URMA Scatter 与 Gather PUT 的批量提交规则相同，同一批次可包含多个 peer；Wait/Test 不会提交暂存批次。
`submitMode` / `batchSize` 的完整语义参见 [TPUT_ASYNC 批量提交](TPUT_ASYNC_zh.md#批量提交)。

## 完成语义（Quiet语义）

连续 Tensor 重载：

- **SDMA**：每次`TGET_ASYNC`都会提交数据传输SQE和用于标记本次操作完成的flag SQE。对其返回的Event调用`Wait`或`Test`时，通过轮询对应flag判断该次`TGET_ASYNC`是否完成；完成后也能保证同一Session中此前提交的所有SDMA操作均已完成。
- **URMA**：`TGET_ASYNC` 立即提交RDMA READ WQE并敲门铃。`Wait` 通过轮询Completion Queue（CQ）等待所有预期的CQE被消费。
- **RDMA**：`TGET_ASYNC` 向 `peer` 选择的队列提交RDMA READ WQE；不同peer/queue分别跟踪完成状态。

- `event.Wait(session)` —阻塞，直到**自上次Wait以来所有已发出的异步操作**全部完成

这意味着多次 `TGET_ASYNC` 调用后，只需对最后一个返回的 `AsyncEvent` 调用一次 `Wait`，即可等待所有pending操作完成（类似shmem的quiet语义）。

RDMA操作涉及不同peer时，必须分别等待每个peer的最后一个Event。

同一Session最多可有64个未完成操作，超过后提交可能产生背压。

wait成功后，所有已发出的 `dstGlobalData` 读入数据均已全部就绪。

URMA Scatter 使用 `session.cfg.urma.cqePolicy`，默认 `OneCqe`，语义同 [Gather PUT 完成策略](TPUT_ASYNC_zh.md#urma-gather完成策略)。
含负载的 `NoCqe` Event 不可等待，须由同一Session的后续 `Final` 确认；允许两个计数均为0的空 `Final` 调用。
成功 Wait/Test 后，才可消费该Event覆盖的本地读取结果。

## SDMA并发与Session所有权

- 同一个Session不能被多个执行流并发使用。
- 共用同一Channel Group的操作必须共享同一个Session。
- 并发Kernel或Kernel内多个独立Session必须使用隔离的Channel Group。
- 重新构建Session或复用Channel Group前，必须先完成此前所有Event。

## 示例

### 单次传输

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/common/pto_tile.hpp>

using namespace pto;

template <typename T>
__global__ AICORE void SimpleGet(__gm__ T *localDst, __gm__ T *remoteSrc,
                                 __gm__ uint8_t *sdmaWorkspace)
{
    using ShapeDyn  = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT        = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
    using ScratchTile = Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);
    GT dstG(localDst,  shape, stride);
    GT srcG(remoteSrc, shape, stride);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::SDMA>(scratchTile, sdmaWorkspace, session)) {
        return;
    }

    auto event = comm::TGET_ASYNC<comm::DmaEngine::SDMA>(dstG, srcG, session);
    (void)event.Wait(session);
}
```

### 连续多次传输（Quiet语义）

```cpp
template <typename T>
__global__ AICORE void BatchGet(__gm__ T *localDstBase, __gm__ T *remoteSrcBase,
                                __gm__ uint8_t *sdmaWorkspace, int nranks)
{
    using ShapeDyn  = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT        = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
    using ScratchTile = Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session)) {
        return;
    }

    comm::AsyncEvent lastEvent;
    for (int rank = 0; rank < nranks; ++rank) {
        GT dstG(localDstBase + rank * 1024, shape, stride);
        GT srcG(remoteSrcBase + rank * 1024, shape, stride);
        lastEvent = comm::TGET_ASYNC(dstG, srcG, session);
    }
    (void)lastEvent.Wait(session);  // 一次 Wait 等待所有 pending 操作
}
```

### URMA示例（NPU_ARCH 3510）

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/common/pto_tile.hpp>

using namespace pto;

template <typename T>
__global__ AICORE void SimpleGetUrma(__gm__ T *localDst, __gm__ T *remoteSrc,
                                     __gm__ uint8_t *urmaWorkspace, uint32_t srcRankId)
{
    using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);
    GT dstG(localDst, shape, stride);
    GT srcG(remoteSrc, shape, stride);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::URMA>(urmaWorkspace, srcRankId, session)) {
        return;
    }

    auto event = comm::TGET_ASYNC<comm::DmaEngine::URMA>(dstG, srcG, session);
    (void)event.Wait(session);
}
```
