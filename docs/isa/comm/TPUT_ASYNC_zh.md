# TPUT_ASYNC

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

`TPUT_ASYNC` 是远程写原语。它发起一次从本地GM到远端GM的传输，并返回 `AsyncEvent` 用于后续等待。

数据流：

`srcGlobalData（本地 GM）` → DMA引擎 → `dstGlobalData（远端 GM）`

## 模板参数

- `engine`：
    - `DmaEngine::SDMA`（默认）。A2/A3为异步传输；Ascend 950为同步实现，参见 [A5上的SDMA路径](#a5上的sdma路径)
    - `DmaEngine::URMA`（Ascend 950PR/Ascend 950DT，仅NPU_ARCH 3510）
    - `DmaEngine::RDMA`（Ascend 950PR/Ascend 950DT，仅NPU_ARCH 3510；当前网卡平台仅支持 HNS1825）

> **注意（布局约束）**
> 各引擎目前都**仅支持扁平连续的逻辑一维tensor**，不支持非一维或非连续布局。

## C++内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`：

### 连续Tensor重载

```cpp
// 全架构可用。URMA/RDMA 使用 session 中绑定的目标rank。
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                               const AsyncSession &session, WaitEvents &... events);

// 仅Ascend 950和CPU仿真可用。URMA/RDMA 用 peer 选择目标rank；SDMA 忽略 peer。
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                               const AsyncSession &session, uint32_t peer,
                               WaitEvents &... events);
```

### URMA Gather重载

```cpp
template <DmaEngine engine = DmaEngine::URMA, typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC(
    __gm__ const GatherDestination *destinations,
    uint32_t destinationCount,
    __gm__ const GatherSegment *sourceSegments,
    uint32_t sourceSegmentCount,
    const AsyncSession &session,
    WaitEvents &... events);
```

该重载将多个非连续本地GM段聚合后写入一个或多个远端连续区间，仅支持 `DmaEngine::URMA`。每个目标描述符自带 `peer`，因此一次调用可写入多个rank，且只返回一个 `AsyncEvent`。参数校验或资源预留失败时返回失败Event，`Wait`/`Test` 前须先检查 `event.valid()`。

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
| `scratchTile` | — | UB scratch tile，用途参见 [scratchTile的作用](#scratchtile的作用)。|
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
// 不绑定目标rank，配合带 peer 的指令重载和Gather重载使用。
template <DmaEngine engine>
PTO_INTERNAL bool BuildAsyncSession(__gm__ uint8_t *workspace,
                                    AsyncSession &session);

// 同上，并传入Gather SIMT路径所需的UB scratch。
template <DmaEngine engine, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(__gm__ uint8_t *workspace,
                                    ScratchTile &scratchTile,
                                    AsyncSession &session);

// 绑定目标rank，配合不带 peer 的指令重载使用。
template <DmaEngine engine>
PTO_INTERNAL bool BuildAsyncSession(__gm__ uint8_t *workspace,
                                    uint32_t destRankId,
                                    AsyncSession &session);
#endif
```

| 参数 | 说明 |
|---|---|
| `scratchTile` | 由kernel分配的UB scratch tile，仅Gather SIMT路径需要。|
| `workspace` | 由调用方分配并传给主机侧 `UrmaWorkspaceManager` 初始化的GM指针。|
| `destRankId` | 连续Tensor重载使用的远端PE rank id。|
| `session` | 输出的 `AsyncSession` 对象。|

连续Tensor重载使用带 `destRankId` 的构建接口；Gather重载使用不绑定rank的构建接口，目标rank由每个 `GatherDestination::peer` 指定。

连续Tensor传输不需要 `scratchTile`，轮询通过 `ld_dev`/`st_dev` 硬件原语直接操作；Gather SIMT路径要求容量不小于 `sizeof(urma::UrmaGatherSimtScratch)`，未提供时自动回退Scalar路径。

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

// 绑定目标rank，配合不带 peer 的指令重载使用。
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
| `qpIdx` | 每个peer选用的QP编号，默认0，需小于Host配置的 `qpNum`。 |

Host通过 `RdmaTransportConfig::qpNum` 设置每个peer的QP数（默认1，各rank须一致）；直接使用后端管理器时设置
`rdma::WorkspaceConfig::qpNum`。不同QP独立完成，需分别等待其Event。

访问多个远端rank时，推荐构建不绑定 peer 的Session，再通过显式 `peer` 参数选择目标：

```cpp
comm::AsyncSession session;
if (comm::BuildAsyncSession<comm::DmaEngine::RDMA>(scratchTile, rdmaWorkspace, myPe, session, syncId)) {
    auto event = comm::TPUT_ASYNC<comm::DmaEngine::RDMA>(dstG, srcG, session, peer);
    (void)event.Wait(session);
}
```

## URMA Gather描述符与数据映射

描述符类型定义如下：

```cpp
struct GatherSegment {
    __gm__ const uint8_t *addr;  // 本地GM段起始地址
    uint32_t length;             // 字节数
};

struct GatherDestination {
    __gm__ uint8_t *addr;        // 远端连续目标区间起始地址
    uint32_t segCount;           // 此目标消费的源段数量
    uint32_t peer;               // 目标rank
};
```

`destinations` 按下标顺序连续消费 `sourceSegments`，并将每个目标消费的源段按顺序拼接到该目标的远端连续区间。例如 `segCount` 依次为2和1时：

```text
destinations[0] <- sourceSegments[0], sourceSegments[1]
destinations[1] <- sourceSegments[2]
```

因此必须满足 `sum(destinations[i].segCount) == sourceSegmentCount`。

## URMA Gather完成策略

提交时机由 `submitMode` 和 `batchSize` 控制，语义同 [批量提交](#批量提交)，但Gather批次不受单一目标rank的限制。完成通知的粒度另由 `session.cfg.urma.cqePolicy` 控制，默认 `OneCqe`。

| `UrmaCqePolicy` | 行为 |
|---|---|
| `OneCqe` | 为本次操作保留完整完成通知，返回的Event可 `Wait`/`Test`。默认值。|
| `Final` | 减少完成通知开销，返回的Event可 `Wait`/`Test`，并覆盖此前尚未确认的 `NoCqe` 操作。|
| `NoCqe` | 不请求完成通知。含负载调用返回的Event不可 `Wait`/`Test`，必须由后续 `Final` 确认。|

## 约束

连续Tensor重载：

- `GlobalSrcData::RawDType == GlobalDstData::RawDType`
- `GlobalSrcData::layout == GlobalDstData::layout`
- 源、目的tensor均须为**扁平连续的逻辑一维**，且目的容量不小于源元素数

URMA Gather重载：

- `destinations`、`sourceSegments` 位于GM；非零数量必须对应非空指针
- 每个源段的 `addr` 非空且 `length > 0`
- 所有本地源范围和远端目标范围必须位于Host初始化阶段注册的MR内
- 每个 `peer` 必须小于Host配置的rank数量
- `sum(destinations[i].segCount) == sourceSegmentCount`
- 同一Session中未结束的连续Tensor batch和Gather batch不能混用
- 同一个Session不能被多个执行流并发使用，多AIV须各自构建Session
- 描述符数组必须在本次调用读取期间有效，源/目标数据必须存活到相关Event完成

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

## A5上的SDMA路径

Ascend 950上，`TPUT_ASYNC<DmaEngine::SDMA>` 是保持接口兼容的同步实现：payload经UB分块搬运，调用返回时传输已完成，返回已完成状态的Event，`Wait` 和 `Test` 立即返回 `true`。该路径不支持 [批量提交](#批量提交)。

同平台的 `TGET_ASYNC<DmaEngine::SDMA>` 仍为异步传输。

## scratchTile的作用

| 场景 | 用途 | 最小可用字节 |
|---|---|---|
| A2/A3 SDMA | SDMA控制元数据与完成标志轮询，不承载payload | 8 |
| Ascend 950 SDMA | payload分块暂存，tile大小即分块粒度 | 8 |
| RDMA | WQE/CQE控制数据，不承载payload | 64 |

类型要求：必须是UB/Vec位置的 `pto::Tile`（`ScratchTile::Loc == TileType::Vec`）。

推荐 `Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>`（256Byte）。Ascend 950 SDMA路径下传输大块payload时，应按典型payload尺寸放大该tile以减少分块次数。

## 批量提交

`AsyncSession::submitMode` 和 `AsyncSession::batchSize` 控制 `TPUT_ASYNC` 立即提交还是暂存后成批提交。两者在 `BuildAsyncSession` 成功后直接赋值，指令签名不变。默认 `IMMEDIATE` + `UINT32_MAX`，即逐条立即提交。

| `AsyncSubmitMode` | 含义 |
|---|---|
| `IMMEDIATE` | 立即提交本次传输。若存在暂存批次，先提交该批次。|
| `DEFER` | 暂存本次传输，暂存数达到 `batchSize` 时自动提交。|
| `DEFER_AND_SUBMIT` | 暂存本次传输并立即提交整个批次。|

- 仅A2/A3的 `TPUT_ASYNC<SDMA>` 和Ascend 950的 `TPUT_ASYNC<URMA>` 支持批量；其余路径忽略 `submitMode`。
- `batchSize` 为0表示禁用批量，此时使用非 `IMMEDIATE` 模式将返回失败Event。
- URMA同一批次内的目标rank必须一致，否则返回失败Event。
- `TGET_ASYNC` 和 `TPUT_ASYNC_NOTIFY` 不加入批次，但会先提交同一Session上已暂存的批次。
- 应等待提交批次时返回的Event，`DEFER` 阶段返回的中间Event不代表批次已提交。

```cpp
comm::AsyncSession session;
if (!comm::BuildAsyncSession<comm::DmaEngine::SDMA>(scratchTile, sdmaWorkspace, session)) {
    return;
}
session.batchSize = 2U;

session.submitMode = comm::AsyncSubmitMode::DEFER;
(void)comm::TPUT_ASYNC(dst0, src0, session);          // 暂存

session.submitMode = comm::AsyncSubmitMode::DEFER_AND_SUBMIT;
auto event = comm::TPUT_ASYNC(dst1, src1, session);   // 暂存并提交整个批次
(void)event.Wait(session);
```

## 完成语义（Quiet语义）

- **SDMA**：每次`TPUT_ASYNC`都会提交数据传输SQE和用于标记本次操作完成的flag SQE。对其返回的Event调用`Wait`或`Test`时，通过轮询对应flag判断该次`TPUT_ASYNC`是否完成；完成后也能保证同一Session中此前提交的所有SDMA操作均已完成。
- **URMA连续Tensor重载**：`TPUT_ASYNC` 立即提交RDMA WRITE WQE并敲门铃。`Wait` 通过轮询Completion Queue（CQ）等待所有预期的CQE被消费。
- **URMA Gather重载**：提交时机由 `submitMode` 控制，完成语义由 `cqePolicy` 控制。`OneCqe`和`Final`返回的Event可等待；含负载的`NoCqe` Event不可等待，必须由后续`Final`完成确认。
- **RDMA**：`TPUT_ASYNC` 向 `peer` 选择的队列提交RDMA WRITE WQE；不同peer/queue分别跟踪完成状态。

- `event.Wait(session)` —阻塞，直到**自上次Wait以来所有已发出的异步操作**全部完成

这意味着多次 `TPUT_ASYNC` 调用后，只需对最后一个返回的 `AsyncEvent` 调用一次 `Wait`，即可等待所有pending操作完成（类似shmem的quiet语义）。

RDMA操作涉及不同peer时，必须分别等待每个peer的最后一个Event。

Gather重载改按其CQE策略：不能等待含负载的 `NoCqe` Event，需由后续 `Final` 确认Session中此前的 `NoCqe` 操作。

同一Session最多可有64个未完成操作，超过后提交可能产生背压。

wait成功后，所有已发出的 `dstGlobalData` 写入均已全部完成。

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
__global__ AICORE void SimplePut(__gm__ T *remoteDst, __gm__ T *localSrc,
                                 __gm__ uint8_t *sdmaWorkspace)
{
    using ShapeDyn  = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT        = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
    using ScratchTile = Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);
    GT dstG(remoteDst, shape, stride);
    GT srcG(localSrc,  shape, stride);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::SDMA>(scratchTile, sdmaWorkspace, session)) {
        return;
    }

    auto event = comm::TPUT_ASYNC<comm::DmaEngine::SDMA>(dstG, srcG, session);
    (void)event.Wait(session);
}
```

### 连续多次传输（Quiet语义）

```cpp
template <typename T>
__global__ AICORE void BatchPut(__gm__ T *remoteDstBase, __gm__ T *localSrc,
                                __gm__ uint8_t *sdmaWorkspace, int nranks)
{
    using ShapeDyn  = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT        = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
    using ScratchTile = Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);
    GT srcG(localSrc, shape, stride);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session)) {
        return;
    }

    comm::AsyncEvent lastEvent;
    for (int rank = 0; rank < nranks; ++rank) {
        GT dstG(remoteDstBase + rank * 1024, shape, stride);
        lastEvent = comm::TPUT_ASYNC(dstG, srcG, session);
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
__global__ AICORE void SimplePutUrma(__gm__ T *remoteDst, __gm__ T *localSrc,
                                     __gm__ uint8_t *urmaWorkspace, uint32_t destRankId)
{
    using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);
    GT dstG(remoteDst, shape, stride);
    GT srcG(localSrc, shape, stride);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::URMA>(urmaWorkspace, destRankId, session)) {
        return;
    }

    auto event = comm::TPUT_ASYNC<comm::DmaEngine::URMA>(dstG, srcG, session);
    (void)event.Wait(session);
}
```

### URMA Gather示例（NPU_ARCH 3510）

下面的示例将四个本地非连续段分别拼接到两个远端连续区间。两个目标可以属于不同rank：

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

__global__ AICORE void GatherPutUrma(
    __gm__ uint8_t *remote0, __gm__ uint8_t *remote1,
    __gm__ const uint8_t *local0, __gm__ const uint8_t *local1,
    __gm__ const uint8_t *local2, __gm__ const uint8_t *local3,
    __gm__ comm::GatherDestination *dsts, __gm__ comm::GatherSegment *segs,
    __gm__ uint8_t *urmaWorkspace, uint32_t peer0, uint32_t peer1)
{
    dsts[0] = {remote0, 2, peer0};
    dsts[1] = {remote1, 2, peer1};

    segs[0] = {local0, 4096};
    segs[1] = {local1, 1024};
    segs[2] = {local2, 2048};
    segs[3] = {local3, 3072};
    comm::urma::DcciCachelines(
        reinterpret_cast<__gm__ uint8_t *>(dsts), 2 * sizeof(comm::GatherDestination));
    comm::urma::DcciCachelines(
        reinterpret_cast<__gm__ uint8_t *>(segs), 4 * sizeof(comm::GatherSegment));
    dsb(DSB_DDR);

    using ScratchTile =
        Tile<TileType::Vec, uint8_t, 1, sizeof(comm::urma::UrmaGatherSimtScratch)>;
    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::URMA>(urmaWorkspace, scratchTile, session)) {
        return;
    }
    session.cfg.urma.cqePolicy = comm::urma::UrmaCqePolicy::Final;

    auto event = comm::TPUT_ASYNC<comm::DmaEngine::URMA>(dsts, 2, segs, 4, session);
    if (event.valid()) {
        (void)event.Wait(session);
    }
}
```

完成后，`remote0` 依次包含 `local0` 和 `local1` 的内容，`remote1` 依次包含 `local2` 和 `local3` 的内容。描述符写入后必须由提交该Session的执行流执行cache clean和 `dsb(DSB_DDR)`，保证URMA读取前描述符已对设备可见。

### URMA Gather延迟提交与NoCqe前缀

需要减少提交开销时，可以先使用 `DEFER + NoCqe` 暂存多个调用，最后用一个无负载 `Final` 调用发布并等待整个前缀：

```cpp
session.submitMode = comm::AsyncSubmitMode::DEFER;
session.batchSize = 4;
session.cfg.urma.cqePolicy = comm::urma::UrmaCqePolicy::NoCqe;

auto noCqeEvent = comm::TPUT_ASYNC<comm::DmaEngine::URMA>(dsts, 2, segs, 4, session);
// 含负载的NoCqe Event不可Wait/Test。

session.submitMode = comm::AsyncSubmitMode::DEFER_AND_SUBMIT;
session.cfg.urma.cqePolicy = comm::urma::UrmaCqePolicy::Final;
auto finalEvent = comm::TPUT_ASYNC<comm::DmaEngine::URMA>(
    nullptr, 0, nullptr, 0, session);
if (finalEvent.valid()) {
    (void)finalEvent.Wait(session);
}
```

`DEFER` 会根据 `batchSize` 自动提交已累积的操作；若需由调用方完全控制提交时机，可将 `batchSize` 设为 `UINT32_MAX`，再用 `DEFER_AND_SUBMIT` 提交。
