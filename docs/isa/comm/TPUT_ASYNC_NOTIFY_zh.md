# TPUT_ASYNC_NOTIFY

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

`TPUT_ASYNC_NOTIFY` 是带通知的异步远程写原语。它先将非空payload从本地GM传输到远端GM，再更新一个
远端32位signal。返回的 `AsyncEvent` 同时表示payload传输和signal更新的完成状态。

数据流：

`srcGlobalData（本地 GM）` → DMA引擎 → `dstGlobalData（远端 GM）` → 更新 `dstSignalData（远端 GM）`

`dstGlobalData` 和 `dstSignalData` 均由调用方传入远端地址。仅更新signal、不传输payload时，应使用
`TNOTIFY`。

## 模板参数

`engine` 在编译期选择DMA后端：

| 引擎 | 平台限制 | 支持的 `NotifyOp` |
|---|---|---|
| `DmaEngine::SDMA`（默认） | A2/A3和A5 | `Set`、`AtomicAdd` |
| `DmaEngine::URMA` | Ascend 950PR/Ascend 950DT，仅NPU_ARCH 3510；要求CANN Toolkit >= 9.1.0 | `Set`、`AtomicAdd` |
| `DmaEngine::RDMA` | Ascend 950PR/Ascend 950DT，仅NPU_ARCH 3510；当前支持HNS1825 RoCE | `Set`、`AtomicAdd` |

## C++内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`：

```cpp
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData,
          typename GlobalSrcData,
          typename GlobalSignalData,
          typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC_NOTIFY(GlobalDstData &dstGlobalData,
                                      GlobalSrcData &srcGlobalData,
                                      GlobalSignalData &dstSignalData,
                                      int32_t signalValue,
                                      NotifyOp notifyOp,
                                      const AsyncSession &session,
                                      uint32_t peer,
                                      WaitEvents &... events);
```

对于URMA和RDMA，`peer` 用于选择目标rank对应的通信队列和远端内存信息，`dstGlobalData` 与
`dstSignalData` 必须属于该peer。SDMA不使用 `peer`，远端地址由GlobalTensor确定。

`events` 是可为空的变参列表。每个事件必须提供无参 `Wait()`；接口在发起payload传输前依次等待这些事件。
`AsyncEvent` 的等待接口为 `Wait(session)`，因此不能作为 `events` 参数传入，应在调用前使用对应Session
显式等待。

## 参数说明

| 参数 | 说明 |
|---|---|
| `dstGlobalData` | payload在远端GM中的目的GlobalTensor。 |
| `srcGlobalData` | payload在本地GM中的源GlobalTensor。 |
| `dstSignalData` | 远端GM中的32位signal，数据类型必须为 `int32_t`。 |
| `signalValue` | `Set` 写入的值，或 `AtomicAdd` 使用的增量。 |
| `notifyOp` | signal更新操作：`NotifyOp::Set` 或 `NotifyOp::AtomicAdd`。 |
| `session` | 为模板参数 `engine` 构建的 `AsyncSession`。 |
| `peer` | URMA和RDMA的目标rank；SDMA不使用该参数。 |
| `events` | 零个或多个前置PTO流水事件。 |

返回值为 `AsyncEvent`。该Event的完成范围同时覆盖payload传输和后续signal更新。

### Signal与 `signalValue`

`comm::Signal` 是单个 `int32_t` signal对应的GlobalTensor别名：

```cpp
using Signal = GlobalTensor<int32_t,
                            Shape<1, 1, 1, 1, 1>,
                            Stride<1, 1, 1, 1, 1>,
                            Layout::ND>;
```

构造 `Signal` 只封装调用方提供的GM地址，不分配或初始化底层内存。调用方应提前分配signal，并按通信协议
设置初始值。`Set` 将signal更新为 `signalValue`；`AtomicAdd` 将 `signalValue` 作为有符号增量。

## 操作语义

单次调用按以下顺序执行：

1. 等待所有 `events`。
2. 将完整payload从 `srcGlobalData` 传输到 `dstGlobalData`。
3. payload传输完成后，按照 `notifyOp` 更新 `dstSignalData`。

`NotifyOp::Set` 的语义为：

$$
\mathrm{signal}^{\mathrm{remote}} = \mathrm{signalValue}
$$

`NotifyOp::AtomicAdd` 的语义为：

$$
\mathrm{signal}^{\mathrm{remote}} \mathrel{+}= \mathrm{signalValue} \quad (\text{原子操作})
$$

`AtomicAdd` 对signal的更新具有原子性。多个生产者可以并发更新同一个signal，最终增量为各次
`signalValue` 之和。该原子性不适用于payload写入；多个生产者并发传输payload时，目的地址范围不得重叠。

多个生产者并发 `Set` 同一个signal时，接口不保证最终值。缺少应用层同步时，不应在同一个signal上并发
混用 `Set`、`AtomicAdd` 或普通store。

`DmaEngine::RDMA` 在HNS1825上通过payload WRITE之后的INT32 SUM Write Reduce实现 `AtomicAdd`。
上述payload先于signal的顺序仅适用于同一次调用，不定义不同Session或不同执行流之间的顺序。

## AsyncSession构建

使用 `include/pto/comm/async_common/async_event_impl.hpp` 中的 `BuildAsyncSession`。该函数按引擎提供不同的
构建接口。构建失败时返回 `false`；只有构建成功的Session才能用于异步指令和Event等待。

### SDMA构建（默认）

```cpp
template <DmaEngine engine = DmaEngine::SDMA, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(
    ScratchTile &scratchTile,
    __gm__ uint8_t *workspace,
    AsyncSession &session,
    uint32_t syncId = 0,
    const sdma::SdmaBaseConfig &baseConfig = {
        sdma::kDefaultSdmaBlockBytes, 0, 1},
    uint32_t channelGroupIdx = sdma::kAutoChannelGroupIdx);
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `scratchTile` | — | 用于SDMA控制元数据的UB scratch tile。 |
| `workspace` | — | 由调用方分配并传给Host侧 `SdmaWorkspaceManager` 初始化的GM指针。 |
| `session` | — | 输出的 `AsyncSession`。 |
| `syncId` | `0` | MTE3/MTE2流水同步事件ID，取值范围为0-7。 |
| `baseConfig` | `{kDefaultSdmaBlockBytes, 0, 1}` | SDMA块字节数、通信块偏移和队列数。 |
| `channelGroupIdx` | `kAutoChannelGroupIdx` | SDMA通道组索引；默认根据 `get_block_idx()` 选择。 |

在A2/A3上，并发AIV必须分别构建Session，并使用不同的Channel Group。默认的
`kAutoChannelGroupIdx` 会按 `get_block_idx()` 选择Group；调用方显式指定Group时也必须保证每个并发AIV
独占一个Group。`queue_num` 可以大于1，但单次 `TPUT_ASYNC_NOTIFY` 的payload和signal只使用该Group中的
第0条queue，以保证payload先于signal；增加 `queue_num` 不会并行拆分此次notify payload。

### URMA构建（仅NPU_ARCH 3510）

```cpp
#ifdef PTO_URMA_SUPPORTED
template <DmaEngine engine>
PTO_INTERNAL bool BuildAsyncSession(__gm__ uint8_t *workspace,
                                    AsyncSession &session);
#endif
```

| 参数 | 说明 |
|---|---|
| `workspace` | 由调用方分配并传给Host侧 `UrmaWorkspaceManager` 初始化的GM指针。 |
| `session` | 输出的 `AsyncSession`。 |

URMA不需要 `scratchTile`，要求CANN Toolkit >= 9.1.0。Session不绑定目标rank，目标由指令的 `peer` 参数
指定。

### RDMA构建（仅NPU_ARCH 3510）

```cpp
#ifdef PTO_RDMA_SUPPORTED
template <DmaEngine engine, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(ScratchTile &scratchTile,
                                    __gm__ uint8_t *workspace,
                                    uint32_t myPe,
                                    AsyncSession &session,
                                    uint32_t syncId = 0,
                                    uint32_t qpIdx = 0);
#endif
```

| 参数 | 说明 |
|---|---|
| `scratchTile` | RDMA使用的UB/Vec scratch tile，至少64字节。 |
| `workspace` | Host侧RDMA初始化流程返回的GM指针。 |
| `myPe` | 本地rank id。 |
| `session` | 输出的 `AsyncSession`。 |
| `syncId` | MTE/Scalar同步事件ID，取值范围为0-7。 |
| `qpIdx` | 每个peer选用的QP编号，默认0，需小于Host配置的 [`qpNum`](TPUT_ASYNC_zh.md#rdma构建仅npu_arch-3510)。 |

RDMA Session不绑定目标peer。调用指令时通过 `peer` 选择目标rank：

```cpp
comm::AsyncSession session;
if (comm::BuildAsyncSession<comm::DmaEngine::RDMA>(
        scratchTile, rdmaWorkspace, myPe, session)) {
    auto event = comm::TPUT_ASYNC_NOTIFY<comm::DmaEngine::RDMA>(
        dstGlobalData, srcGlobalData, remoteSignal, 1,
        comm::NotifyOp::Set, session, peer);
    (void)event.Wait(session);
}
```

### scratchTile

`scratchTile` 是SDMA和RDMA Session使用的临时UB工作区，不用于存放用户payload。它必须是UB/Vec内存中的
`pto::Tile`，并在相关Event完成前保持有效。

- SDMA要求至少8字节可用空间，常用类型为
  `Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>`（256字节）。
- RDMA要求至少64字节可用空间。

## 约束

- `GlobalSrcData::RawDType` 必须等于 `GlobalDstData::RawDType`。
- `GlobalSrcData::layout` 必须等于 `GlobalDstData::layout`。
- 源和目的payload tensor必须是扁平、连续的逻辑一维tensor。
- 目的tensor的元素容量不得小于源tensor的元素数。
- payload大小必须大于0；仅更新signal时应使用 `TNOTIFY`。
- `dstSignalData` 必须恰好包含远端GM中的一个 `int32_t`，地址非空且按4字节对齐。调用方负责分配和初始化。
- payload目的地址范围不得与 `dstSignalData` 重叠。
- 对于URMA和RDMA，payload目的地址与signal必须属于同一个目标peer；本地payload、远端payload和远端
  signal的完整地址范围都必须位于Host初始化阶段注册的内存区域内。
- URMA payload无单次大小上限：超过256 MB（单WQE传输上限）的payload由实现自动拆分为多个WQE，对调用方透明，且signal仍保证在所有payload分片之后最后完成。单次RDMA payload不得超过 `0x7fffffff` 字节。
- SDMA workspace必须由Host侧 `SdmaWorkspaceManager` 初始化；URMA workspace必须由Host侧
  `UrmaWorkspaceManager` 初始化。
- 传给 `UrmaWorkspaceManager::Init()` 的对称数据buffer必须是可由HCCL注册的设备内存；分配方式遵循
  当前CANN/HCCL运行时对注册内存的要求。
- Session、workspace和scratch tile的生命周期必须覆盖相关Event的完成阶段。

## 完成语义

- `event.Wait(session)` 阻塞等待Event完成。
- `event.Test(session)` 非阻塞检测Event是否完成。
- 两个接口都必须使用发起操作时的同一个Session。
- 成功完成同时覆盖完整payload传输及其后的signal更新。

同一Session、同一后端队列中的异步操作按提交顺序完成。连续提交多次操作后，等待该队列最后一次操作的
Event，也会覆盖此前尚未完成的操作。URMA或RDMA访问不同peer时，各peer的队列独立，调用方必须分别等待
每个peer的最后一个Event；调用 `Wait` 或 `Test` 时无需再次传入peer。

接收端观察到signal更新时，对应payload已经传输到远端GM，但该接口不保证接收端已有的payload缓存副本
同步更新。读取payload前，调用方负责按照目标平台和运行时的内存一致性规则保证payload可见性。

## 并发与Session所有权

- A2/A3 SDMA：不同AIV使用独立Session和Channel Group时，可以并发访问同一rank或不同rank。
- URMA：不同AIV访问不同peer时可以并发；访问同一peer时必须串行。
- 并发payload范围不得重叠；共享signal应使用 `AtomicAdd`，`Set` 应使用独立signal。

- 同一个Session不能被多个执行流并发使用。
- 对A2/A3 SDMA，每个并发AIV必须使用独立Session和独立Channel Group；不得让多个AIV同时提交到同一个
  Group。`queue_num` 为 `N` 时，合法Group数最多为 `kSdmaMaxChannelGroups / N`。
- 多个AIV并发 `Set` 时，应为每个生产者使用独立signal；使用同一个signal做完成计数时应使用
  `AtomicAdd`。无论哪种模式，各AIV的payload目的地址范围都不得重叠。
- 对URMA，即使调用方基于同一workspace构建了不同Session，对同一peer/QP的提交也必须串行；
  不同peer使用独立队列。
- 重新构建Session或复用后端队列前，必须先完成此前全部Event。
- 使用URMA时，释放通信资源前必须完成每个活动peer/QP的最后一个Event，并同步所有使用该
  `CommContext` 的Host stream。若通过 `existingComm` 传入外部HCCL communicator，还必须确保其
  他使用者已经停止，先销毁该communicator以释放Channel/MR，再调用 `DestroyComm`、`Reset`、重新
  `BuildComm` 或析构 `CommContext`。

## 示例

以下示例假设Host通信运行时已经完成远端地址转换，并初始化相应引擎的workspace。

### SDMA Set

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/common/pto_tile.hpp>

using namespace pto;

template <typename T>
__global__ AICORE void PutAndNotifySdma(__gm__ T *remoteDst,
                                       __gm__ T *localSrc,
                                       __gm__ int32_t *remoteSignalPtr,
                                       __gm__ uint8_t *sdmaWorkspace,
                                       uint32_t peer)
{
    using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
    using ScratchTile =
        Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);
    GT dstGlobalData(remoteDst, shape, stride);
    GT srcGlobalData(localSrc, shape, stride);
    comm::Signal remoteSignal(remoteSignalPtr);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::SDMA>(
            scratchTile, sdmaWorkspace, session)) {
        return;
    }

    auto event = comm::TPUT_ASYNC_NOTIFY<comm::DmaEngine::SDMA>(
        dstGlobalData, srcGlobalData, remoteSignal, 1,
        comm::NotifyOp::Set, session, peer);
    (void)event.Wait(session);
}
```

### SDMA AtomicAdd

SDMA `AtomicAdd` 的Session和tensor构建方式与上一示例相同：

```cpp
comm::AsyncSession session;
if (comm::BuildAsyncSession<comm::DmaEngine::SDMA>(
        scratchTile, sdmaWorkspace, session)) {
    auto event = comm::TPUT_ASYNC_NOTIFY<comm::DmaEngine::SDMA>(
        dstGlobalData, srcGlobalData, remoteSignal, 1,
        comm::NotifyOp::AtomicAdd, session, peer);
    (void)event.Wait(session);
}
```

### URMA Set

URMA Session不绑定目标rank，调用指令时通过 `peer` 指定目标：

```cpp
comm::AsyncSession session;
if (comm::BuildAsyncSession<comm::DmaEngine::URMA>(
        urmaWorkspace, session)) {
    auto event = comm::TPUT_ASYNC_NOTIFY<comm::DmaEngine::URMA>(
        dstGlobalData, srcGlobalData, remoteSignal, 1,
        comm::NotifyOp::Set, session, peer);
    (void)event.Wait(session);
}
```

### URMA AtomicAdd

URMA `AtomicAdd` 使用相同的Session构建方式：

```cpp
comm::AsyncSession session;
if (comm::BuildAsyncSession<comm::DmaEngine::URMA>(
        urmaWorkspace, session)) {
    auto event = comm::TPUT_ASYNC_NOTIFY<comm::DmaEngine::URMA>(
        dstGlobalData, srcGlobalData, remoteSignal, 1,
        comm::NotifyOp::AtomicAdd, session, peer);
    (void)event.Wait(session);
}
```

### RDMA Set与AtomicAdd

RDMA通过 `peer` 选择目标rank。以下示例用 `Set` 赋值；将同一次调用的操作改为 `NotifyOp::AtomicAdd`，
即可在signal原值上累加 `signalValue`：

```cpp
comm::AsyncSession session;
if (comm::BuildAsyncSession<comm::DmaEngine::RDMA>(
        scratchTile, rdmaWorkspace, myPe, session)) {
    auto event = comm::TPUT_ASYNC_NOTIFY<comm::DmaEngine::RDMA>(
        dstGlobalData, srcGlobalData, remoteSignal, 1,
        comm::NotifyOp::Set, session, peer);
    (void)event.Wait(session);
}
```

### 接收端

```cpp
comm::Signal ready(localSignalPtr);
comm::TWAIT(ready, 1, comm::WaitCmp::EQ);

// 按目标平台和运行时的内存一致性规则保证payload可见后，再读取payload。
```
