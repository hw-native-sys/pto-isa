# TPUT_ASYNC

## 概述

`TPUT_ASYNC` 是异步远程写原语：启动从本地 GM 到远端 GM 的 DMA 传输后立即返回 `AsyncEvent`，不阻塞。稍后通过事件等待传输完成。

支持两种 DMA 引擎：SDMA（默认，所有目标均可用）和 URMA（硬件 RDMA，仅 Ascend950 / NPU_ARCH 3510 可用）。

## 机制

`TPUT_ASYNC` 启动从本地 GM 到远端 GM 的 DMA 传输后立即返回：

```
srcGlobalData（本地 GM） → DMA 引擎 → dstGlobalData（远端 GM）
```

`AsyncSession` 管理引擎无关的异步状态。发出一个或多个异步操作后，调用 `event.Wait(session)` 阻塞直到所有 pending 操作完成（quiet 语义）。

不同引擎的完成机制：

- **SDMA**：只提交数据传输 SQE，flag SQE 延迟到 `Wait` 时提交，通过轮询 flag 判断完成。
- **URMA**：立即提交 RDMA WRITE WQE 并敲门铃，`Wait` 轮询 Completion Queue 直到所有预期的 CQE 被消费。

## 模板参数

- `engine`：
    - `DmaEngine::SDMA`（默认）
    - `DmaEngine::URMA`（Ascend950，仅 NPU_ARCH 3510）

> **注意（SDMA 路径）**
> `TPUT_ASYNC` 配合 `DmaEngine::SDMA` 目前**仅支持扁平连续的逻辑一维 tensor**。
> 当前 SDMA 异步实现不支持非一维或非连续布局。

## C++ 内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`：

```cpp
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                               const AsyncSession &session, WaitEvents &... events);
```

`AsyncSession` 是引擎无关的会话对象。使用 `BuildAsyncSession<engine>()` 构建一次后，传递给所有异步调用和事件等待。模板参数 `engine` 在编译期选择 DMA 后端。

## 输入

|| 操作数 | 类型 | 说明 |
||--------|------|------|
|| `dstGlobalData` | `GlobalTensor` | 远端目标，必须为扁平连续一维 |
|| `srcGlobalData` | `GlobalTensor` | 本地源，必须为扁平连续一维 |
|| `session` | `AsyncSession` | 引擎无关的会话对象 |
|| `WaitEvents...` | `RecordEvent...` | 发指令前要等待的事件 |

## 预期输出

|| 结果 | 类型 | 说明 |
||------|------|------|
|| `AsyncEvent` | event | 后续用于 `Wait` 调用的句柄 |

## 副作用

此操作启动从本地 GM 到远端 GM 的 DMA 传输。完成时机延后到 `Wait` 调用。

## AsyncSession 构建

使用 `include/pto/comm/async/async_event_impl.hpp` 中的 `BuildAsyncSession`。
该函数有两个重载——分别用于 SDMA 和 URMA，参数列表不同。

### SDMA 构建（默认）

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
| `scratchTile` | — | 用于 SDMA 控制元数据的 UB scratch tile（参见 [scratchTile 的作用](#scratchtile-的作用)）。|
| `workspace` | — | 由主机侧 `SdmaWorkspaceManager` 分配的 GM 指针。|
| `session` | — | 输出的 `AsyncSession` 对象。|
| `syncId` | `0` | MTE3/MTE2 管道同步事件 ID（0-7）。若 kernel 在相同 ID 上使用了其他管道屏障，则需覆盖此值。|
| `baseConfig` | `{kDefaultSdmaBlockBytes, 0, 1}` | `{block_bytes, comm_block_offset, queue_num}`。适用于大多数单队列传输场景。|
| `channelGroupIdx` | `kAutoChannelGroupIdx` | SDMA 通道组索引。默认内部使用 `get_block_idx()` 映射到当前 AI Core。多 block 或自定义通道映射场景下需覆盖此值。|

### URMA 构建（仅 NPU_ARCH 3510）

> URMA（User-level RDMA Memory Access）是 Ascend950（NPU_ARCH 3510）上的硬件加速 RDMA 传输引擎。

```cpp
#ifdef PTO_URMA_SUPPORTED
template <DmaEngine engine>
PTO_INTERNAL bool BuildAsyncSession(__gm__ uint8_t *workspace,
                                    uint32_t destRankId,
                                    AsyncSession &session);
#endif
```

| 参数 | 说明 |
|---|---|
| `workspace` | 由主机侧 `UrmaWorkspaceManager` 分配的 GM 指针。|
| `destRankId` | 此会话通信的远端 PE rank id。对于 `TPUT_ASYNC`，这是数据写入的目标 rank。|
| `session` | 输出的 `AsyncSession` 对象。|

URMA 不需要 `scratchTile`——轮询通过 `ld_dev`/`st_dev` 硬件原语直接操作。

## 约束

### 类型约束

- `GlobalSrcData::RawDType == GlobalDstData::RawDType`
- `GlobalSrcData::layout == GlobalDstData::layout`

### 传输约束

- SDMA 和 URMA 路径均要求源 tensor 为**扁平连续的逻辑一维**
- 若不满足一维连续要求，当前实现返回无效 async event（`handle == 0`）

### 内存约束

- SDMA workspace 必须是由主机侧 `SdmaWorkspaceManager` 分配的有效 GM 指针
- URMA workspace 必须是由主机侧 `UrmaWorkspaceManager` 分配的有效 GM 指针
- URMA 仅在 NPU_ARCH 3510（Ascend950）上可用
- 传给 `UrmaWorkspaceManager::Init()` 的对称数据缓冲区必须由大页内存支撑（使用 `ACL_MEM_MALLOC_HUGE_ONLY` 分配）

## scratchTile 的作用

`scratchTile` **不是**用于存放用户数据负载的暂存缓冲区。
它被转换为 `TmpBuffer`，用作临时 UB 工作区，用于：

- 写入/读取 SDMA 控制字（flag、sq_tail、channel_info）
- 轮询事件完成标志
- 完成时提交队列尾部

实际数据负载直接在 GM 缓冲区之间传输；`scratchTile` 仅用于控制和同步元数据。

## scratchTile 类型与大小约束

- 必须是 `pto::Tile` 类型
- 必须是 UB/Vec tile（`ScratchTile::Loc == TileType::Vec`）
- 可用字节数至少为 `sizeof(uint64_t)`（8 字节）

推荐使用：`Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>`（256B）。

## 完成语义（Quiet 语义）

不同引擎的底层完成机制不同，但用户侧的 quiet 语义行为一致：

- **SDMA**：`TPUT_ASYNC` 仅提交数据传输 SQE，flag SQE 延迟到 `Wait` 时提交，通过轮询 flag 判断完成。
- **URMA**：`TPUT_ASYNC` 立即提交 RDMA WRITE WQE 并敲门铃。`Wait` 通过轮询 Completion Queue（CQ）等待所有预期的 CQE 被消费。

- `event.Wait(session)` — 阻塞，直到**自上次 Wait 以来所有已发出的异步操作**全部完成

这意味着多次 `TPUT_ASYNC` 调用后，只需对最后一个返回的 `AsyncEvent` 调用一次 `Wait`，即可等待所有 pending 操作完成（类似 shmem 的 quiet 语义）。

wait 成功后，所有已发出的 `dstGlobalData` 写入均已全部完成。

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

### 批量传输（Quiet 语义）

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

### URMA 示例（NPU_ARCH 3510）

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

## 目标Profile限制

- SDMA 在所有目标上可用。URMA 仅在 Ascend950（Ascend 9xx）上可用。
- CPU 模拟器不支持异步通信操作。
- `AsyncSession` 是引擎无关的；切换引擎需要重新编译。

## 相关页面

- 通信概述：[通信与运行时](../other/communication-and-runtime_zh.md)
- 同步对应：[TPUT](./TPUT_zh.md)
- 异步读：[TGET_ASYNC](./TGET_ASYNC_zh.md)
- 指令集：[其他与通信](../other/README_zh.md)
