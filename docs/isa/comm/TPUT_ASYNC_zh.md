# TPUT_ASYNC

## 简介

`TPUT_ASYNC` 是异步远程写原语。它启动一次从本地 GM 到远端 GM 的传输，并立即返回 `AsyncEvent`。

数据流：

`srcGlobalData（本地 GM）` → DMA 引擎 → `dstGlobalData（远端 GM）`

## 模板参数

- `engine`：
    - `DmaEngine::SDMA`（默认）
    - `DmaEngine::URMA`（待实现）

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

`AsyncSession` 是引擎无关的会话对象。使用 `BuildAsyncSession<engine>()` 构建一次后，传递给所有异步调用和事件等待。模板参数 `engine` 在编译期选择 DMA 后端，使代码对未来引擎（URMA、CCU 等）保持前向兼容。

## AsyncSession 构建

使用 `include/pto/comm/async/async_event_impl.hpp` 中的 `BuildAsyncSession`：

```cpp
template <DmaEngine engine = DmaEngine::SDMA, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(ScratchTile &scratchTile,
                                    __gm__ uint8_t *workspace,
                                    AsyncSession &session,
                                    uint32_t syncId = 0,
                                    const sdma::SdmaBaseConfig &baseConfig = {32 * 1024, 0, 1},
                                    uint32_t channelGroupIdx = sdma::kAutoChannelGroupIdx);
```

带默认值的参数说明：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `syncId` | `0` | MTE3/MTE2 管道同步事件 ID（0-7）。若 kernel 在相同 ID 上使用了其他管道屏障，则需覆盖此值。|
| `baseConfig` | `{32*1024, 0, 1}` | `{block_bytes, comm_block_offset, queue_num}`。适用于大多数单队列传输场景。|
| `channelGroupIdx` | `kAutoChannelGroupIdx` | SDMA 通道组索引。默认内部使用 `get_block_idx()` 映射到当前 AI Core。多 block 或自定义通道映射场景下需覆盖此值。|

## 约束

- `GlobalSrcData::RawDType == GlobalDstData::RawDType`
- `GlobalSrcData::layout == GlobalDstData::layout`
- SDMA 路径要求源 tensor 为**扁平连续的逻辑一维**
- workspace 必须是由主机侧 `SdmaWorkspaceManager` 分配的有效 GM 指针

若不满足一维连续要求，当前实现返回无效 async event（`handle == 0`）。

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

## 完成语义

使用 `AsyncEvent` 同步：

- `event.Wait(session)` — 阻塞直到传输完成

wait 成功后，对 `dstGlobalData` 的写入已全部完成。

## 示例

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

