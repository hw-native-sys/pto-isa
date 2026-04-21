# 异步通信指令详解（TPUT_ASYNC / TGET_ASYNC / BuildAsyncSession）

## TPUT_ASYNC — 异步远程写

启动 GM→GM DMA 传输，立即返回 `AsyncEvent`。

```cpp
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
AsyncEvent TPUT_ASYNC(GlobalDstData &dst, GlobalSrcData &src,
                      const AsyncSession &session, WaitEvents&... events);
```

## TGET_ASYNC — 异步远程读

启动远端 GM→本地 GM DMA 传输。

```cpp
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
AsyncEvent TGET_ASYNC(GlobalDstData &dst, GlobalSrcData &src,
                      const AsyncSession &session, WaitEvents&... events);
```

---

## BuildAsyncSession — 构建异步会话

### SDMA 构建（默认）

```cpp
template <DmaEngine engine = DmaEngine::SDMA, typename ScratchTile>
bool BuildAsyncSession(ScratchTile &scratchTile, __gm__ uint8_t *workspace,
                       AsyncSession &session,
                       uint32_t syncId = 0,
                       const sdma::SdmaBaseConfig &baseConfig = {sdma::kDefaultSdmaBlockBytes, 0, 1},
                       uint32_t channelGroupIdx = sdma::kAutoChannelGroupIdx);
```

| 参数 | 说明 |
|------|------|
| `scratchTile` | 用于 SDMA 控制元数据的 UB scratch tile（非数据负载），推荐 `Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>`（256B） |
| `workspace` | 由 Host 侧 `SdmaWorkspaceManager` 分配的 GM 指针 |
| `syncId` | MTE3/MTE2 管道同步事件 ID（0-7），避免与 kernel 内其他管道屏障冲突 |
| `baseConfig` | `{block_bytes, comm_block_offset, queue_num}`，默认适用于单队列场景 |
| `channelGroupIdx` | SDMA 通道组索引，默认使用 `get_block_idx()` 映射 |

### URMA 构建（仅 Ascend950 / NPU_ARCH 3510）

```cpp
bool BuildAsyncSession(__gm__ uint8_t *workspace, uint32_t destRankId, AsyncSession &session);
```

---

## 异步约束

- **仅支持扁平连续的逻辑一维 tensor**（非一维返回无效 event）
- SDMA workspace 必须由 Host 侧 `SdmaWorkspaceManager` 分配
- URMA workspace 必须由 Host 侧 `UrmaWorkspaceManager` 分配
- URMA 需要大页内存（`ACL_MEM_MALLOC_HUGE_ONLY`），小页分配导致注册失败
- `scratchTile` 仅用于控制元数据，不是数据暂存缓冲

---

## 完成语义（Quiet 语义）

- `event.Wait(session)` 阻塞直到**自上次 Wait 以来所有已发出的异步操作**全部完成
- 多次异步调用后，只需对最后一个 `AsyncEvent` 调用一次 `Wait`
- 类似 shmem 的 quiet 语义

---

## 完整示例

```cpp
// 构建会话
using ScratchTile = Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;
ScratchTile scratchTile;
TASSIGN(scratchTile, 0x0);

comm::AsyncSession session;
if (!comm::BuildAsyncSession<comm::DmaEngine::SDMA>(scratchTile, sdmaWorkspace, session)) {
    return;
}

// 批量传输 + 一次 Wait
comm::AsyncEvent lastEvent;
for (int rank = 0; rank < nranks; ++rank) {
    GT dstG(remoteDst + rank * size, shape, stride);
    lastEvent = comm::TPUT_ASYNC(dstG, srcG, session);
}
(void)lastEvent.Wait(session);  // 等待所有 pending 操作完成
```
