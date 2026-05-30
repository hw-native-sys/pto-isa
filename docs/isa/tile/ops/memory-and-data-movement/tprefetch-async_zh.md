# pto.tprefetch_async

## 简介

`TPREFETCH_ASYNC` 通过 SDMA CMO（Cache Maintenance Operation, opcode=6）将 Global Memory (GM/HBM) 中的数据异步预取到 NPU L2 Cache。与将数据搬入 UB 的 `TPREFETCH` 不同，`TPREFETCH_ASYNC` 只将数据预热到 L2 Cache，不占用数据对应的 UB 空间，后续依赖的 `TLOAD` 可以从 L2 命中。

该指令是面向计算侧的内存访问/缓存提示接口。虽然内部使用 SDMA CMO 路径，公开 API 仍位于 `pto` 命名空间中。

## 数据流

```text
GM / HBM  --(SDMA CMO prefetch)-->  L2 Cache
                                         |
                                         +-- 后续 TLOAD 命中 L2
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
namespace pto {

template <typename GlobalData, typename... WaitEvents>
PTO_INST comm::AsyncEvent TPREFETCH_ASYNC(GlobalData &src, PrefetchAsyncContext &ctx,
                                          WaitEvents &... events);

} // namespace pto
```

`PrefetchAsyncContext` 保存由 Host 侧 `SdmaWorkspaceManager::Init` 初始化后的 SDMA workspace 指针：

```cpp
pto::PrefetchAsyncContext ctx(workspace);
auto evt = pto::TPREFETCH_ASYNC(srcGlobal, ctx);
evt.Wait(ctx.session);
```

指令内部使用默认参数构造 SDMA session（`channelGroupIdx = get_block_idx()`、`syncId = 0`、`queue_num = 1`），并保存在 `PrefetchAsyncContext` 中。后续消费者依赖预取结果时，使用返回的 `comm::AsyncEvent` 和 `ctx.session` 等待完成。

### 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `src` | `GlobalData&` | 需要预取到 L2 的 GlobalTensor 区域 |
| `ctx` | `PrefetchAsyncContext&` | 计算侧预取上下文，包含 SDMA workspace 基址以及内部构造的 `comm::AsyncSession` |
| `events...` | `WaitEvents&...` | 可选同步事件 |

### 返回值

返回 `comm::AsyncEvent`，用于跟踪异步预取完成状态。后续 `TLOAD` 依赖预取结果时，调用 `evt.Wait(ctx.session)` 等待完成。

## 约束

- 源数据必须位于 Global Memory (GM/HBM)。
- GlobalTensor 必须是平坦连续的一维布局。
- SDMA workspace 需要在 kernel 启动前由 Host 侧初始化，并传入 kernel。
- `PrefetchAsyncContext` 内部持有 256-byte UB scratch tile 和 `AsyncSession`，用于构造 SDMA 元数据并等待事件完成。
- SDMA CMO 按 cache line 粒度工作，非对齐范围由硬件处理。
- CPU simulation 后端中该指令为空操作，返回空 `AsyncEvent`。

## 与 TPREFETCH 对比

| 维度 | `pto::TPREFETCH` | `pto::TPREFETCH_ASYNC` |
|------|------------------|------------------------|
| 数据流 | GM 到 UB | GM 到 L2 Cache |
| 硬件路径 | MTE (`copy_gm_to_ubuf`) | SDMA CMO (opcode=6) |
| UB 占用 | 需要目标 Tile | 数据不占用 UB，仅内部使用 scratch |
| 同步方式 | 同步 | 异步 (`AsyncEvent`) |
| 典型用途 | 小数据预取到 UB | 大数据或跨阶段数据预热到 L2 |
