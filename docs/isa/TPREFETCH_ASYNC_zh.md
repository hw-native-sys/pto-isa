# TPREFETCH_ASYNC

## 简介

`TPREFETCH_ASYNC` 通过 SDMA CMO (Cache Maintenance Operation, opcode=6) 将 Global Memory (GM/HBM) 中的数据异步预热到 NPU L2 Cache。与将数据搬入 UB 的 `TPREFETCH` 不同，`TPREFETCH_ASYNC` 只将数据预热到 L2 Cache，数据本身不占用 UB 空间，后续依赖的 `TLOAD` 可以从 L2 命中。

## 数据流

```text
GM / HBM  ──(SDMA CMO prefetch)──>  L2 Cache
                                       │
                                       └── 后续 TLOAD 命中 L2
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`，位于 `pto` 命名空间下：

```cpp
namespace pto {

template <typename GlobalData, typename... WaitEvents>
PTO_INST comm::AsyncEvent TPREFETCH_ASYNC(GlobalData &src, PrefetchAsyncContext &ctx,
                                          WaitEvents &... events);

} // namespace pto
```

`PrefetchAsyncContext` 持有 SDMA workspace 指针，该指针需在 Host 侧通过 `SdmaWorkspaceManager::Init` 预先初始化：

```cpp
pto::PrefetchAsyncContext ctx(workspace);
auto evt = pto::TPREFETCH_ASYNC(srcGlobal, ctx);
evt.Wait(ctx.session);
```

指令内部使用默认参数构造的 SdmaAsyncSession (`channelGroupIdx = get_block_idx()`、`syncId = 0`、`queue_num = 1`)。后续消费者依赖预取结果时，使用返回的 `comm::AsyncEvent` 和 `ctx.session` 等待完成。

### 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `src` | `GlobalData&` | 需要预取到 L2 的 `GlobalTensor` 区域 |
| `ctx` | `PrefetchAsyncContext&` | 计算侧预取上下文，包含 SDMA workspace 基址以及内部构造的 `comm::AsyncSession` |
| `events...` | `WaitEvents&...` | 可选同步事件 |

### 返回值

返回 `comm::AsyncEvent`，用于跟踪异步预取完成状态。后续 `TLOAD` 依赖预取结果时，调用 `evt.Wait(ctx.session)` 等待完成后再消费预取数据。

## 约束

- 源数据必须位于 Global Memory (GM/HBM 地址空间)。
- 源 `GlobalTensor` 必须是平坦连续的一维布局；否则该指令会返回无效 async event (`handle == 0`)。
- SDMA workspace 需要在 kernel 启动前由 Host 侧初始化，并传入 kernel。
- Auto 模式（`__PTO_AUTO__`）下该指令为空操作桩函数，仅保证 API 兼容，返回无效 async event（`handle == 0`）。
- CPU 仿真后端中该指令为空操作，返回无效 async event（`handle == 0`）。

## 说明

- `PrefetchAsyncContext` 内部持有一个 256B 的 UB scratch tile，仅用于承载 SDMA 控制元数据，不承载预取数据本身；同时保存返回事件所使用的 `AsyncSession`。
- SDMA CMO 按 cache line 粒度工作，非对齐范围由硬件处理。

## 与 TPREFETCH 对比

| 维度 | `pto::TPREFETCH` | `pto::TPREFETCH_ASYNC` |
|------|------------------|------------------------|
| 数据流 | GM 到 UB | GM 到 L2 Cache |
| 硬件路径 | MTE (`copy_gm_to_ubuf`) | SDMA CMO (opcode=6) |
| UB 占用 | 需要目标 Tile | 数据不占用 UB，仅使用 256B scratch 用于 SQE 构造 |
| 同步方式 | 同步 | 异步 (`AsyncEvent`) |
| 典型用途 | 小数据预取到 UB | 大数据或跨阶段数据 L2 预热 |

## 示例

### 基础用法

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

__global__ AICORE void my_kernel(__gm__ half *src, __gm__ half *dst,
                                 __gm__ uint8_t *workspace)
{
    using GShape = Shape<1, 1, 1, 1, 16384>;
    using GStride = Stride<16384, 16384, 16384, 16384, 1>;
    GlobalTensor<half, GShape, GStride> srcGlobal(src);

    PrefetchAsyncContext ctx(workspace);
    auto evt = TPREFETCH_ASYNC(srcGlobal, ctx);
    evt.Wait(ctx.session);

    using TileData = Tile<TileType::Vec, half, 128, 128, BLayout::RowMajor>;
    TileData tile;
    TLOAD(tile, srcGlobal);
}
```
