# 开发模式详解

## 模式 1：P2P 通信

最基础的模式，使用 TPUT/TGET 在两个 NPU 间传输数据。

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

__global__ AICORE void P2PSendKernel(__gm__ half *local_data, __gm__ half *remote_addr)
{
    using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using Global = GlobalTensor<half, ShapeDyn, StrideDyn, Layout::ND>;
    using TileData = Tile<TileType::Vec, half, 128, 256, BLayout::RowMajor, -1, -1>;

    ShapeDyn shape(1, 1, 1, 128, 256);
    StrideDyn stride(128 * 256, 128 * 256, 128 * 256, 256, 1);

    Global srcG(local_data, shape, stride);
    Global dstG(remote_addr, shape, stride);

    TileData stagingTile(128, 256);
    TASSIGN(stagingTile, 0x0);

    comm::TPUT(dstG, srcG, stagingTile);
}
```

---

## 模式 2：集合通信

使用内置的集合通信指令（适合标准场景）。

```cpp
template <typename T, int NRANKS>
__global__ AICORE void ReduceKernel(__gm__ T *group_ptrs[NRANKS], __gm__ T *result, int my_rank)
{
    using TileT = Tile<TileType::Vec, T, 1, 1024>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,1024>,
                                 Stride<1024,1024,1024,1024,1>, Layout::ND>;

    GTensor tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) tensors[i] = GTensor(group_ptrs[i]);

    comm::ParallelGroup<GTensor> group(tensors, NRANKS, my_rank);
    GTensor dstG(result);
    TileT accTile, recvTile;
    comm::TREDUCE(group, dstG, accTile, recvTile, comm::ReduceOp::Sum);
}
```

---

## 模式 3：自定义集合通信（TPUT + TNOTIFY/TWAIT）

当内置集合通信指令不满足需求时（如 ReduceScatter + AllGather 组合实现 AllReduce），使用底层指令组合。

### 方式 A：使用 TPUT\<AtomicAdd\>（推荐，一步完成 RS + Reduce）

每个 rank 将自己的数据通过 `TPUT<AtomicAdd>` 直接累加到 owner rank 的输出缓冲区，无需独立的 Reduce 阶段。

```cpp
// ReduceScatter：使用 TPUT<AtomicAdd> 直接累加到 owner
AICORE inline void ReduceScatterViaTput(__gm__ half *local_src, __gm__ half *remote_dst,
                                        TileData &pingTile, TileData &pongTile)
{
    Global srcG(local_src, shape, stride);
    Global dstG(remote_dst, shape, stride);

    // TPUT<AtomicAdd> 自动处理流水线同步，内部分块滑动
    comm::TPUT<AtomicType::AtomicAdd>(dstG, srcG, pingTile, pongTile);
}

// AllGather：使用 TPUT<AtomicNone> 直接写到远端
AICORE inline void AllGatherViaTput(__gm__ half *local_src, __gm__ half *remote_dst,
                                    TileData &pingTile, TileData &pongTile)
{
    Global srcG(local_src, shape, stride);
    Global dstG(remote_dst, shape, stride);

    comm::TPUT(dstG, srcG, pingTile, pongTile);
}
```

### 方式 B：使用 TLOAD/TSTORE_IMPL（更底层，需手动流水线同步）

需要在 TLOAD 和 TSTORE_IMPL 之间手动插入 `set_flag`/`wait_flag` 做流水线同步。适合需要在传输间插入自定义逻辑的场景。

```cpp
// ReduceScatter：手动流水线 + AtomicAdd
AICORE inline void ReduceScatterManual(__gm__ half *src_addr, __gm__ half *dst_addr,
                                       TileData &pingTile, TileData &pongTile, int pp_count)
{
    bool use_ping = (pp_count % 2 == 0);
    TileData &curTile = use_ping ? pingTile : pongTile;
    event_t curEv = use_ping ? EVENT_ID0 : EVENT_ID1;

    Global srcG(src_addr, shape, stride);
    Global dstG(dst_addr, shape, stride);

    TLOAD(curTile, srcG);
    set_flag(PIPE_MTE2, PIPE_MTE3, curEv);
    wait_flag(PIPE_MTE2, PIPE_MTE3, curEv);
    TSTORE_IMPL<TileData, Global, AtomicType::AtomicAdd>(dstG, curTile);
    set_flag(PIPE_MTE3, PIPE_MTE2, curEv);
    wait_flag(PIPE_MTE3, PIPE_MTE2, curEv);
}

// AllGather：手动流水线 + 普通写
AICORE inline void AllGatherManual(__gm__ half *src_addr, __gm__ half *dst_addr,
                                   TileData &tile)
{
    Global srcG(src_addr, shape, stride);
    Global dstG(dst_addr, shape, stride);

    TLOAD(tile, srcG);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE_IMPL<TileData, Global, AtomicType::AtomicNone>(dstG, tile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}
```

### 方式选择

| 方式 | 优点 | 缺点 | 适用 |
|------|------|------|------|
| TPUT\<AtomicAdd\> | 代码简洁，自动流水线同步 | 灵活性低 | 标准 RS/AG 场景 |
| TLOAD/TSTORE_IMPL | 可插入自定义逻辑 | 需手动 set_flag/wait_flag | 需要精细控制的场景 |

---

## 模式 4：通算融合（计算+通信重叠）

将计算 kernel 和通信 kernel 分别部署在不同的 AICore Block 上，通过 Stream 并行和队列同步实现重叠。

```
computeStream: [GEMM Block 0] [GEMM Block 1] ... [GEMM Block N]
                     │              │                    │
                  Enqueue        Enqueue             Enqueue
                     │              │                    │
                     ▼              ▼                    ▼
commStream:    [RS: poll queues, TPUT<AtomicAdd>] → [Barrier] → [AG: TPUT<AtomicNone>]
```

### 关键设计要素

1. **双 Stream**：计算流（Cube kernel）和通信流（Vec kernel）并行执行
2. **就绪队列**：计算完成后将 tile 索引入队，通信 kernel 轮询出队
3. **信号矩阵**：跨 rank 同步，确保 RS 阶段完成后才开始 AG
4. **Phase Barrier**：多阶段执行的 rank 间同步

### 就绪队列设计（SPSC 无锁队列）

```cpp
// 生产者端（计算 kernel）：
PerBlockQueueEnqueueFast(cached_queue, tile_idx, local_slot);

// 消费者端（通信 kernel）：使用 TTEST 硬件指令轮询
comm::Signal sig(const_cast<__gm__ int32_t *>(&queue->count));
if (!comm::TTEST(sig, local_head + 1, comm::WaitCmp::GE)) {
    return -1;  // 无新数据
}
```
