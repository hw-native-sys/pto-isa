# 信号与同步设计

## 信号矩阵布局

通信算子通常需要一个信号矩阵来协调多 rank 同步。典型布局：

```
信号矩阵（每个 rank 一份，存放在通信窗口/对称内存中）：
┌──────────────────────────────────────────────────────┐
│ [0..MAX_RANKS-1]     跨 rank 阶段计数器（RS/AG done）│
│ [MAX_RANKS]          本地广播标志（block 0 → all）   │
│ [MAX_RANKS+1]        intra-rank block 到达计数器     │
│ ...（按需扩展更多阶段）                              │
└──────────────────────────────────────────────────────┘
```

**设计原则**：
- 每个 phase 使用独立的信号区域，避免信号复用导致竞争
- 使用 `NotifyOp::AtomicAdd` 实现计数型同步（多方通知一方）
- 使用 `NotifyOp::Set` 实现标志型同步（一方通知多方）

---

## 跨 Rank Barrier 模式

```cpp
AICORE inline void DeviceBarrier(__gm__ DeviceContext *ctx, __gm__ int32_t *signal_base,
                                 int phase, int my_rank, int nranks,
                                 int block_idx, int num_comm_blocks)
{
    // 步骤 1：Intra-rank barrier（所有 block 必须到达）
    __gm__ int32_t *intra_counter = signal_base + INTRA_RANK_OFFSET + phase;
    if (block_idx != 0) {
        comm::Signal arrSig(intra_counter);
        comm::TNOTIFY(arrSig, 1, comm::NotifyOp::AtomicAdd);
    } else {
        if (num_comm_blocks > 1) {
            comm::Signal arrSig(intra_counter);
            comm::TWAIT(arrSig, num_comm_blocks - 1, comm::WaitCmp::GE);
        }
    }
    pipe_barrier(PIPE_ALL);

    // 步骤 2：Cross-rank barrier（仅 block 0 执行）
    if (block_idx == 0) {
        for (int r = 0; r < nranks; r++) {
            if (r == my_rank) continue;
            __gm__ int32_t *remote_sig = GetRemotePtr(ctx, signal_base + my_rank, r);
            comm::Signal sig(remote_sig);
            comm::TNOTIFY(sig, 1, comm::NotifyOp::AtomicAdd);
        }
        for (int r = 0; r < nranks; r++) {
            if (r == my_rank) continue;
            comm::Signal sig(signal_base + r);
            comm::TWAIT(sig, 1, comm::WaitCmp::GE);
        }
        __gm__ int32_t *local_flag = signal_base + LOCAL_FLAG_OFFSET + phase;
        comm::Signal localSig(local_flag);
        comm::TNOTIFY(localSig, 1, comm::NotifyOp::Set);
    } else {
        __gm__ int32_t *local_flag = signal_base + LOCAL_FLAG_OFFSET + phase;
        comm::Signal localSig(local_flag);
        comm::TWAIT(localSig, 1, comm::WaitCmp::GE);
    }
    pipe_barrier(PIPE_ALL);
}
```

---

## Intra-kernel 同步模式

通信 kernel 内部不同阶段之间使用 `pipe_barrier(PIPE_ALL)` 分隔：

```cpp
// 阶段 1：ReduceScatter
ReduceScatterPhase(...);
pipe_barrier(PIPE_ALL);

// 阶段 2：跨 rank barrier
DeviceBarrier(...);

// 阶段 3：AllGather
AllGatherPhase(...);
pipe_barrier(PIPE_ALL);
```

---

## 手动流水线同步

当使用底层 `TLOAD`/`TSTORE_IMPL` 而非高层 `TPUT`/`TGET` 时，需要手动管理流水线同步：

```cpp
TLOAD(tile, srcG);
set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
TSTORE_IMPL<TileData, Global, AtomicType::AtomicNone>(dstG, tile);
set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
```

**注意**：TPUT/TGET 内部已处理流水线同步，直接使用时无需手动 set_flag/wait_flag。仅在使用底层 `TLOAD`/`TSTORE_IMPL` 构建自定义流水线时才需要。
