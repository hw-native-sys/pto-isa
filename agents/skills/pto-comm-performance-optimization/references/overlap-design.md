# 通算重叠（Overlap）详解

## 基本原理

将计算和通信部署在不同硬件资源上并行执行：
- **计算**：Cube 核（矩阵运算）或 Vec 核（向量运算）
- **通信**：MTE 引擎（GM→UB→GM）或 DMA 引擎（GM→GM）

```
时间 →

无重叠：
[── Compute ──][── Comm ──]  Total = T_comp + T_comm

有重叠：
[── Compute ──────────────]
    [── Comm ─────────]     Total ≈ max(T_comp, T_comm)
```

---

## 实现模式：双 Stream + 队列调度

```
computeStream (Cube blocks):
  ┌────┐  ┌────┐  ┌────┐
  │Tile│→│Tile│→│Tile│→ ...
  │ 0  │  │ 1  │  │ 2  │
  └──┬─┘  └──┬─┘  └──┬─┘
     │       │       │
   enqueue enqueue enqueue    ← 计算完成即入队
     │       │       │
     ▼       ▼       ▼
commStream (Vec blocks):
  poll → TPUT → poll → TPUT → ...  ← 立即开始传输
```

**关键设计要素**：

1. **就绪队列**：每个计算 block 一个 SPSC 队列，无锁设计
2. **计算完成即传输**：不等待所有 tile 计算完成
3. **TTEST 轮询**：通信 kernel 使用硬件指令轮询队列

---

## 重叠效率度量

```
重叠效率 = 1 - (实际总时间 - max(T_comp, T_comm)) / min(T_comp, T_comm)
```

- **100%**：完美重叠，总时间 = max(T_comp, T_comm)
- **0%**：无重叠，总时间 = T_comp + T_comm

### 计算方法

```cpp
float pipe_total_us = MeasurePipelined();
float comp_only_us = MeasureComputeOnly();
float comm_est_us = pipe_total_us - comp_only_us;
float speedup = (comp_only_us + comm_est_us) / pipe_total_us;
printf("Overlap speedup: %.2fx\n", speedup);
```

---

## 分块粒度选择

| 粒度 | 优点 | 缺点 |
|------|------|------|
| 细粒度（小 Tile） | 更早开始通信，更好重叠 | 同步开销大，队列管理复杂 |
| 粗粒度（大 Tile） | 同步少，传输效率高 | 计算完成前通信空闲 |

**推荐**：选择使通信开始时间 < 第一个 Tile 计算完成时间的最大 Tile 大小。

---

## 多 Block 负载均衡

多 Block 并行时，如果工作分配不均，最慢的 Block 决定总时间。

### Row-level 均分（推荐）

```cpp
int total_rows = tile_count * ROWS_PER_TILE * (nranks - 1);
int rows_per_block = (total_rows + num_blocks - 1) / num_blocks;
int my_start = block_idx * rows_per_block;
int my_end = min((block_idx + 1) * rows_per_block, total_rows);

while (cur_row < my_end) {
    int flat_transfer = cur_row / ROWS_PER_TILE;
    int row_in_tile = cur_row % ROWS_PER_TILE;
    AgTransferRows(reduced_output, ctx, stride, remote_rank, row_offset, nrows);
    cur_row += nrows;
}
```

### Block 数量选择

| 因素 | 影响 |
|------|------|
| AICore 数量 | Block 数 ≤ 可用 AICore 数 |
| 数据量 | 数据太少不值得多 Block |
| 同步开销 | Block 越多，barrier 中 intra-rank 同步越贵 |
| 通信带宽 | 多 Block 可能竞争同一链路 |

**经验值**：
- 通信 kernel：4~24 blocks
- 计算 kernel：24 blocks（占满 Cube 核）
