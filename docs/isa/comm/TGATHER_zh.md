# TGATHER

## 简介

Gather 操作：调用方 NPU（根节点）从并行组中所有 rank 收集数据，并沿 **DIM_3**（行维度）拼接到本地输出缓冲区。

只有根节点需要执行 `TGATHER`。非根节点只需确保在操作期间其源缓冲区已就绪且保持有效。在非根节点上调用 `TGATHER` 属于未定义行为。

**大 Tile 支持**：当 GlobalTensor 在行和/或列方向超出 UB Tile 容量时，传输将通过二维滑动自动分块——与其他 PTO-COMM 指令采用相同机制。

## 数学语义

每个 rank $r$ 的源数据形状为 $(D_0, D_1, D_2, H, W)$。gather 沿 DIM_3 拼接所有 $N$ 个 rank 的数据：

$$\mathrm{dst}_{d_0, d_1, d_2,\; r \cdot H + i,\; j} = \mathrm{src}^{(r)}_{d_0, d_1, d_2,\; i,\; j} \quad \forall\, r \in [0, N),\; i \in [0, H),\; j \in [0, W)$$

目标 tensor 的形状为 $(D_0, D_1, D_2, N \times H, W)$。

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../../assembly/PTO-AS_zh.md)。

同步形式：

```text
tgather %group, %dst : (!pto.group<...>, !pto.memref<...>)
```

降级时会为 GM→UB→GM 数据路径引入 UB 暂存 Tile；C++ 内建接口需要显式传入 `stagingTileData`（或 `pingTile` / `pongTile`）操作数。

## C++ 内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`：

```cpp
// 基础 gather（单暂存 Tile）
template <typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                             TileData &stagingTileData, WaitEvents&... events);

// 乒乓 gather（使用两个暂存 Tile 实现双缓冲）
template <typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                             TileData &pingTile, TileData &pongTile, WaitEvents&... events);
```

## 约束

- **类型约束**：
    - `ParallelGroup::value_type::RawDType` 必须等于 `GlobalDstData::RawDType`。
    - `TileData::DType` 必须等于 `GlobalDstData::RawDType`。
- **内存约束**：
    - `dstGlobalData` 必须指向本地内存（当前 NPU），且足够容纳所有 rank 拼接后的结果。具体要求：`dstGlobalData.GetShape(DIM_3)` 必须 $\geq N \times H$，其中 $H$ 为每个 rank 的 `GetShape(DIM_3)`。
    - 若 `dstGlobalData.GetShape(DIM_3) > N × H`，则只写入前 `N × H` 行，其余行保持不变。
    - `stagingTileData`（或 `pingTile` / `pongTile`）必须预先在 UB 中分配。
- **ParallelGroup 约束**：
    - `parallelGroup.tensors[r]` 必须指向 rank `r` 的源缓冲区（从根节点视角看到的远端 GM）。
    - `parallelGroup.GetRootIdx()` 标识调用方 NPU 为 gather 根节点。
    - 所有源 tensor 假定具有相同的形状和步幅；否则行为未定义。
- **分块模式约束**（源数据超出单个 UB Tile 时）：
    - 若 `TileData` 具有静态 `ValidRow`，则每个 rank 源数据的 `GetShape(DIM_3)` 必须能被 `ValidRow` 整除。如需支持不足一行的情况，请使用 `DYNAMIC` ValidRow 的 Tile。
    - 若 `TileData` 具有静态 `ValidCol`，则 `GetShape(DIM_4)` 必须能被 `ValidCol` 整除。如需支持不足一列的情况，请使用 `DYNAMIC` ValidCol 的 Tile。

## 示例

### 基础 Gather（单暂存 Tile）

每个 rank 提供 `ROWS × COLS` 的数据，根节点将其收集到 `NRANKS * ROWS` 行中。
Tile 大小（`TILE_ROWS × TILE_COLS`）可小于每 rank 的数据——此时实现会自动沿 DIM_3 和 DIM_4 通过二维滑动进行分块传输。

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int ROWS, int COLS, int TILE_ROWS, int TILE_COLS, int NRANKS>
void gather(__gm__ T* group_addrs[NRANKS], __gm__ T* result, int my_rank) {
    using TileT    = Tile<TileType::Vec, T, TILE_ROWS, TILE_COLS, BLayout::RowMajor, -1, -1>;
    using GPerRank = GlobalTensor<T, Shape<1,1,1,ROWS,COLS>,
                                  BaseShape2D<T, ROWS, COLS, Layout::ND>, Layout::ND>;
    using GResult  = GlobalTensor<T, Shape<1,1,1,NRANKS*ROWS,COLS>,
                                  BaseShape2D<T, NRANKS*ROWS, COLS, Layout::ND>, Layout::ND>;

    GPerRank tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) tensors[i] = GPerRank(group_addrs[i]);

    comm::ParallelGroup<GPerRank> group(tensors, NRANKS, my_rank);
    GResult dstG(result);
    TileT stagingTile(TILE_ROWS, TILE_COLS);
    comm::TGATHER(group, dstG, stagingTile);
}
```

### 乒乓 Gather（双缓冲）

使用两个 UB Tile，将下一块的 TLOAD（MTE2）与当前块的 TSTORE（MTE3）重叠执行。

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int ROWS, int COLS, int TILE_ROWS, int TILE_COLS, int NRANKS>
void gather_pingpong(__gm__ T* group_addrs[NRANKS], __gm__ T* result, int my_rank) {
    using TileT    = Tile<TileType::Vec, T, TILE_ROWS, TILE_COLS, BLayout::RowMajor, -1, -1>;
    using GPerRank = GlobalTensor<T, Shape<1,1,1,ROWS,COLS>,
                                  BaseShape2D<T, ROWS, COLS, Layout::ND>, Layout::ND>;
    using GResult  = GlobalTensor<T, Shape<1,1,1,NRANKS*ROWS,COLS>,
                                  BaseShape2D<T, NRANKS*ROWS, COLS, Layout::ND>, Layout::ND>;

    GPerRank tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) tensors[i] = GPerRank(group_addrs[i]);

    comm::ParallelGroup<GPerRank> group(tensors, NRANKS, my_rank);
    GResult dstG(result);
    TileT pingTile(TILE_ROWS, TILE_COLS);
    TileT pongTile(TILE_ROWS, TILE_COLS);
    // 乒乓模式：将 TLOAD 与 TSTORE 重叠执行以提升吞吐量
    comm::TGATHER(group, dstG, pingTile, pongTile);
}
```
