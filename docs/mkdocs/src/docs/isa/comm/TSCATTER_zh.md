<!-- Generated from `docs/isa/comm/TSCATTER_zh.md` -->

# TSCATTER

## 简介

Scatter 操作：调用方 NPU（根节点）将本地源 tensor 沿 **DIM_3**（行维度）拆分后分发到并行组中所有 rank。该操作是 `TGATHER` 的逆操作。

只有根节点需要执行 `TSCATTER`。非根节点只需确保在操作期间其目标缓冲区已分配且可写。在非根节点上调用 `TSCATTER` 属于未定义行为。

**大 Tile 支持**：当每 rank 的数据在行和/或列方向超出 UB Tile 容量时，传输将通过二维滑动自动分块。

## 数学语义

本地源 tensor 的形状为 $(D_0, D_1, D_2, N \times H, W)$，其中 $N$ 为 rank 总数，每个 rank 接收 $H$ 行。操作完成后：

$$\mathrm{dst}^{(r)}_{d_0, d_1, d_2,\; i,\; j} = \mathrm{src}^{\mathrm{local}}_{d_0, d_1, d_2,\; r \cdot H + i,\; j} \quad \forall\, r \in [0, N),\; i \in [0, H),\; j \in [0, W)$$

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../../assembly/PTO-AS_zh.md)。

同步形式：

```text
tscatter %group, %src : (!pto.group<...>, !pto.memref<...>)
```

降级时会为 GM→UB→GM 数据路径引入 UB 暂存 Tile；C++ 内建接口需要显式传入 `stagingTileData`（或 `pingTile` / `pongTile`）操作数。

## C++ 内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`：

```cpp
// 基础 scatter（单暂存 Tile）
template <typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                              TileData &stagingTileData, WaitEvents&... events);

// 乒乓 scatter（使用两个暂存 Tile 实现双缓冲）
template <typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                              TileData &pingTile, TileData &pongTile, WaitEvents&... events);
```

## 约束

- **类型约束**：
    - `ParallelGroup::value_type::RawDType` 必须等于 `GlobalSrcData::RawDType`。
    - `TileData::DType` 必须等于 `GlobalSrcData::RawDType`。
- **内存约束**：
    - `srcGlobalData` 必须指向本地内存（当前 NPU），且足够容纳所有 rank 的数据。具体要求：`srcGlobalData.GetShape(DIM_3)` 必须 $\geq N \times H$，其中 $H$ 为每个 rank 的 `GetShape(DIM_3)`。
    - 若 `srcGlobalData.GetShape(DIM_3) > N × H`，则只读取前 `N × H` 行，其余行被忽略。
    - `stagingTileData`（或 `pingTile` / `pongTile`）必须预先在 UB 中分配。
- **ParallelGroup 约束**：
    - `parallelGroup.tensors[r]` 必须指向 rank `r` 的目标缓冲区（从根节点视角看到的远端 GM）。
    - `parallelGroup.GetRootIdx()` 标识调用方 NPU 为 scatter 根节点。
    - 所有目标 tensor 假定具有相同的形状和步幅；否则行为未定义。
- **分块模式约束**（每 rank 数据超出单个 UB Tile 时）：
    - 若 `TileData` 具有静态 `ValidRow`，则每个 rank 目标数据的 `GetShape(DIM_3)` 必须能被 `ValidRow` 整除。如需支持不足一行的情况，请使用 `DYNAMIC` ValidRow 的 Tile。
    - 若 `TileData` 具有静态 `ValidCol`，则 `GetShape(DIM_4)` 必须能被 `ValidCol` 整除。如需支持不足一列的情况，请使用 `DYNAMIC` ValidCol 的 Tile。

## 示例

### 基础 Scatter（单暂存 Tile）

根节点拥有 `NRANKS * ROWS` 行、宽度为 `COLS` 的数据，每个 rank 接收 `ROWS × COLS`，沿 DIM_3 拆分。
Tile 大小可小于每 rank 的数据——此时实现会自动通过二维滑动进行分块传输。

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int ROWS, int COLS, int TILE_ROWS, int TILE_COLS, int NRANKS>
void scatter(__gm__ T* local_data, __gm__ T* group_addrs[NRANKS], int my_rank) {
    using TileT    = Tile<TileType::Vec, T, TILE_ROWS, TILE_COLS, BLayout::RowMajor, -1, -1>;
    using GPerRank = GlobalTensor<T, Shape<1,1,1,ROWS,COLS>,
                                  BaseShape2D<T, ROWS, COLS, Layout::ND>, Layout::ND>;
    using GSource  = GlobalTensor<T, Shape<1,1,1,NRANKS*ROWS,COLS>,
                                  BaseShape2D<T, NRANKS*ROWS, COLS, Layout::ND>, Layout::ND>;

    GPerRank tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) tensors[i] = GPerRank(group_addrs[i]);

    comm::ParallelGroup<GPerRank> group(tensors, NRANKS, my_rank);
    GSource srcG(local_data);
    TileT stagingTile(TILE_ROWS, TILE_COLS);
    comm::TSCATTER(group, srcG, stagingTile);
}
```

### 乒乓 Scatter（双缓冲）

使用两个 UB Tile，将下一块的 TLOAD（MTE2）与当前块的 TSTORE（MTE3）重叠执行。

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int ROWS, int COLS, int TILE_ROWS, int TILE_COLS, int NRANKS>
void scatter_pingpong(__gm__ T* local_data, __gm__ T* group_addrs[NRANKS], int my_rank) {
    using TileT    = Tile<TileType::Vec, T, TILE_ROWS, TILE_COLS, BLayout::RowMajor, -1, -1>;
    using GPerRank = GlobalTensor<T, Shape<1,1,1,ROWS,COLS>,
                                  BaseShape2D<T, ROWS, COLS, Layout::ND>, Layout::ND>;
    using GSource  = GlobalTensor<T, Shape<1,1,1,NRANKS*ROWS,COLS>,
                                  BaseShape2D<T, NRANKS*ROWS, COLS, Layout::ND>, Layout::ND>;

    GPerRank tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) tensors[i] = GPerRank(group_addrs[i]);

    comm::ParallelGroup<GPerRank> group(tensors, NRANKS, my_rank);
    GSource srcG(local_data);
    TileT pingTile(TILE_ROWS, TILE_COLS);
    TileT pongTile(TILE_ROWS, TILE_COLS);
    // 乒乓模式：将 TLOAD 与 TSTORE 重叠执行以提升吞吐量
    comm::TSCATTER(group, srcG, pingTile, pongTile);
}
```

