<!-- Generated from `docs/isa/comm/TBROADCAST_zh.md` -->

# TBROADCAST

## 简介

将当前 NPU 的数据广播到并行组中所有 rank。调用方 NPU 为根节点，其数据将被复制到所有其他 NPU。

只有根节点需要执行 `TBROADCAST`。非根节点只需确保在操作期间其目标缓冲区已分配且可写。在非根节点上调用 `TBROADCAST` 属于未定义行为。

**大 Tile 支持**：当 GlobalTensor 在行和/或列方向超出 UB（统一缓冲区）Tile 容量时，传输将通过二维滑动自动分块。

## 数学语义

操作完成后：

$$ \mathrm{dst}^{(k)}_{i,j} = \mathrm{src}^{(\text{root})}_{i,j} \quad \forall k \in [0, N) $$

其中 $N$ 为 rank 总数，`root` 为调用方 NPU。

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../../assembly/PTO-AS_zh.md)。

同步形式：

```text
tbroadcast %group, %src : (!pto.group<...>, !pto.memref<...>)
```

降级时会为 GM→UB→GM 数据路径引入 UB 暂存 Tile；C++ 内建接口需要显式传入 `stagingTileData`（或 `pingTile` / `pongTile`）操作数。

## C++ 内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`：

```cpp
// 基础广播（单暂存 Tile）
template <typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TBROADCAST(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                                TileData &stagingTileData, WaitEvents&... events);

// 乒乓广播（使用两个暂存 Tile 实现双缓冲）
template <typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TBROADCAST(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                                TileData &pingTile, TileData &pongTile, WaitEvents&... events);
```

## 约束

- **类型约束**：
    - `ParallelGroup::value_type::RawDType` 必须等于 `GlobalSrcData::RawDType`。
    - `TileData::DType` 必须等于 `GlobalSrcData::RawDType`。
- **内存约束**：
    - `srcGlobalData` 必须指向本地内存（当前 NPU）。
    - `stagingTileData`（或 `pingTile` / `pongTile`）必须预先在 UB 中分配。
- **ParallelGroup 约束**：
    - `parallelGroup.tensors[k]` 必须指向 rank `k` 的目标缓冲区（从根节点视角看到的远端 GM）。
    - `parallelGroup.GetRootIdx()` 标识调用方 NPU 为广播根节点。
    - 所有目标 tensor 假定具有相同的形状和步幅。
- **分块模式约束**（数据超出单个 UB Tile 时）：
    - 若 `TileData` 具有静态 `ValidRow`，则 `GetShape(DIM_3)` 必须能被 `ValidRow` 整除。如需支持不足一行的情况，请使用 `DYNAMIC` ValidRow 的 Tile。
    - 若 `TileData` 具有静态 `ValidCol`，则 `GetShape(DIM_4)` 必须能被 `ValidCol` 整除。如需支持不足一列的情况，请使用 `DYNAMIC` ValidCol 的 Tile。

## 示例

### 基础广播

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int ROWS, int COLS, int TILE_ROWS, int TILE_COLS, int NRANKS>
void broadcast(__gm__ T* group_addrs[NRANKS], __gm__ T* my_data, int my_rank) {
    // Tile 维度可以与 tensor 维度不同。
    // 二维滑动分块路径会自动在行和列两个方向进行分块。
    using TileT = Tile<TileType::Vec, T, TILE_ROWS, TILE_COLS, BLayout::RowMajor, -1, -1>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,ROWS,COLS>,
                                 BaseShape2D<T, ROWS, COLS, Layout::ND>, Layout::ND>;

    GTensor tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) {
        tensors[i] = GTensor(group_addrs[i]);
    }

    comm::ParallelGroup<GTensor> group(tensors, NRANKS, my_rank);
    GTensor srcG(my_data);
    TileT stagingTile(TILE_ROWS, TILE_COLS);

    // 当前 NPU 将自身数据广播到所有其他 NPU
    comm::TBROADCAST(group, srcG, stagingTile);
}
```

### 乒乓广播（双缓冲）

使用两个 UB Tile，将下一块的 TLOAD 与当前块的 TSTORE 重叠执行。

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int ROWS, int COLS, int TILE_ROWS, int TILE_COLS, int NRANKS>
void broadcast_pingpong(__gm__ T* group_addrs[NRANKS], __gm__ T* my_data, int my_rank) {

    using TileT = Tile<TileType::Vec, T, TILE_ROWS, TILE_COLS, BLayout::RowMajor, -1, -1>;
    using GPerRank = GlobalTensor<T, Shape<1,1,1,ROWS,COLS>,
                                  BaseShape2D<T, ROWS, COLS, Layout::ND>, Layout::ND>;

    GPerRank tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) {
        tensors[i] = GPerRank(group_addrs[i]);
    }

    comm::ParallelGroup<GPerRank> group(tensors, NRANKS, my_rank);
    GPerRank srcG(my_data);
    TileT pingTile(TILE_ROWS, TILE_COLS);
    TileT pongTile(TILE_ROWS, TILE_COLS);

    // 乒乓模式：将 TLOAD 与 TSTORE 重叠执行以提升吞吐量
    comm::TBROADCAST(group, srcG, pingTile, pongTile);
}
```
