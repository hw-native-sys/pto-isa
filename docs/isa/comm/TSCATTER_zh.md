# TSCATTER

`TSCATTLER` 是[通信与运行时](../other/communication-and-runtime_zh.md)指令集的一部分。

## 概述

集合 Scatter 操作：根节点 NPU 将本地源 tensor 沿 DIM_3（行维度）拆分后分发到并行组中所有 rank。该操作是 `TGATHER` 的逆操作。只有根节点执行 `TSCATTLER`；非根节点只需确保目标缓冲区已分配。调用 `TSCATTLER` 的非根节点属于未定义行为。

当每个 rank 的数据超出 UB Tile 容量时，传输会自动通过二维滑动分块。

## 机制

本地源 tensor 的形状为 $(D_0, D_1, D_2, N \times H, W)$，其中 $N$ 为 rank 总数，每个 rank 接收 $H$ 行。操作完成后：

$$\mathrm{dst}^{(r)}_{d_0, d_1, d_2,\; i,\; j} = \mathrm{src}^{\mathrm{local}}_{d_0, d_1, d_2,\; r \cdot H + i,\; j} \quad \forall\, r \in [0, N),\; i \in [0, H),\; j \in [0, W)$$

## 语法

### PTO 汇编形式

```text
tscatter %group, %src : (!pto.group<...>, !pto.memref<...>)
```

Lowering 会引入 UB 暂存 Tile。C++ 内建接口显式暴露这些 Tile。

## C++ 内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`：

```cpp
// 基础 scatter — 单个暂存 Tile
template <typename ParallelGroupType, typename GlobalSrcData,
          typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                           TileData &stagingTileData, WaitEvents&... events);

// 乒乓 scatter — 两个暂存 Tile 实现双缓冲
template <typename ParallelGroupType, typename GlobalSrcData,
          typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                           TileData &pingTile, TileData &pongTile, WaitEvents&... events);
```

## 输入

|| 操作数 | 类型 | 说明 |
||--------|------|------|
|| `parallelGroup` | `ParallelGroup` | 并行组描述符；`GetRootIdx()` 标识 scatter 根节点 |
|| `srcGlobalData` | `GlobalTensor` | 根节点上的本地源缓冲区；必须包含所有 rank 的数据 |
|| `stagingTileData` | `Tile` | GM→UB→GM 传输路径上的 UB 暂存 Tile |
|| `pingTile` / `pongTile` | `Tile` | 双缓冲用的两个 UB 暂存 Tile |
|| `WaitEvents...` | `RecordEvent...` | 发指令前要等待的事件 |

## 预期输出

|| 结果 | 类型 | 说明 |
||------|------|------|
|| `RecordEvent` | event | 标记 scatter 完成的事件令牌 |

## 副作用

本指令从根节点的全局内存读取数据并写入所有 rank 的全局内存。通过返回的事件令牌建立同步边界。

## 约束

### 类型约束

- `ParallelGroup::value_type::RawDType` 必须等于 `GlobalSrcData::RawDType`
- `TileData::DType` 必须等于 `GlobalSrcData::RawDType`

### 内存约束

- `srcGlobalData` 必须指向本地内存，且足够容纳所有 rank 的数据。具体要求：`srcGlobalData.GetShape(DIM_3)` 必须 $\geq N \times H$
- 若 `srcGlobalData.GetShape(DIM_3) > N \times H`，只读取前 $N \times H$ 行，其余行被忽略
- `stagingTileData` / `pingTile` / `pongTile` 必须预先在 UB 中分配

### 并行组约束

- `parallelGroup.tensors[r]` 必须指向 rank `r` 的目标缓冲区（从根节点视角看到的远端 GM）
- `parallelGroup.GetRootIdx()` 标识调用方 NPU 为 scatter 根节点
- 所有目标 Tensor 必须具有相同的形状和步幅

### 分块约束

当每个 rank 的数据超出单个 UB Tile 的行或列时：

- 若 `TileData` 具有静态 `ValidRow`，每个 rank 目标数据的 `GetShape(DIM_3)` 必须能被 `ValidRow` 整除。如需支持不足整行，应使用动态 `ValidRow` 的 Tile
- 若 `TileData` 具有静态 `ValidCol`，`GetShape(DIM_4)` 必须能被 `ValidCol` 整除。如需支持不足整列，应使用动态 `ValidCol` 的 Tile

## 目标Profile限制

- 集合通信在 A2/A3 和 A5 上支持。CPU 模拟器不支持集合通信。
- 大数据量传输建议使用乒乓双缓冲，以重叠通信与计算。
- `TSCATTLER` 需要正确初始化的 `ParallelGroup`，覆盖所有参与的 NPU。

## 示例

### 基础 scatter

根节点拥有 `NRANKS × ROWS` 行、宽度为 `COLS` 的数据，每个 rank 接收 `ROWS × COLS`：

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
    for (int i = 0; i < NRANKS; ++i)
        tensors[i] = GPerRank(group_addrs[i]);

    comm::ParallelGroup<GPerRank> group(tensors, NRANKS, my_rank);
    GSource srcG(local_data);
    TileT stagingTile(TILE_ROWS, TILE_COLS);

    comm::TSCATTER(group, srcG, stagingTile);
}
```

### 乒乓 scatter

```cpp
template <typename T, int ROWS, int COLS, int TILE_ROWS, int TILE_COLS, int NRANKS>
void scatter_pingpong(__gm__ T* local_data, __gm__ T* group_addrs[NRANKS], int my_rank) {
    using TileT    = Tile<TileType::Vec, T, TILE_ROWS, TILE_COLS, BLayout::RowMajor, -1, -1>;
    using GPerRank = GlobalTensor<T, Shape<1,1,1,ROWS,COLS>,
                                     BaseShape2D<T, ROWS, COLS, Layout::ND>, Layout::ND>;
    using GSource  = GlobalTensor<T, Shape<1,1,1,NRANKS*ROWS,COLS>,
                                     BaseShape2D<T, NRANKS*ROWS, COLS, Layout::ND>, Layout::ND>;

    GPerRank tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i)
        tensors[i] = GPerRank(group_addrs[i]);

    comm::ParallelGroup<GPerRank> group(tensors, NRANKS, my_rank);
    GSource srcG(local_data);
    TileT pingTile(TILE_ROWS, TILE_COLS);
    TileT pongTile(TILE_ROWS, TILE_COLS);

    comm::TSCATTER(group, srcG, pingTile, pongTile);
}
```

## 相关页面

- 通信概述：[通信与运行时](../other/communication-and-runtime_zh.md)
- 逆操作：[TGATHER](./TGATHER_zh.md)
- 集合通信：[TBROADCAST](./TBROADCAST_zh.md)、[TGATHER](./TGATHER_zh.md)、[TREDUCE](./TREDUCE_zh.md)
- 点对点通信：[TGET](./TGET_zh.md)、[TPUT](./TPUT_zh.md)
- 指令集：[其他与通信](../other/README_zh.md)
