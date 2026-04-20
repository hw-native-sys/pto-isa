# TBROADCAST

`TBROADCAST` 是[通信与运行时](../other/communication-and-runtime_zh.md)指令集的一部分。

## 概述

`TBROADCAST` 把当前 NPU 作为根节点的数据广播到并行组中的所有 rank。调用方 NPU 作为根节点，其数据会被复制到组内所有其他 NPU。

只有根节点执行广播。非根节点只需保证目标缓冲区在操作期间已分配且可写。在非根节点上执行 `TBROADCAST` 属于未定义行为。

当 GlobalTensor 超过 UB Tile 容量时，传输会自动按二维滑动方式分块。

## 机制

`TBROADCAST` 将根节点 NPU 源缓冲区的数据复制到并行组内所有其他 NPU 的对应目标缓冲区。数据路径以 UB 作为暂存区：GM → UB → GM。

对于包含 $N$ 个 rank 的组中编号为 $k$ 的节点，广播完成后：

$$ \mathrm{dst}^{(k)}_{i,j} = \mathrm{src}^{(\text{root})}_{i,j} \quad \forall k \in [0, N) $$

其中 `root` 为执行广播的 NPU。

## 语法

### PTO 汇编形式

```text
|tbroadcast %group, %src : (!pto.group<...>, !pto.memref<...>)
```

汇编形式接收一个并行组和一个源内存引用。UB 暂存 Tile 在 lowering 阶段引入；C++ 内建接口显式暴露这些 Tile。

## C++ 内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`：

```cpp
// 基础广播 — 单个暂存 Tile
template <typename ParallelGroupType, typename GlobalSrcData,
          typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TBROADCAST(ParallelGroupType &parallelGroup,
                                GlobalSrcData &srcGlobalData,
                                TileData &stagingTileData,
                                WaitEvents&... events);

// 乒乓广播 — 两个暂存 Tile，用于双缓冲
template <typename ParallelGroupType, typename GlobalSrcData,
          typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TBROADCAST(ParallelGroupType &parallelGroup,
                                GlobalSrcData &srcGlobalData,
                                TileData &pingTile,
                                TileData &pongTile,
                                WaitEvents&... events);
```

## 输入

| | 操作数 | 类型 | 说明 |
|---|---------|------|-------------|
| | `parallelGroup` | `ParallelGroup` | 并行组描述符；`GetRootIdx()` 标识广播根节点 |
| | `srcGlobalData` | `GlobalTensor` | 根节点上的源数据；必须指向本地 GM |
| | `stagingTileData` | `Tile` | GM→UB→GM 传输路径上的 UB 暂存 Tile |
| | `pingTile` / `pongTile` | `Tile` | 双缓冲用的两个 UB 暂存 Tile |
| | `WaitEvents...` | `RecordEvent...` | 发指令前要等待的事件 |

## 预期输出

| | 结果 | 类型 | 说明 |
|---|--------|------|-------------|
| | `RecordEvent` | event | 标记广播完成的事件令牌，具体语义取决于异步变体 |

## 副作用

本指令对所有 rank 的全局内存做读和写操作。通过返回的事件令牌建立同步边。

## 约束

### 类型约束

- `ParallelGroup::value_type::RawDType` 必须等于 `GlobalSrcData::RawDType`
- `TileData::DType` 必须等于 `GlobalSrcData::RawDType`

### 内存约束

- `srcGlobalData` 必须指向根节点本地内存
- `stagingTileData`（或 `pingTile`/`pongTile`）必须预先在 UB 中分配

### 并行组约束

- `parallelGroup.tensors[k]` 必须指向 rank `k` 的目标缓冲区
- `parallelGroup.GetRootIdx()` 必须标识当前调用方是广播根节点
- 所有目标 Tensor 必须具有相同的形状和步幅

### 分块约束

当 GlobalTensor 在行方向或列方向超过单个 UB Tile 时：

- 若 `TileData` 具有静态 `ValidRow`，`GetShape(DIM_3)` 必须能被 `ValidRow` 整除。若需支持不足整行的边界情况，应使用动态 `ValidRow` 的 Tile。
- 若 `TileData` 具有静态 `ValidCol`，`GetShape(DIM_4)` 必须能被 `ValidCol` 整除。若需支持不足整列的边界情况，应使用动态 `ValidCol` 的 Tile。

## 目标Profile限制

- 集合通信在 A2/A3 和 A5 上支持。CPU 模拟器不支持集合通信。
- 大数据量传输建议使用乒乓双缓冲形式，以重叠下一次加载与当前块存储，提高吞吐率。
- `TBROADCAST` 需要正确初始化的 `ParallelGroup`，覆盖所有参与的 NPU。

## 示例

### 基础广播

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int ROWS, int COLS, int TILE_ROWS, int TILE_COLS, int NRANKS>
void broadcast(__gm__ T* group_addrs[NRANKS], __gm__ T* my_data, int my_rank) {
    using TileT = Tile<TileType::Vec, T, TILE_ROWS, TILE_COLS, BLayout::RowMajor, -1, -1>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,ROWS,COLS>,
                                     BaseShape2D<T, ROWS, COLS, Layout::ND>, Layout::ND>;

    GTensor tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i)
        tensors[i] = GTensor(group_addrs[i]);

    comm::ParallelGroup<GTensor> group(tensors, NRANKS, my_rank);
    GTensor srcG(my_data);
    TileT stagingTile(TILE_ROWS, TILE_COLS);

    // Root NPU broadcasts its data to all others
    comm::TBROADCAST(group, srcG, stagingTile);
}
```

### 乒乓双缓冲

使用两个 UB 暂存 Tile 来重叠下一块的加载与当前块的存储：

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int ROWS, int COLS, int TILE_ROWS, int TILE_COLS, int NRANKS>
void broadcast_pingpong(__gm__ T* group_addrs[NRANKS], __gm__ T* my_data, int my_rank) {
    using TileT = Tile<TileType::Vec, T, TILE_ROWS, TILE_COLS, BLayout::RowMajor, -1, -1>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,ROWS,COLS>,
                                     BaseShape2D<T, ROWS, COLS, Layout::ND>, Layout::ND>;

    GTensor tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i)
        tensors[i] = GTensor(group_addrs[i]);

    comm::ParallelGroup<GTensor> group(tensors, NRANKS, my_rank);
    GTensor srcG(my_data);
    TileT pingTile(TILE_ROWS, TILE_COLS);
    TileT pongTile(TILE_ROWS, TILE_COLS);

    // Overlaps TLOAD and TSTORE for better throughput
    comm::TBROADCAST(group, srcG, pingTile, pongTile);
}
```

## 相关页面

- 通信概述：[通信与运行时](../other/communication-and-runtime_zh.md)
- 集合通信：[TGATHER](./TGATHER_zh.md)、[TSCATTLER](./TSCATTER_zh.md)、[TREDUCE](./TREDUCE_zh.md)
- 点对点通信：[TGET](./TGET_zh.md)、[TPUT](./TPUT_zh.md)
- 指令集：[其他与通信](../other/README_zh.md)
- 机器模型：[排序与同步](../machine-model/ordering-and-synchronization_zh.md)
