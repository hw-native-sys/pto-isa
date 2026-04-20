# TREDUCE

`TREDUCE` 是[通信与运行时](../other/communication-and-runtime_zh.md)指令集的一部分。

## 概述

归约操作：从并行组中所有 rank 收集数据，并在根节点 NPU 上执行逐元素归约。只有根节点执行 `TREDUCE`；非根节点只需确保源缓冲区已就绪。调用 `TREDUCE` 的非根节点属于未定义行为。

当 GlobalTensor 在行或列方向超出 UB Tile 容量时，归约会自动通过二维滑动分块。

## 机制

`TREDUCE` 从并行组内所有 rank 收集源数据，并在根节点 NPU 的目标缓冲区中完成归约。对有效区域中每个元素 `(i, j)`：

$$ \mathrm{dst}^{\mathrm{local}}_{i,j} = \bigoplus_{r=0}^{N-1} \mathrm{src}^{(r)}_{i,j} $$

其中 $N$ 为 rank 总数，$\oplus$ 为归约运算符。

### 支持的归约运算符

|| 运算符 | 说明 |
||--------|------|
|| `Sum` | 加法归约 |
|| `Max` | 最大值 |
|| `Min` | 最小值 |

## 语法

### PTO 汇编形式

```text
treduce %group, %dst {op = #pto.reduce_op<Sum>} : (!pto.group<...>, !pto.memref<...>)
treduce %group, %dst {op = #pto.reduce_op<Max>} : (!pto.group<...>, !pto.memref<...>)
```

Lowering 会引入累加器和接收 Tile。C++ 内建接口显式暴露这些 Tile。

## C++ 内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`：

```cpp
// 基础归约 — 累加 Tile + 接收 Tile
template <typename ParallelGroupType, typename GlobalDstData,
          typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TREDUCE(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                            TileData &accTileData, TileData &recvTileData,
                            ReduceOp op, WaitEvents&... events);

// 乒乓归约 — 累加 Tile + ping/pong Tile 实现双缓冲
template <typename ParallelGroupType, typename GlobalDstData,
          typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TREDUCE(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                            TileData &accTileData, TileData &pingTileData, TileData &pongTileData,
                            ReduceOp op, WaitEvents&... events);
```

## 输入

|| 操作数 | 类型 | 说明 |
||--------|------|------|
|| `parallelGroup` | `ParallelGroup` | 并行组描述符；`GetRootIdx()` 标识归约根节点 |
|| `dstGlobalData` | `GlobalTensor` | 根节点上的本地目标缓冲区 |
|| `accTileData` | `Tile` | 用于部分归约的 UB 累加 Tile |
|| `recvTileData` | `Tile` | 用于接收远端数据的 UB 接收 Tile |
|| `pingTileData` / `pongTileData` | `Tile` | 乒乓双缓冲用的两个 UB Tile |
|| `op` | `ReduceOp` | 归约运算符（`Sum`、`Max`、`Min` 等） |
|| `WaitEvents...` | `RecordEvent...` | 发指令前要等待的事件 |

## 预期输出

|| 结果 | 类型 | 说明 |
||------|------|------|
|| `RecordEvent` | event | 标记归约完成的事件令牌 |

## 副作用

本指令从所有 rank 的全局内存读取数据并写入根节点的全局内存。通过返回的事件令牌建立同步边界。

## 约束

### 类型约束

- `ParallelGroup::value_type::RawDType` 必须等于 `GlobalDstData::RawDType`
- `TileData::DType` 必须等于 `GlobalDstData::RawDType`

### 内存约束

- `dstGlobalData` 必须指向本地地址（根节点 NPU）
- `accTileData` 和 `recvTileData`（或 `accTileData`、`pingTileData`、`pongTileData`）必须预先在 UB 中分配

### 并行组约束

- `parallelGroup.tensors[r]` 必须指向 rank `r` 的源缓冲区（从根节点视角看到的远端 GM）
- `parallelGroup.GetRootIdx()` 标识调用方 NPU 为归约根节点
- 所有源 Tensor 必须具有相同的形状和步幅

### 分块约束

当 GlobalTensor 在行方向或列方向超过单个 UB Tile 时：

- 若 `TileData` 具有静态 `ValidRow`，`GetShape(DIM_3)` 必须能被 `ValidRow` 整除。如需支持不足整行，应使用动态 `ValidRow` 的 Tile
- 若 `TileData` 具有静态 `ValidCol`，`GetShape(DIM_4)` 必须能被 `ValidCol` 整除。如需支持不足整列，应使用动态 `ValidCol` 的 Tile

## 目标Profile限制

- 集合通信在 A2/A3 和 A5 上支持。CPU 模拟器不支持集合通信。
- 大数据量传输建议使用乒乓双缓冲，以重叠通信与计算。
- `TREDUCE` 需要正确初始化的 `ParallelGroup`，覆盖所有参与的 NPU。

## 示例

### 求和归约

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SIZE, int NRANKS>
void reduce_sum(__gm__ T* group_addrs[NRANKS], __gm__ T* result, int my_rank) {
    using TileT   = Tile<TileType::Vec, T, 1, SIZE>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SIZE>,
                                     BaseShape2D<T, 1, SIZE, Layout::ND>, Layout::ND>;

    GTensor tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i)
        tensors[i] = GTensor(group_addrs[i]);

    comm::ParallelGroup<GTensor> group(tensors, NRANKS, my_rank);
    GTensor dstG(result);
    TileT accTile, recvTile;

    comm::TREDUCE(group, dstG, accTile, recvTile, comm::ReduceOp::Sum);
}
```

### 最大值归约

```cpp
template <typename T, int SIZE, int NRANKS>
void reduce_max(__gm__ T* group_addrs[NRANKS], __gm__ T* result, int my_rank) {
    using TileT   = Tile<TileType::Vec, T, 1, SIZE>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SIZE>,
                                     BaseShape2D<T, 1, SIZE, Layout::ND>, Layout::ND>;

    GTensor tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i)
        tensors[i] = GTensor(group_addrs[i]);

    comm::ParallelGroup<GTensor> group(tensors, NRANKS, my_rank);
    GTensor dstG(result);
    TileT accTile, recvTile;

    comm::TREDUCE(group, dstG, accTile, recvTile, comm::ReduceOp::Max);
}
```

## 相关页面

- 通信概述：[通信与运行时](../other/communication-and-runtime_zh.md)
- 集合通信：[TBROADCAST](./TBROADCAST_zh.md)、[TGATHER](./TGATHER_zh.md)、[TSCATTLER](./TSCATTER_zh.md)
- 点对点通信：[TGET](./TGET_zh.md)、[TPUT](./TPUT_zh.md)
- 指令集：[其他与通信](../other/README_zh.md)
