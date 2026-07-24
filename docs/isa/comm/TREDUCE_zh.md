# TREDUCE

## 简介

Reduce操作：从多个远端NPU收集数据并在本地执行逐元素归约。

只有根节点需要执行 `TREDUCE`。非根节点只需确保在操作期间其源缓冲区已就绪且保持有效。在非根节点上调用 `TREDUCE` 属于未定义行为。

**大Tile支持**：当GlobalTensor在行和/或列方向超出UB Tile容量时，归约操作将通过二维滑动自动分块。

## 数学语义

对有效区域内每个元素 `(i, j)`：

$$\mathrm{dst}^{\mathrm{local}}_{i,j} = \bigoplus_{r=0}^{N-1} \mathrm{src}^{(r)}_{i,j}$$

其中 $N$ 为rank总数，$\oplus$ 为归约运算（求和、取最大值、取最小值等）。

## 汇编语法

同步形式：

```text
treduce %group, %dst {op = #pto.reduce_op<Sum>} : (!pto.group<...>, !pto.memref<...>)
treduce %group, %dst {op = #pto.reduce_op<Max>} : (!pto.group<...>, !pto.memref<...>)
```

降级时会为reduce流水线引入内部累加Tile和接收Tile；C++内建接口需要显式传入 `accTileData`、`recvTileData`（或 `accTileData`、`pingTileData`、`pongTileData`）操作数。

## 模板参数

- `engine`：
    - `CollEngine::AIV`（默认）
    - `CollEngine::CCU`（Ascend 950PR/Ascend 950DT，仅NPU_ARCH 3510）

## C++内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`：

```cpp
// 基础 reduce（累加 Tile + 接收 Tile）
template <CollEngine engine = CollEngine::AIV,
          typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... Args>
PTO_INST RecordEvent TREDUCE(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                              TileData &accTileData, TileData &recvTileData, ReduceOp op, Args&... args);

// 乒乓 reduce（累加 Tile + ping/pong Tile 实现双缓冲）
template <CollEngine engine = CollEngine::AIV,
          typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... Args>
PTO_INST RecordEvent TREDUCE(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                              TileData &accTileData, TileData &pingTileData, TileData &pongTileData,
                              ReduceOp op, Args&... args);
```

当 `engine == CollEngine::CCU` 时，可变参数的第一个参数必须是包含CKE slot虚拟地址和gate mask的 `CcuTriggerContext`。AIV kernel触发CKE gate，实际的reduce数据路径在CCU引擎上执行。

## 约束

- **类型约束**：
    - `ParallelGroup::value_type::RawDType` 必须等于 `GlobalDstData::RawDType`。
    - `TileData::DType` 必须等于 `GlobalDstData::RawDType`。
- **内存约束**：
    - `dstGlobalData` 必须指向本地内存（当前NPU）。
    - `accTileData`、`recvTileData`（或 `accTileData`、`pingTileData`、`pongTileData`）必须为预先分配的UB Tile。
- **ParallelGroup约束**：
    - `parallelGroup.tensors[r]` 必须指向rank `r` 的源缓冲区（从根节点视角看到的远端GM）。
    - `parallelGroup.GetRootIdx()` 标识调用方NPU为reduce根节点。
    - 所有源tensor假定具有相同的形状和步幅。
- **分块模式约束**（数据超出单个UB Tile时）：
    - 若 `TileData` 具有静态 `ValidRow`，则 `GetShape(DIM_3)` 必须能被 `ValidRow` 整除。如需支持不足一行的情况，请使用 `DYNAMIC` ValidRow的Tile。
    - 若 `TileData` 具有静态 `ValidCol`，则 `GetShape(DIM_4)` 必须能被 `ValidCol` 整除。如需支持不足一列的情况，请使用 `DYNAMIC` ValidCol的Tile。

> **CCU路径**：与AIV路径（仅根节点调用 `TREDUCE`）不同，CCU路径要求所有rank通过宿主侧 `HcclCcuKernelRegister` / `HcclCcuKernelLaunch` 注册并启动CCU kernel。完整示例参见 `tests/npu/a5/comm/st/testcase/treduce_ccu/`。

## 示例

### 基础求和归约

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SIZE, int NRANKS>
void reduce_sum(__gm__ T* group_addrs[NRANKS], __gm__ T* result, int my_rank) {
    using TileT   = Tile<TileType::Vec, T, 1, SIZE>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SIZE>,
                                 BaseShape2D<T, 1, SIZE, Layout::ND>, Layout::ND>;

    GTensor tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) tensors[i] = GTensor(group_addrs[i]);

    comm::ParallelGroup<GTensor> group(tensors, NRANKS, my_rank);
    GTensor dstG(result);
    TileT accTile, recvTile;
    comm::TREDUCE(group, dstG, accTile, recvTile, comm::ReduceOp::Sum);
}
```

### 最大值归约

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SIZE, int NRANKS>
void reduce_max(__gm__ T* group_addrs[NRANKS], __gm__ T* result, int my_rank) {
    using TileT   = Tile<TileType::Vec, T, 1, SIZE>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SIZE>,
                                 BaseShape2D<T, 1, SIZE, Layout::ND>, Layout::ND>;

    GTensor tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) tensors[i] = GTensor(group_addrs[i]);

    comm::ParallelGroup<GTensor> group(tensors, NRANKS, my_rank);
    GTensor dstG(result);
    TileT accTile, recvTile;
    comm::TREDUCE(group, dstG, accTile, recvTile, comm::ReduceOp::Max);
}
```
