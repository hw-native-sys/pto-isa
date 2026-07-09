# TREDUCE

## 简介

Reduce 操作：从多个远端 NPU 收集数据并在本地执行逐元素归约。

只有根节点需要执行 `TREDUCE`。非根节点只需确保在操作期间其源缓冲区已就绪且保持有效。在非根节点上调用 `TREDUCE` 属于未定义行为。

**大 Tile 支持**：当 GlobalTensor 在行和/或列方向超过 UB Tile 容量时，归约会自动通过二维滑动进行分块传输。

## 数学语义

对有效区域内每个元素 `(i, j)`：

$$\mathrm{dst}^{\mathrm{local}}_{i,j} = \bigoplus_{r=0}^{N-1} \mathrm{src}^{(r)}_{i,j}$$

其中 $N$ 为 rank 总数，$\oplus$ 为归约运算（求和、取最大值、取最小值等）。

## 模板参数

- `engine`：
    - `CollEngine::AIV` (默认)
    - `CollEngine::CCU` (Ascend950，仅 NPU_ARCH 3510)

## C++ 内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`：

```cpp
// 基础 reduce (累加 Tile + 接收 Tile)
template <CollEngine engine = CollEngine::AIV,
          typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... Args>
PTO_INST RecordEvent TREDUCE(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                              TileData &accTileData, TileData &recvTileData, ReduceOp op, Args&... args);

// Ping-pong reduce (累加 Tile + ping/pong Tile 实现双缓冲)
template <CollEngine engine = CollEngine::AIV,
          typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... Args>
PTO_INST RecordEvent TREDUCE(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                              TileData &accTileData, TileData &pingTileData, TileData &pongTileData,
                              ReduceOp op, Args&... args);
```

当 `engine == CollEngine::CCU` 时，可变参数的第一个参数必须是包含 CKE slot 虚拟地址和 gate mask 的 `CcuTriggerContext`。AIV kernel 触发 CKE gate，实际的 reduce 数据路径在 CCU 引擎上执行。

## 约束

- **类型约束**：
    - `ParallelGroup::value_type::RawDType` 必须等于 `GlobalDstData::RawDType`。
    - `TileData::DType` 必须等于 `GlobalDstData::RawDType`。
- **内存约束**：
    - `dstGlobalData` 必须指向本地内存（当前 NPU）。
    - `accTileData`、`recvTileData` (或 `accTileData`、`pingTileData`、`pongTileData`) 必须为预先分配的 UB Tile。
- **ParallelGroup 约束**：
    - `parallelGroup.tensors[r]` 必须指向 rank `r` 的源缓冲区（从根节点视角看到的远端 GM）。
    - `parallelGroup.GetRootIdx()` 标识调用方 NPU 为 reduce 根节点。
    - 所有源 tensor 假定具有相同的形状和步幅；否则行为未定义。
- **分块模式约束**（数据超出单个 UB Tile 时）：
    - 若 `TileData` 具有静态 `ValidRow`，则 `GetShape(DIM_3)` 必须能被 `ValidRow` 整除。如需支持不足一行的情况，请使用 `DYNAMIC` ValidRow 的 Tile。
    - 若 `TileData` 具有静态 `ValidCol`，则 `GetShape(DIM_4)` 必须能被 `ValidCol` 整除。如需支持不足一列的情况，请使用 `DYNAMIC` ValidCol 的 Tile。

> **CCU 路径**：与 AIV 路径（仅根节点调用 `TREDUCE`）不同，CCU 路径要求所有 rank 通过 Host 侧 `HcclCcuKernelRegister` / `HcclCcuKernelLaunch` 注册并启动 CCU kernel。完整示例参见 `tests/npu/a5/comm/st/testcase/treduce_ccu/`。

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

    // 栈上分配的 tensor 数组
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

### Ping-pong 归约（双缓冲）

在累加 Tile 之外再使用 ping/pong 两个接收 Tile，将"取下一个 rank 数据"的 TLOAD 与"归约当前 rank 数据"的向量运算重叠执行。

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SIZE, int NRANKS>
void reduce_sum_pingpong(__gm__ T* group_addrs[NRANKS], __gm__ T* result, int my_rank) {
    using TileT   = Tile<TileType::Vec, T, 1, SIZE>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SIZE>,
                                 BaseShape2D<T, 1, SIZE, Layout::ND>, Layout::ND>;

    GTensor tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) tensors[i] = GTensor(group_addrs[i]);

    comm::ParallelGroup<GTensor> group(tensors, NRANKS, my_rank);
    GTensor dstG(result);
    TileT accTile, pingTile, pongTile;

    comm::TREDUCE(group, dstG, accTile, pingTile, pongTile, comm::ReduceOp::Sum);
}
```

