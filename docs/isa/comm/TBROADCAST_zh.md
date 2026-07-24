# TBROADCAST

## 简介

将当前NPU的数据广播到并行组中所有rank。调用方NPU为根节点，其数据将被复制到所有其他NPU。

只有根节点需要执行 `TBROADCAST`。非根节点只需确保在操作期间其目标缓冲区已分配且可写。在非根节点上调用 `TBROADCAST` 属于未定义行为。

**大Tile支持**：当GlobalTensor在行和/或列方向超出UB（统一缓冲区）Tile容量时，传输将通过二维滑动自动分块。

## 数学语义

操作完成后：

$$ \mathrm{dst}^{(k)}_{i,j} = \mathrm{src}^{(\text{root})}_{i,j} \quad \forall k \in [0, N) $$

其中 $N$ 为rank总数，`root` 为调用方NPU。

## 汇编语法

同步形式：

```text
tbroadcast %group, %src : (!pto.group<...>, !pto.memref<...>)
```

降级时会为GM→UB→GM数据路径引入UB暂存Tile；C++内建接口需要显式传入 `stagingTileData`（或 `pingTile` / `pongTile`）操作数。

## 模板参数

- `engine`：
    - `CollEngine::AIV`（默认）
    - `CollEngine::CCU`（Ascend 950PR/Ascend 950DT，仅NPU_ARCH 3510）

## C++内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`：

```cpp
// 基础广播（单暂存 Tile）
template <CollEngine engine = CollEngine::AIV,
          typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... Args>
PTO_INST RecordEvent TBROADCAST(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                                TileData &stagingTileData, Args&... args);

// 乒乓广播（使用两个暂存 Tile 实现双缓冲）
template <CollEngine engine = CollEngine::AIV,
          typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... Args>
PTO_INST RecordEvent TBROADCAST(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                                TileData &pingTile, TileData &pongTile, Args&... args);
```

当 `engine == CollEngine::CCU` 时，可变参数的第一个参数必须是包含CKE slot虚拟地址和gate mask的 `CcuTriggerContext`。AIV kernel触发CKE gate，实际的广播数据路径在CCU引擎上执行。

## 约束

- **类型约束**：
    - `ParallelGroup::value_type::RawDType` 必须等于 `GlobalSrcData::RawDType`。
    - `TileData::DType` 必须等于 `GlobalSrcData::RawDType`。
- **内存约束**：
    - `srcGlobalData` 必须指向本地内存（当前NPU）。
    - `stagingTileData`（或 `pingTile` / `pongTile`）必须预先在UB中分配。
- **ParallelGroup约束**：
    - `parallelGroup.tensors[k]` 必须指向rank `k` 的目标缓冲区（从根节点视角看到的远端GM）。
    - `parallelGroup.GetRootIdx()` 标识调用方NPU为广播根节点。
    - 所有目标tensor假定具有相同的形状和步幅。
- **分块模式约束**（数据超出单个UB Tile时）：
    - 若 `TileData` 具有静态 `ValidRow`，则 `GetShape(DIM_3)` 必须能被 `ValidRow` 整除。如需支持不足一行的情况，请使用 `DYNAMIC` ValidRow的Tile。
    - 若 `TileData` 具有静态 `ValidCol`，则 `GetShape(DIM_4)` 必须能被 `ValidCol` 整除。如需支持不足一列的情况，请使用 `DYNAMIC` ValidCol的Tile。

> **CCU路径**：与AIV路径（仅根节点调用 `TBROADCAST`）不同，CCU路径要求所有rank通过宿主侧 `HcclCcuKernelRegister` / `HcclCcuKernelLaunch` 注册并启动CCU kernel。完整示例参见 `tests/npu/a5/comm/st/testcase/tbroadcast_ccu/`。

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

使用两个UB Tile，将下一块的TLOAD与当前块的TSTORE重叠执行。

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
