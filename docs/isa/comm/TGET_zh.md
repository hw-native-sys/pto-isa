# TGET

## 产品支持情况<a name="section73648168211"></a>

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：支持
<!-- end id3 -->


## 简介

远程读操作：将远端NPU的数据读取到本地内存。数据通过UB Tile作为中间暂存缓冲区进行传输。

当GlobalTensor超出UB Tile容量时，TGET将自动执行**二维滑动**——沿行（DIM_3）和列（DIM_4）分块以适配Tile，并遍历所有外层维度（DIM_0、DIM_1、DIM_2）。

## 数学语义

对有效区域内每个元素 `(i, j)`：

$$\mathrm{dst}^{\mathrm{local}}_{i,j} = \mathrm{src}^{\mathrm{remote}}_{i,j}$$

数据流：`srcGlobalData（远端 GM）` → `stagingTileData（UB）` → `dstGlobalData（本地 GM）`

## C++内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`

### 单Tile（自动分块）

```cpp
template <typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGET(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                          TileData &stagingTileData, WaitEvents&... events);
```

### 乒乓双缓冲

使用两个暂存Tile，将相邻块的TLOAD与TSTORE重叠执行，隐藏DMA传输延迟。

```cpp
template <typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGET(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                          TileData &pingTile, TileData &pongTile, WaitEvents&... events);
```

## 约束

- **类型约束**：
    - `GlobalSrcData::RawDType` 必须等于 `GlobalDstData::RawDType`。
    - `TileData::DType` 必须等于 `GlobalSrcData::RawDType`。
    - `GlobalSrcData::layout` 必须等于 `GlobalDstData::layout`。
- **内存约束**：
    - `srcGlobalData` 必须指向远端地址（源NPU）。
    - `dstGlobalData` 必须指向本地地址（当前NPU）。
    - `stagingTileData` / `pingTile` / `pongTile` 必须预先在统一缓冲区中分配。
- **有效区域**：
    - 传输大小由 `GlobalTensor` 的形状决定（自动分块以适配Tile）。
- **乒乓约束**：
    - `pingTile` 和 `pongTile` 必须具有相同的类型和维度。
    - 必须位于不重叠的UB偏移处。

## 示例

### 基础用法

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_tget(__gm__ T* local_data, __gm__ T* remote_addr) {
    using TileT   = Tile<TileType::Vec, T, 16, 16>;
    using GShape  = Shape<1, 1, 1, 16, 16>;
    using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
    using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

    GTensor srcG(remote_addr);
    GTensor dstG(local_data);
    TileT stagingTile;
    TASSIGN(stagingTile, 0);

    // 基础远程读
    comm::TGET(dstG, srcG, stagingTile);
}
```

### 乒乓双缓冲

```cpp
// (rows, cols) 构造函数要求 ValidRow/ValidCol 为 DYNAMIC
using PingPongTileT = Tile<TileType::Vec, T, 64, 64, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
constexpr size_t tileUBBytes = ((64 * 64 * sizeof(T) + 1023) / 1024) * 1024;
PingPongTileT pingTile(64, 64);
PingPongTileT pongTile(64, 64);
TASSIGN(pingTile, 0);
TASSIGN(pongTile, tileUBBytes);  // 不重叠的 UB 区域

// 将 TLOAD[i+1] 与 TSTORE[i] 重叠执行以提升流水线利用率
comm::TGET(dstG, srcG, pingTile, pongTile);
```
