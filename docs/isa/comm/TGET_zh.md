# TGET

## 简介

远程读操作：将远端 NPU 的数据读取到本地内存。数据通过 UB Tile 作为中间暂存缓冲区进行传输。

当 GlobalTensor 超出 UB Tile 容量时，TGET 将自动执行**二维滑动**——沿行（DIM_3）和列（DIM_4）分块以适配 Tile，并遍历所有外层维度（DIM_0、DIM_1、DIM_2）。

## 数学语义

对有效区域内每个元素 `(i, j)`：

$$\mathrm{dst}^{\mathrm{local}}_{i,j} = \mathrm{src}^{\mathrm{remote}}_{i,j}$$

数据流：`srcGlobalData（远端 GM）` → `stagingTileData（UB）` → `dstGlobalData（本地 GM）`

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../../assembly/PTO-AS_zh.md)。

同步形式：

```text
tget %dst_local, %src_remote : (!pto.memref<...>, !pto.memref<...>)
```

降级时会为 GM→UB→GM 数据路径引入 UB 暂存 Tile；C++ 内建接口需要显式传入 `stagingTileData`（或 `pingTile` / `pongTile`）操作数。

## C++ 内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`

### 单 Tile（自动分块）

```cpp
template <typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGET(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                          TileData &stagingTileData, WaitEvents&... events);
```

### 乒乓双缓冲

使用两个暂存 Tile，将相邻块的 TLOAD 与 TSTORE 重叠执行，隐藏 DMA 传输延迟。

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
    - `srcGlobalData` 必须指向远端地址（源 NPU）。
    - `dstGlobalData` 必须指向本地地址（当前 NPU）。
    - `stagingTileData` / `pingTile` / `pongTile` 必须预先在统一缓冲区中分配。
- **有效区域**：
    - 传输大小由 `GlobalTensor` 的形状决定（自动分块以适配 Tile）。
- **乒乓约束**：
    - `pingTile` 和 `pongTile` 必须具有相同的类型和维度。
    - 必须位于不重叠的 UB 偏移处。

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
constexpr size_t tileUBBytes = ((64 * 64 * sizeof(float) + 1023) / 1024) * 1024;
TileT pingTile(64, 64);
TileT pongTile(64, 64);
TASSIGN(pingTile, 0);
TASSIGN(pongTile, tileUBBytes);  // 不重叠的 UB 区域

// 将 TLOAD[i+1] 与 TSTORE[i] 重叠执行以提升流水线利用率
comm::TGET(dstG, srcG, pingTile, pongTile);
```

