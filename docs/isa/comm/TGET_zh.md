# pto.tget

## 简介

`TGET` 是远程读原语：把远端 NPU 上的 GM 数据读到当前 NPU 的本地 GM。`pto.tget` 是 IR 形式，`TGET` 是 C++ intrinsic 形式，两者描述的是同一条通信指令。

数据路径为：

```text
远端 GM -> 暂存 Tile（UB） -> 本地 GM
```

当 `GlobalTensor` 的行或列超出单个 UB Tile 的容量时，`TGET` 会自动沿 `DIM_3` 和 `DIM_4` 做二维滑动分块，不需要手工把传输拆成小块。

## 数学语义

对有效区域中的每个元素 `(i, j)`：

$$ \mathrm{dst}^{\mathrm{local}}_{i,j} = \mathrm{src}^{\mathrm{remote}}_{i,j} $$

## 汇编语法

PTO-AS 形式：

```text
pto.tget %dst_local, %src_remote : (!pto.memref<...>, !pto.memref<...>)
```

lowering 会引入 UB 暂存 Tile 来承接 GM→UB→GM 的路径，因此 C++ 接口要求显式传入 `stagingTileData`，或在双缓冲场景下传入 `pingTile` / `pongTile`。

## C++ 内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`：

### 单暂存 Tile

```cpp
template <typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGET(GlobalDstData &dstGlobalData,
                          GlobalSrcData &srcGlobalData,
                          TileData &stagingTileData,
                          WaitEvents&... events);
```

### 乒乓双缓冲

```cpp
template <typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGET(GlobalDstData &dstGlobalData,
                          GlobalSrcData &srcGlobalData,
                          TileData &pingTile,
                          TileData &pongTile,
                          WaitEvents&... events);
```

## 约束

!!! warning "约束"
    ### 类型约束

    - `GlobalSrcData::RawDType` 必须等于 `GlobalDstData::RawDType`
    - `TileData::DType` 必须等于 `GlobalSrcData::RawDType`
    - `GlobalSrcData::layout` 必须等于 `GlobalDstData::layout`

    ### 内存约束

    - `srcGlobalData` 必须指向远端地址（源 NPU）
    - `dstGlobalData` 必须指向本地地址（当前 NPU）
    - `stagingTileData`、`pingTile`、`pongTile` 必须预先在 UB 中分配

    ### 乒乓约束

    - `pingTile` 与 `pongTile` 的类型和维度必须一致
    - 两者必须位于不重叠的 UB 偏移

## 示例

### 基础形式

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void remote_read(__gm__ T* local_data, __gm__ T* remote_addr) {
    using TileT   = Tile<TileType::Vec, T, 16, 16>;
    using GShape  = Shape<1, 1, 1, 16, 16>;
    using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
    using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

    GTensor srcG(remote_addr);
    GTensor dstG(local_data);
    TileT stagingTile;
    TASSIGN(stagingTile, 0);

    comm::TGET(dstG, srcG, stagingTile);
}
```

### 大张量自动分块

```cpp
using GShape  = Shape<1, 1, 1, 4096, 4096>;
using GStride = BaseShape2D<T, 4096, 4096, Layout::ND>;
using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

GTensor srcG(remote_addr);
GTensor dstG(local_data);
TileT stagingTile(64, 64);
TASSIGN(stagingTile, 0);

comm::TGET(dstG, srcG, stagingTile);
```

### 乒乓双缓冲

```cpp
constexpr size_t tileUBBytes = ((64 * 64 * sizeof(float) + 1023) / 1024) * 1024;
TileT pingTile(64, 64);
TileT pongTile(64, 64);
TASSIGN(pingTile, 0);
TASSIGN(pongTile, tileUBBytes);

comm::TGET(dstG, srcG, pingTile, pongTile);
```

## 相关页面

- [通信与运行时](communication-runtime_zh.md)
- [TPUT](./TPUT_zh.md)
- [TBROADCAST](./TBROADCAST_zh.md)
- [TGATHER](./TGATHER_zh.md)
