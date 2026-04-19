# pto.tget / TGET

## 概述

`TGET` 是远程读原语：把远端 NPU 上的 GM 数据读到当前 NPU 的本地 GM。`pto.tget` 是 IR 形式，`TGET` 是 C++ intrinsic 形式，两者描述的是同一条通信指令。

数据路径为：

```text
远端 GM -> 暂存 Tile（UB） -> 本地 GM
```

当 `GlobalTensor` 的行或列超出单个 UB Tile 的容量时，`TGET` 会自动沿 `DIM_3` 和 `DIM_4` 做二维滑动分块，不需要手工把传输拆成小块。

只有本地 NPU 执行 TGET；远端 NPU 是被动的。

## 语法

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

## 输入

| 操作数 | 类型 | 描述 |
|--------|------|------|
| `dstGlobalData` | `GlobalTensor` | 本地目标，必须指向本地 GM |
| `srcGlobalData` | `GlobalTensor` | 远端源，必须指向远端 NPU 的 GM |
| `stagingTileData` | `Tile` | UB 暂存 Tile，用于 GM→UB→GM 传输路径 |
| `pingTile` / `pongTile` | `Tile` | 用于乒乓双缓冲的两个 UB 暂存 Tile |
| `WaitEvents...` | `RecordEvent...` | 在发起 GET 前需要等待的事件 |

## 预期输出

| 结果 | 类型 | 描述 |
|------|------|------|
| `RecordEvent` | event | 标记远程读取完成的事件令牌 |

## 副作用

此操作从远端 GM 读取数据并写入本地 GM。它通过返回的事件令牌建立同步边界。

## 约束

### 类型约束

- `GlobalSrcData::RawDType` 必须等于 `GlobalDstData::RawDType`
- `TileData::DType` 必须等于 `GlobalSrcData::RawDType`
- `GlobalSrcData::layout` 必须等于 `GlobalDstData::layout`

### 内存约束

- `srcGlobalData` 必须指向远端地址（源 NPU）
- `dstGlobalData` 必须指向本地地址（当前 NPU）
- `stagingTileData`、`pingTile`、`pongTile` 必须预先在 UB 中分配

### 传输约束

- 传输大小由 `GlobalTensor` 的 shape 决定；自动分块以适配 UB 暂存缓冲区
- 自动分块时，行（`DIM_3`）和列（`DIM_4`）会根据需要细分

### 乒乓约束

- `pingTile` 与 `pongTile` 的类型和维度必须一致
- 两者必须位于不重叠的 UB 偏移

## 目标Profile限制

- 点对点通信仅在 A2/A3 和 A5 Profile 上支持。CPU 模拟不支持远程内存访问。
- 传输大张量时请使用乒乓双缓冲，以重叠连续传输，提高流水线利用率。
- `TGET` 需要有效的远端 GM 地址；远端 NPU 必须已分配对应的内存区域。

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

- [通信与运行时](../other/communication-and-runtime_zh.md)
- 逆操作：[TPUT](./TPUT_zh.md)
- 集合通信：[TBROADCAST](./TBROADCAST_zh.md)、[TGATHER](./TGATHER_zh.md)、[TSCATTLER](./TSCATTER_zh.md)、[TREDUCE](./TREDUCE_zh.md)
- 指令集：[其他与通信](../other/README_zh.md)
