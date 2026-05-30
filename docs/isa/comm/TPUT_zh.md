# pto.tput

## 简介

`TPUT` 是远程写原语：把当前 NPU 本地 GM 中的数据写到远端 NPU 的 GM。它通过 UB 中的暂存 Tile 完成 GM→UB→GM 路径。

当 `GlobalTensor` 的行或列超出单个 UB Tile 容量时，`TPUT` 会自动沿 `DIM_3` 和 `DIM_4` 做二维滑动分块。

## 数学语义

对有效区域内每个元素 `(i, j)`：

$$ \mathrm{dst}^{\mathrm{remote}}_{i,j} = \mathrm{src}^{\mathrm{local}}_{i,j} $$

## 汇编语法

PTO-AS 形式：

```text
tput %dst_remote, %src_local : (!pto.memref<...>, !pto.memref<...>)
```

lowering 会为 GM→UB→GM 路径引入 UB 暂存 Tile，因此 C++ 接口要求显式传入 `stagingTileData`，或在双缓冲场景下传入 `pingTile` / `pongTile`。

## C++ 内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`：

### 单暂存 Tile

```cpp
template <AtomicType atomicType = AtomicType::AtomicNone,
          typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TPUT(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                          TileData &stagingTileData, WaitEvents&... events);
```

### 乒乓双缓冲

```cpp
template <AtomicType atomicType = AtomicType::AtomicNone,
          typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TPUT(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                          TileData &pingTile, TileData &pongTile, WaitEvents&... events);
```

### 运行时原子模式

```cpp
template <typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TPUT(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                          TileData &stagingTileData, AtomicType atomicType, WaitEvents&... events);
```

## 约束

!!! warning "约束"
    ### 类型约束

    - `GlobalSrcData::RawDType` 必须等于 `GlobalDstData::RawDType`
    - `TileData::DType` 必须等于 `GlobalSrcData::RawDType`
    - `GlobalSrcData::layout` 必须等于 `GlobalDstData::layout`

    ### 内存约束

    - `dstGlobalData` 必须指向远端地址（目标 NPU）
    - `srcGlobalData` 必须指向本地地址（当前 NPU）
    - `stagingTileData`、`pingTile`、`pongTile` 必须预先在 UB 中分配

    ### 原子与双缓冲约束

    - 当前接口支持 `AtomicNone` 与 `AtomicAdd`
    - `pingTile` 与 `pongTile` 的类型和维度必须一致
    - 两者必须位于不重叠的 UB 偏移处

## 示例

### 基础形式

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_tput(__gm__ T* local_data, __gm__ T* remote_addr) {
    using TileT   = Tile<TileType::Vec, T, 16, 16>;
    using GShape  = Shape<1, 1, 1, 16, 16>;
    using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
    using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

    GTensor srcG(local_data);
    GTensor dstG(remote_addr);
    TileT stagingTile;
    TASSIGN(stagingTile, 0);

    comm::TPUT(dstG, srcG, stagingTile);
    comm::TPUT<AtomicType::AtomicAdd>(dstG, srcG, stagingTile);
}
```

### 乒乓双缓冲

```cpp
constexpr size_t tileUBBytes = ((64 * 64 * sizeof(float) + 1023) / 1024) * 1024;
TileT pingTile(64, 64);
TileT pongTile(64, 64);
TASSIGN(pingTile, 0);
TASSIGN(pongTile, tileUBBytes);

comm::TPUT(dstG, srcG, pingTile, pongTile);
```

### 运行时指定原子模式

```cpp
comm::TPUT(dstG, srcG, stagingTile, AtomicType::AtomicAdd);
```

## 相关页面

- [通信与运行时](communication-runtime_zh.md)
- [TGET](./TGET_zh.md)
- [TSCATTER](./TSCATTER_zh.md)
