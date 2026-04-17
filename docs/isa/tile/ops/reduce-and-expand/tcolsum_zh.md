# pto.tcolsum

`pto.tcolsum` 属于[归约与扩展](../../reduce-and-expand_zh.md)指令集。

## 概述

对每一列按行求和。

## 机制

设：

- `R = src.GetValidRow()`
- `C = src.GetValidCol()`

则对 `0 <= j < C`：

$$ \mathrm{dst}_{0,j} = \sum_{i=0}^{R-1} \mathrm{src}_{i,j} $$

它把 `(R, C)` 压成 `(1, C)`，保留列、折叠行。`isBinary` 用来选择实现路径：二叉树累加还是顺序累加。

## 语法

同步形式：

```text
%dst = tcolsum %src {isBinary = false} : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tcolsum %src : !pto.tile<...> -> !pto.tile<...>
%dst = pto.tcolsum %src, %tmp {isBinary = false} : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tcolsum ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
pto.tcolsum ins(%src, %tmp {isBinary = false} : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TCOLSUM(TileDataOut &dst, TileDataIn &src, WaitEvents &... events);

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TCOLSUM(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, bool isBinary, WaitEvents &... events);
```

## 输入

- `src`：源 tile
- `tmp`：在二叉路径里参与归约的临时 tile
- `isBinary`：选择二叉树累加还是顺序累加
- `dst`：目标 tile

## 预期输出

- `dst[0,j]`：第 `j` 列所有行元素的和

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- `dst` 与 `src` 必须为 `TileType::Vec`
- 二者都必须使用标准 ND 布局：行主且非分形
- `dst` 与 `src` 元素类型必须一致
- 运行时要求：
  - `src.GetValidCol() == dst.GetValidCol()`
  - `src.GetValidRow() != 0`
  - `src.GetValidCol() != 0`

### A2A3

- 支持类型：`half`、`float`、`int16_t`、`int32_t`
- `tmp` 必须是 `TileType::Vec`，使用标准 ND 布局，且元素类型与 `src`、`dst` 一致

### A5

- A5 共享列归约检查允许：`half`、`float`、`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`bfloat16_t`
- 已检查到的 A5 `TCOLSUM` 路径中，`tmp` 主要用于二叉累加路径

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## 性能

当前仓内没有把 `tcolsum` 单列成公开 cost table，但它和 `trowsum` 一样，应视为多阶段归约路径，而不是普通逐元素二元算术。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 1, 16>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TCOLSUM(dst, src, tmp, /*isBinary=*/false);
}
```

## 相关页面

- 指令集总览：[归约与扩展](../../reduce-and-expand_zh.md)
- 上一条指令：[pto.trowsum](./trowsum_zh.md)
