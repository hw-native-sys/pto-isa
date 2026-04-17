# pto.tcolargmax

`pto.tcolargmax` 属于[归约与扩展](../../reduce-and-expand_zh.md)指令集。

## 概述

返回每一列最大值所在的行索引。

## 机制

设：

- `R = src.GetValidRow()`
- `C = src.GetValidCol()`

则对 `0 <= j < C`：

$$ \mathrm{dst}_{0,j} = \underset{0 \le i < R}{\operatorname{argmax}} \; \mathrm{src}_{i,j} $$

输出 tile 只保留一行，因此 `dst[0,j]` 表示“第 `j` 列的最大值落在哪一行”。若同一列有多个相同最大值，具体选哪一个索引由实现决定。

## 语法

同步形式：

```text
%dst = tcolargmax %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tcolargmax %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tcolargmax ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TCOLARGMAX(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp, WaitEvents&... events);
```

## 输入

- `src`：源 tile
- `tmp`：A2A3 路径可能使用的临时 tile
- `dst`：目标索引 tile

## 预期输出

- `dst[0,j]`：第 `j` 列最大值所在的行索引

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- `dst` 与 `src` 必须为 `TileType::Vec`
- 源元素类型支持：`half`、`float`、`int32_t`、`int16_t`
- 目标元素类型支持：`uint32_t`、`int32_t`
- 运行时要求：
  - `src.GetValidRow() != 0`
  - `src.GetValidCol() != 0`
  - `src.GetValidCol() == dst.GetValidCol()`

### A2A3

- `src` 必须是标准 ND 布局：row-major 且非分形
- `dst` 可为单行 ND，或 `Rows == 1` 的等价非分形形式
- 当 `srcValidRow > elementPerRepeat` 时，`tmp` 会被用于中间索引与比较值保存

### A5

- A5 路径接口仍保留 `tmp`，但当前实现里不实际使用它

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## 性能

当前仓内没有把 `tcolargmax` 单列成公开 cost table。它应视为列索引归约路径，而不是普通数值归约。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1>;
  using DstT = Tile<TileType::Vec, uint32_t, 1, 256, BLayout::RowMajor, -1, -1>;
  using TmpT = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1>;
  SrcT src(16, 255);
  DstT dst(1, 255);
  TmpT tmp(1, 32);
  TCOLARGMAX(dst, src, tmp);
}
```

## 相关页面

- 指令集总览：[归约与扩展](../../reduce-and-expand_zh.md)
- [TCOLMAX](./tcolmax_zh.md)
