# pto.trowmax

`pto.trowmax` 属于[归约与扩展](../../reduce-and-expand_zh.md)指令集。

## 概述

对每一行按列取最大值。

## 机制

设：

- `R = src.GetValidRow()`
- `C = src.GetValidCol()`

则对 `0 <= i < R`：

$$ \mathrm{dst}_{i,0} = \max_{0 \le j < C} \mathrm{src}_{i,j} $$

它把 `(R, C)` 压成 `(R, 1)`，保留行索引、折叠列方向。lowering 过程中通常会引入 `tmp`。

## 语法

同步形式：

```text
%dst = trowmax %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.trowmax %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.trowmax ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWMAX(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents &... events);
```

## 输入

- `src`：源 tile
- `tmp`：临时 tile
- `dst`：目标 tile

## 预期输出

- `dst[i,0]`：第 `i` 行所有列元素中的最大值

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- `dst` 与 `src` 必须都为 `TileType::Vec`
- `src` 必须使用标准 ND 布局：行主且非分形
- `dst` 可以是 ND，或 `Cols == 1` 的 DN 布局
- `dst` 与 `src` 元素类型必须一致
- 运行时要求：
  - `src.GetValidRow() != 0`
  - `src.GetValidCol() != 0`
  - `src.GetValidRow() == dst.GetValidRow()`

### A2A3

- 支持类型：`half`、`float`、`int32_t`、`int16_t`
- 实现同时接受 ND 输出和 `Cols == 1` 的 DN 输出
- 当前实现路径会把 `tmp` 传入后端调用，但文档不额外引入 checked implementation 没声明的 `tmp` 限制

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## 性能

当前仓内没有把 `trowmax` 单列成公开 cost table。它应视为行归约类路径，而不是普通二元逐元素算术。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TROWMAX(dst, src, tmp);
}
```

## 相关页面

- 指令集总览：[归约与扩展](../../reduce-and-expand_zh.md)
- 上一条指令：[pto.tcolmax](./tcolmax_zh.md)
- 下一条指令：[pto.trowmin](./trowmin_zh.md)
