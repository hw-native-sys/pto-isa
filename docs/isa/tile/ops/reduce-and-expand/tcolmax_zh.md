# pto.tcolmax

`pto.tcolmax` 属于[归约与扩展](../../reduce-and-expand_zh.md)指令集。

## 概述

对每一列按行取最大值。

## 机制

设：

- `R = src.GetValidRow()`
- `C = src.GetValidCol()`

则对 `0 <= j < C`：

$$ \mathrm{dst}_{0,j} = \max_{0 \le i < R} \mathrm{src}_{i,j} $$

它把 `(R, C)` 压成 `(1, C)`，保留列、折叠行。

## 语法

同步形式：

```text
%dst = tcolmax %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tcolmax %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tcolmax ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TCOLMAX(TileDataOut &dst, TileDataIn &src, WaitEvents &... events);
```

## 输入

- `src`：源 tile
- `dst`：目标 tile

## 预期输出

- `dst[0,j]`：第 `j` 列所有行元素中的最大值

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- `dst` 与 `src` 必须为 `TileType::Vec`
- 二者都必须使用标准 ND 布局：行主且非分形
- 二者元素类型必须一致
- 运行时要求：`src.GetValidCol() == dst.GetValidCol()`
- 若 `src.GetValidRow() == 0` 或 `src.GetValidCol() == 0`，实现会直接返回

### A2A3

- 支持类型：`half`、`float`、`int16_t`、`int32_t`

### A5

- 支持类型：`half`、`float`、`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`bfloat16_t`

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## 性能

当前仓内没有把 `tcolmax` 单列成公开 cost table。它应视为列归约路径，而不是普通逐元素算术。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 1, 16>;
  SrcT src;
  DstT dst;
  TCOLMAX(dst, src);
}
```

## 相关页面

- 指令集总览：[归约与扩展](../../reduce-and-expand_zh.md)
- [TCOLARGMAX](./tcolargmax_zh.md)
- [TCOLMIN](./tcolmin_zh.md)
