# pto.trowexpand

`pto.trowexpand` 属于[归约与扩展](../../reduce-and-expand_zh.md)指令集。

## 概述

把源 tile 每一行的第一个元素广播到整行。

## 机制

设 `R = dst.GetValidRow()`、`C = dst.GetValidCol()`。对 `0 <= i < R` 且 `0 <= j < C`：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,0} $$

也就是说，`trowexpand` 先从每一行抽出一个标量，再沿列方向复制回整行。它是行广播的最基础形式，后面的 `trowexpandadd`、`trowexpandmax`、`trowexpandexpdif` 都是在这个语义上继续组合。

## 语法

同步形式：

```text
%dst = trowexpand %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.trowexpand %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.trowexpand ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPAND(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events);
```

## 输入

- `src`：源 tile
- `dst`：目标 tile

## 预期输出

- `dst`：每一行都被 `src[i,0]` 填满的广播结果

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- `dst` 和 `src` 都必须是 `TileType::Vec`
- `src` 与 `dst` 都必须是标准 ND 非分形布局：row-major 且 `SLayout::NoneBox`
- 支持的数据类型在 A2A3 / A5 上都覆盖：`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`half`、`bfloat16_t`、`float`

### 运行时有效区域

- A2A3：若 `dstValidRow`、`dstValidCol`、`srcValidRow`、`srcValidCol` 中任一为 0，直接提前返回
- A5：要求 `srcValidRow == dstValidRow`，并要求 `srcValidRow != 0 && srcValidCol != 0`

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## 性能

当前仓内没有把 `trowexpand` 单列成公开 cost table。若代码依赖具体延迟，应把它视为目标 profile 相关的广播 / 重排路径。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TROWEXPAND(dst, src);
}
```

## 相关页面

- 指令集总览：[归约与扩展](../../reduce-and-expand_zh.md)
- 上一条指令：[pto.trowargmin](./trowargmin_zh.md)
- 下一条指令：[pto.trowexpanddiv](./trowexpanddiv_zh.md)
