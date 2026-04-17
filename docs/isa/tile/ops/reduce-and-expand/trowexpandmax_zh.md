# pto.trowexpandmax

`pto.trowexpandmax` 属于[归约与扩展](../../reduce-and-expand_zh.md)指令集。

## 概述

把“每行一个标量”的向量广播到整行，再与 `src0` 做逐元素最大值。

## 机制

设 `R = dst.GetValidRow()`、`C = dst.GetValidCol()`。记 `s_i` 为第 `i` 行对应的广播标量。则：

$$ \mathrm{dst}_{i,j} = \max(\mathrm{src0}_{i,j}, s_i) $$

常见用途是按行阈值裁剪、数值稳定化前的按行上界展开等。

## 语法

同步形式：

```text
%dst = trowexpandmax %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.trowexpandmax %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.trowexpandmax ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMAX(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMAX(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp,
                                   WaitEvents &... events);
```

## 输入

- `src0`：逐元素主输入 tile
- `src1`：提供“每行一个标量”的广播源
- `tmp`：部分实现路径会用到的临时 tile
- `dst`：目标 tile

## 预期输出

- `dst[i,j] = max(src0[i,j], src1[i,0])`

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- `dst/src0/src1` 的元素类型必须一致，当前实现只支持 `half` 或 `float`
- `dst` 必须是 row-major
- `src1` 需要表达“每行一个标量”这一角色

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## 性能

当前仓内没有把 `trowexpandmax` 单列成公开 cost table，应视为目标 profile 相关的广播组合路径。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using MatT = Tile<TileType::Vec, float, 16, 16>;
  using RowBiasT = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor>;
  MatT src0, dst;
  RowBiasT src1;
  TROWEXPANDMAX(dst, src0, src1);
}
```

## 相关页面

- 指令集总览：[归约与扩展](../../reduce-and-expand_zh.md)
- 上一条指令：[pto.trowexpandadd](./trowexpandadd_zh.md)
- 下一条指令：[pto.trowexpandmin](./trowexpandmin_zh.md)
