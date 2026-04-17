# pto.trowexpandadd

`pto.trowexpandadd` 属于[归约与扩展](../../reduce-and-expand_zh.md)指令集。

## 概述

把“每行一个标量”的向量广播到整行，再与 `src0` 做逐元素加法。

## 机制

设 `R = dst.GetValidRow()`、`C = dst.GetValidCol()`。记 `s_i` 为第 `i` 行对应的广播标量。则：

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} + s_i $$

这里的 `src1` 在语义上表示“每行一个标量”。它的物理布局在不同 backend 上可以有不止一种，但可见结果始终等价于“先行广播，再逐元素加”。

## 语法

同步形式：

```text
%dst = trowexpandadd %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.trowexpandadd %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.trowexpandadd ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDADD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDADD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp,
                                   WaitEvents &... events);
```

## 输入

- `src0`：逐元素主输入 tile
- `src1`：提供“每行一个标量”的广播源
- `tmp`：某些实现路径会用到的临时 tile
- `dst`：目标 tile

## 预期输出

- `dst[i,j] = src0[i,j] + src1[i,0]`

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- `dst/src0/src1` 的元素类型必须一致，当前实现只支持 `half` 或 `float`
- `dst` 必须是 row-major
- `src1` 需要表达“每行一个标量”这一角色

### 广播侧常见形态

- 单列广播向量
- 或者每行一段固定 32B 数据的打包形式

具体支持哪种物理布局，取决于目标 backend。

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## 性能

当前仓内没有把 `trowexpandadd` 单列成公开 cost table，应视为目标 profile 相关的广播组合路径。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  using RowVecT = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor>;
  SrcT src0;
  DstT dst;
  RowVecT src1;
  TROWEXPANDADD(dst, src0, src1);
}
```

## 相关页面

- 指令集总览：[归约与扩展](../../reduce-and-expand_zh.md)
- 上一条指令：[pto.trowexpandsub](./trowexpandsub_zh.md)
- 下一条指令：[pto.trowexpandmax](./trowexpandmax_zh.md)
