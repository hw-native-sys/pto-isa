# pto.tcolexpand

`pto.tcolexpand` 属于[归约与扩展](../../reduce-and-expand_zh.md)指令集。

## 概述

把源 tile 每一列的第一个元素广播到整列。

## 机制

设 `R = dst.GetValidRow()`、`C = dst.GetValidCol()`。对 `0 <= i < R` 且 `0 <= j < C`：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{0,j} $$

它是列广播的基础形式：先从每一列取出一个标量，再沿行方向复制回整列。后续 `tcolexpandadd`、`tcolexpandmax`、`tcolexpandexpdif` 都建立在这个语义之上。

## 语法

同步形式：

```text
%dst = tcolexpand %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tcolexpand %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tcolexpand ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPAND(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events);
```

## 输入

- `src`：源 tile
- `dst`：目标 tile

## 预期输出

- `dst`：每一列都被 `src[0,j]` 填满的广播结果

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- 迭代域由 `dst.GetValidRow()` / `dst.GetValidCol()` 决定。
- 这条指令要求源和目标在 shape / layout 上满足列广播的合法条件。

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## Target-Profile 限制

- `pto.tcolexpand` 在 CPU 仿真、A2/A3 和 A5 上保留一致的 PTO 可见语义，但具体支持子集仍取决于 profile。

## 性能

当前仓内没有把 `tcolexpand` 单列成公开 cost table。若代码依赖具体延迟，应把它视为目标 profile 相关的列广播路径。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TCOLEXPAND(dst, src);
}
```

## 相关页面

- 指令集总览：[归约与扩展](../../reduce-and-expand_zh.md)
- 上一条指令：[pto.tcolmin](./tcolmin_zh.md)
- 下一条指令：[pto.tcolexpanddiv](./tcolexpanddiv_zh.md)
