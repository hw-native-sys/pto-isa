# pto.tcolexpandmin

`pto.tcolexpandmin` 属于[归约与扩展](../../reduce-and-expand_zh.md)指令集。

## 概述

把“每列一个标量”的向量广播到整列，再与 `src0` 做逐元素最小值。

## 机制

设 `R = dst.GetValidRow()`、`C = dst.GetValidCol()`。记 `s_j` 为第 `j` 列对应的广播标量。则：

$$ \mathrm{dst}_{i,j} = \min(\mathrm{src0}_{i,j}, s_j) $$

## 语法

同步形式：

```text
%dst = tcolexpandmin %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tcolexpandmin %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tcolexpandmin ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDMIN(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

## 输入

- `src0`：逐元素主输入 tile
- `src1`：提供“每列一个标量”的广播源
- `dst`：目标 tile

## 预期输出

- `dst[i,j] = min(src0[i,j], src1[0,j])`

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

- `TileDataDst::DType`、`TileDataSrc1::DType` 当前实现只支持 `half`、`float`
- `dst` 必须是 row-major
- `src1` 应覆盖每一列的广播标量
- 具体布局 / 分形约束由 backend 决定

## 异常与非法情形

- 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## 性能

当前仓内没有把 `tcolexpandmin` 单列成公开 cost table，应视为目标 profile 相关的列广播组合路径。

## 示例

```text
%dst = pto.tcolexpandmin %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

## 相关页面

- 指令集总览：[归约与扩展](../../reduce-and-expand_zh.md)
- 上一条指令：[pto.tcolexpandmax](./tcolexpandmax_zh.md)
- 下一条指令：[pto.tcolexpandsub](./tcolexpandsub_zh.md)
