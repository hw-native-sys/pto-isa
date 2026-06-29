# pto.trowexpanddiv

`pto.trowexpanddiv` 属于[归约与扩展](../../reduce-and-expand_zh.md)指令集。

## 概述

把“每行一个标量”的向量广播到整行，再作为除数参与逐元素除法。

## 机制

逐行广播除法：用逐行广播操作数除 `src0` 的每一行元素。

### Mode 1 — 每行一个标量（ColMajor 扩展操作数）

![TROWEXPANDDIV Mode 1 tile operation](../../../../figures/isa/TROWEXPANDDIV.svg)

令 `R = dst.GetValidRow()`，`C = dst.GetValidCol()`。令 `s_i` 为扩展操作数中第 `i` 行对应的标量。

对 `0 <= i < R` 且 `0 <= j < C`：

$$
\mathrm{dst}_{i,j} = \frac{\mathrm{src0}_{i,j}}{s_i}
$$

### Mode 2 — 每行 32 字节块（RowMajor 扩展操作数）

![TROWEXPANDDIV Mode 2 tile operation](../../../../figures/isa/TROWEXPANDDIV_mode2.svg)

令 `b_i` 为扩展操作数中第 `i` 行对应的 32 字节数据块。该数据块包含 `32 / sizeof(T)` 个元素，并在行内重复使用。

对 `0 <= i < R` 且 `0 <= j < C`：

$$
\mathrm{dst}_{i,j} = \frac{\mathrm{src0}_{i,j}}{b_i[\,j \bmod (32 / \mathit{sizeof}(T))\,]}
$$

## 语法

同步形式：

```text
%dst = trowexpanddiv %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.trowexpanddiv %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.trowexpanddiv ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDDIV(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDDIV(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp, WaitEvents &... events);
```

## 输入

- `src0`：逐元素主输入 tile
- `src1`：提供“每行一个标量”的广播源
- `tmp`：部分实现路径会用到的临时 tile
- `dst`：目标 tile

## 预期输出

- `dst[i,j] = src0[i,j] / src1[i,0]`

## 副作用

除产生目标 tile 外，没有额外架构副作用。

## 约束

!!! warning "约束"
    - `TileDataDst::DType == TileDataSrc0::DType == TileDataSrc1::DType`。

    - `TileDataDst::DType`、`TileDataSrc0::DType` 和 `TileDataSrc1::DType` 必须是以下类型之一：`half`、`float`、`int16`、`int32`（A2、A3 和 A5），`uint16`、`uint32`（A5）。

    - `TileDataDst` 必须是 RowMajor。

    - `src0` 和 `src1` 中必须恰好有一个操作数的有效形状与 `dst` 相同。该操作数是完整尺寸操作数；另一个操作数是逐行广播的扩展操作数。

    - 完整尺寸操作数必须是 RowMajor。

    - Mode 1：扩展操作数为 ColMajor，有效列数为 1，有效行数等于 `dst.GetValidRow()`。

    - Mode 2：扩展操作数为 RowMajor，有效行数等于 `dst.GetValidRow()`，有效列数为 `32 / sizeof(T)`：16 位类型为 16，32 位类型为 8。

    - 精确的布局、fractal 和对齐约束与目标后端有关；请参见 `include/pto/npu/*/TRowExpand*.hpp` 下的后端头文件。

## 异常与非法情形

!!! danger "异常与非法情形"
    - 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## 性能

当前仓内没有把 `trowexpanddiv` 单列成公开 cost table，应视为目标 profile 相关的广播组合路径。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, half, 16, 16>;
  using RowVecT = Tile<TileType::Vec, half, 16, 1, BLayout::ColMajor, 1, DYNAMIC, SLayout::NoneBox>;
  TileT src0, dst;
  RowVecT src1(16);
  TROWEXPANDDIV(dst, src0, src1);
}
```

## 相关页面

- 指令集总览：[归约与扩展](../../reduce-and-expand_zh.md)
- 上一条指令：[pto.trowexpand](./trowexpand_zh.md)
- 下一条指令：[pto.trowexpandmul](./trowexpandmul_zh.md)
