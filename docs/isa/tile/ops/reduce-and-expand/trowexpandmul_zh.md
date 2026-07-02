# pto.trowexpandmul

`pto.trowexpandmul` 属于[归约与扩展](../../reduce-and-expand_zh.md)指令集。

## 概述

把“每行一个标量”的向量广播到整行，再与 `src0` 做逐元素乘法。

## 机制

逐行广播乘法：将 `src0` 的每一行元素乘以逐行广播操作数。

### Mode 1 — 每行一个标量（ColMajor 扩展操作数）

![TROWEXPANDMUL Mode 1 tile operation](../../../../figures/isa/TROWEXPANDMUL.svg)

令 `R = dst.GetValidRow()`，`C = dst.GetValidCol()`。令 `s_i` 为扩展操作数中第 `i` 行对应的标量。

对 `0 <= i < R` 且 `0 <= j < C`：

$$
\mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \cdot s_i
$$

### Mode 2 — 每行 32 字节块（RowMajor 扩展操作数）

![TROWEXPANDMUL Mode 2 tile operation](../../../../figures/isa/TROWEXPANDMUL_mode2.svg)

令 `b_i` 为扩展操作数中第 `i` 行对应的 32 字节数据块。该数据块包含 `32 / sizeof(T)` 个元素，并在行内重复使用。

对 `0 <= i < R` 且 `0 <= j < C`：

$$
\mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \cdot b_i[\,j \bmod (32 / \mathit{sizeof}(T))\,]
$$

## 语法

同步形式：

```text
%dst = trowexpandmul %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.trowexpandmul %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.trowexpandmul ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMUL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMUL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp, WaitEvents &... events);
```

## 输入

- `src0`：逐元素主输入 tile
- `src1`：提供“每行一个标量”的广播源
- `tmp`：部分实现路径会用到的临时 tile
- `dst`：目标 tile

## 预期输出

- `dst[i,j] = src0[i,j] * src1[i,0]`

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


### 临时 Tile

C++ API 提供显式传入 `TileDataTmp &tmp` 的重载。该重载仅支持**模式 1**（ColMajor 扩展操作数，每行标量）。

- **A2A3**：tmp Tile 作为广播缓冲区使用。ColMajor 扩展操作数的每行标量值通过 `vbrcb` 指令广播到 tmp 缓冲区，为每行创建一个 32 字节块，然后在二元运算中作为扩展操作数使用。`vbrcb` 指令的 repeat stride 为 8 个块（256 字节），每个 repeat 处理 8 行。最小 tmp 大小计算：
    - 公共参数：`R = dst.GetValidRow()`，`T = TileDataDst::DType`。
    - 当 `R < 256` 时：
      $$ \text{tmpSize} = \left\lceil\frac{R}{8}\right\rceil \times 256 \text{ 字节} $$
    - 当 `R >= 256` 时，操作采用循环方式，每次循环最多 30 个 repeat（240 行）。tmp 缓冲区在各循环间复用，每次循环需要：
      $$ \text{tmpSize} = 30 \times 256 = 7680 \text{ 字节} $$
    - 对于任何模式 1 调用，一个紧凑的形状无关上界为 **8 KB**（8192 字节）。
    - 不带 `tmp` 的 3 参数重载支持模式 1 和模式 2。对于模式 1，使用内部 8 KB 缓冲区（`TMP_UB_OFFSET`）。对于模式 2，不需要广播缓冲区。
- **A5**：`tmp` Tile 被接受但不使用（`[[maybe_unused]]`）。A5 硬件通过 `vlds` 指令的广播模式原生支持行广播，因此不需要临时缓冲区。

## 异常与非法情形

!!! danger "异常与非法情形"
    - 非法操作数组合、不支持的数据类型、不合法布局或不支持的 target-profile 模式，会被 verifier 或后端实现拒绝。

## 性能

当前仓内没有把 `trowexpandmul` 单列成公开 cost table，应视为目标 profile 相关的广播组合路径。

## 示例

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, half, 16, 16>;
  using RowVecT = Tile<TileType::Vec, half, 16, 1, BLayout::ColMajor, 1, DYNAMIC, SLayout::NoneBox>;
  TileT src0, dst;
  RowVecT src1(16);
  TROWEXPANDMUL(dst, src0, src1);
}
```

## 相关页面

- 指令集总览：[归约与扩展](../../reduce-and-expand_zh.md)
- 上一条指令：[pto.trowexpanddiv](./trowexpanddiv_zh.md)
- 下一条指令：[pto.trowexpandsub](./trowexpandsub_zh.md)
