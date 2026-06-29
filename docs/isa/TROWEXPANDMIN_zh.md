# TROWEXPANDMIN

## 指令示意图

### 模式 1 — 每行标量（ColMajor src1）

![TROWEXPANDMIN 模式 1 tile operation](../figures/isa/TROWEXPANDMIN.svg)

### 模式 2 — 每行 32 字节块（RowMajor src1）

![TROWEXPANDMIN 模式 2 tile operation](../figures/isa/TROWEXPANDMIN_mode2.svg)

## 简介

行广播最小值：将 `src0` 的每一行与扩展操作数的每行标量取最小值。

指令支持两种模式，由扩展操作数的布局决定（当 `src0` 与 `dst` 形状匹配时为 `src1`，当 `src1` 与 `dst` 形状匹配时为 `src0`）：

- **模式 1**：扩展操作数为 **ColMajor** 布局，单列（每行一个标量）。每个标量广播到整行。
- **模式 2**：扩展操作数为 **RowMajor** 布局，每行 `32 / sizeof(T)` 列（每行一个 32 字节块）。每个 32 字节块在向量重复步长内自然重复，提供行级广播。

## 数学语义

设 `R = dst.GetValidRow()` 和 `C = dst.GetValidCol()`。

### 模式 1

设 `s_i` 为从扩展操作数中获取的每行标量（每行一个值，ColMajor 布局）。

对于 `0 <= i < R` 和 `0 <= j < C`：

$$
\mathrm{dst}_{i,j} = \min(\mathrm{src0}_{i,j}, s_i)
$$

### 模式 2

设 `b_i` 为第 `i` 行从扩展操作数中获取的 32 字节块（RowMajor 布局，每行 `32 / sizeof(T)` 个值）。该块在每个向量重复步长内自然重复。

对于 `0 <= i < R` 和 `0 <= j < C`：

$$
\mathrm{dst}_{i,j} = \min(\mathrm{src0}_{i,j}, b_i[\,j \bmod (32 / \mathit{sizeof}(T))\,])
$$

## 汇编语法

同步形式：

```text
%dst = trowexpandmin %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.trowexpandmin %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.trowexpandmin ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMIN(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMIN(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp, WaitEvents &... events);
```

## 约束

- `TileDataDst::DType == TileDataSrc0::DType == TileDataSrc1::DType`
- `TileDataDst::DType`、`TileDataSrc0::DType`、`TileDataSrc1::DType` 必须是以下之一：`half`、`float`、`int16`、`int32`适用于A2, A3和A5，`uint16`、`uint32`适用于A5。
- `TileDataDst` 必须为 **RowMajor**（`TileDataDst::isRowMajor == true`）。
- `src0` 或 `src1` 中必须恰好一个与 `dst` 的有效形状相同（即 `validRow == dst.validRow` 且 `validCol == dst.validCol`），该操作数为全尺寸操作数。另一个操作数为**扩展操作数**（行广播源）。
- 全尺寸操作数必须为 **RowMajor**（`isRowMajor == true`）。

### 模式 1 — 扩展操作数为 ColMajor（每行标量）

当扩展操作数为 **ColMajor**（`isRowMajor == false`）时：

- 其有效列数必须为 **1**（每行一个标量）：`srcX.GetValidCol() == 1`。
- 其有效行数必须等于 `dst.GetValidRow()`：`srcX.GetValidRow() == dst.GetValidRow()`。

### 模式 2 — 扩展操作数为 RowMajor（每行 32 字节块）

当扩展操作数为 **RowMajor**（`isRowMajor == true`）时：

- 其有效列数必须为 **32 / sizeof(T)**（每行一个 32 字节块）：`srcX.GetValidCol() == 32 / sizeof(T)`。
  - 对于 `half` / `int16` / `uint16`：`validCol == 16`。
  - 对于 `float` / `int32` / `uint32`：`validCol == 8`。
- 其有效行数必须等于 `dst.GetValidRow()`：`srcX.GetValidRow() == dst.GetValidRow()`。

### 其他目标特定约束

具体的布局、分形和对齐约束可能因后端目标而异。参见 `include/pto/npu/*/TRowExpand*.hpp` 下的后端头文件。

## 示例

参见 `docs/isa/` 和 `docs/coding/tutorials/` 中的相关示例。

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.trowexpandmin %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.trowexpandmin %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = trowexpandmin %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.trowexpandmin ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
