# TROWEXPANDSUB

## 指令示意图

### 模式 1 — 每行标量（ColMajor src1）

![TROWEXPANDSUB 模式 1 tile operation](../figures/isa/TROWEXPANDSUB.svg)

### 模式 2 — 每行 32字节块（RowMajor src1）

![TROWEXPANDSUB 模式 2 tile operation](../figures/isa/TROWEXPANDSUB_mode2.svg)

## 简介

行广播减法：从 `src0` 的每一行中减去扩展操作数的每行标量。

指令支持两种模式，由扩展操作数的布局决定（当 `src0` 与 `dst` 形状匹配时为 `src1`，当 `src1` 与 `dst` 形状匹配时为 `src0`）：

- **模式 1**：扩展操作数为 **ColMajor** 布局，单列（每行一个标量）。每个标量广播到整行。
- **模式 2**：扩展操作数为 **RowMajor** 布局，每行 `32 / sizeof(T)` 列（每行一个 32字节块）。每个 32字节块在向量重复步长内自然重复，提供行级广播。

## 数学语义

设 `R = dst.GetValidRow()` 和 `C = dst.GetValidCol()`。

### 模式 1

设 `s_i` 为从扩展操作数中获取的每行标量（每行一个值，ColMajor 布局）。

对于 `0 <= i < R` 和 `0 <= j < C`：

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} - s_i $$

### 模式 2

设 `b_i` 为第 `i` 行从扩展操作数中获取的 32字节块（RowMajor 布局，每行 `32 / sizeof(T)` 个值）。该块在每个向量重复步长内自然重复。

对于 `0 <= i < R` 和 `0 <= j < C`：

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} - b_i[\,j \bmod (32 / \mathit{sizeof}(T))\,] $$

## 汇编语法

同步形式：

```text
%dst = trowexpandsub %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.trowexpandsub %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.trowexpandsub ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDSUB(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDSUB(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp, WaitEvents &... events);
```

## 约束

- `TileDataDst::DType == TileDataSrc0::DType == TileDataSrc1::DType`
- `TileDataDst::DType`、`TileDataSrc0::DType`、`TileDataSrc1::DType` 必须是以下之一：`half`、`float`、`int16`、`int32`（适用于A2、A3和A5），`uint16`、`uint32`（仅适用于A5）。
- `TileDataDst` 必须为 **RowMajor**（`TileDataDst::isRowMajor == true`）。
- `src0` 或 `src1` 中必须恰好一个与 `dst` 的有效形状相同（即 `validRow == dst.validRow` 且 `validCol == dst.validCol`），该操作数为全尺寸操作数。另一个操作数为**扩展操作数**（行广播源）。
- 全尺寸操作数必须为 **RowMajor**（`isRowMajor == true`）。

### 模式 1 — 扩展操作数为 ColMajor（每行标量）

当扩展操作数为 **ColMajor**（`isRowMajor == false`）时：

- 其有效列数必须为 **1**（每行一个标量）：`srcX.GetValidCol() == 1`。
- 其有效行数必须等于 `dst.GetValidRow()`：`srcX.GetValidRow() == dst.GetValidRow()`。

### 模式 2 — 扩展操作数为 RowMajor（每行 32字节块）

当扩展操作数为 **RowMajor**（`isRowMajor == true`）时：

- 其有效列数必须为 **32 / sizeof(T)**（每行一个32字节块）：`srcX.GetValidCol() == 32 / sizeof(T)`。
  - 对于 `half` / `int16` / `uint16`：`validCol == 16`。
  - 对于 `float` / `int32` / `uint32`：`validCol == 8`。
- 其有效行数必须等于 `dst.GetValidRow()`：`srcX.GetValidRow() == dst.GetValidRow()`。

### 其他目标特定约束

具体的布局、分形和对齐约束可能因后端目标而异。参见 `include/pto/npu/*/TRowExpand*.hpp` 下的后端头文件。

### 临时 Tile

C++ API 提供了显式传入 `TileDataTmp &tmp` 的重载。该重载仅支持**模式 1**（ColMajor 扩展操作数，每行标量）。

- **A2A3**：tmp Tile 作为广播缓冲区使用。ColMajor 扩展操作数的每行标量值通过 `vbrcb` 指令广播到 tmp 缓冲区，为每行创建一个32字节块，然后在二元运算中作为扩展操作数使用。`vbrcb` 指令的 repeat stride 为 8 个块（256字节），每个 repeat 处理 8 行。最小 tmp 大小计算：
    - **公共参数**：
        - `R = dst.GetValidRow()`，`T = TileDataDst::DType`。
    - 当 `R < 256` 时：
        $$ \text{tmpSize} = \left\lceil\frac{R}{8}\right\rceil \times 256 \text{ 字节} $$
    - 当 `R >= 256` 时：
        - 操作采用循环方式，每次循环最多 30 个 repeat（240 行）。tmp 缓冲区在各循环间复用，每次循环需要：
        $$ \text{tmpSize} = 30 \times 256 = 7680 \text{ 字节} $$
    - 对于任何模式 1 调用，一个紧凑的形状无关上界为 **8KB**（8192字节）。
    - 不带 `tmp` 的 3 参数重载支持模式 1 和模式 2。对于模式 1，使用内部8KB缓冲区（`TMP_UB_OFFSET`）。对于模式 2，不需要广播缓冲区。
- **A5**：`tmp` Tile 被接受但不使用（`[[maybe_unused]]`）。A5 硬件通过 `vlds` 指令的广播模式原生支持行广播，因此不需要临时缓冲区。

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, half, 16, 16>;
  using RowVecT = Tile<TileType::Vec, half, 16, 1, BLayout::ColMajor, 1, DYNAMIC, SLayout::NoneBox>;

  TileT src0, dst;
  RowVecT src1(16);
  TROWEXPANDSUB(dst, src0, src1);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, half, 16, 16>;
  using RowVecT = Tile<TileType::Vec, half, 16, 1, BLayout::ColMajor, 1, DYNAMIC, SLayout::NoneBox>;

  TileT src0, dst;
  RowVecT src1(16);
  TASSIGN(src0, 0x1000);
  TASSIGN(dst,  0x2000);
  TASSIGN(src1, 0x3000);
  TROWEXPANDSUB(dst, src0, src1);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.trowexpandsub %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.trowexpandsub %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = trowexpandsub %src0, %src1 : !pto.tile<...>, !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.trowexpandsub ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
