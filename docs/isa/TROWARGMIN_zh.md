# TROWARGMIN

## 指令示意图

![TROWMIN tile operation](../figures/isa/TROWMIN.svg)

## 简介

获取每行最小值对应列索引。

## 数学语义

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. For `0 <= i < R`:

$$ \mathrm{dst}_{i,0} = \min_{0 \le j < C} j_{i} $$

## 汇编语法

PTO-AS 形式：参见 `docs/grammar/PTO-AS.md`.

同步形式：

```text
%dst = trowargmin %src : !pto.tile<...> -> !pto.tile<...>
```
Lowering may introduce internal scratch tiles; the C++ intrinsic requires an explicit `tmp` operand.

### IR Level 1（SSA）

```text
%dst = pto.trowargmin %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### IR Level 2（DPS）

```text
pto.trowargmin ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWARGMIN(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp, WaitEvents&... events);
```

## 约束

实现检查 (NPU):

- A2A3:
  - Tile location: `dst` and `src` must be `TileType::Vec`.
  - Tile 布局 of `src`: ND fractal (`isRowMajor` and `SLayout::NoneBox`).
  - Tile 布局 of `dst`:
    - **紧凑模式**：DN 布局的一维 Tile，例如 `Tile<TileType::Vec, T, ROWS, 1, BLayout::ColMajor, ValidRows, 1>`，此时ROWS要做到32b对齐。
    - **传统模式**：ND 布局的二维 Tile，例如 `Tile<TileType::Vec, T, ROWS, COLS, BLayout::RowMajor, ValidRows, 1>`。
  - 源数据类型: `half` or `float`.
  - 目标数据类型：`uint32_t` or `int32_t`.
  - 运行期有效区域检查:
    - `srcValidCol != 0` and `srcValidRow != 0`.
- A5:
  - 源数据类型: `half` or `float`.
  - 目标数据类型：`uint32_t` or `int32_t`.
  - No explicit runtime assertions on `validRow/validCol` in the implementation; the loops use `src.GetValidRow()` and `src.GetValidCol()`.
  - `tmp`临时Tile不使用，仅做兼容。

### A3 `tmp`临时Tile相关说明

* `tmp`临时Tile在`srcValidCol <= ElementPerRepeat`时不使用，`srcValidCol > ElementPerRepeat`时需要使用。
* `tmp` tile的行数和`src` tile的行数相同。
* 按以下公式根据`src` tile的`validCol`算出`tmp` tile所需stride：

```text
repeats = ceil(validCol / elementPerRepeat)
stride = ceil(repeats * 2 / elementPerBlock) * elementPerBlock + ceil(repeats / elementPerBlock) * elementPerBlock
```

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TROWARGMIN(dst, src, tmp);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(tmp, 0x3000);
  TROWARGMIN(dst, src, tmp);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.trowargmin %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.trowargmin %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = trowargmin %src : !pto.tile<...> -> !pto.tile<...>
# IR Level 2 (DPS)
pto.trowargmin ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

