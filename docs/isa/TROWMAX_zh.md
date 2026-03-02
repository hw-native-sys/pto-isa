# TROWMAX

## 指令示意图

![TROWMAX tile operation](../figures/isa/TROWMAX.svg)

## 简介

通过取列间最大值来归约每一行。

## 数学语义

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. For `0 <= i < R`:

$$ \mathrm{dst}_{i,0} = \max_{0 \le j < C} \mathrm{src}_{i,j} $$

## 汇编语法

PTO-AS 形式：参见 [docs/grammar/PTO-AS_zh.md](../grammar/PTO-AS_zh.md)。

同步形式：

```text
%dst = trowmax %src : !pto.tile<...> -> !pto.tile<...>
```
Lowering may introduce internal scratch tiles; the C++ intrinsic requires an explicit `tmp` operand.

### IR Level 1（SSA）

```text
%dst = pto.trowmax %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### IR Level 2（DPS）

```text
pto.trowmax ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWMAX(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp, WaitEvents&... events);
```

## 约束

实现检查 (NPU):

- A2A3:
  - Tile location: `dst` and `src` must be `TileType::Vec`.
  - Tile 布局 of `src`: ND fractal (`isRowMajor` and `SLayout::NoneBox`).
  - Tile 布局 of `dst`:
    - **推荐**: DN layout Tile of 1D, e.g., `Tile<TileType::Vec, T, ROWS, 1, BLayout::ColMajor, ValidRows, 1>`
    - **将移除**: ND layout Tile of 2D, e.g., `Tile<TileType::Vec, T, ROWS, COLS, BLayout::RowMajor, ValidRows, 1>`
  - 数据类型: `half` or `float`.
  - 数据类型一致性: `dst.DType == src.DType`.
  - 运行期有效区域检查:
    - `srcValidCol != 0` and `srcValidRow != 0`.
    - `srcValidRow == dstValidRow` (the output valid row must match the input valid row).
- A5:
  - 数据类型: `half` or `float`.
  - 数据类型一致性: `dst.DType == src.DType`.
  - No explicit runtime assertions on `validRow/validCol` in the implementation; the loops use `src.GetValidRow()` and `src.GetValidCol()`.

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
  TROWMAX(dst, src, tmp);
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
  TROWMAX(dst, src, tmp);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.trowmax %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.trowmax %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = trowmax %src : !pto.tile<...> -> !pto.tile<...>
# IR Level 2 (DPS)
pto.trowmax ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

