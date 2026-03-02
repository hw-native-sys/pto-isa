# TROWPROD


## Tile 操作图示

<!-- ![TROWPROD tile operation](../figures/isa/TROWPROD.svg) -->
<!-- TODO: 添加 TROWPROD 图示 -->

## 简介

对每行元素进行乘积归约。

## 数学定义

设 `R = src.GetValidRow()` 且 `C = src.GetValidCol()`。对于 `0 <= i < R`：

$$ \mathrm{dst}_{i,0} = \prod_{j=0}^{C-1} \mathrm{src}_{i,j} $$

## 汇编语法

PTO-AS 形式：参见 `docs/grammar/PTO-AS.md`。

同步形式：

```text
%dst = trowprod %src : !pto.tile<...> -> !pto.tile<...>
```
Lowering 可能引入内部临时 tile；C++ 内建函数需要显式的 `tmp` 操作数。

### IR Level 1 (SSA)

```text
%dst = pto.trowprod %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### IR Level 2 (DPS)

```text
pto.trowprod ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建函数

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWPROD(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp, WaitEvents&... events);
```

## 约束条件

NPU 实现检查：

- A2A3:
  - Tile 位置：`dst` 和 `src` 必须为 `TileType::Vec`。
  - `src` 的 Tile 布局：ND 分形（`isRowMajor` 且 `SLayout::NoneBox`）。
  - `dst` 的 Tile 布局：
    - **推荐**：一维 DN 布局 Tile，例如 `Tile<TileType::Vec, T, ROWS, 1, BLayout::ColMajor, ValidRows, 1>`
    - **将被移除**：二维 ND 布局 Tile，例如 `Tile<TileType::Vec, T, ROWS, COLS, BLayout::RowMajor, ValidRows, 1>`
  - 数据类型：`half`、`float`。
  - DType 一致性：`dst.DType == src.DType`。
  - 运行时有效性检查：
    - `srcValidCol != 0` 且 `srcValidRow != 0`。
    - `srcValidRow == dstValidRow`（输出有效行数必须与输入有效行数匹配）。
  - `tmp` 必须与 `src` 具有相同的形状。

## 实现说明

与使用 `vcadd`/`vcgadd` 指令的 TROWSUM 不同，TROWPROD 使用 `vmul` 进行二分归约，因为 A2A3 没有 `vcmul` 指令。实现步骤：

1. 将相邻的 repeat 对相乘，结果存入 `tmp`
2. 对 `tmp` 迭代执行二分乘法归约
3. 持续归约直到每行只剩一个元素

## 示例

### Auto 模式

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
  TROWPROD(dst, src, tmp);
}
```

### Manual 模式

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
  TROWPROD(dst, src, tmp);
}
```

## ASM 形式示例

### Auto 模式

```text
# Auto 模式：编译器/运行时管理的放置和调度。
%dst = pto.trowprod %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual 模式

```text
# Manual 模式：在发出指令前显式绑定资源。
# Tile 操作数可选：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.trowprod %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = trowprod %src : !pto.tile<...> -> !pto.tile<...>
# IR Level 2 (DPS)
pto.trowprod ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
