# TROWPROD

## 指令示意图

![TROWPROD tile operation](../figures/isa/TROWPROD.svg)

## 简介

对每行元素进行乘积归约。

## 数学定义

设 `R = src.GetValidRow()` 且 `C = src.GetValidCol()`。对于 `0 <= i < R`：

$$ \mathrm{dst}_{i,0} = \prod_{j=0}^{C-1} \mathrm{src}_{i,j} $$

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

同步形式：

```text
%dst = trowprod %src : !pto.tile<...> -> !pto.tile<...>
```

降级可能引入内部临时 tile；C++ 内建函数需要显式的 `tmp` 操作数。

### AS Level 1 (SSA)

```text
%dst = pto.trowprod %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.trowprod ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建函数

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWPROD(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents &... events);
```

## 约束条件

### 通用约束或检查

- `dst` 和 `src` 必须均为 `TileType::Vec`。
- `src` 必须使用标准 ND 布局：行主且非分形（`BLayout::RowMajor`、`SLayout::NoneBox`）。
- `dst` 必须使用以下两种非分形布局之一：
  - ND 布局（`BLayout::RowMajor`、`SLayout::NoneBox`），或
  - 列数严格为 1 的 DN 布局（`BLayout::ColMajor`、`SLayout::NoneBox`、`Cols == 1`）。
- `dst` 和 `src` 的元素类型必须一致。
- 运行时有效区域检查：
  - `src.GetValidRow() != 0`
  - `src.GetValidCol() != 0`
  - `src.GetValidRow() == dst.GetValidRow()`
- 内建接口签名要求显式传入 `tmp` 操作数。

### A5 实现检查

- 支持的元素类型：`half`、`float`、`int32_t`、`int16_t`。
- 当前检查到的实现路径中，实际受约束的是 `src` 和 `dst`。
- 当前实现路径中，没有额外要求 `tmp` 必须满足特定 shape/layout 约束。

## 实现说明

`TROWPROD` 在当前代码库中遵循已实现的 A5 后端路径。该实现会在校验 `src` / `dst` 约束后，直接完成按行乘积归约。

C++ 内建接口中仍然保留 `tmp` 参数，以保持接口形式一致：

1. `tmp` 仍然保留在内建接口签名和 AS lowering 形式中。
2. 当前检查到的实现路径中，实际被约束的是 `src` 和 `dst`。
3. 如果后续该指令的其他后端实现对 `tmp` 引入额外要求，文档应再按对应实现同步更新。

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
# AS Level 2 (DPS)
pto.trowprod ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
