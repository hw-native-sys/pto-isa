# TCOLSUM

## 指令示意图

![TCOLSUM tile operation](../../../../figures/isa/TCOLSUM.svg)

## 简介

通过对行求和来归约每一列。

## 数学语义

设 `R = src.GetValidRow()`，`C = src.GetValidCol()`。对 `0 <= j < C`：

$$ \mathrm{dst}_{0,j} = \sum_{i=0}^{R-1} \mathrm{src}_{i,j} $$

`isBinary` 选择实现路径（二叉树累加 vs. 顺序累加）。

## 汇编语法

同步形式：

```text
%dst = tcolsum %src {isBinary = false} : !pto.tile<...> -> !pto.tile<...>
```
降阶时可能引入内部临时 Tile；C++ 内建接口需要显式传入 `tmp` 操作数。

### AS Level 1（SSA）

```text
%dst = pto.tcolsum %src : !pto.tile<...> -> !pto.tile<...>
%dst = pto.tcolsum %src, %tmp {isBinary = false} : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tcolsum ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
pto.tcolsum ins(%src, %tmp {isBinary = false} : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TCOLSUM(TileDataOut &dst, TileDataIn &src, WaitEvents &... events);

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TCOLSUM(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, bool isBinary, WaitEvents &... events);
```

## 约束

### 通用约束或检查

- `dst` 和 `src` 必须为 `TileType::Vec`。
- `dst` 和 `src` 必须使用标准 ND 布局：行主且非分形（`BLayout::RowMajor`、`SLayout::NoneBox`）。
- `dst` 和 `src` 的元素类型必须一致。
- 运行时检查：
    - `src.GetValidCol() == dst.GetValidCol()`
    - `src.GetValidRow() != 0`（为零时实现静默返回，不执行计算）
    - `src.GetValidCol() != 0`（为零时实现静默返回，不执行计算）
    - `src.GetValidCol()` 必须不大于按 `src` 元素计的 `tmp` 行跨度（即 `tmp.RowStride * sizeof(TmpDType) / sizeof(DType) >= src.GetValidCol()`）
- `isBinary` 选择已检查到的后端路径：
    - `true`：使用 `tmp` 做二叉树累加
    - `false`：直接在 `dst` 上做顺序累加

### A2A3 实现检查

- 支持的元素类型：`half`、`float`、`int16_t`、`int32_t`。
- `tmp` 必须为 `TileType::Vec`，且使用标准 ND 布局：行主且非分形（`BLayout::RowMajor`、`SLayout::NoneBox`）。
- `tmp` 的元素类型必须与 `src` 和 `dst` 一致。
- 若 `src.GetValidRow() == 0` 或 `src.GetValidCol() == 0`，实现会直接返回。

### A5 实现检查

- A5 共享列归约检查允许的元素类型为：`half`、`float`、`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`bfloat16_t`。
- 已检查到的 A5 `TCOLSUM` 路径中，`tmp` 仍只用于二叉累加路径；`TCOLSUM_IMPL` 中没有额外显式加入 `tmp` 的编译期类型/布局断言。

## 临时空间

### 无 `tmp`（2 参数重载：`TCOLSUM(dst, src)`）

不需要 `tmp`。A2A3 和 A5 均使用直接在 `dst` 上的顺序累加。

### 带 `tmp` 和 `isBinary`（4 参数重载：`TCOLSUM(dst, src, tmp, isBinary)`）

#### A2A3

- 当 `isBinary = true` 时：`tmp` **被使用**于二叉树累加。`src` 中相邻行对被求和到 `tmp`，然后 `tmp` 被递归折半直到仅剩单行。
  - `tmp` 必须与 `src`/`dst` 具有相同的元素类型。
  - `tmp` 必须是 `TileType::Vec`，行主序，非分形。
  - `tmp.GetValidCol() >= src.GetValidCol()`（以元素数计，考虑 `tmp` 跨度）。
  - `tmp` 至少需要 `ceil(src.GetValidRow() / 2)` 行。
- 当 `isBinary = false` 时：`tmp` 被接受但实现使用直接在 `dst` 上的顺序累加；`tmp` 未被主动使用。

#### A5

- 当 `isBinary = true` 时：`tmp` **被使用**于基于向量寄存器的二叉树累加（使用 UB 存储）。
  - `tmp` 必须与 `src`/`dst` 具有相同的元素类型。
  - `tmp.GetValidCol() >= src.GetValidCol()`（以元素数计，考虑 `tmp` 跨度）。
  - `tmp` 至少需要 `ceil(src.GetValidRow() / 2)` 行。
- 当 `isBinary = false` 时：`tmp` 未被主动使用；实现使用通过 `TColReduceInstr` 的顺序归约。

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 1, 16>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TCOLSUM(dst, src, tmp, /*isBinary=*/false);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 1, 16>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(tmp, 0x3000);
  TCOLSUM(dst, src, tmp, /*isBinary=*/false);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tcolsum %src : !pto.tile<...> -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tcolsum %src : !pto.tile<...> -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = tcolsum %src {isBinary = false} : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tcolsum ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
