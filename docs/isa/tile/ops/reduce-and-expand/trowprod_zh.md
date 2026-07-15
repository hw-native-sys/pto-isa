# TROWPROD

## 指令示意图

![TROWPROD tile operation](../../../../figures/isa/TROWPROD.svg)

## 简介

对每行元素进行乘积归约。

## 数学定义

设 `R = src.GetValidRow()` 且 `C = src.GetValidCol()`。对于 `0 <= i < R`：

$$ \mathrm{dst}_{i,0} = \prod_{j=0}^{C-1} \mathrm{src}_{i,j} $$

## 汇编语法

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
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

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

## 临时空间

### A2A3

`tmp` **被使用**作为逐行累加器缓冲区。对于每一行，实现将 `tmp` 初始化为 `1.0`，然后使用 `vmul` 将 `src` 数据的各个块乘入 `tmp`。所有块累加完成后，标量模式流水线读取 `tmp` 元素并计算最终乘积。

- `tmp` 必须与 `src`/`dst` 具有相同的元素类型。
- `tmp` 大小：至少 1 行和 `BLOCK_BYTE_SIZE / sizeof(T)` 列（即 1 个块：`float`/`int32_t` 为 8 个元素，`half`/`int16_t` 为 16 个元素）。
- 安全的默认设置：将 `tmp` 设为与 `src` 相同的形状。

### A5

`tmp` 被接口接受但 A5 实现**不使用**。A5 后端使用基于向量寄存器的归约（`vmul` + `vintlv` 进行树形归约），不需要暂存 Tile 存储。`tmp` 仅为了与 A2A3 的 API 兼容性而保留在 C++ 内建接口签名中。

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
