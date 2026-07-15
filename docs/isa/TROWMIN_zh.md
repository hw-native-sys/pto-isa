# TROWMIN

## 指令示意图

![TROWMIN tile operation](../figures/isa/TROWMIN.svg)

## 简介

通过取列间最小值来归约每一行。

## 数学语义

设 `R = src.GetValidRow()`，`C = src.GetValidCol()`。对 `0 <= i < R`：

$$ \mathrm{dst}_{i,0} = \min_{0 \le j < C} \mathrm{src}_{i,j} $$

## 汇编语法

同步形式：

```text
%dst = trowmin %src : !pto.tile<...> -> !pto.tile<...>
```
降低时可能引入内部临时 Tile；C++ 内建接口需要显式传入 `tmp` 操作数。

### AS Level 1（SSA）

```text
%dst = pto.trowmin %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.trowmin ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWMIN(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents &... events);
```

## 约束

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

### A2A3 实现检查

- 支持的元素类型：`half`、`float`、`int32_t`、`int16_t`。
- 实现同时接受 ND 输出和 `Cols == 1` 的 DN 输出。
- 运行时检查遵循共享的行归约检查路径：
    - `src.GetValidRow() != 0`
    - `src.GetValidCol() != 0`
    - `src.GetValidRow() == dst.GetValidRow()`

## 临时空间

### A2A3

`tmp` **被使用**作为行最小值归约的暂存存储。

- 对于**整数**类型（`int32_t`、`int16_t`）：`tmp` 用作逐行累加器缓冲区（1 个块）。对于每一行，`tmp` 初始化为最大可表示值，然后通过 `vmin` 累加 `src` 的各个块。最终最小值在标量模式下从 `tmp` 读取。
  - `tmp` 大小：至少 1 行和 `BLOCK_BYTE_SIZE / sizeof(T)` 列（`int32_t` 为 8，`int16_t` 为 16）。
- 对于**浮点**类型（`float`、`half`）：`tmp` 用于通过 `vcmin`/`vcgmin` 的二叉树归约。
  - 安全的默认设置：将 `tmp` 设为与 `src` 相同的形状。

### A5

`tmp` 被接口接受但 A5 实现**不使用**。A5 后端使用基于向量寄存器的归约（`vcmin` 指令），不需要暂存 Tile 存储。`tmp` 仅为了与 A2A3 的 API 兼容性而保留在 C++ 内建接口签名中。

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
  TROWMIN(dst, src, tmp);
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
  TROWMIN(dst, src, tmp);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.trowmin %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.trowmin %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = trowmin %src : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.trowmin ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
