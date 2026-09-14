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

降低时可能引入内部临时Tile；C++内建接口需要显式传入 `tmp` 操作数。

### AS Level 1（SSA）

```text
%dst = pto.trowmin %src, %tmp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.trowmin ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWMIN(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents &... events);
```

## 约束

### 通用约束或检查

- `dst` 和 `src` 必须均为 `TileType::Vec`。
- `src` 必须使用标准ND布局：行主且非分形（`BLayout::RowMajor`、`SLayout::NoneBox`）。
- `dst` 必须使用以下两种非分形布局之一：
    - ND布局（`BLayout::RowMajor`、`SLayout::NoneBox`），或
    - 列数严格为1的DN布局（`BLayout::ColMajor`、`SLayout::NoneBox`、`Cols == 1`）。
- `dst` 和 `src` 的元素类型必须一致。
- 运行时有效区域检查：
    - `src.GetValidRow() != 0`
    - `src.GetValidCol() != 0`
    - `src.GetValidRow() == dst.GetValidRow()`
- 内建接口签名要求显式传入 `tmp` 操作数。

### Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品实现检查

- 支持的元素类型：`half`、`float`、`int32_t`、`int16_t`。

### Ascend 950PR/Ascend 950DT实现检查

- 支持的元素类型：`half`、`float`、`int32_t`、`int64_t`、`uint64_t`、`int16_t`、`int8_t`、`uint8_t`。
- 对于 `int64_t` / `uint64_t`：
    - 输出有效列数应为 1；只写入每个有效行的第 0 列，保留其余物理填充。
    - ND 输出要求物理 `Cols % 4 == 0`；DN 输出要求物理 `Cols == 1` 且 `Rows % 4 == 0`。有效行数不必是 4 的倍数。
    - 行步长由物理形状决定，见[形状与布局约定](conventions_zh.md)。

## 临时空间

### Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品

`tmp` **被使用**作为行最小值归约的暂存存储。

- 对于**整数**类型（`int32_t`、`int16_t`）：`tmp` 用作逐行累加器缓冲区（1个块）。对于每一行，`tmp` 初始化为最大可表示值，然后通过 `vmin` 累加 `src` 的各个块。最终最小值在标量模式下从 `tmp` 读取。
  - `tmp` 大小：至少1行和 `BLOCK_BYTE_SIZE / sizeof(T)` 列（`int32_t` 为8，`int16_t` 为16）。
- 对于**浮点**类型（`float`、`half`）：`tmp` 用于通过 `vcmin`/`vcgmin` 的二叉树归约。
  - 安全的默认设置：将 `tmp` 设为与 `src` 相同的形状。

### Ascend 950PR/Ascend 950DT

`tmp` 被接口接受但不使用。64 位整数使用精确整数归约，不经过浮点转换。

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

### 64 位 ND 输出（Ascend 950PR/Ascend 950DT）

输出物理形状为 `[64,4]`、有效形状为 `[64,1]`，每隔 32 字节写入一个 8 字节结果。

```cpp
#include <cstdint>
#include <pto/pto-inst.hpp>

using namespace pto;

void example_int64() {
  using SrcT = Tile<TileType::Vec, int64_t, 64, 16>;
  using DstT = Tile<TileType::Vec, int64_t, 64, 4, BLayout::RowMajor, 64, 1>;
  SrcT src, tmp;
  DstT dst;
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

### PTO汇编形式

```text
%dst = trowmin %src : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.trowmin ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
