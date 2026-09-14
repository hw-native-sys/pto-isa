# TCMP

## 指令示意图

![TCMP tile operation](../figures/isa/TCMP.svg)

## 简介

比较两个Tile并写入一个打包的谓词掩码。

## 数学语义

概念上，对于有效区域中的每个元素 `(i, j)`，定义一个谓词：

$$ p_{i,j} = \left(\mathrm{src0}_{i,j}\ \mathrm{cmpMode}\ \mathrm{src1}_{i,j}\right) $$

谓词掩码使用实现定义的打包布局存储在 `dst` 中。

## 汇编语法

同步形式：

```text
%dst = tcmp %src0, %src1 {cmpMode = #pto.cmp<EQ>} : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tcmp %src0, %src1{cmpMode = #pto<cmp xx>}: (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tcmp ins(%src0, %src1{cmpMode = #pto<cmp xx>}: !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp` 和 `include/pto/common/type.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCMP(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, CmpMode cmpMode, WaitEvents &... events);
```

## 约束

- **实现检查 （Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品）**:
    - 输入类型必须是以下之一：`int32_t`、`half`、`float`。
    - 输出类型必须是 `uint8_t`。
    - `src0/src1/dst` tile位置必须是 `TileType::Vec`。
    - 静态有效边界：`TileDataSrc::ValidRow <= TileDataSrc::Rows` 且 `TileDataSrc::ValidCol <= TileDataSrc::Cols`。
    - 运行时：`src0` 与 `src1` 的有效行列数分别相等，且 `src0.GetValidRow() == dst.GetValidRow()`。
    - 目标有效列数表示打包容量，不要求等于源有效列数。
    - 对于 `int32_t` 输入，支持 `EQ` 和 `NE`；`NE` 对相等比较结果取反，其他模式使用 `EQ` 路径。
- **实现检查 (Ascend 950PR/Ascend 950DT)**:
    - 输入类型必须是以下之一：`uint32_t`、`int32_t`、`int64_t`、`uint64_t`、`uint16_t`、`int16_t`、`uint8_t`、`int8_t`、`float`、`half`、`bfloat16_t`。
    - 输出为打包谓词字节，可使用 RowMajor `uint8_t` 掩码 Tile。
    - 已实现（参见 `include/pto/npu/a5/TCmp.hpp`）。
    - 迭代域为 `src0.GetValidRow()` / `src0.GetValidCol()`；`src1` 须提供对应有效元素。目标有效列数表示打包容量，不决定比较次数。
- **掩码编码**:
    - 64 位输入（Ascend 950PR/Ascend 950DT）：列 `j` 的比较结果存于该行第 `j / 8` 字节的第 `j % 8` 位，低位在前。
    - 对 `uint8_t` 掩码，有效形状可设为 `[R, ceil(C / 8)]`，物理 `Cols` 按 32 字节对齐，其中 `[R,C]` 为源有效形状。行地址按目标物理步长计算；最后一个有效位之后的填充值未指定。
    - 掩码tile被解释为目标定义布局中的打包谓词位。

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using MaskT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  SrcT src0, src1;
  MaskT mask(16, 2);
  TCMP(mask, src0, src1, CmpMode::GT);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using MaskT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  SrcT src0, src1;
  MaskT mask(16, 16);
  TASSIGN(src0, 0x1000);
  TASSIGN(src1, 0x2000);
  TASSIGN(mask, 0x3000);
  TCMP(mask, src0, src1, CmpMode::GT);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tcmp %src0, %src1{cmpMode = #pto<cmp xx>}: (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tcmp %src0, %src1{cmpMode = #pto<cmp xx>}: (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO汇编形式

```text
%dst = tcmp %src0, %src1 {cmpMode = #pto.cmp<EQ>} : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tcmp ins(%src0, %src1{cmpMode = #pto<cmp xx>}: !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
