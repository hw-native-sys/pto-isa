# pto.ttrans

`pto.ttrans` 属于[布局与重排](../../layout-and-rearrangement_zh.md)指令集。

## 概述

使用实现定义的临时 Tile 进行转置。对于二维 Tile，在有效转置域上满足 `dst_{i,j} = src_{j,i}`。确切的形状/布局及转置域取决于目标硬件。

## 机制

对于二维 Tile，在有效转置域上：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{j,i} $$

确切的形状/布局及转置域取决于目标硬件。

## 语法

### PTO-AS

参见 [PTO-AS 规范](../../../../assembly/PTO-AS_zh.md)。

同步形式：

```text
%dst = ttrans %src : !pto.tile<...> -> !pto.tile<...>
```

降低时可能引入内部临时 Tile；C++ 内建接口需要显式传入 `tmp` 操作数。

### AS Level 1（SSA）

```mlir
%dst = pto.ttrans %src : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.ttrans ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TTRANS(TileDataDst &dst, TileDataSrc &src, TileDataTmp &tmp, WaitEvents & ... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| dst | 输出 Tile | 目标 Tile |
| src | 输入 Tile | 源 Tile |
| tmp | 临时 Tile | C++ API 需要 tmp，但某些实现可能不使用它 |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| dst | Tile | 转置后的 Tile |

## 副作用

确切的形状/布局及转置域取决于目标硬件。

## 约束

A2/A3 实现检查：`sizeof(TileDataSrc::DType) == sizeof(TileDataDst::DType)`，源布局必须是行主序（`TileDataSrc::isRowMajor`），元素大小必须是 `1`、`2` 或 `4` 字节。支持按元素宽度限制的元素类型：4 字节为 `uint32_t`、`int32_t`、`float`；2 字节为 `uint16_t`、`int16_t`、`half`、`bfloat16_t`；1 字节为 `uint8_t`、`int8_t`。转置大小取自 `src.GetValidRow()` / `src.GetValidCol()`。

A5 实现检查：`sizeof(TileDataSrc::DType) == sizeof(TileDataDst::DType)`，对输入和输出的主维度强制执行 32 字节对齐约束（行主序检查 `Cols * sizeof(T) % 32 == 0`，列主序检查 `Rows * sizeof(T) % 32 == 0`）。支持按元素宽度限制的元素类型：4 字节为 `uint32_t`、`int32_t`、`float`；2 字节为 `uint16_t`、`int16_t`、`half`、`bfloat16_t`；1 字节为 `uint8_t`、`int8_t`。实现在静态 Tile 形状（`TileDataSrc::Rows/Cols`）上运算，不参考 `GetValidRow/GetValidCol`。

临时 Tile：C++ API 需要 `tmp`，但某些实现可能不使用它。

ConvTile：支持在 `TileType::Vec` 上的 ConvTile 格式转换，元素大小必须是 `1`、`2` 或 `4` 字节。元素类型限制为 `uint32_t`、`int32_t`、`float`、`uint16_t`、`int16_t`、`half`、`bfloat16_t`、`uint8_t`、`int8_t`。支持 ConvTile 从 `NCHW` 到 `NC1HWC0` 的变换，其中 `C1 == (C + C0 - 1)/C0`，HW 满足对齐要求，即 `H*W*sizeof(T)==0`，C0 对应 `c0_size`，即 `C0 * sizeof(T) == 32`，C0 也可以为 4。支持 ConvTile 从 `NC1HWC0` 到 `FRACTAL_Z` 的变换，其中 `N1 == (N + N0 - 1)/N0`，N0 为 16。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| 源布局要求 | - | 行主序 | - |
| 32 字节对齐 | - | - | 是 |

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TTRANS(dst, src, tmp);
}
```

### C++ 手动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(tmp, 0x3000);
  TTRANS(dst, src, tmp);
}
```

### PTO-AS

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.ttrans %src : !pto.tile<...> -> !pto.tile<...>
# 手动模式：先显式绑定资源，再发射指令。
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.ttrans %src : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.ttrans ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## 相关页面

- [布局与重排指令集](../../layout-and-rearrangement_zh.md)

![TTRANS tile operation](../../../../figures/isa/TTRANS.svg)
