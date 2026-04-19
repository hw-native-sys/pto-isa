# pto.textract

`pto.textract` 属于[布局与重排](./tile/ops/layout-and-rearrangement/textract_zh.md)指令集。

## 概述

从较大的源 Tile 中提取较小的子 Tile，概念上从较大的 `src` Tile 中，以 `(indexRow, indexCol)` 为起点复制一个较小窗口到 `dst`，确切的映射取决于 tile 布局。

## 机制

设 `R = dst.GetValidRow()` 和 `C = dst.GetValidCol()`。对于 `0 <= i < R` 和 `0 <= j < C`：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{\mathrm{indexRow}+i,\; \mathrm{indexCol}+j} $$

## 语法

### PTO-AS

同步形式：

```text
%dst = textract %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```mlir
%dst = pto.textract %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

### AS Level 2（DPS）

```mlir
pto.textract ins(%src, %idxrow, %idxcol : !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);
```

## 输入

| 操作数 | 角色 | 说明 |
| --- | --- | --- |
| `src` | - | 源 Tile |
| `indexRow` | - | 行偏移量 |
| `indexCol` | - | 列偏移量 |
| `fp` | - | FP 缩放 Tile（TEXTRACT_FP 形式） |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `dst` | - | 提取后的子 Tile |

## 副作用

提取操作不会产生额外的副作用。

## 约束

### 通用约束或检查

- `DstTileData::DType` 必须等于 `SrcTileData::DType`。
- 运行时边界检查：
    - `indexRow + DstTileData::Rows <= SrcTileData::Rows`
    - `indexCol + DstTileData::Cols <= SrcTileData::Cols`

### A2A3 实现检查

- 支持的元素类型：`int8_t`、`half`、`bfloat16_t`、`float`。
- 源布局必须满足以下已检查到的 A2A3 提取布局之一：
    - `(SFractal == ColMajor && isRowMajor)`，或
    - `(SFractal == RowMajor && !isRowMajor)`。
- 在以 `TileType::Left` 为目标的 GEMV 场景中，已检查到的源布局还允许 `(SrcTileData::Rows == 1 && SrcTileData::isRowMajor)`。
- 目标必须是 `TileType::Left` 或 `TileType::Right`，并具有目标支持的布局配置。

### A5 实现检查

- 支持的元素类型：`int8_t`、`hifloat8_t`、`float8_e5m2_t`、`float8_e4m3_t`、`half`、`bfloat16_t`、`float`、`float4_e2m1x2_t`、`float4_e1m2x2_t`、`float8_e8m0_t`。
- 源布局必须满足以下已检查到的 A5 提取布局之一：
    - 对于 `Left` / `Right`：`(SFractal == ColMajor && isRowMajor)` 或 `(SFractal == RowMajor && !isRowMajor)`
    - 对于 `ScaleLeft`：`(SFractal == RowMajor && isRowMajor)`
    - 对于 `ScaleRight`：`(SFractal == ColMajor && !isRowMajor)`
- 在以 `Left` 为目标的 GEMV 场景中，已检查到的源布局还允许 `(SrcTileData::Rows == 1 && SrcTileData::isRowMajor)`。
- 目标支持 `TileType::Mat -> TileType::Left/Right/Scale`、`TileType::Acc -> TileType::Mat`（含 relu、标量量化、向量量化形式），以及特定的 `TileType::Vec -> TileType::Mat` 提取路径。
- 向量量化形式额外要求提供 `FpTileData` 缩放操作数，对应 `TEXTRACT_FP(...)` 接口。

## 异常与非法情形

- 当 `DstTileData::DType` 与 `SrcTileData::DType` 不匹配时行为未定义。
- 当 `indexRow + DstTileData::Rows > SrcTileData::Rows` 时行为未定义。
- 当 `indexCol + DstTileData::Cols > SrcTileData::Cols` 时行为未定义。

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| int8_t 提取 | ✓ | ✓ | ✓ |
| float 提取 | ✓ | ✓ | ✓ |
| bfloat16 提取 | ✓ | ✓ | ✓ |
| 向量量化形式 | ✗ | ✗ | ✓ |

## 示例

### C++ 自动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Mat, float, 16, 16, BLayout::RowMajor, 16, 16, SLayout::ColMajor>;
  using DstT = TileLeft<float, 16, 16>;
  SrcT src;
  DstT dst;
  TEXTRACT(dst, src, /*indexRow=*/0, /*indexCol=*/0);
}
```

### C++ 手动模式

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Mat, float, 16, 16, BLayout::RowMajor, 16, 16, SLayout::ColMajor>;
  using DstT = TileLeft<float, 16, 16>;
  SrcT src;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TEXTRACT(dst, src, /*indexRow=*/0, /*indexCol=*/0);
}
```

### PTO-AS

```text
%dst = textract %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
```

AS Level 2 (DPS)：

```mlir
pto.textract ins(%src, %idxrow, %idxcol : !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```

![TEXTRACT tile operation](../figures/isa/TEXTRACT.svg)

## 相关页面

- 指令集总览：[布局与重排](./tile/ops/layout-and-rearrangement/textract_zh.md)
