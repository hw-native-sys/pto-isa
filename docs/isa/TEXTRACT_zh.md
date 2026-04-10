# TEXTRACT

## 指令示意图

![TEXTRACT tile operation](../figures/isa/TEXTRACT.svg)

## 简介

从较大的源 Tile 中提取较小的子 Tile。

## 数学语义

概念上从较大的 `src` Tile 中，以 `(indexRow, indexCol)` 为起点复制一个较小窗口到 `dst`。确切的映射取决于 tile 布局。

设 `R = dst.GetValidRow()` 和 `C = dst.GetValidCol()`。对于 `0 <= i < R` 和 `0 <= j < C`：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{\mathrm{indexRow}+i,\; \mathrm{indexCol}+j} $$

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

同步形式：

```text
%dst = textract %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.textract %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
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

## 示例

### 自动（Auto）

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

### 手动（Manual）

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

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.textract %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.textract %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = textract %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.textract ins(%src, %idxrow, %idxcol : !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```

