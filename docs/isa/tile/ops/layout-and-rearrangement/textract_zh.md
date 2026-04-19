# pto.textract

`pto.textract` 属于[布局与重排](../../layout-and-rearrangement_zh.md)指令集。

## 概述

`TEXTRACT` 从较大的源 Tile 中提取一个较小的子 Tile 到目标位置。概念上从较大的 `src` Tile 中，以 `(indexRow, indexCol)` 为起点复制一个较小窗口到 `dst`。确切的映射取决于 tile 布局。

## 机制

设 `R = dst.GetValidRow()` 和 `C = dst.GetValidCol()`。对于 `0 <= i < R` 和 `0 <= j < C`：

$$\mathrm{dst}_{i,j} = \mathrm{src}_{\mathrm{indexRow}+i,\; \mathrm{indexCol}+j}$$

## 语法

### PTO-AS

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
| `src` | 输入 Tile | 源 Tile |
| `indexRow` | 起始行索引 | 从源 Tile 的哪一行开始提取 |
| `indexCol` | 起始列索引 | 从源 Tile 的哪一列开始提取 |
| `dst` | 输出 Tile | 目标 Tile |

## 预期输出

| 结果 | 类型 | 说明 |
| --- | --- | --- |
| `dst` | Tile | 从源 Tile 指定位置提取的子 Tile |

## 副作用

提取的子 Tile 会被写入到 `dst` 中，`dst` 原来的内容会被覆盖。

## 约束

- `DstTileData::DType` 必须等于 `SrcTileData::DType`
- 运行时边界检查：`indexRow + DstTileData::Rows <= SrcTileData::Rows`
- 运行时边界检查：`indexCol + DstTileData::Cols <= SrcTileData::Cols`
- A2/A3 支持的元素类型为 `int8_t`、`half`、`bfloat16_t`、`float`
- A2/A3 源布局必须满足 `(SFractal == ColMajor && isRowMajor)` 或 `(SFractal == RowMajor && !isRowMajor)`，在以 `TileType::Left` 为目标的 GEMV 场景中还允许 `(SrcTileData::Rows == 1 && SrcTileData::isRowMajor)`
- A2/A3 目标必须是 `TileType::Left` 或 `TileType::Right`
- A5 支持的元素类型为 `int8_t`、`hifloat8_t`、`float8_e5m2_t`、`float8_e4m3_t`、`half`、`bfloat16_t`、`float`、`float4_e2m1x2_t`、`float4_e1m2x2_t`、`float8_e8m0_t`
- A5 源布局对于 `Left` / `Right` 必须满足 `(SFractal == ColMajor && isRowMajor)` 或 `(SFractal == RowMajor && !isRowMajor)`，对于 `ScaleLeft` 必须满足 `(SFractal == RowMajor && isRowMajor)`，对于 `ScaleRight` 必须满足 `(SFractal == ColMajor && !isRowMajor)`
- A5 目标支持 `TileType::Mat -> TileType::Left/Right/Scale`、`TileType::Acc -> TileType::Mat`（含 relu、标量量化、向量量化形式），以及特定的 `TileType::Vec -> TileType::Mat` 提取路径

## 异常与非法情形

- 如果提取区域超出源 Tile 范围，行为未定义
- 如果源/目标布局不兼容，backend 可能报错

## Target-Profile 限制

| 特性 | CPU Simulator | A2/A3 | A5 |
| --- | :---: | :---: | :---: |

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

## 相关页面

- [TEXTRACT_FP](./textract-fp_zh.md)
- 指令集总览：[布局与重排](../../layout-and-rearrangement_zh.md)
