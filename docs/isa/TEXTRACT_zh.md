# TEXTRACT

## 指令示意图

![TEXTRACT tile operation](../figures/isa/TEXTRACT.svg)

## 简介

从较大的源Tile中提取较小的子Tile。

## 数学语义

概念上从较大的 `src` Tile中，以 `(indexRow, indexCol)` 为起点复制一个较小窗口到 `dst`。确切的映射取决于tile布局。

设 `R = dst.GetValidRow()` 和 `C = dst.GetValidCol()`。对于 `0 <= i < R` 和 `0 <= j < C`：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{\mathrm{indexRow}+i,\; \mathrm{indexCol}+j} $$

## 汇编语法

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

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

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
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, FpTileData &fp,
                              uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);
```

`TEXTRACT_FP(...)` 为历史 fp 量化形式保留源码兼容入口，并直接映射到无 `mode` 的
`TEXTRACT_IMPL(dst, src, fp, indexRow, indexCol)` 路径。规范同名 `TEXTRACT(..., fp, ...)`
重载仅在 `FpTileData::Loc == TileType::Scaling` 时参与匹配。
规范接口还提供显式 `AccToVecMode` 形式，用于目标支持的 Acc-to-Vec 路由。

## 约束

### 通用约束或检查

- 对于同 dtype 抽取/布局路径，`DstTileData::DType` 必须等于 `SrcTileData::DType`。
  Acc 转换和量化路径使用下述后端特定 dtype 组合。
- 运行时边界检查：
    - `indexRow + DstTileData::Rows <= SrcTileData::Rows`
    - `indexCol + DstTileData::Cols <= SrcTileData::Cols`

### Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品实现检查

- 支持的元素类型：`int8_t`、`half`、`bfloat16_t`、`float`。
- 源布局必须满足以下已检查到的Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品提取布局之一：
    - `(SFractal == ColMajor && isRowMajor)`，或
    - `(SFractal == RowMajor && !isRowMajor)`。
- 在以 `TileType::Left` 为目标的GEMV场景中，已检查到的源布局还允许 `(SrcTileData::Rows == 1 && SrcTileData::isRowMajor)`。
- 目标必须是 `TileType::Left` 或 `TileType::Right`，并具有目标支持的布局配置。

### Ascend 950PR/Ascend 950DT实现检查

- 支持的元素类型：`int8_t`、`hifloat8_t`、`float8_e5m2_t`、`float8_e4m3_t`、`half`、`bfloat16_t`、`float`、`float4_e2m1x2_t`、`float4_e1m2x2_t`、`float8_e8m0_t`。
- 源布局必须满足以下已检查到的Ascend 950PR/Ascend 950DT提取布局之一：
    - 对于 `Left` / `Right`：`(SFractal == ColMajor && isRowMajor)` 或 `(SFractal == RowMajor && !isRowMajor)`
    - 对于 `ScaleLeft`：`(SFractal == RowMajor && isRowMajor)`
    - 对于 `ScaleRight`：`(SFractal == ColMajor && !isRowMajor)`
- 在以 `Left` 为目标的GEMV场景中，已检查到的源布局还允许 `(SrcTileData::Rows == 1 && SrcTileData::isRowMajor)`。
- 目标支持 `TileType::Mat -> TileType::Left/Right/Scale`、`TileType::Acc -> TileType::Mat`（含relu、标量量化、向量量化形式）、`TileType::Acc -> TileType::Vec`，以及特定的 `TileType::Vec -> TileType::Mat` 提取路径。
- 规范向量量化 `TEXTRACT(..., fp, ...)` 形式额外要求提供 `FpTileData` Scaling 操作数。
  `TEXTRACT_FP(...)` 仍作为源码兼容历史 alias 保留，并由所选后端实现继续检查合法性。
- 向量量化 Acc-to-Vec 形式仅在存在对应后端实现的目标上暴露
  （A5、kirin9030、kirinX90 和 CPU 模拟器），接受
  `mode = AccToVecMode::{SingleModeVec0, SingleModeVec1, DualModeSplitM, DualModeSplitN}`。
- 对于 `TileType::Acc -> TileType::Vec`，当目标为32位类型（`float`/`int32_t`）且使用 `DualModeSplitN` 时，切分前的 `ValidCol` 必须是 `32` 的整数倍。

### Vec → Vec 抽取路径

除上述 `Mat/Acc -> ...` 路径外，`TEXTRACT` 还支持 `TileType::Vec -> TileType::Vec` 抽取路径（ND 与 NZ 布局），由 `CheckTExtractVecToVecCommon` 强制：

- `DstTileData::DType` 必须等于 `SrcTileData::DType`。
- 支持的元素类型（A2A3 与 A5 均同）：`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`half`、`bfloat16_t`、`float`（任意 1/2/4 字节标准类型）。该集合与主 tile 路径不同：新增 `uint8_t`/`int16_t`/`uint16_t`/`int32_t`/`uint32_t`，且在 A5 上**不含** fp8/fp4 类型。
- ND 路径：源/目标行步进须 32 字节对齐；`Dst` 行/列不得超过 `Src`。

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

### PTO汇编形式

```text
%dst = textract %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.textract ins(%src, %idxrow, %idxcol : !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```
