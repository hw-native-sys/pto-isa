# TINSERT

## 指令示意图

![TINSERT tile operation](../../../../figures/isa/TINSERT.svg)

## 简介

在 `(indexRow, indexCol)` 偏移处将源子Tile插入目标Tile。概念上是 `TEXTRACT` 的逆操作。

`TINSERT` 用于：

- Acc → Mat插入（含可选的relu、标量量化或向量量化）
- Acc → Vec插入（含可选的 `AccToVecMode`、relu、标量量化或向量量化） *(Ascend 950PR/Ascend 950DT)*
- Vec → Mat插入（ND和NZ布局） *(Ascend 950PR/Ascend 950DT)*
- Vec → Vec插入（ND和NZ布局） *(Ascend 950PR/Ascend 950DT)*
- NZ split插入（`SPLIT2`、`SPLIT4`） *(Ascend 950PR/Ascend 950DT)*

## 数学语义

设 `R = src.GetValidRow()` 和 `C = src.GetValidCol()`。对于 `0 <= i < R` 和 `0 <= j < C`：

$$
\mathrm{dst}_{\mathrm{indexRow}+i,\;\mathrm{indexCol}+j} = \mathrm{src}_{i,j}
$$

## 汇编语法

同步形式：

```text
%dst = tinsert %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tinsert %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tinsert ins(%src, %idxrow, %idxcol : !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                             uint16_t indexRow, uint16_t indexCol,
                             WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode,
          typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                             uint16_t indexRow, uint16_t indexCol,
                             WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, AccToVecMode mode,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                             uint16_t indexRow, uint16_t indexCol,
                             WaitEvents &... events);

template <typename DstTileData, typename SrcTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                             uint64_t preQuantScalar,
                             uint16_t indexRow, uint16_t indexCol,
                             WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, AccToVecMode mode,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                             uint64_t preQuantScalar,
                             uint16_t indexRow, uint16_t indexCol,
                             WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename FpTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TINSERT_FP(DstTileData &dst, SrcTileData &src,
                                FpTileData &fp,
                                uint16_t indexRow, uint16_t indexCol,
                                WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename FpTileData,
          AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                             FpTileData &fp,
                             uint16_t indexRow, uint16_t indexCol,
                             WaitEvents &... events);

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRINX90)
template <TInsertMode mode, typename DstTileData, typename SrcTileData,
          typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                             uint16_t indexRow = 0, uint16_t indexCol = 0,
                             WaitEvents &... events);
#endif
```

## 约束

### 通用约束 / 检查

- `TINSERT` 有以下重载族：
    - 普通插入：`TINSERT(dst, src, indexRow, indexCol)`
    - relu形式：`TINSERT<..., reluMode>(dst, src, indexRow, indexCol)`
    - 累加器到向量形式：`TINSERT<..., mode, reluMode>(dst, src, indexRow, indexCol)`
    - 标量量化形式：`TINSERT<..., reluMode>(dst, src, preQuantScalar, indexRow, indexCol)` 和 `TINSERT<..., mode, reluMode>(dst, src, preQuantScalar, indexRow, indexCol)`
    - 向量量化形式：`TINSERT_FP<..., reluMode>(dst, src, fp, indexRow, indexCol)` 和 `TINSERT<..., FpTileData, mode, reluMode>(dst, src, fp, indexRow, indexCol)`
    - NZ split形式 *(仅Ascend 950PR/Ascend 950DT/kirin9030/kirinX90)*：`TINSERT<TInsertMode::SPLIT2>(dst, src, indexRow, indexCol)` 或 `TINSERT<TInsertMode::SPLIT4>(...)`
- `reluMode` 取值为 `ReluPreMode::{NoRelu, NormalRelu}`。
- `mode` 取值为 `AccToVecMode::{SingleModeVec0, SingleModeVec1, DualModeSplitM, DualModeSplitN}`。
- 运行时边界：`indexRow + src.ValidRow <= dst.Rows` 且 `indexCol + src.ValidCol <= dst.Cols`。

### Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品实现检查

- 支持的tile类型对：仅 `TileType::Acc → TileType::Mat`。
- 源布局必须为 `(BFractal: ColMajor, SFractal: RowMajor)`。
- 目标布局必须为 `(BFractal: ColMajor, SFractal: RowMajor)`，且 `SFractalSize == 512`。
- `Dst.Cols * sizeof(DstDType)` 必须是 `32` 字节的倍数且非零。
- **普通 / relu**（非量化）支持的dtype对：
    - `float` Acc → `half`、`bfloat16_t`
- **标量量化** 支持的dtype对：
    - `float` Acc → `int8_t`
    - `int32_t` Acc → `int8_t`、`uint8_t`、`half`、`int16_t`
- **向量量化**（`TINSERT_FP`）支持的dtype对：
    - `float` Acc → `int8_t`、`uint8_t`
    - `int32_t` Acc → `int8_t`、`uint8_t`、`half`、`int16_t`
- 向量量化要求提供 `FpTileData` 缩放操作数（`TileType::Scaling`）。

### Ascend 950PR/Ascend 950DT实现检查

除了Acc → Mat路径外，Ascend 950PR/Ascend 950DT还支持Acc → Vec、Vec → Vec、Vec → Mat和NZ split路径。

- **Acc → Mat**（`TileType::Acc → TileType::Mat`）：
    - 源Acc类型必须是 `float` 或 `int32_t`；源布局必须为 `(BFractal: ColMajor, SFractal: RowMajor)`。
    - 目标布局必须为 `(!isRowMajor, SFractal: RowMajor)`（NZ格式）。
    - **非量化**（普通 / relu）目标类型：
        - `float` Acc → `half`、`bfloat16_t`、`float`
        - `int32_t` Acc → `int32_t`
    - **标量量化** 目标类型：
        - `float` Acc → `int8_t`、`uint8_t`、`hifloat8_t`、`half`、`bfloat16_t`、`float8_e4m3_t`
        - `int32_t` Acc → `int8_t`、`uint8_t`、`half`、`bfloat16_t`
    - **向量量化**（`TINSERT_FP`）目标类型：与标量量化相同。

- **Acc → Vec**（`TileType::Acc → TileType::Vec`）：
    - 源Acc类型必须是 `float` 或 `int32_t`；源布局必须为 `(BFractal: ColMajor, SFractal: RowMajor)`。
    - **非量化**（普通 / relu）目标类型：
        - `float` Acc → `half`、`bfloat16_t`、`float`
        - `int32_t` Acc → `int32_t`
    - **标量量化** 目标类型：
        - `float` Acc → `int8_t`、`uint8_t`、`hifloat8_t`、`half`、`bfloat16_t`、`float8_e4m3_t`
        - `int32_t` Acc → `int8_t`、`uint8_t`、`half`、`bfloat16_t`
    - **向量量化**（`TINSERT_FP` / `TINSERT` 含 `FpTileData`）目标类型：与标量量化相同。
    - 目标布局必须为以下之一：NZ-to-NZ（`!isRowMajor, SFractal: RowMajor`）、NZ-to-ND（`isRowMajor, SFractal: NoneBox`）或NZ-to-DN（`!isRowMajor, SFractal: NoneBox`）。
    - `AccToVecMode` 选择 `SingleModeVec0`、`SingleModeVec1`、`DualModeSplitM` 或 `DualModeSplitN`。
    - 双目标模式（`DualModeSplitM`、`DualModeSplitN`）要求 `QuantMode_t::NoQuant` 且不支持NZ-to-DN路径。
    - 对于32位目标类型（`float`/`int32_t`），使用 `DualModeSplitN` 时切分前的 `ValidCol` 必须是 `32` 的整数倍。
    - 目标stride必须非零且 `dstStride * sizeof(dstType)` 必须是 `32` 字节的倍数。

- **Vec → Vec**（`TileType::Vec → TileType::Vec`）：
    - `DstTileData::DType` 必须等于 `SrcTileData::DType`。
    - 支持的元素类型：`half`、`bfloat16_t`、`float`、`int32_t`、`int8_t`、`hifloat8_t`、`float8_e4m3_t`、`float8_e5m2_t`、`float8_e8m0_t`、`float4_e2m1x2_t`、`float4_e1m2x2_t`。
    - 源和目标布局必须匹配（均ND或均NZ）。
    - ND路径：源有效区域必须在目标边界内。分发选择 `copy_ubuf_to_ubuf`（对齐）、`vlds`/`vsts`（stride对齐、未对齐validCol）、`vlds`/`vstus`（未对齐stride或indexCol）或标量拷贝（1×1元素）。
    - NZ路径：源列数不得超目标列数。使用 `ComputeNZBlockParams` 进行分形块 `copy_ubuf_to_ubuf`。

- **Vec → Mat**（`TileType::Vec → TileType::Mat`，UB → L1）：
    - `DstTileData::DType` 必须等于 `SrcTileData::DType`。
    - 支持的元素类型：`half`、`bfloat16_t`、`float`、`int32_t`、`int8_t`、`hifloat8_t`、`float8_e4m3_t`、`float8_e5m2_t`、`float8_e8m0_t`、`float4_e2m1x2_t`、`float4_e1m2x2_t`。
    - ND路径：源必须为 `isRowMajor`；使用 `copy_ubuf_to_cbuf`。每行数据字节数必须与 `BLOCK_BYTE_SIZE`（32字节）对齐。
    - NZ路径：源必须为 `(!isRowMajor, SFractal: RowMajor)`；使用 `ComputeNZBlockParams` 进行分形块 `copy_ubuf_to_cbuf`。

- **NZ Split**（`TInsertMode::SPLIT2` / `TInsertMode::SPLIT4`，仅Ascend 950PR/Ascend 950DT/kirin9030/kirinX90）：
    - 目标必须为 `TileType::Mat`；源必须为 `TileType::Vec`。
    - `DstTileData::DType` 必须等于 `SrcTileData::DType`。
    - 源必须为NZ格式：`(!isRowMajor, SFractal: RowMajor)`。
    - 支持的元素类型：`half`、`bfloat16_t`、`float`、`int32_t`、`int8_t`、`hifloat8_t`、`float8_e4m3_t`、`float8_e5m2_t`、`float8_e8m0_t`、`float4_e2m1x2_t`、`float4_e1m2x2_t`。
    - `validRow` 对齐到 `FRACTAL_NZ_ROW`（16）用于burst计算。
    - 将 `copy_ubuf_to_cbuf` 的总burst拆分为2或4个子传输，每个处理 `totalBurstNum / SplitCount` 列块。

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

// Vec -> Mat 插入（NZ 布局）
void example_auto() {
  using SrcT = Tile<TileType::Vec, half, 16, 32, BLayout::ColMajor, 16, 32, SLayout::RowMajor>;
  using DstT = Tile<TileType::Mat, half, 16, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
  SrcT src;
  DstT dst(16, 32);
  TINSERT(dst, src, /*indexRow=*/0, /*indexCol=*/0);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

// Vec -> Mat 插入（NZ 布局，手动缓冲分配）
void example_manual() {
  using SrcT = Tile<TileType::Vec, half, 16, 32, BLayout::ColMajor, 16, 32, SLayout::RowMajor>;
  using DstT = Tile<TileType::Mat, half, 16, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
  SrcT src;
  DstT dst(16, 32);
  TASSIGN(src, 0x0);
  TASSIGN(dst, 0x0);
  TINSERT(dst, src, /*indexRow=*/0, /*indexCol=*/0);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tinsert %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tinsert %src, %idxrow, %idxcol : (!pto.tile<...>, dtype, dtype) -> !pto.tile<...>
```

### PTO汇编形式

```text
%dst = tinsert %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tinsert ins(%src, %idxrow, %idxcol : !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```
