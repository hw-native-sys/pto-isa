# TINSERT

## 指令示意图

![TINSERT tile operation](../../../../figures/isa/TINSERT.svg)

## 简介

在 `(indexRow, indexCol)` 偏移处将源子 Tile 插入目标 Tile。概念上是 `TEXTRACT` 的逆操作。

`TINSERT` 用于：

- Acc → Mat 插入（含可选的 relu、标量量化或向量量化）
- Acc → Vec 插入（含可选的 `AccToVecMode`、relu、标量量化或向量量化） *(A5)*
- Vec → Mat 插入（ND 和 NZ 布局） *(A5)*
- Vec → Vec 插入（ND 和 NZ 布局） *(A5)*
- NZ split 插入（`SPLIT2`、`SPLIT4`） *(A5)*

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

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

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
    - relu 形式：`TINSERT<..., reluMode>(dst, src, indexRow, indexCol)`
    - 累加器到向量形式：`TINSERT<..., mode, reluMode>(dst, src, indexRow, indexCol)`
    - 标量量化形式：`TINSERT<..., reluMode>(dst, src, preQuantScalar, indexRow, indexCol)` 和 `TINSERT<..., mode, reluMode>(dst, src, preQuantScalar, indexRow, indexCol)`
    - 向量量化形式：`TINSERT_FP<..., reluMode>(dst, src, fp, indexRow, indexCol)` 和 `TINSERT<..., FpTileData, mode, reluMode>(dst, src, fp, indexRow, indexCol)`
    - NZ split 形式 *(仅 A5/kirin9030/kirinX90)*：`TINSERT<TInsertMode::SPLIT2>(dst, src, indexRow, indexCol)` 或 `TINSERT<TInsertMode::SPLIT4>(...)`
- `reluMode` 取值为 `ReluPreMode::{NoRelu, NormalRelu}`。
- `mode` 取值为 `AccToVecMode::{SingleModeVec0, SingleModeVec1, DualModeSplitM, DualModeSplitN}`。
- 运行时边界：`indexRow + src.ValidRow <= dst.Rows` 且 `indexCol + src.ValidCol <= dst.Cols`。

### A2A3 实现检查

- 支持的 tile 类型对：仅 `TileType::Acc → TileType::Mat`。
- 源布局必须为 `(BFractal: ColMajor, SFractal: RowMajor)`。
- 目标布局必须为 `(BFractal: ColMajor, SFractal: RowMajor)`，且 `SFractalSize == 512`。
- `Dst.Cols * sizeof(DstDType)` 必须是 `32` 字节的倍数且非零。
- **普通 / relu**（非量化）支持的 dtype 对：
    - `float` Acc → `half`、`bfloat16_t`
- **标量量化** 支持的 dtype 对：
    - `float` Acc → `int8_t`
    - `int32_t` Acc → `int8_t`、`uint8_t`、`half`、`int16_t`
- **向量量化**（`TINSERT_FP`）支持的 dtype 对：
    - `float` Acc → `int8_t`、`uint8_t`
    - `int32_t` Acc → `int8_t`、`uint8_t`、`half`、`int16_t`
- 向量量化要求提供 `FpTileData` 缩放操作数（`TileType::Scaling`）。

### A5 实现检查

除了 Acc → Mat 路径外，A5 还支持 Acc → Vec、Vec → Vec、Vec → Mat 和 NZ split 路径。

- **Acc → Mat**（`TileType::Acc → TileType::Mat`）：
    - 源 Acc 类型必须是 `float` 或 `int32_t`；源布局必须为 `(BFractal: ColMajor, SFractal: RowMajor)`。
    - 目标布局必须为 `(!isRowMajor, SFractal: RowMajor)`（NZ 格式）。
    - **非量化**（普通 / relu）目标类型：
        - `float` Acc → `half`、`bfloat16_t`、`float`
        - `int32_t` Acc → `int32_t`
    - **标量量化** 目标类型：
        - `float` Acc → `int8_t`、`uint8_t`、`hifloat8_t`、`half`、`bfloat16_t`、`float8_e4m3_t`
        - `int32_t` Acc → `int8_t`、`uint8_t`、`half`、`bfloat16_t`
    - **向量量化**（`TINSERT_FP`）目标类型：与标量量化相同。

- **Acc → Vec**（`TileType::Acc → TileType::Vec`）：
    - 源 Acc 类型必须是 `float` 或 `int32_t`；源布局必须为 `(BFractal: ColMajor, SFractal: RowMajor)`。
    - **非量化**（普通 / relu）目标类型：
        - `float` Acc → `half`、`bfloat16_t`、`float`
        - `int32_t` Acc → `int32_t`
    - **标量量化** 目标类型：
        - `float` Acc → `int8_t`、`uint8_t`、`hifloat8_t`、`half`、`bfloat16_t`、`float8_e4m3_t`
        - `int32_t` Acc → `int8_t`、`uint8_t`、`half`、`bfloat16_t`
    - **向量量化**（`TINSERT_FP` / `TINSERT` 含 `FpTileData`）目标类型：与标量量化相同。
    - 目标布局必须为以下之一：NZ-to-NZ（`!isRowMajor, SFractal: RowMajor`）、NZ-to-ND（`isRowMajor, SFractal: NoneBox`）或 NZ-to-DN（`!isRowMajor, SFractal: NoneBox`）。
    - `AccToVecMode` 选择 `SingleModeVec0`、`SingleModeVec1`、`DualModeSplitM` 或 `DualModeSplitN`。
    - 双目标模式（`DualModeSplitM`、`DualModeSplitN`）要求 `QuantMode_t::NoQuant` 且不支持 NZ-to-DN 路径。
    - 对于 32 位目标类型（`float`/`int32_t`），使用 `DualModeSplitN` 时切分前的 `ValidCol` 必须是 `32` 的整数倍。
    - 目标 stride 必须非零且 `dstStride * sizeof(dstType)` 必须是 `32` 字节的倍数。

- **Vec → Vec**（`TileType::Vec → TileType::Vec`）：
    - `DstTileData::DType` 必须等于 `SrcTileData::DType`。
    - 支持的元素类型：`half`、`bfloat16_t`、`float`、`int32_t`、`int8_t`、`hifloat8_t`、`float8_e4m3_t`、`float8_e5m2_t`、`float8_e8m0_t`、`float4_e2m1x2_t`、`float4_e1m2x2_t`。
    - 源和目标布局必须匹配（均 ND 或均 NZ）。
    - ND 路径：源有效区域必须在目标边界内。分发选择 `copy_ubuf_to_ubuf`（对齐）、`vlds`/`vsts`（stride 对齐、未对齐 validCol）、`vlds`/`vstus`（未对齐 stride 或 indexCol）或标量拷贝（1×1 元素）。
    - NZ 路径：源列数不得超目标列数。使用 `ComputeNZBlockParams` 进行分形块 `copy_ubuf_to_ubuf`。

- **Vec → Mat**（`TileType::Vec → TileType::Mat`，UB → L1）：
    - `DstTileData::DType` 必须等于 `SrcTileData::DType`。
    - 支持的元素类型：`half`、`bfloat16_t`、`float`、`int32_t`、`int8_t`、`hifloat8_t`、`float8_e4m3_t`、`float8_e5m2_t`、`float8_e8m0_t`、`float4_e2m1x2_t`、`float4_e1m2x2_t`。
    - ND 路径：源必须为 `isRowMajor`；使用 `copy_ubuf_to_cbuf`。每行数据字节数必须与 `BLOCK_BYTE_SIZE`（32 字节）对齐。
    - NZ 路径：源必须为 `(!isRowMajor, SFractal: RowMajor)`；使用 `ComputeNZBlockParams` 进行分形块 `copy_ubuf_to_cbuf`。

- **NZ Split**（`TInsertMode::SPLIT2` / `TInsertMode::SPLIT4`，仅 A5/kirin9030/kirinX90）：
    - 目标必须为 `TileType::Mat`；源必须为 `TileType::Vec`。
    - `DstTileData::DType` 必须等于 `SrcTileData::DType`。
    - 源必须为 NZ 格式：`(!isRowMajor, SFractal: RowMajor)`。
    - 支持的元素类型：`half`、`bfloat16_t`、`float`、`int32_t`、`int8_t`、`hifloat8_t`、`float8_e4m3_t`、`float8_e5m2_t`、`float8_e8m0_t`、`float4_e2m1x2_t`、`float4_e1m2x2_t`。
    - `validRow` 对齐到 `FRACTAL_NZ_ROW`（16）用于 burst 计算。
    - 将 `copy_ubuf_to_cbuf` 的总 burst 拆分为 2 或 4 个子传输，每个处理 `totalBurstNum / SplitCount` 列块。

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

### PTO 汇编形式

```text
%dst = tinsert %src[%r0, %r1] : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tinsert ins(%src, %idxrow, %idxcol : !pto.tile_buf<...>, dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```
