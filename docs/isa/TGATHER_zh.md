# TGATHER

## 指令示意图

![TGATHER tile operation](../figures/isa/TGATHER.svg)

## 简介

使用索引Tile或编译时掩码模式来收集/选择元素。

## 数学语义

基于索引的gather（概念性定义）：

设 `R = dst.GetValidRow()`，`C = dst.GetValidCol()`。对于 `0 <= i < R` 且 `0 <= j < C`：

$$ \mathrm{dst}_{i,j} = \mathrm{src0}\!\left[\mathrm{indices}_{i,j}\right] $$

`indices[i,j]` 是相对于 `src0.data()` 的扁平元素偏移，单位是源元素，不是字节或行号；物理填充也占用偏移。调用者须保证索引非负、位于已分配的源 Tile 内，并指向已初始化的数据。实现不执行索引边界检查。

### 64 位模式 gather 的输出（A5 / Ascend 950PR/Ascend 950DT）

每个源有效行选择列 `offset, offset + factor, ...`，且列号必须小于 `C = src.GetValidCol()`：

| 模式 | factor | offset |
| --- | ---: | ---: |
| `P0101` / `P1010` | 2 | 0 / 1 |
| `P0001` / `P0010` / `P0100` / `P1000` | 4 | 0 / 1 / 2 / 3 |
| `P1111` | 1 | 0 |

每行选中数 `N = C > offset ? ceil((C - offset) / factor) : 0`。各行结果连续拼接到 `dst.data()`，总数为 `src.GetValidRow() * N`，行间不按 `DstTileData::Cols` 插入填充。调用者须提供足够的连续物理容量，并选择能够表示这段连续输出的有效形状。源有效行数为 0 或 `N == 0` 时不写入。

例如源物理形状 `[2,8]`、有效形状 `[2,5]` 使用 `P0101` 时，每行选择列 0、2、4，输出为连续 6 个元素；可用物理 `[1,8]`、有效 `[1,6]` 的目标承接，末尾 2 个填充元素保持不变。

## 汇编语法

基于索引的gather：

```text
%dst = tgather %src0, %indices : !pto.tile<...> -> !pto.tile<...>
```

基于掩码模式的gather：

```text
%dst = tgather %src {maskPattern = #pto.mask_pattern<P0101>} : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tgather %src, %indices : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
%dst = pto.tgather %src {maskPattern = #pto.mask_pattern<P0101>}: !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tgather ins(%src, %indices : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
pto.tgather ins(%src, {maskPattern = #pto.mask_pattern<P0101>} : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

### 基于索引的Gather

```cpp
template <typename TileDataD, typename TileDataS0, typename TileDataS1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(TileDataD &dst, TileDataS0 &src0, TileDataS1 &src1, TileDataTmp &tmp, WaitEvents &... events);
```

### 基于掩码模式的Gather

```cpp
template <typename DstTileData, typename SrcTileData, MaskPattern maskPattern = MaskPattern::P1111,
          auto gatherType = GatherAxis::GATHER_ROW, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(DstTileData &dst, SrcTileData &src, WaitEvents &... events);
```

### 基于比较的Gather（TGather_cmp）

收集满足与每行阈值标量比较条件的元素的索引。

```cpp
template <typename TileDataD, typename TileDataS, typename TileDataS1, typename TileDataC, typename TileDataTmp, CmpMode cmpMode, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(TileDataD &dst, TileDataS &src0, TileDataS1 &k_value, TileDataC &cdst, TileDataTmp &tmp, int offset, WaitEvents &... events);
```

对于 `src0` 的每一行 `i`，使用 `cmpMode`（GT或EQ）将每个元素 `src0[i, j]` 与阈值 `k_value[i]` 比较。匹配元素的索引被收集到 `dst[i]` 中。每行的匹配数量存储在 `cdst[i]` 中。`offset` 参数指定起始索引值。

#### 基于比较的Gather约束

- **Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品**:
    - `TileDataD::DType` 必须是 `int32_t` 或 `uint32_t`。
    - `TileDataS::DType` 必须是 `float`、`half`，或 `int32_t`（仅EQ模式）。
    - `TileDataS1::DType` 必须是 `int32_t` 或 `uint32_t`。
    - `cmpMode` 必须是 `CmpMode::GT` 或 `CmpMode::EQ`。
- **Ascend 950PR/Ascend 950DT**:
    - `TileDataD::DType` 必须是 `int32_t` 或 `uint32_t`。
    - `TileDataS::DType` 必须是 `int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`half` 或 `float`。
    - `TileDataS1::DType` 必须是 `uint16_t` 或 `uint32_t`。
    - `cmpMode` 必须是 `CmpMode::GT` 或 `CmpMode::EQ`。

## 约束

- **基于索引的gather：实现检查 （Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品）**:
    - `sizeof(DstTileData::DType)` 必须是2或4字节（b16/b32）。
    - `sizeof(Src1TileData::DType)` 必须是4字节（b32: `int32_t`、`uint32_t`）。
    - `DstTileData::DType` 必须与 `Src0TileData::DType` 类型相同。
    - `TmpTileData::DType` 必须与 `Src1TileData::DType` 类型相同。
    - `src1.GetValidCol() == TmpTileData::Cols` 且 `src1.GetValidRow() == TmpTileData::Rows`。
    - `dst.GetValidCol() == DstTileData::Cols`（连续的目标存储）。
- **基于索引的gather：实现检查 (Ascend 950PR/Ascend 950DT)**:
    - `DstTileData::DType` 必须为 1、2、4 或 8 字节类型：`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`int64_t`、`uint64_t`、`half`、`bfloat16_t`、`float` 之一。
    - `Src1TileData::DType` 必须是 `int16_t`、`uint16_t`、`int32_t`、`uint32_t` 之一。
    - `DstTileData::DType` 必须与 `Src0TileData::DType` 类型相同。
    - 64 位数据要求 32 位索引（`int32_t` 或 `uint32_t`）。
    - 64 位索引 gather 支持 RowMajor/ColMajor 的目标和索引 Tile，并按各自物理行列步长访问；有效列数可以小于物理列数。索引 Tile 的有效区域须覆盖目标有效区域。
- **基于掩码模式的gather：实现检查 （Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品）**:
    - 源元素大小必须是`2`或`4`字节。
    - `SrcTileData::DType`/`DstTileData::DType` 必须是 `int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`half`、`bfloat16_t` 或 `float` 之一。
    - `dst` 和 `src` 必须都是 `TileType::Vec` 且行主序。
    - `sizeof(dst element) == sizeof(src element)` 且 `dst.GetValidCol() == DstTileData::Cols`（连续的目标存储）。
- **基于掩码模式的gather：实现检查 (Ascend 950PR/Ascend 950DT)**:
    - 源元素大小必须是`1`、`2`、`4`或`8`字节。
    - `dst` 和 `src` 必须都是 `TileType::Vec` 且行主序。
    - `SrcTileData::DType`/`DstTileData::DType` 必须是 `int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`int64_t`、`uint64_t`、`half`、`bfloat16_t`、`float`、`float8_e4m3_t`、`float8_e5m2_t` 或 `hifloat8_t` 之一。
    - `sizeof(dst element) == sizeof(src element)`。输出按连续流写入；目标容量必须足够，目标有效形状不会截断写入。
    - 64 位模式 gather 只支持 `GatherAxis::GATHER_ROW`，具体输出长度与排列见数学语义。
- **基于比较的gather：实现检查**：类型与 `cmpMode` 约束详见 [C++内建接口 → 基于比较的Gather约束](#基于比较的gather约束) 一节。
- **边界 / 有效性**:
    - 索引边界不通过显式运行时断言进行验证；超出范围的索引行为由目标定义。
- **临时Tile**:
    - **基于索引的Gather （Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品）**：C++ API需要显式传入 `tmp` Tile。`TileDataTmp::DType` 必须与 `TileDataS1::DType` 类型相同（`int32_t` 或 `uint32_t`）。`src1.GetValidRow() == TileDataTmp::Rows` 且 `src1.GetValidCol() == TileDataTmp::Cols`。tmp Tile用于存放b16源类型的 `vmuls` 中间结果供 `vgather` 使用；对于b32源类型，结果直接写入 `dst`，但API仍需要 `tmp`。
    - **基于索引的Gather (Ascend 950PR/Ascend 950DT)**：`tmp` Tile被接受但不使用。Ascend 950PR/Ascend 950DT硬件无需临时缓冲区即可处理基于索引的Gather。
    - **基于比较的Gather （Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品）**：C++ API需要显式传入 `tmp` Tile，该Tile作为三个内部区域的合并暂存缓冲区：
        1. **cmpsTmp**（比较结果位图）：偏移量0，以 `uint8_t` 存储，大小 = `TileDataTmp::Rows × TileDataTmp::Cols`字节。
        2. **indexTmp**（索引数组）：偏移量 = `TileDataTmp::Rows × TileDataTmp::Cols × sizeof(uint8_t)`，以 `TileDataD::DType` 存储，大小 = `TileDataS::Rows × TileDataS::Cols × sizeof(TileDataD::DType)`字节。
        3. **cvtTmp**（转换后的k值数组）：偏移量 = `TileDataTmp::Rows × TileDataTmp::Cols × sizeof(uint8_t)` + `TileDataS::Rows × TileDataS::Cols × sizeof(TileDataD::DType)`，以 `TileDataS::DType` 存储，大小 = `TileDataS::Rows × sizeof(TileDataS::DType)`字节。
        最小tmp大小必须满足：
        $$ \text{tmpSize} \ge \text{Rows}_\text{tmp} \times \text{Cols}_\text{tmp} + \text{Rows}_\text{src} \times \text{Cols}_\text{src} \times \text{sizeof(DType}_\text{dst}\text{)} + \text{Rows}_\text{src} \times \text{sizeof(DType}_\text{src}\text{)} $$
    - **基于比较的Gather (Ascend 950PR/Ascend 950DT)**：`tmp` Tile被接受但不使用。Ascend 950PR/Ascend 950DT硬件无需临时缓冲区即可处理基于比较的Gather。

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using IdxT = Tile<TileType::Vec, int32_t, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  using TmpT = Tile<TileType::Vec, int32_t, 16, 16>;
  SrcT src0;
  IdxT idx;
  DstT dst;
  TmpT tmp;
  TGATHER(dst, src0, idx, tmp);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 1, 128>;
  SrcT src;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TGATHER<DstT, SrcT, MaskPattern::P0101>(dst, src);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tgather %src, %indices : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tgather %src, %indices : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO汇编形式

```text
%dst = pto.tgather %src, %indices : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tgather ins(%src, %indices : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
