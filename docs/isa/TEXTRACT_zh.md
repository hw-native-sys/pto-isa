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

template <STPhase Phase, typename DstTileData, typename SrcTileData, typename FpTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <STPhase Phase, typename DstTileData, typename SrcTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <STPhase Phase, typename DstTileData, typename SrcTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <STPhase Phase, typename DstTileData, typename SrcTileData, typename FpTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);
```

`STPhase` 重载把 unit flag（单元标志）写入 L0C 搬出指令，用于与 `TMATMUL<AccPhase>` 配对完成
Cube 到 Fixpipe 的硬件同步，从而省去显式的 `set_flag`/`wait_flag`。
仅在存在对应后端实现的目标上暴露（Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品、
Ascend 950PR/Ascend 950DT 和 CPU 模拟器），适用范围与配对规则见下方实现检查。

`TEXTRACT_FP(...)` 为历史 fp 量化形式保留源码兼容入口，并直接映射到无 `mode` 的
`TEXTRACT_IMPL(dst, src, fp, indexRow, indexCol)` 路径。规范同名 `TEXTRACT(..., fp, ...)`
重载仅在 `FpTileData::Loc == TileType::Scaling` 时参与匹配。
规范接口还提供显式 `AccToVecMode` 形式，用于目标支持的 Acc-to-Vec 路由。
`TEXTRACT_FP` 同样提供 `STPhase` 版本，与 `TMOV_FP` / `TSTORE_FP` 对齐，语义与规范接口一致。

## 约束

### 通用约束或检查

- `STPhase` 重载（unit flag）仅支持 `TileType::Acc -> TileType::Mat`（L0C→L1）路径；
  目标为 `TileType::Vec` 时编译期报错。该重载在
  Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品、Ascend 950PR/Ascend 950DT 与 CPU 模拟器上暴露。
- 搬出侧的取值规则与累加侧不同，不是简单对应关系：
  产生该 L0C 结果的 `TMATMUL` 必须已经是 `AccPhase::Final`，即数据已就绪；
  `STPhase::Final` 用于最后一次搬出并释放 unit flag；
  `STPhase::Partial` 只用于同一块 L0C 分多次搬出时的非末次那几条，它不释放 unit flag。
  把 `STPhase::Partial` 与 `AccPhase::Partial` 配对会让 fixpipe 等待一个不会到来的标志而挂死，
  该现象已在 Ascend 950PR 仿真器上复现。
- Acc→Mat 搬出已通过 A3 上板测试，`tmov_acc2mat` 覆盖了 `STPhase::Final` 和
  `STPhase::Partial` 后接 `STPhase::Final` 的场景。
  Ascend 950PR 仿真测试已通过；在 Ascend 950PR 板机、CANN 9.2.0 环境下，
  `textract` 的 7 个定向用例（`case1`、`case21`–`case26`）全部通过，max diff 均为 0，
  覆盖 NZ512/NZ1024、`Final`、`Partial` 后接 `Final`、`Unspecified` 和 K 切分累加。
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

除上述 `Mat/Acc -> ...` 路径外，`TEXTRACT` 还支持 `TileType::Vec -> TileType::Vec` 抽取路径（ND 与 NZ 布局）。A2A3 使用 `CheckTExtractVecToVecCommon`，A5 在 `TEXTRACT_IMPL` 中单独检查：

- `DstTileData::DType` 必须等于 `SrcTileData::DType`。
- A2A3 元素类型：`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`half`、`bfloat16_t`、`float`。
- A5 元素类型：`int8_t`、`int32_t`、`half`、`bfloat16_t`、`float`、`hifloat8_t`、`float8_e4m3_t`、`float8_e5m2_t`、`float8_e8m0_t`、`float4_e2m1x2_t`、`float4_e1m2x2_t`。A5 不支持此路径的 `uint8_t`、`int16_t`、`uint16_t`、`uint32_t` 或 64 位整数。
- A5 ND fp4 路径以打包元素（每个 1 字节包含两个 fp4 值）计数；行步长、静态有效列字节数以及列偏移字节数须 32 字节对齐。
- ND 路径：源/目标行步进须 32 字节对齐；`Dst` 行/列不得超过 `Src`。
- A5 ND Vec→Vec 路径先检查 `indexRow + dst.GetValidRow() <= SrcTileData::Rows` 和 `indexCol + dst.GetValidCol() <= SrcTileData::Cols`；通过检查后，目标有效行数或列数为 0 时直接返回，不读取源或写入目标。对齐与非对齐列偏移均遵循此规则；这不扩展该路径的类型支持，仍不支持 int64。

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

带 unit flag 的 L0C→L1 搬出（Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品、Ascend 950PR/Ascend 950DT）：

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_unit_flag() {
  TileLeft<half, 32, 32> a;
  TileRight<half, 32, 32> b;
  TileAcc<float, 32, 32> c;
  Tile<TileType::Mat, float, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor> l1;
  TASSIGN(a, 0x0);
  TASSIGN(b, 0x0);
  TASSIGN(c, 0x0);
  TASSIGN(l1, 0x2000);
  // 数据就绪后再搬出：两条指令之间不需要显式 set_flag/wait_flag
  TMATMUL<AccPhase::Final>(c, a, b);
  TEXTRACT<STPhase::Final>(l1, c, /*indexRow=*/0, /*indexCol=*/0);

  // 同一块 L0C 分多次搬出时，非末次用 Partial 不释放 unit flag，末次用 Final 释放
  // TEXTRACT<STPhase::Partial>(l1, c, /*indexRow=*/0, /*indexCol=*/0);
  // TEXTRACT<STPhase::Final>(l1, c, /*indexRow=*/0, /*indexCol=*/0);
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
