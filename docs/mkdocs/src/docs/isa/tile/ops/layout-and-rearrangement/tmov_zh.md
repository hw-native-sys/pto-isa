<!-- Generated from `docs/isa/tile/ops/layout-and-rearrangement/tmov_zh.md` -->

# TMOV

## 指令示意图

![TMOV tile operation](../figures/isa/TMOV.svg)

## 简介

在 Tile 之间移动/复制，可选通过模板参数和重载选择实现定义的转换模式。

`TMOV` 用于：

- Vec -> Vec 移动
- Mat -> Left/Right/Bias/Scaling/Scale（微缩放）移动（取决于目标）
- Acc -> Mat/Vec 移动（取决于目标）

## 数学语义

概念上在有效区域上将元素从 `src` 复制或转换到 `dst`。确切的转换取决于所选模式和目标。

对于纯复制情况：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} $$

## 汇编语法

PTO-AS 形式：参见 [PTO-AS 规范](../assembly/PTO-AS_zh.md)。

PTO AS 设计建议将 `TMOV` 拆分为一系列操作：

```text
%left  = tmov.m2l %mat  : !pto.tile<...> -> !pto.tile<...>
%right = tmov.m2r %mat  : !pto.tile<...> -> !pto.tile<...>
%bias  = tmov.m2b %mat  : !pto.tile<...> -> !pto.tile<...>
%scale = tmov.m2s %mat  : !pto.tile<...> -> !pto.tile<...>
%vec   = tmov.a2v %acc  : !pto.tile<...> -> !pto.tile<...>
%v1    = tmov.v2v %v0   : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tmov.s2d %src  : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tmov ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp` 和 `include/pto/common/constants.hpp`：

```cpp
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, FpTileData &fp, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, WaitEvents &... events);
```

## 约束

### 通用约束或检查

- `TMOV` 包含以下重载族：
  - 普通移动：`TMOV(dst, src)`
  - relu 形式：`TMOV<..., reluMode>(dst, src)`
  - 累加器到向量形式：`TMOV<..., mode, reluMode>(dst, src)`
  - 向量量化形式：`TMOV<..., FpTileData, mode, reluMode>(dst, src, fp)`
  - 标量量化形式：`TMOV<..., reluMode>(dst, src, preQuantScalar)` 和 `TMOV<..., mode, reluMode>(dst, src, preQuantScalar)`
- `reluMode` 取值为 `ReluPreMode::{NoRelu, NormalRelu}`。
- `mode` 取值为 `AccToVecMode::{SingleModeVec0, SingleModeVec1, DualModeSplitM, DualModeSplitN}`。

### A2A3 实现检查

- 形状必须匹配：`SrcTileData::Rows == DstTileData::Rows` 且 `SrcTileData::Cols == DstTileData::Cols`。
- 支持的 Tile 类型对在编译期限制为：
  - `TileType::Mat -> TileType::Left/Right/Bias/Scaling`
  - `TileType::Vec -> TileType::Vec`
  - `TileType::Acc -> TileType::Mat`
- 对于 `TileType::Mat -> TileType::Bias`：
  - 支持的源/目标 dtype 对为 `int32_t -> int32_t`、`float -> float`、`half -> float`
  - 源行数必须为 `1`
  - `SrcTileData::Cols * sizeof(SrcType)` 必须按 `64` 字节对齐
- 对于 `TileType::Mat -> TileType::Scaling`：
  - 目标 dtype 必须与源 dtype 相同，且必须为 `uint64_t`
  - 源行数必须为 `1`
  - `SrcTileData::Cols * sizeof(SrcType)` 必须按 `128` 字节对齐
- 对于 `TileType::Acc -> TileType::Mat`：
  - 额外执行 `CheckTMovAccToMat<...>` 编译期检查
  - 普通/relu 形式使用 `GetCastPreQuantMode<SrcDType, DstDType>()` 推导的 cast pre-quant 模式
  - 标量量化形式使用 `GetScalarPreQuantMode<SrcDType, DstDType>()`
  - 向量量化形式要求提供 `FpTileData` 操作数，且 `FpTileData::Loc == TileType::Scaling`，并使用 `GetVectorPreQuantMode<SrcDType, DstDType>()`

### A5 实现检查

- `CommonCheck()` 要求：
  - 目标/源 dtype 必须相同
  - 支持的元素类型为 `int8_t`、`hifloat8_t`、`float8_e5m2_t`、`float8_e4m3_t`、`half`、`bfloat16_t`、`float`、`float4_e2m1x2_t`、`float4_e1m2x2_t`
  - 源布局必须满足以下之一：
    - `(SrcTileData::SFractal == SLayout::ColMajor && SrcTileData::isRowMajor)`
    - `(SrcTileData::SFractal == SLayout::RowMajor && !SrcTileData::isRowMajor)`
    - `SrcTileData::isRowMajor`
- `CommonCheckMX()` 用于 MX 路径时要求源/目标 dtype 相同，并支持 `float8_e8m0_t`。
- 支持的路径包括：
  - `TileType::Mat -> TileType::Left/Right/Bias/Scaling/ScaleLeft/ScaleRight`
  - `TileType::Vec -> TileType::Vec/TileType::Mat`
  - `TileType::Acc -> TileType::Vec/TileType::Mat`
  - A5 实现中处理的特定 `ND -> ZZ` 及相关内部路径变体
- 对于 `TileType::Mat -> TileType::Bias`：
  - 支持的 dtype 对为 `int32_t -> int32_t`、`float -> float`、`half -> float`、`bfloat16_t -> float`
  - 源行数必须为 `1`
  - `DstTileData::Cols * sizeof(DstType)` 必须按 `64` 字节对齐
  - bias table 占用 `DstTileData::Cols * sizeof(DstType)` 不得超过 `4096` 字节
- 对于 `TileType::Mat -> TileType::Scaling`：
  - 源行数必须为 `1`
  - `DstTileData::Cols * sizeof(DstType)` 必须按 `128` 字节对齐
  - fixpipe buffer 占用 `DstTileData::Cols * sizeof(DstType)` 不得超过 `4096` 字节
- 对于 `TileType::Acc -> TileType::Vec`：
  - `mode` 用于选择 `SingleModeVec0`、`SingleModeVec1`、`DualModeSplitM` 或 `DualModeSplitN`
  - 双目标模式要求 `QuantMode_t::NoQuant`
  - 双目标模式不支持 `nz2dn` 路径
  - 目标 stride 必须非零，且 `dstStride * sizeof(dstType)` 必须是 `32` 字节的整数倍
- 对于 `TileType::Acc -> TileType::Mat`：
  - 目标 stride 必须非零，且 `dstStride * sizeof(dstType)` 必须是 `32` 字节的整数倍
  - 支持通过对应重载启用 relu/标量量化/向量量化形式

## 示例

### 自动（Auto）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TMOV(dst, src);
}
```

### 手动（Manual）

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Mat, float, 16, 16, BLayout::RowMajor, 16, 16, SLayout::ColMajor>;
  using DstT = TileLeft<float, 16, 16>;
  SrcT mat;
  DstT left;
  TASSIGN(mat, 0x1000);
  TASSIGN(left, 0x2000);
  TMOV(left, mat);
}
```

## 汇编示例（ASM）

### 自动模式

```text
# 自动模式：由编译器/运行时负责资源放置与调度。
%dst = pto.tmov.s2d %src  : !pto.tile<...> -> !pto.tile<...>
```

### 手动模式

```text
# 手动模式：先显式绑定资源，再发射指令。
# 可选（当该指令包含 tile 操作数时）：
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tmov.s2d %src  : !pto.tile<...> -> !pto.tile<...>
```

### PTO 汇编形式

```text
%dst = pto.tmov.s2d %src  : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tmov ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
