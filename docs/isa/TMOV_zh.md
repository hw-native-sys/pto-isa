# TMOV

## 指令示意图

![TMOV tile operation](../figures/isa/TMOV.svg)

## 简介

在Tile之间移动/复制，可选通过模板参数和重载选择实现定义的转换模式。

`TMOV` 用于：

- Vec -> Vec移动
- Mat -> Left/Right/Bias/Scaling/Scale(Microscaling) 移动（目标相关）
- Acc -> Mat/Vec移动（目标相关）

## 数学语义

在有效区域内将元素从 `src` 复制或转换到 `dst`。具体转换取决于所选模式和目标。

对于纯复制情况：

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} $$

### ND → NZ（为Cube Unit重新打包数据）

Cube Unit以 **NZ**（Normal-ZigZag）fractal格式消费操作数：Tile被划分为 `C0 × C0` fractal，每个fractal以 `BLayout = ColMajor`（"N"——列主序外层块）和 `SLayout = RowMajor`（"Z"——fractal内行主序）存储。`TMOV(dstNZ, src)` 将RowMajor `Vec`/`Mat` Tile（`NoneBox`）重新打包为NZ布局。无需 `tmp`。

| 操作数（GM/L1侧） | `BLayout` | `SLayout` | 含义 |
|-------------------|-----------|-----------|------|
| Left（A，NT）      | `ColMajor` | `RowMajor` | 标准NZ |
| Right（B，NT）     | `RowMajor` | `ColMajor` | 转置NZ |

`CompactMode::RowPlusOne` 目标（`Rows = Vec_S0 + 1`）是避免 `vsstb` scatter上UB bank冲突的标准用法。

### X → ZZ（microscaling指数重新打包）

对于MXFP8/MXFP4矩阵乘，每组E8M0指数须以 **ZZ** 格式送达Cube的scale操作数：`[16,2]` 块（32Byte = 16列 × 2行组），按Cube每个fractal列对一个块的方式线性化。两种变体：

**ND → ZZ**（`grp_axis = 1`，默认）：源指数为ND分组（axis-1）。

$$D_{ZZ}[r_b, c, q, \delta] = D_{ND}[16r_b + \delta][2c + q]$$

其中 $r_b \in [0, R/16)$，$c \in [0, C/2)$，$q \in [0,2)$，$\delta \in [0,16)$。

**DN → ZZ**（`grp_axis = 0`）：源指数为DN分组（axis-0）。等价于转置后ND→ZZ：

$$E_{ZZ}[c_b, p, q, \delta] = E_{DN}^{T}[16c_b + q][2p + \delta] = E_{DN}[2p + \delta][16c_b + q]$$

其中 $c_b \in [0, N/16)$，$p \in [0, \hat M/2)$，$q \in [0,16)$，$\delta \in \{0,1\}$，$\hat M = M/32$。

对于固定的 $(c_b, p)$，ZZ块的32Byte来自两段连续的16Byte源数据（$E_{DN}[2p]$ 和 $E_{DN}[2p+1]$）；因此DN→ZZ使用连续加载 + `vintlv`（比ND→ZZ的 `vgather2` 更高效）。

### `tmp` Tile的作用

仅 **X→ZZ** 转换接受 `tmp` 操作数（3参数重载）。ND→ZZ用作 `vgather2` 索引缓冲；DN→ZZ为接口一致接受但**不访问**（`vsstb` scatter无需scratch）。ND→NZ无 `tmp`。

## 汇编语法

PTO AS设计建议将 `TMOV` 拆分为一组操作：

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

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp` 和 `include/pto/common/constants.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

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

### ND → NZ / X → ZZ重载

```cpp
// ND -> NZ（2 参数，无 tmp）
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, WaitEvents &...events);

// X -> ZZ（3 参数，带 tmp）。grp_axis=1（默认）= ND->ZZ；grp_axis=0 = DN->ZZ。
template <typename DstTileData, typename SrcTileData, typename TmpTileData, typename... WaitEvents,
          std::enable_if_t<is_tile_data_v<TmpTileData>, int> = 0>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, TmpTileData &tmp, WaitEvents &...events);

template <int grp_axis, typename DstTileData, typename SrcTileData, typename TmpTileData, typename... WaitEvents,
          std::enable_if_t<is_tile_data_v<TmpTileData>, int> = 0>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, TmpTileData &tmp, WaitEvents &...events);
```

| 重载 | `grp_axis` | 转换 | 是否使用 `tmp`？ |
|------|-----------|------|----------------|
| `TMOV(dst, src)` | — | ND → NZ | 否 |
| `TMOV(dst, src, tmp)` | 1（默认） | ND → ZZ | 是（vgather2索引缓冲） |
| `TMOV<0>(dst, src, tmp)` | 0 | DN → ZZ | 否（仅为接口一致接受） |

## 约束

### 通用约束 / 检查

- `TMOV` 有以下重载族：
    - 纯移动：`TMOV(dst, src)`
    - relu形式：`TMOV<..., reluMode>(dst, src)`
    - 累加器到向量形式：`TMOV<..., mode, reluMode>(dst, src)`
    - 向量量化形式：`TMOV<..., FpTileData, mode, reluMode>(dst, src, fp)`
    - 标量量化形式：`TMOV<..., reluMode>(dst, src, preQuantScalar)` 和 `TMOV<..., mode, reluMode>(dst, src, preQuantScalar)`
- `reluMode` 取值为 `ReluPreMode::{NoRelu, NormalRelu}`。
- `mode` 取值为 `AccToVecMode::{SingleModeVec0, SingleModeVec1, DualModeSplitM, DualModeSplitN}`。

### Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品实现检查

- 形状须匹配：`SrcTileData::Rows == DstTileData::Rows` 且 `SrcTileData::Cols == DstTileData::Cols`。
- 支持的Tile类型对在编译期限制为：
    - `TileType::Mat -> TileType::Left/Right/Bias/Scaling`
    - `TileType::Vec -> TileType::Vec`
    - `TileType::Acc -> TileType::Mat`
- 对于 `TileType::Mat -> TileType::Bias`：
    - 支持的源/目标dtype对为 `int32_t -> int32_t`、`float -> float`、`half -> float`
    - 源行数必须为 `1`
    - `SrcTileData::Cols * sizeof(SrcType)` 必须64字节对齐
- 对于 `TileType::Mat -> TileType::Scaling`：
    - 目标dtype须等于源dtype且为 `uint64_t`
    - 源行数必须为 `1`
    - `SrcTileData::Cols * sizeof(SrcType)` 必须128字节对齐
- 对于 `TileType::Acc -> TileType::Mat`：
    - 额外执行 `CheckTMovAccToMat<...>` 编译期检查
    - 纯/relu形式使用由 `GetCastPreQuantMode<SrcDType, DstDType>()` 推导的类型转换前量化模式
    - 标量量化形式使用 `GetScalarPreQuantMode<SrcDType, DstDType>()`
    - 向量量化形式要求 `FpTileData` 操作数 `FpTileData::Loc == TileType::Scaling`，使用 `GetVectorPreQuantMode<SrcDType, DstDType>()`

### Ascend 950PR/Ascend 950DT实现检查

- `CommonCheck()` 要求：
    - 目标/源dtype必须相同
    - 支持的元素类型为 `int8_t`、`hifloat8_t`、`float8_e5m2_t`、`float8_e4m3_t`、`half`、`bfloat16_t`、`float`、`float4_e2m1x2_t`、`float4_e1m2x2_t`
    - 源布局须满足以下之一：
        - `(SrcTileData::SFractal == SLayout::ColMajor && SrcTileData::isRowMajor)`
        - `(SrcTileData::SFractal == SLayout::RowMajor && !SrcTileData::isRowMajor)`
        - `SrcTileData::isRowMajor`
- `CommonCheckMX()` 用于MX路径，要求源/目标dtype一致且支持 `float8_e8m0_t`。
- 支持的路径包括：
    - `TileType::Mat -> TileType::Left/Right/Bias/Scaling/ScaleLeft/ScaleRight`
    - `TileType::Vec -> TileType::Vec/TileType::Mat`
    - `TileType::Acc -> TileType::Vec/TileType::Mat`
    - 特定的 `ND -> ZZ` 等内部路径变体由Ascend 950PR/Ascend 950DT实现处理
- 对于 `TileType::Mat -> TileType::Bias`：
    - 支持的dtype对为 `int32_t -> int32_t`、`float -> float`、`half -> float`、`bfloat16_t -> float`
    - 源行数必须为 `1`
    - `DstTileData::Cols * sizeof(DstType)` 必须64字节对齐
    - bias表占用 `DstTileData::Cols * sizeof(DstType)` 不得超过 `4096` 字节
- 对于 `TileType::Mat -> TileType::Scaling`：
    - 源行数必须为 `1`
    - `DstTileData::Cols * sizeof(DstType)` 必须128字节对齐
    - fixpipe缓冲占用 `DstTileData::Cols * sizeof(DstType)` 不得超过 `4096` 字节
- 对于 `TileType::Acc -> TileType::Vec`：
    - `mode` 选择 `SingleModeVec0`、`SingleModeVec1`、`DualModeSplitM` 或 `DualModeSplitN`
    - 双目标模式要求 `QuantMode_t::NoQuant`
    - 双目标模式不支持 `nz2dn` 路径
    - 对于32位目标类型（`float`/`int32_t`），使用 `DualModeSplitN` 时 `ValidCol`（分裂前）须为32的倍数
    - 目标stride须非零且 `dstStride * sizeof(dstType)` 须为32字节的倍数
- 对于 `TileType::Acc -> TileType::Mat`：
    - 目标stride须非零且 `dstStride * sizeof(dstType)` 须为32字节的倍数
    - relu/标量量化/向量量化形式通过相应重载支持

## 示例

### ND → NZ（数据）— (128, 256) BF16

```cpp
// 源：128 行 × 256 列 BF16 RowMajor Vec Tile（ND）。
// 目标：Cube Unit 的 NZ fractal Mat Tile（Left 操作数）。
constexpr uint32_t R = 128, C = 256;
using SrcT = Tile<TileType::Vec, bfloat16_t, R, C, BLayout::RowMajor, R, C, SLayout::NoneBox>;
using DstT = Tile<TileType::Mat, bfloat16_t, R, C, BLayout::ColMajor, R, C, SLayout::RowMajor>;
SrcT src; DstT dst;
TMOV(dst, src);   // ND -> NZ，无 tmp
```

**ND→NZ的 `tmp`：** 无——2参数重载通过 `vsstb` 原地重新打包。

### ND → ZZ（指数）— `tmp` 尺寸推导

给定量化输入形状 $M \times N$（组大小 $G = 32$），ND分组E8M0指数Tile形状为：

| 量 | 值 |
|----------|------|
| 指数行数 | $\mathrm{validRow} = M$（每输入行一个指数行） |
| 指数列数 | $\mathrm{validCol} = N/G = N/32$（每32元素列组一个指数） |
| 行块数 | $r_b = \lceil M/16 \rceil$ |
| 块对数 | $P = \mathrm{validCol}/2 = N/64$ |

`tmp` 缓冲保存 `GenerateB8IndicesZZToUB` 使用的 `vgather2` B16索引缓冲：

$$\boxed{\mathrm{tmpBytes} = \bigl(16 + r_b \cdot P + 16\bigr) \times 2 = \left(32 + \left\lceil\tfrac{M}{16}\right\rceil \cdot \tfrac{N}{64}\right) \times 2}$$

- 16 B16 lanes头部空间 + $r_b \times P$ 个gather索引 + 16 B16 lanes尾部空间。
- `tmp` dtype = `uint8_t`（E8M0），形状 `1 × ⌈tmpBytes⌉`。

**示例：** $M = 128$，$N = 256$ → 指数Tile $128 \times 8$，$r_b = 8$，$P = 4$：

$$\mathrm{tmpBytes} = (32 + 8 \times 4) \times 2 = 128\ \mathrm{B}$$

### DN → ZZ（指数）— `tmp`

DN分组指数形状为 $\hat M \times N$，其中 $\hat M = M/32$。`TMOV<0>` 为接口一致接受 `tmp` 操作数但**不访问它**（`vsstb` scatter无需scratch）。任意非零尺寸Tile即可满足签名。

```cpp
// DN 分组 e8 指数（M̂×N）-> ZZ fractal scale Tile。
TMOV<0>(e8ZzTile, e8DnTile, tmpTile);   // grp_axis=0 = DN->ZZ；tmp 未使用
```

### 在MX量化中的用法（参考）

`TQUANT` 生成量化数据 + E8M0指数后，两个 `TMOV` 将它们重新打包供Cube Unit使用。完整流水线见 `TQUANT.md` / `TQUANT_DN.md`。

```cpp
// MXFP8 DN 流水线：量化 128×256 BF16 Tile，然后为 Cube 重新打包。
constexpr uint32_t M = 128, N = 256, G = 32, Mhat = M / G;   // Mhat = 4
// Tiles
using SrcT   = Tile<TileType::Vec, bfloat16_t, M, N, BLayout::RowMajor>;
using Fp8T   = Tile<TileType::Vec, int8_t, M, N, BLayout::RowMajor>;
using E8DnT  = Tile<TileType::Vec, uint8_t, Mhat, N, BLayout::RowMajor>;        // 4×256
using E8ZzT  = Tile<TileType::Mat, uint8_t, N, Mhat, BLayout::ColMajor, N, Mhat, SLayout::RowMajor>;
using Fp8NzT = Tile<TileType::Mat, int8_t, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor>;
// 1. 量化（DN 分组）
TQUANT<0, MxQuantAlg::OcpMxFp8E4M3>(fp8Tile, srcTile, &e8DnTile, &maxTile, &scalingTile);
// 2. 数据重新打包 ND->NZ（2 参数，无 tmp）
TMOV(fp8NzTile, fp8Tile);
// 3. 指数重新打包 DN->ZZ（3 参数，tmp 接受但未使用）
TMOV<0>(e8ZzTile, e8DnTile, tmpTile);
```

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TMOV(dst, src);
}
```

### Manual

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

## ASM形式示例

### Auto模式

```text
# Auto 模式：由编译器/运行时管理资源放置与调度。
%dst = pto.tmov.s2d %src  : !pto.tile<...> -> !pto.tile<...>
```

### Manual模式

```text
# Manual 模式：须先显式绑定资源再发射指令。
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tmov.s2d %src  : !pto.tile<...> -> !pto.tile<...>
```

### PTO汇编形式

```text
%dst = pto.tmov.s2d %src  : !pto.tile<...> -> !pto.tile<...>
# AS Level 2（DPS）
pto.tmov ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
