# TQUANT

## 指令示意图

![TQUANT tile operation](../figures/isa/TQUANT.svg)

## 简介

将高精度Tile（`FP32` / `BF16` / `FP16`）量化为低精度格式，同时生成量化的数据Tile以及辅助的每组指数 / 最大值 / 缩放Tile。目标格式、缩放算法和分组轴均为编译期模板参数。

| 目标格式族 | 格式 | 分组方式 | 缩放算法 |
|-----------|------|---------|---------|
| **Microscaling (MX)** | MXFP8 (e4m3)、MXFP4 (e2m1) | 每32个元素共享一个指数 | OCP、NV |
| **Integer** | INT8（对称 / 非对称） | 每Tile一个scale（+ 可选offset） | 仿射 |

## 量化流程

### MX格式（3阶段，组大小G = 32）

对于Tile $x \in \mathbb{R}^{M \times N}$，沿 `grp_axis` 分组（ND：axis-1/列；DN：axis-0/行）：

| 阶段 | 操作 | 输出 |
|------|------|------|
| **1. 组内最大值** | $m_g = \max_{i \in g} \|x_i\|$ | `max`（scratch，FP） |
| **2. 指数 + 缩放** | $s_g = \mathrm{biasedExp}(m_g) - e_{\max}$；$\alpha_g = 2^{254 - s_g}$ | `exp`（E8M0，1字节(Byte)/组）、`scaling`（scratch，FP） |
| **3. 缩放 + 类型转换** | $q_i = \mathrm{clip}_{[-V_{\max},V_{\max}]}(x_i \cdot \alpha_g) \to$ 目标格式 | `dst`（FP8 / 打包FP4） |

- $e_{\max}$ = 目标格式最大指数（e4m3为8，e2m1为1）。
- $V_{\max}$ = 目标格式MAX_NORM（e4m3为448，e2m1为6）。
- 阶段1–2使用精确的IEEE-754位操作（无FP `log`/`floor`）；阶段3使用硬件类型转换 + 随机舍入（`SPR.CTRL[50]=1`）。
- **ND**（"normal direction"，`grp_axis=1`）：每32个连续**列**为一组——默认/标准分组方式。**DN**（`grp_axis=0`）：每32个连续**行**为一组——转置式axis-0分组；指数Tile形状 `M̂×N`，`M̂ = M/32`。

### Integer INT8（仿射，5阶段类型转换）

$$q_i = \mathrm{round}\!\left(\frac{x_i}{\mathrm{scale}}\right) + \mathrm{offset}, \qquad q_i \in [-128, 127]$$

无分组结构；`scale`（和非对称 `offset`）为每Tile的FP32标量/向量。为避免二次舍入，Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品上的类型转换链为 `FP32 → S32 → FP32 → FP16 → INT8`（5阶段，通过 `tmp` Tile）；Ascend 950PR/Ascend 950DT使用原生广播 + 类型转换（无需 `tmp`）。输入必须为 **FP32**。

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp`。
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

### MX—分组式（`grp_axis` + `MxQuantAlg`）—推荐

```cpp
template <int grp_axis, auto mx_alg, typename TileDataOut, typename TileDataSrc,
          typename TileDataExp, typename TileDataMax, typename TileDataScaling, typename... WaitEvents>
PTO_INST RecordEvent TQUANT(TileDataOut &dst, TileDataSrc &src, TileDataExp *exp, TileDataMax *max,
                            TileDataScaling *scaling, WaitEvents &...events);
```

| 模板参数 | 取值 | 含义 |
|---------|------|------|
| `grp_axis` | `0` = DN（axis-0分组）、`1` = ND（axis-1分组） | 量化分组轴 |
| `mx_alg` | `MxQuantAlg::OcpMxFp8E4M3`、`NvMxFp8E4M3`、`OcpMxFp4E2M1`、`NvMxFp4E2M1` | 格式 + 缩放算法 |

### MX—ND旧版（`QuantType` + `QuantScaleAlg`）

```cpp
template <auto quant_type, typename ...Tiles, auto scale_alg = QuantScaleAlg::OCP, typename... WaitEvents>
PTO_INST RecordEvent TQUANT(TileDataOut &dst, TileDataSrc &src, TileDataExp *exp, TileDataMax *max,
                            TileDataScaling *scaling, WaitEvents &...events);

// 带显式 ZZ 指数存储模式
template <auto quant_type, auto store_mode, typename ...Tiles, typename... WaitEvents>
PTO_INST RecordEvent TQUANT(TileDataOut &dst, TileDataSrc &src, TileDataExp *exp, TileDataMax *max,
                            TileDataScaling *scaling, TileDataExp *exp_zz, WaitEvents &...events);
```

| `quant_type` | 格式 | `scale_alg` |
|--------------|------|-------------|
| `QuantType::MXFP8` | e4m3 + E8M0 | OCP / NV |
| `QuantType::MXFP4_E2M1` | e2m1 + E8M0 | OCP / NV |

### Integer INT8

```cpp
// 对称
template <auto quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataPara, typename... WaitEvents>
PTO_INST RecordEvent TQUANT(TileDataOut &dst, TileDataSrc &src, TileDataPara &scale,
                            TileDataPara *offset = nullptr, WaitEvents &...events);
// 带 scratch（A2/A3）
template <auto quant_type, typename ...Tiles, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TQUANT(TileDataOut &dst, TileDataSrc &src, TileDataPara &scale, TileDataTmp &tmp,
                            TileDataPara *offset = nullptr, WaitEvents &...events);
```

| `quant_type` | `offset` | `dst` dtype | 模式 |
|--------------|----------|-----------|------|
| `QuantType::INT8_SYM` | `nullptr` | `int8_t` | 对称（$q = \mathrm{round}(x/\mathrm{scale})$） |
| `QuantType::INT8_ASYM` | 提供 | `uint8_t` | 非对称（$q = \mathrm{round}(x/\mathrm{scale}) + \mathrm{offset}$） |

> 带 `tmp` 的重载（`dst, src, scale, tmp, offset`）用于Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品接口对齐。Ascend 950PR/Ascend 950DT不使用 `tmp`；Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品上 `tmp` 必须为 $M \times N$ FP32（S32类型转换中间结果）。

## Tile尺寸与数据类型

对于输入Tile形状 $M \times N$（dtype $T \in \{\mathrm{FP32}, \mathrm{BF16}, \mathrm{FP16}\}$），组大小 $G = 32$：

### MXFP8（e4m3）

| Tile | dtype | 形状（ND） | 形状（DN） | 字节数 |
|------|-------|-----------|-----------|--------|
| `src` | $T$ | $M \times N$ | $M \times N$ | $M \cdot N \cdot \mathrm{sizeof}(T)$ |
| `dst` | `int8_t`（e4m3别名） | $M \times N$ | $M \times N$ | $M \cdot N$ |
| `exp` | `uint8_t`（E8M0） | $M \times N/32$ | $M/32 \times N$ | $M \cdot N / 32$ |
| `max`（scratch） | $T$ | $M \times N/32$ | $M/32 \times N$ | $M \cdot N / 32 \cdot \mathrm{sizeof}(T)$ |
| `scaling`（scratch） | $T$ | $M \times N/32$ | $M/32 \times N$ | $M \cdot N / 32 \cdot \mathrm{sizeof}(T)$ |

### MXFP4（e2m1）

同MXFP8，但：
| Tile | dtype | 字节数 |
|------|-------|--------|
| `dst` | `float4_e2m1x2_t`（每字节打包2个e2m1） | $M \cdot N / 2$ |

> **输入限制：** MXFP4仅接受 **FP16/BF16**（不支持FP32）。

### INT8

| Tile | dtype | 形状 | 字节数 |
|------|-------|------|--------|
| `src` | `float32_t` | $M \times N$ | $M \cdot N \cdot 4$ |
| `dst`（SYM） | `int8_t` | $M \times N$ | $M \cdot N$ |
| `dst`（ASYM） | `uint8_t` | $M \times N$ | $M \cdot N$ |
| `scale` | FP32标量/向量 | 每Tile | — |
| `offset`（ASYM） | FP32标量/向量 | 每Tile | — |
| `tmp`（仅Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品） | FP32 | $M \times N$ | $M \cdot N \cdot 4$ |

> **`tmp` Tile（仅Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品）：** 必须与 `src` **同尺寸**（$M \times N$ FP32 = $4MN$ 字节），保存FP32→S32类型转换中间结果（Atlas A3 训练系列产品/Atlas A3 推理系列产品无原地 `tcvt`）。Ascend 950PR/Ascend 950DT接受同名 `tmp` 参数以保持接口一致，但**不使用它**（Ascend 950PR/Ascend 950DT原生 `vlds BRC_B32` 广播）。

## 约束

| 约束 | 适用范围 | 原因 |
|------|---------|------|
| $M \bmod 16 = 0$ | ND MX（ZZ布局） | 16行ZZ块 |
| $M \bmod 32 = 0$ | DN MX | axis-0组整除 |
| $M \bmod 64 = 0$ | DN MX + ZZ转换 | δ 配对（$\hat M / 2$ 为整数） |
| $N \bmod 32 = 0$ | 所有MX | 组大小 $G = 32$ |
| $N \bmod 64 = 0$ | ND MX + ZZ转换 | 指数组数为偶数 |
| $M \cdot N \le 59461$ | MX（UB 256KB） | 复用后的缓冲预算 |
| BF16/FP16：`validCols % 32 != 0` → 零填充至 `StaticCols` | MX B16路径 | 组对齐 |

## 输出布局与布局转换

TQUANT默认输出 **ND**（行主序）。Cube Unit消费两种fractal布局，由独立的 `TMOV` 指令生成：

| 输出 | 原生（TQUANT） | Cube布局 | 转换 |
|------|---------------|-----------|------|
| FP8 / FP4数据 | ND | NZ（ColMajor+RowMajor fractal） | `TMOV(dstNZ, dst)`（2参数） |
| E8M0指数（ND分组） | ND | ZZ（zigzag，`[16,2]` 块） | `TMOV(e8Zz, e8, tmp)`（3参数） |
| E8M0指数（DN分组） | DN | ZZ | `TMOV<0>(e8Zz, e8Dn, tmp)`（3参数，`grp_axis=0`） |

DN数据的FP8 mantissa与ND共享相同的物理地址（`(r,c)` 元素完全相同），因此2参数 `TMOV` ND→NZ对DN数据同样适用。仅**指数**路径不同（DN→ZZ通过 `TMOV<0>`）。详见 `TQUANT_DN.md`。

## 支持的输入dtype

| 格式 | 可接受输入dtype | 说明 |
|------|-----------------|------|
| MXFP8 | FP32、BF16、FP16 | FP32：原地FP8输出（4:1）。BF16/FP16：源零填充至 `StaticCols`；类型转换前先上转为FP32（无直接b16→e4m3）。 |
| MXFP4（e2m1） | **仅FP16、BF16**（不支持FP32） | 输出 `float4_e2m1x2_t`（打包）。 |
| INT8（sym/asym） | **仅FP32** | SYM→`int8_t`，ASYM→`uint8_t`。Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品需 `tmp` = src尺寸。 |

## 数学语义

除另有说明外，语义在有效区域内定义，目标相关行为标记为实现定义。

## 汇编语法

### AS Level 1（SSA）

```text
%dst = pto.tquant %src, %qp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tquant ins(%src, %qp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## ASM形式示例

### Auto模式

```text
# Auto 模式：由编译器/运行时管理资源放置与调度。
%dst = pto.tquant %src, %qp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual模式

```text
# Manual 模式：须先显式绑定资源再发射指令。
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tquant %src, %qp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO汇编形式

```text
%dst = pto.tquant %src, %qp : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
# AS Level 2（DPS）
pto.tquant ins(%src, %qp : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## 示例

```cpp
// MXFP8，DN 分组（axis-0），OCP 缩放
TQUANT<0, MxQuantAlg::OcpMxFp8E4M3>(fp8Tile, srcTile, &e8DnTile, &maxTile, &scalingTile);

// MXFP8，ND 分组（旧版）
TQUANT<QuantType::MXFP8>(fp8Tile, srcTile, &e8NdTile, &maxTile, &scalingTile);

// MXFP4 E2M1，DN，NV 缩放
TQUANT<0, MxQuantAlg::NvMxFp4E2M1>(fp4Tile, srcTile, &e8DnTile, &maxTile, &scalingTile);

// INT8 对称
TQUANT<QuantType::INT8_SYM>(int8Tile, srcTile, scale);

// 完整 MXFP8 DN 流水线：量化 + 为 Cube 转换布局
TQUANT<0, MxQuantAlg::OcpMxFp8E4M3>(fp8Tile, srcTile, &e8DnTile, &maxTile, &scalingTile);
TMOV(fp8NZTile, fp8Tile);                  // 数据 ND→NZ
TMOV<0>(e8ZzTile, e8DnTile, tmpTile);      // 指数 DN→ZZ
```

详见 `TQUANT_DN.md`（DN→ZZ转换）和 `tests/npu/a5/src/st/testcase/tquant_dn/`（完整ST示例）。
