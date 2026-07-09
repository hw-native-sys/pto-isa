# TQUANT

## 指令示意图

![TQUANT tile operation](../figures/isa/TQUANT.svg)

## 简介

将高精度 Tile（`FP32` / `BF16` / `FP16`）量化为低精度格式，同时生成量化的数据 Tile 以及辅助的每组指数 / 最大值 / 缩放 Tile。目标格式、缩放算法和分组轴均为编译期模板参数。

| 目标格式族 | 格式 | 分组方式 | 缩放算法 |
|-----------|------|---------|---------|
| **Microscaling (MX)** | MXFP8 (e4m3)、MXFP4 (e2m1) | 每 32 个元素共享一个指数 | OCP、NV |
| **Integer** | INT8（对称 / 非对称） | 每 Tile 一个 scale（+ 可选 offset） | 仿射 |

## 量化流程

### MX 格式（3 阶段，组大小 G = 32）

对于 Tile $x \in \mathbb{R}^{M \times N}$，沿 `grp_axis` 分组（ND：axis-1/列；DN：axis-0/行）：

| 阶段 | 操作 | 输出 |
|------|------|------|
| **1. 组内最大值** | $m_g = \max_{i \in g} \|x_i\|$ | `max`（scratch，FP） |
| **2. 指数 + 缩放** | $s_g = \mathrm{biasedExp}(m_g) - e_{\max}$；$\alpha_g = 2^{254 - s_g}$ | `exp`（E8M0，1 字节(Byte)/组）、`scaling`（scratch，FP） |
| **3. 缩放 + 类型转换** | $q_i = \mathrm{clip}_{[-V_{\max},V_{\max}]}(x_i \cdot \alpha_g) \to$ 目标格式 | `dst`（FP8 / 打包 FP4） |

- $e_{\max}$ = 目标格式最大指数（e4m3 为 8，e2m1 为 1）。
- $V_{\max}$ = 目标格式 MAX_NORM（e4m3 为 448，e2m1 为 6）。
- 阶段 1–2 使用精确的 IEEE-754 位操作（无 FP `log`/`floor`）；阶段 3 使用硬件类型转换 + 随机舍入（`SPR.CTRL[50]=1`）。
- **ND**（"normal direction"，`grp_axis=1`）：每 32 个连续**列**为一组——默认/标准分组方式。**DN**（`grp_axis=0`）：每 32 个连续**行**为一组——转置式 axis-0 分组；指数 Tile 形状 `M̂×N`，`M̂ = M/32`。

### Integer INT8（仿射，5 阶段类型转换）

$$q_i = \mathrm{round}\!\left(\frac{x_i}{\mathrm{scale}}\right) + \mathrm{offset}, \qquad q_i \in [-128, 127]$$

无分组结构；`scale`（和非对称 `offset`）为每 Tile 的 FP32 标量/向量。为避免二次舍入，A2/A3 上的类型转换链为 `FP32 → S32 → FP32 → FP16 → INT8`（5 阶段，通过 `tmp` Tile）；A5 使用原生广播 + 类型转换（无需 `tmp`）。输入必须为 **FP32**。

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`。

### MX — 分组式（`grp_axis` + `MxQuantAlg`）— 推荐

```cpp
template <int grp_axis, auto mx_alg, typename TileDataOut, typename TileDataSrc,
          typename TileDataExp, typename TileDataMax, typename TileDataScaling, typename... WaitEvents>
PTO_INST RecordEvent TQUANT(TileDataOut &dst, TileDataSrc &src, TileDataExp *exp, TileDataMax *max,
                            TileDataScaling *scaling, WaitEvents &...events);
```

| 模板参数 | 取值 | 含义 |
|---------|------|------|
| `grp_axis` | `0` = DN（axis-0 分组）、`1` = ND（axis-1 分组） | 量化分组轴 |
| `mx_alg` | `MxQuantAlg::OcpMxFp8E4M3`、`NvMxFp8E4M3`、`OcpMxFp4E2M1`、`NvMxFp4E2M1` | 格式 + 缩放算法 |

### MX — ND 旧版（`QuantType` + `QuantScaleAlg`）

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

> 带 `tmp` 的重载（`dst, src, scale, tmp, offset`）用于 A2/A3 接口对齐。A5 不使用 `tmp`；A2/A3 上 `tmp` 必须为 $M \times N$ FP32（S32 类型转换中间结果）。

## Tile 尺寸与数据类型

对于输入 Tile 形状 $M \times N$（dtype $T \in \{\mathrm{FP32}, \mathrm{BF16}, \mathrm{FP16}\}$），组大小 $G = 32$：

### MXFP8（e4m3）

| Tile | dtype | 形状（ND） | 形状（DN） | 字节数 |
|------|-------|-----------|-----------|--------|
| `src` | $T$ | $M \times N$ | $M \times N$ | $M \cdot N \cdot \mathrm{sizeof}(T)$ |
| `dst` | `int8_t`（e4m3 别名） | $M \times N$ | $M \times N$ | $M \cdot N$ |
| `exp` | `uint8_t`（E8M0） | $M \times N/32$ | $M/32 \times N$ | $M \cdot N / 32$ |
| `max`（scratch） | $T$ | $M \times N/32$ | $M/32 \times N$ | $M \cdot N / 32 \cdot \mathrm{sizeof}(T)$ |
| `scaling`（scratch） | $T$ | $M \times N/32$ | $M/32 \times N$ | $M \cdot N / 32 \cdot \mathrm{sizeof}(T)$ |

### MXFP4（e2m1）

同 MXFP8，但：
| Tile | dtype | 字节数 |
|------|-------|--------|
| `dst` | `float4_e2m1x2_t`（每字节打包 2 个 e2m1） | $M \cdot N / 2$ |

> **输入限制：** MXFP4 仅接受 **FP16/BF16**（不支持 FP32）。

### INT8

| Tile | dtype | 形状 | 字节数 |
|------|-------|------|--------|
| `src` | `float32_t` | $M \times N$ | $M \cdot N \cdot 4$ |
| `dst`（SYM） | `int8_t` | $M \times N$ | $M \cdot N$ |
| `dst`（ASYM） | `uint8_t` | $M \times N$ | $M \cdot N$ |
| `scale` | FP32 标量/向量 | 每 Tile | — |
| `offset`（ASYM） | FP32 标量/向量 | 每 Tile | — |
| `tmp`（仅 A2/A3） | FP32 | $M \times N$ | $M \cdot N \cdot 4$ |

> **`tmp` Tile（仅 A2/A3）：** 必须与 `src` **同尺寸**（$M \times N$ FP32 = $4MN$ 字节），保存 FP32→S32 类型转换中间结果（A3 无原地 `tcvt`）。A5 接受同名 `tmp` 参数以保持接口一致，但**不使用它**（A5 原生 `vlds BRC_B32` 广播）。

## 约束

| 约束 | 适用范围 | 原因 |
|------|---------|------|
| $M \bmod 16 = 0$ | ND MX（ZZ 布局） | 16 行 ZZ 块 |
| $M \bmod 32 = 0$ | DN MX | axis-0 组整除 |
| $M \bmod 64 = 0$ | DN MX + ZZ 转换 | δ 配对（$\hat M / 2$ 为整数） |
| $N \bmod 32 = 0$ | 所有 MX | 组大小 $G = 32$ |
| $N \bmod 64 = 0$ | ND MX + ZZ 转换 | 指数组数为偶数 |
| $R \cdot C \le 59461$ | MX（UB 256KB） | 复用后的缓冲预算 |
| BF16/FP16：`validCols % 32 != 0` → 零填充至 `StaticCols` | MX B16 路径 | 组对齐 |

## 输出布局与布局转换

TQUANT 默认输出 **ND**（行主序）。Cube Unit 消费两种 fractal 布局，由独立的 `TMOV` 指令生成：

| 输出 | 原生（TQUANT） | Cube 布局 | 转换 |
|------|---------------|-----------|------|
| FP8 / FP4 数据 | ND | NZ（ColMajor+RowMajor fractal） | `TMOV(dstNZ, dst)`（2 参数） |
| E8M0 指数（ND 分组） | ND | ZZ（zigzag，`[16,2]` 块） | `TMOV(e8Zz, e8, tmp)`（3 参数） |
| E8M0 指数（DN 分组） | DN | ZZ | `TMOV<0>(e8Zz, e8Dn, tmp)`（3 参数，`grp_axis=0`） |

DN 数据的 FP8 mantissa 与 ND 共享相同的物理地址（`(r,c)` 元素完全相同），因此 2 参数 `TMOV` ND→NZ 对 DN 数据同样适用。仅**指数**路径不同（DN→ZZ 通过 `TMOV<0>`）。详见 `TQUANT_DN.md`。

## 支持的输入 dtype

| 格式 | 可接受输入 dtype | 说明 |
|------|-----------------|------|
| MXFP8 | FP32、BF16、FP16 | FP32：原地 FP8 输出（4:1）。BF16/FP16：源零填充至 `StaticCols`；类型转换前先上转为 FP32（无直接 b16→e4m3）。 |
| MXFP4（e2m1） | **仅 FP16、BF16**（不支持 FP32） | 输出 `float4_e2m1x2_t`（打包）。 |
| INT8（sym/asym） | **仅 FP32** | SYM→`int8_t`，ASYM→`uint8_t`。A2/A3 需 `tmp` = src 尺寸。 |

## 数学语义

除另有说明外，语义在有效区域内定义，目标相关行为标记为实现定义。

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

详见 `TQUANT_DN.md`（DN→ZZ 转换）和 `tests/npu/a5/src/st/testcase/tquant_dn/`（完整 ST 示例）。
