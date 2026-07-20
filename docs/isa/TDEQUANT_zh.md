# TDEQUANT

## 简介

将低精度量化Tile（`S8` / `S16`）反量化为高精度 `FP32` Tile。对每个元素执行**仿射反量化**（affine dequantization），即 `TQUANT` 整数量化（INT8对称 / 非对称）的逆运算：

$$ \mathrm{dst}_{i,j} = (\mathrm{src}_{i,j} - \mathrm{offset}_{i}) \cdot \mathrm{scale}_{i} $$

`scale` 与 `offset` 为**每行一组**的FP32参数（在列方向广播到整行），因此可在一条指令内完成"去偏移 + 反缩放"的完整反量化。

## 数学语义

概念上，对于有效区域中的每个元素 `(i, j)`：

$$ \mathrm{dst}_{i,j} = \bigl(\mathrm{src}_{i,j} - \mathrm{offset}_{i}\bigr) \cdot \mathrm{scale}_{i} $$

- `src`：量化后的整数码（`S8` 或 `S16`）。
- `scale`、`offset`：每行的FP32反量化参数（按行索引 `i` 选取参数组，沿列方向广播到整行）；`scale` 的有效列数 `paraCols = max(1, scale.GetValidCol())`，参数列下标 `paraCol = min(j, paraCols - 1)` 仅用于在参数Tile多列时钳位读取列下标（典型用法每行1个标量，`paraCols = 1`），参数组本身由行 `i` 决定。
- 与 `TQUANT` 整数仿射量化互逆：`TQUANT` 中 $q = \mathrm{round}(x / \mathrm{scale}) + \mathrm{offset}$，故 $x = (q - \mathrm{offset}) \cdot \mathrm{scale}$。

> 除非另有说明，语义在有效区域内定义，目标相关行为标记为实现定义。`scale` 与 `offset` 均为ISA可见的Tile操作数（非编译器scratch）。

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp`。
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
template <typename TileDataDst, typename TileDataSrc, typename TileDataPara, typename... WaitEvents>
PTO_INST RecordEvent TDEQUANT(TileDataDst &dst, TileDataSrc &src, TileDataPara &scale, TileDataPara &offset,
                              WaitEvents &...events);
```

| 参数 | 方向 | 含义 |
|------|------|------|
| `dst` | 输出 | 反量化结果Tile，`FP32`，行主序 |
| `src` | 输入 | 量化源Tile，`S8` 或 `S16`，行主序，有效形状与 `dst` 相同 |
| `scale` | 输入 | 每行反缩放系数，`FP32`，沿列广播 |
| `offset` | 输入 | 每行零点偏移，`FP32`，沿列广播 |
| `events...` | 输入 | 等待事件（`WaitEvents`），指令前隐式 `TSYNC` |

`scale` 与 `offset` 必须为同一类型（`TileDataPara`），且其dtype与 `dst` 一致（均为 `FP32`）。

## Tile尺寸与数据类型

对于源/目的Tile有效形状 $M \times N$：

| Tile | dtype | 有效形状 | 布局 | 说明 |
|------|-------|---------|------|------|
| `dst` | `float32_t` | $M \times N$ | RowMajor | 反量化结果 |
| `src` | `int8_t` 或 `int16_t` | $M \times N$ | RowMajor | 量化整数码；dtype决定unpack路径 |
| `scale` | `float32_t` | $M \times 1$（每行） | ColMajor / 行广播 | 沿列广播（`BRC_B32`） |
| `offset` | `float32_t` | $M \times 1$（每行） | ColMajor / 行广播 | 沿列广播（`BRC_B32`） |

> `scale`/`offset` 的有效行数必须等于 `dst` 的有效行数；列方向以32字节块为单位广播，故典型用法为每行1个标量（形状 $M \times 1$）。

## 支持的输入dtype

| 源dtype | 目的dtype | scale/offset dtype | 说明 |
|----------|-----------|--------------------|------|
| `S8` (`int8_t`) | `FP32` | `FP32` | `UNPK4_B8` 解包后类型转换为FP32 |
| `S16` (`int16_t`) | `FP32` | `FP32` | `UNPK_B16` 解包后类型转换为FP32 |

> `dst`、`scale`、`offset` 必须dtype一致且均为 `FP32`；`src` 仅支持 `S8` / `S16`。其它dtype组合为非法（实现内 `static_assert` 拦截）。

## 实现说明

TDEQUANT在向量流水线（`PIPE_V`）上执行，无需 `tmp` scratch Tile（与 `TQUANT` 在Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品上的5阶段类型转换链不同）：

1. **加载并解包 `src`**：`S8` 经 `UNPK4_B8`、`S16` 经 `UNPK_B16` 解包，再 `vcvt` 转为FP32（kirinX90上 `S8` 走 `US_B8` + 交错路径）。
2. **广播加载参数**：`scale`、`offset` 经 `vlds ... BRC_B32` 以32字节块广播到整行。
3. **计算**：`vsub(dst, src, offset)` 后 `vmul(dst, dst, scale)`，即先去偏移、再反缩放。

## 编码

TDEQUANT属于TEPL（Tile Elementwise Pipeline）复合变换类指令：

```text
BSTART.TEPL TDEQUANT, DataType +
B.DATR(optional) +
B.DIM LB0 +
B.DIM (LB1/LB2 for 2D) +
B.IOT
```

| 字段 | 取值 |
|------|------|
| Mode | 3（复合变换） |
| Function | 11 |
| TileOp | `0x6B` |
| 操作数（`B.IOT`） | `dst, src, scale, offset` |

## 约束

| 约束 | 适用范围 | 原因 |
|------|---------|------|
| `dst`、`src` 必须行主序 | 所有目标 | 反量化按行广播参数 |
| `dst` 与 `src` 有效形状相同（$M \times N$） | 所有目标 | 逐元素一一对应 |
| `scale`、`offset` 有效行数 == `dst` 有效行数 | 所有目标 | 每行一组参数 |
| `dst`/`scale`/`offset` dtype必须一致且为 `FP32` | 所有目标 | 反量化输出精度 |
| `src` ∈ {`S8`, `S16`} | 所有目标 | 整数量化码字宽 |

## 示例

```cpp
// src: int8/int16 量化码；scale/offset: 每行 FP32 参数
TDEQUANT(dstTile, srcTile, scaleTile, offsetTile);
```

完整ST示例见 `tests/npu/a5/src/st/testcase/tdequant/`（A5）、`tests/npu/a2a3/src/st/testcase/tdequant/`（A2/A3）及 `tests/cpu/st/testcase/tdequant/`（CPU参考实现）。
