# TREM

## 产品支持情况<a name="section73648168211"></a>

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：支持
<!-- end id3 -->


## 指令示意图

![TREM tile operation](../figures/isa/TREM.svg)

## 简介

两个Tile的逐元素余数运算。除数非零时，非零结果的符号与除数相同。

## 数学语义

对有效区域内除数非零的每个元素 `(i, j)`：

$$\mathrm{dst}_{i,j} = \mathrm{remainder}(\mathrm{src0}_{i,j}, \mathrm{src1}_{i,j}) = \mathrm{src0}_{i,j} - \mathrm{floor}(\frac{\mathrm{src0}_{i,j}}{\mathrm{src1}_{i,j}}) \times \mathrm{src1}_{i,j}$$

当除数（`src1`）非零时，非零结果的符号与除数相同；整除时结果为零。
例如，`remainder(-7, 3) = 2`、`remainder(7, -3) = -2`、`remainder(-6, 3) = 0`。
在Ascend 950PR/Ascend 950DT上，`int64_t` 遵循上述向下取整定义，在除数非零时余数语义与 `int32_t` 一致。

**注意**：这与 `TFMOD` 不同，`TFMOD` 的结果符号与被除数（`src0`）相同。

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
template <auto PrecisionType = RemAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0,
          typename TileDataSrc1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TREM(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp, WaitEvents &... events);
```

`PrecisionType`可指定以下值：

* `RemAlgorithm::DEFAULT`：普通算法，速度快但精度较低。
* `RemAlgorithm::HIGH_PRECISION`：高精度算法，仅支持`float`类型。

## 高精度算法（TREM）

`RemAlgorithm::HIGH_PRECISION` 运行与 TREMS 相同的 11 次迭代、以 $2^{24}$ 为基的逐位剥离和相同的 staged 循环倒置（算法与共享切分预算见 [TREMS 高精度章节](TREMS_zh.md#高精度算法)）。差异全部来自除数是张量：

- **每遍重载除数**：每一遍从原始 tile 重新读取 `|src1|`（`vabs` 到暂存的 `div` 区域），因为缩放过的除数不能喂给下一遍。
- **逐元素的次正规区间**：标量除数时“除数是否次正规”是宿主侧常量；此处第 1 次迭代逐元素判定 —— 在 `|src1| <= minNormal` 的通道上把除数与状态同乘 $2^{24}$（打包掩码比较 + 选择），后处理再精确地对这些通道乘回 $2^{-24}$。
- **异常修正为逐元素选择**：`src1 = ±inf → src0`；随后（必须在其后）`src0 = ±inf ∧ src1 = ±inf → NaN`；然后除数在保证域内时 `src0 = ±0 → src0`。
- **多一个暂存区域**：未缩放的 `|src1|` 行跨该遍的全部迭代存活，因此 TREM 的 tmp 基于 $192C$（$C=1$ 时 272 个 float），而 TREMS 为 $128C$ —— 见 [TREMS](TREMS_zh.md#tmp-占用) 的 `tmpCols` 分段方程。

**除数域**：`src1` 元素须非零且非 NaN。$\pm\infty$ 没问题（$x \bmod \pm\infty = x$）。保证域是 $|y| \ge 1$（或 $\pm\infty$）：余数符号修正探测 $r \cdot y$，当 $|r \cdot y| < 2^{-149}$ 时下溢为 $\pm 0$ 从而漏掉修正 —— 这是自 A5 实现继承的限制。更小的除数遵循 A5 迭代算术及该限制。

## 约束

- **实现检查 （Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品）**:
    - `TileData::DType` 必须是以下之一：`float`， `float32_t`， `int32_t`。
    - Tile布局必须是行主序（`TileData::isRowMajor`）。
    - Tile位置必须是向量（`TileData::Loc == TileType::Vec`）。
    - 运行时：`src0`、`src1` 和 `dst` tiles应具有相同的 `validRow/validCol`。
    - 原地操作（`dst` 与 `src0` 分配到同一UB区域）在两种精度模式下均支持；`src1` 必须与 `dst` 不重叠。
    - `tmp` 必须至少有 1 个有效行和 `TileDataDst::RowStride + 8 * ceil(TileDataDst::RowStride / 256) + 8` 个有效列，依次用于中间结果、按 32 字节对齐的压缩比较掩码和 32 字节的地址缓冲区。
- **实现检查 (Ascend 950PR/Ascend 950DT)**:
    - `TileData::DType` 必须是以下之一：`half`， `float`， `int16_t`， `uint16_t`， `int32_t`， `uint32_t`， `int64_t`， `uint64_t`。
    - Tile布局必须是行主序（`TileData::isRowMajor`）。
    - Tile位置必须是向量（`TileData::Loc == TileType::Vec`）。
    - 运行时：`src0`、`src1` 和 `dst` tiles应具有相同的 `validRow/validCol`。
    - 注意：`tmp` 参数在Ascend 950PR/Ascend 950DT上被接受但不使用。
- **有效区域**:
    - 该操作使用 `dst.GetValidRow()` / `dst.GetValidCol()` 作为迭代域。
- **除零**:
    - 在Ascend 950PR/Ascend 950DT上，`int64_t` 和 `uint64_t` 在除数为零时分别返回 `-1` 和 `UINT64_MAX`（64 位全为 1）。
    - 行为由目标定义；CPU仿真在调试构建中会断言。
- **高精度算法**:
    - 仅在Ascend 950PR/Ascend 950DT上对 `float` 类型有效；`PrecisionType` 选项在Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品上将被忽略。

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using TmpT = Tile<TileType::Vec, float, 1, 32>;
  TileT dst, src0, src1;
  TmpT tmp;
  TREM(dst, src0, src1, tmp);

  // 高精度（仅 float）：tmp 按 TREMS 切分预算取宽，
  // 例如 8x448 tile -> C = 7, tmpCols = 192*7 + 8*ceil(7/4) + 8 = 1368
  Tile<TileType::Vec, float, 8, 448> hpDst, hpSrc0, hpSrc1;
  Tile<TileType::Vec, float, 1, 1368> hpTmp;
  TREM<RemAlgorithm::HIGH_PRECISION>(hpDst, hpSrc0, hpSrc1, hpTmp);
}
```
