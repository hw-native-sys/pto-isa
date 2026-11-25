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

![TREM tile operation](../../figures/isa/TREM.svg)

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

`RemAlgorithm::HIGH_PRECISION` 运行与 TREMS 相同的 11 次迭代、以 $2^{24}$ 为基的逐位剥离和相同的 staged 循环倒置（算法与共享切分预算见 [TREMS 高精度章节](TREMS.md#高精度算法)）。差异全部来自除数是张量（详见[附录](#section_trem_appendix)）：

- **每遍重载除数**：每一遍从原始 tile 重新读取 `|src1|`（`vabs` 到暂存的 `div` 区域），因为缩放过的除数不能喂给下一遍。
- **逐元素的次正规区间**：标量除数时“除数是否次正规”是宿主侧常量；此处第 1 次迭代逐元素判定 —— 在 `|src1| <= minNormal` 的通道上把除数与状态同乘 $2^{24}$（打包掩码比较 + 选择），后处理再精确地对这些通道乘回 $2^{-24}$。
- **异常修正为逐元素选择**：`src1 = ±inf → src0`；随后（必须在其后）`src0 = ±inf ∧ src1 = ±inf → NaN`；然后除数在保证域内时 `src0 = ±0 → src0`。
- **多一个暂存区域**：未缩放的 `|src1|` 行跨该遍的全部迭代存活，因此 TREM 的 tmp 基于 $192C$（$C=1$ 时 912 个 float，含 640 float 的补偿窗口；更宽的行按 $\min(C,8)$ 增长），而 TREMS 为 $128C$ —— 见 [TREMS](TREMS.md#tmp-占用) 的 `tmpCols` 分段方程。

**除数域**：`src1` 元素须非零且非 NaN。$\pm\infty$ 没问题（遵循向下取整语义，见 [TREMS 特殊值表](TREMS.md)）。保证域是 $|y| \ge 1$（或 $\pm\infty$）：余数符号修正探测 $r \cdot y$，当 $|r \cdot y| < 2^{-149}$ 时下溢为 $\pm 0$ 从而漏掉修正 —— 这是自 A5 实现继承的限制。更小的除数遵循 A5 迭代算术及该限制。

## 支持的 Tile 尺寸（A2/A3）

TREM 比 TREMS 多一个 `tmp` 区域（未缩放的 `|src1|` 行），且 int DEFAULT 路径的全部 float 域转换都走 `tmp`（两个输入 tile 均只读）。各模式的 `tmp` 列数，其中 $R$ 为 dst 行跨度（元素数）、$\mathit{maskF} = 8\lceil R/256\rceil$、$C = \max(\lfloor N/64 \rfloor, 1)$：

$$
\mathit{tmpCols}(N)=
\begin{cases}
R+8\lceil R/256\rceil+8, & \text{DEFAULT},\ \texttt{float} \\[2pt]
3R+8\lceil R/256\rceil+8, & \text{DEFAULT},\ \texttt{int32\_t} \\[2pt]
272+640, & \text{HIGH\_PRECISION},\ C=1 \\[2pt]
192C+8\lceil C/4\rceil+8+640\min(C,8), & \text{HIGH\_PRECISION},\ C\ge 2
\end{cases}
$$

HIGH_PRECISION 的 $C=1$ 分支含 64 float 的逐向量状态缓冲；$640\min(C,8)$ 项为末次迭代精确化的分批补偿窗口（8 位 Veltkamp 拆分、结果有限性守护，详见 TREMS.md）。192 KB 预算与 `validCols/64 <= 255` 下各行的最宽合法 `N`：

| 行数 | TREM float | TREM int32 | TREM HP |
|---|---|---|---|
| 1 | 12160 | 8128 | 7296 |
| 2 | 6976 | 5440 | 4864 |
| 4 | 3712 | 3264 | 2880 |
| 8 | 1920 | 1792 | 1600 |
| 16 | 960 | 960 | 832 |
| 32 | 448 | 448 | 448 |

更宽的 tile 需拆分为多次调用。两种精度模式均支持原地 `dst == src0`（`src1` 必须保持不重叠）。

## 约束

- **实现检查 （Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品）**:
    - `TileData::DType` 必须是以下之一：`float`， `float32_t`， `int32_t`。
    - Tile布局必须是行主序（`TileData::isRowMajor`）。
    - Tile位置必须是向量（`TileData::Loc == TileType::Vec`）。
    - 运行时：`src0`、`src1` 和 `dst` tiles应具有相同的 `validRow/validCol`。
    - 原地操作（`dst` 与 `src0` 分配到同一UB区域）在两种精度模式下均支持；`src1` 必须与 `dst` 不重叠。
    - `tmp` 必须至少有 1 个有效行和 `TileDataDst::RowStride + 8 * ceil(TileDataDst::RowStride / 256) + 8` 个有效列（float；依次用于中间结果、按 32 字节对齐的压缩比较掩码和 32 字节的地址缓冲区），或 `3 * TileDataDst::RowStride + 8 * ceil(TileDataDst::RowStride / 256) + 8` 个有效列（int32_t —— int 路径将被除数、除数的 float 域中间量与商链全部保存在 `tmp`，两个输入 tile 均只读）。
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
  // 例如 8x448 tile -> C = 7, tmpCols = 192*7 + 8*ceil(7/4) + 8 + 640*7 = 5848
  Tile<TileType::Vec, float, 8, 448> hpDst, hpSrc0, hpSrc1;
  Tile<TileType::Vec, float, 1, 5848> hpTmp;
  TREM<RemAlgorithm::HIGH_PRECISION>(hpDst, hpSrc0, hpSrc1, hpTmp);
}
```

## 附录：向量除数下的 TREM（A2/A3）<a name="section_trem_appendix"></a>

共享高精度核心的数学规范 —— 语义与定义域、以 $2^{24}$ 为基的数字剥离梯、精确补偿的末次迭代、有限性守卫、次正规区间与后处理 —— 见 [TREMS 附录](TREMS.md#section_hp_appendix)，此处不再重复。本附录说明除数为张量时的差异。

- **每遍重载除数。** 每一遍都从*原始* tile 重读除数 $b_i = \mathrm{RN}(|\mathit{src1}| \cdot 2^{e(i)})$（`vabs` + 2 的幂缩放），绝不复用上一遍的缩放缓冲：梯级的每一级都必须是 $|y|$ 本身的精确缩放，复用缩放过的除数会使缩放因子复合、破坏 A.2 的梯。因此未缩放的 $|\mathit{src1}|$ 行跨整遍存活，这就是 TREM 的 tmp 基于 $192C$（对比 TREMS 的 $128C$）多出的那一个暂存区域。

- **逐元素次正规区间。** 除数是否次正规在此是逐通道性质而非宿主常量：第 1 次迭代仅在 $|\mathit{src1}| \le \mathit{minNormal}$ 的通道上施加 TREMS 附录（A.5）的 $2^{24}$ 提升（打包掩码比较 + 选择），后处理仅对这些通道乘回 $2^{-24}$。

- **后处理顺序。** TREMS 附录（A.6）的链逐元素执行、除数读自 $\mathit{src1}$：符号恢复 → 次正规缩回 → $|\mathit{src1}| = \pm\infty$ 选择 $x$ → 双无穷选择 NaN → $x = \pm 0$ 选择 $x$ → 余数符号修正（$r \cdot y < 0 \Rightarrow r + y$）→ $|y| = \pm\infty$ 通道上的向下取整无穷选择（同号 $x$、异号 $x + y$、$\pm 0$ 保持 $x$）→ $|x| = \pm\infty$ 上重施加 NaN。NaN 重施加放在最后，否则向下取整无穷选择会用被除数覆盖双无穷 NaN 通道。

- **逐通道等价。** 与 TREMS 附录（A.7）相同，逐向量、staged 与分块调度求值同一张逐通道就近舍入数据流图，各组织产出按位一致的逐通道结果。
