# TREMS

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

![TREMS tile operation](../figures/isa/TREMS.svg)

## 简介

与标量的逐元素余数：`remainder(src, scalar)`。标量除数非零时，非零结果的符号与除数相同。

## 数学语义

对有效区域内除数非零的每个元素 `(i, j)`：

$$\mathrm{dst}_{i,j} = \mathrm{src}_{i,j} - \mathrm{floor}(\frac{\mathrm{src}_{i,j}}{\mathrm{scalar}}) \times \mathrm{scalar}$$

当 `scalar` 非零时，非零结果的符号与 `scalar` 相同；整除时结果为零。
例如，`remainder(-7, 3) = 2`、`remainder(7, -3) = -2`、`remainder(-6, 3) = 0`。
在Ascend 950PR/Ascend 950DT上，`int64_t` 遵循上述向下取整定义，在除数非零时余数语义与 `int32_t` 一致。

## C++内建接口

声明于 `include/pto/common/pto_instr.hpp`：
> 公共包含头为 `<pto/pto-inst.hpp>`，内部声明位于 `pto/common/pto_instr.hpp`。

```cpp
template <auto PrecisionType = RemSAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TREMS(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar, TileDataTmp &tmp,
                           WaitEvents &...events);
```

`PrecisionType`可指定以下值：

* `RemSAlgorithm::DEFAULT`：普通算法，速度快但精度较低。
* `RemSAlgorithm::HIGH_PRECISION`：高精度算法，速度较慢，仅支持`float`类型。

## 高精度算法

高精度模式计算同样的向下取整求余，但把结果解析到完整 `float32` 精度，而不是在 $|x/s| > 2^{24}$ 后退化（默认路径的一次成形乘积 $q \cdot s$ 在 $x$ 的量级上舍入，随后的减法灾难性抵消，只剩 $\pm(1\ldots 2)\,\mathrm{ulp}(x)$ —— 在 `FLT_MAX` 边界附近甚至翻转为 $\pm\infty$）。特殊值遵循 IEEE 语义（即 numpy 的 `x % s`）：

| x | s | 结果 |
|---|---|------|
| $\pm\infty$ | 有限值 | NaN |
| 有限值 | $\pm\infty$ | x（若 x 也是 $\pm\infty$ 则为 NaN） |
| $\pm 0$ | $s \ne 0$ | $\pm 0$（保留 x 的符号） |
| NaN | 任意 | NaN |

算法为以 $2^{24}$ 为基的逐位剥离长除法。在 $\mathit{state} = |x|$、$\mathit{div} = |s|$ 上工作；f32 商的动态范围可达 $\sim 2^{276}$，11 次迭代（$\lceil 276/24 \rceil = 11$）从高位起按约 24 bit 一段剥离指数差：

| 迭代（自高向低执行） | 除数缩放 $\mathit{div} \cdot 2^{e}$ |
|---|---|
| 11 | $2^{240}$ |
| 10 | $2^{216}$ |
| 9 | $2^{192}$ |
| 8 | $2^{168}$ |
| 7 | $2^{144}$ |
| 6 | $2^{120}$ |
| 5 | $2^{96}$ |
| 4 | $2^{72}$ |
| 3 | $2^{48}$ |
| 2 | $2^{24}$ |
| 1 | $2^{0}$（不缩放）+ 次正规区间 |

$$
\begin{aligned}
b &= \mathit{div} \cdot 2^{e(i)} \\[2pt]
k &= \begin{cases} \lfloor \mathit{state}/b \rfloor & \text{求余 remainder} \\[2pt]
      \operatorname{trunc}(\mathit{state}/b) & \text{fmod} \end{cases} \\[2pt]
\mathit{state} &\leftarrow \mathit{state} - k \cdot b
\end{aligned}
$$

$b$ 是除数的精确 2 的幂缩放；$k$ 为整数，在 f32 中精确（int32 转换饱和，相当于极端指数差下免费的商钳制）；减法只在该量级的 chunk 上有 $\le 1$ ulp 舍入 —— 从不在 $|x|$ 的量级上舍入 —— 从而抵消掉指数差的最高约 24 bit。第 1 次迭代额外在 $|s| \le \mathit{minNormal}$ 时把除数与状态同乘 $2^{24}$（后处理再乘回 $2^{-24}$），执行最终不缩放的 $k = \lfloor \mathit{state}/|s| \rfloor$，并做次正规尺度的符号修正。后处理恢复被除数符号、应用上表的异常修正，并修正余数符号（$\mathit{state} \cdot s < 0 \Rightarrow \mathit{state} \leftarrow \mathit{state} + s$）。

算法的严格数学规范见[附录](#section_hp_appendix)。

存在两种执行组织方式（issue #2 的循环倒置）。逐向量路径一次一个 64 元素向量，在暂存 state 缓冲中跑完全部 11 次迭代 —— 小 tile、重叠 tile 和行尾的回退路径。staged 路径**每个满向量每遍只推进一次迭代**，逐向量状态暂存于 `dst`；在 A2/A3 上每一遍中的每条操作都是一条覆盖该行全部满向量的 repeat 指令：

```
seed:      dst[row] = |src[row]|                     (vabs, rep = fullPerRow)
第 11→2 遍: 对每一行: 在 dst[row] 上原地做 SolveScale 迭代  (rep = fullPerRow)
末遍(第1遍): 按 min(8, C) 个向量分批执行补偿减法（补偿窗口随分批取宽）
post:      对每一行: HpPostS
行尾:      每行不足 64 元素的尾部走逐向量回退
```

当 `validRows * (validCols / 64) >= TREM_HP_STAGED_MIN_VECS`（48）且 `dst` tile 不与输入重叠时启用 staged；`validCols / 64 <= 255`（8 bit repeatTimes）始终适用 —— 更宽的 tile 请拆分成多次调用。

### 末次迭代精确化

最后一次剥离迭代以单次舍入语义计算 `state -= k * b`。A5 用融合 `vmula` 一步完成；A2/A3 没有融合乘加（`vmla` 先对乘积舍入），因此用 Veltkamp 常数 $257 = 2^8+1$ 拆分 $b$（$b_H \le 8$ 位有效数字，$b_H + b_L = b$ 精确），并把小于 $2^{24}$ 的商数字 $k$ 拆成三个 8 位数字 $k_1 2^{16} + k_2 2^8 + k_3$。每个部分积 $k_i \cdot b_{\{H,L\}}$ 不超过 24 位、均精确；一串标准 TwoSum 把 $k \cdot b$ 精确重组为 `(hi, lo)`，两部分先后相减只舍入一次，得到真余数（真余数总是可表示的）。staged 末遍按 $\min(C, 8)$ 个向量分批（`TREM_HP_FUSED_CHUNK_A2A3`）。结果有限性守卫（$|r| \le \text{FLT\_MAX}$；NaN 比较为假）把罕见的近 FLT_MAX 溢出通道与 $|b| > \text{FLT\_MAX}/257$ 区域（拆分本身溢出）引到普通两步结果，其误差不超过被除数的半个 ulp。这使得 $4580025 \bmod -7.25$ 之类得到精确结果（$-3.0$ 而非 $-3.25$），$\text{FLT\_MAX} \bmod \text{FLT\_MAX}/2^{23}$ 精确为 $0$（而非 NaN），并让富尾数除数在 $|b|$ 达 $\text{FLT\_MAX}/257$ 时仍得到真余数。

## 高精度切分预算

### tmp tile 布局

高精度暂存是 `tmp` tile 上一个按容量取宽的窗口，$C = \max(\lfloor N/64 \rfloor, 1)$ 个向量（$N$ = 有效列数）。生命周期不相交的缓冲共享存储：

| 区域 | float 偏移 | 生命周期 |
|---|---|---|
| `bTmp` | `[0, 64C)` | 单次迭代内 —— 缩放后的除数，每迭代由宿主侧标量 `vector_dup` |
| `k` | `[64C, 128C)` | 商；乘积 `vmul` 之后兼作通用临时量（符号因子、选择操作数、探测值、修正候选） |
| `cmpMask` | `[128C, +maskF)` | 打包比较位掩码，从 `vcmp` 直到消费它的 `vsel` |
| `addrBuf` | `+8` | `cmpMask` 的完整 64 位地址，伴随每次 `set_cmpmask` |
| `state` | `+64`，仅 $C = 1$ | 逐向量回退跨 11 次迭代的状态 |
| `comp` | `+640\,\min(C,8)$ | 末次迭代的补偿减法（`HpFusedSub`），按 $\min(C,8)$ 个向量分批：8 位 Veltkamp 拆分、三个 8 位商数字、六个精确部分积与 $k \cdot b$ 的 TwoSum 重组 |

商/临时量共用是安全的：`k` 在读取它的 `vmul` 处即死亡，后处理的探测值在被 `vcmp` 消费后才被候选覆盖 —— 每次交接都以 `pipe_barrier` 分隔（原地 `dst == src0` 操作是 A2/A3 的既有习惯）。staged 路径的状态在 `dst` 中，`state` 绝不随 $C$ 增长。

### tmp 占用

$$
\mathit{tmpCols}(N)=
\begin{cases}
208+640, & \text{TREMS},\ C=1 \\[2pt]
128C+8\lceil C/4\rceil+8+640\min(C,8), & \text{TREMS},\ C\ge 2 \\[2pt]
272+640, & \text{TREM},\ C=1 \\[2pt]
192C+8\lceil C/4\rceil+8+640\min(C,8), & \text{TREM},\ C\ge 2
\end{cases}
\qquad
C=\max\Bigl(\Bigl\lfloor \frac{N}{64}\Bigr\rfloor,\ 1\Bigr)
$$

其中 $8\lceil C/4\rceil$ 为打包比较掩码（按 32 字节对齐填充）；$C=1$ 的常量分支折入了 64 个 float 的逐向量 state 缓冲。TREM 比 TREMS 多一个区域 —— 未缩放的 `|src1|` 行必须跨该遍的全部迭代存活作为重新缩放的源（它不能放进会被第 1 次迭代覆盖的 `bTmp`，也不能放进会被商中途覆盖的 `k`）。

### 各 tile 占用与最大切分

数据 tile 共 $n_{\text{tiles}} = 2$（TREMS）或 3（TREM）块 $M \times N$：

$$
\mathit{UBUF}(M,N) \;=\;
\underbrace{n_{\text{tiles}}\cdot 4MN}_{\text{数据 tile}} \;+\;
\underbrace{4\,\mathit{tmpCols}(N)}_{\text{tmp}}
\;\le\; 196608\ \mathrm{B}, \qquad C \le 255
$$

算例 —— 每个 tile 的字节大小：

| tile $[M, N]$ | dst | src | tmp | 总 UBUF | 占 196608 B | 分发（$M \cdot C$ 对 48） |
|---|---|---|---|---|---|---|
| $[1, 64]$ | 256 | 256 | 3392 | 3904 | 2% | 逐向量（1） |
| $[32, 64]$ | 8192 | 8192 | 3392 | 19776 | 10% | 逐向量（32） |
| $[12, 256]$ | 12288 | 12288 | 12352 | 36928 | 19% | staged（48） |
| $[8, 448]$ | 14336 | 14336 | 21600 | 50272 | 26% | staged（56） |
| $[2, 2048]$ | 16384 | 16384 | 37152 | 69920 | 36% | staged（64） |
| $[1, 8192]$ | 32768 | 32768 | 87072 | 152608 | 78% | staged（128） |
| $[8, 2432]$ | 77312 | 77312 | 38528 | 196224 | 100% | staged（304） |
| $[1, 10880]$ | 43520 | 43520 | 108032 | 195968 | 100% | staged（170） |

每个行数下最大的打包 $W$（TREMS；经验公式 $W \lesssim 21760/(R+1)$）：

| 行数 R | 1 | 2 | 3 | 4 | 7 | 8 | 12 | 16 | 32 |
|---|---|---|---|---|---|---|---|---|---|
| 最大 W | 10880 | 7296 | 5440 | 4352 | 2688 | 2432 | 1664 | 1280 | 640 |

TREM（$W \lesssim 14400/(R+1)$）：相同行数下最大 W 分别为 7296 / 4864 / 3648 / 2880 / 1792 / 1600 / 1088 / 832 / 448；单行合法最大宽度为 `[1, 7296]`（196544 B）。

切分设计清单：
- 先定工作宽度 $N$ —— 它单独决定 tmp 大小；行数再由 $M \le (196608 - 4\,\mathit{tmpCols}(N))\,/\,(4N \cdot n_{\text{tiles}})$ 约束。
- 若需要 staged 路径：$C \le 255$ 且 $M \cdot C \ge 48$。
- 超出预算的按列分块：每次调用处理一个 tile。
- DEFAULT 路径的 tmp 契约小得多（`dstRowStride + maskF + 8` 个 float），同一 tile 可能对 DEFAULT 合法而对 HIGH_PRECISION 非法；请按实际分发的精度模式核对预算。

## 支持的 Tile 尺寸（A2/A3）

DEFAULT 路径的 `tmp` 列数，其中 $R$ 为 dst 行跨度（元素数）、$\mathit{maskF} = 8\lceil R/256\rceil$（int 路径将被除数的 float 域中间量与商链保存在 `tmp` —— 输入 tile 只读）：

$$
\mathit{tmpCols}_{\text{DEFAULT}}(N)=
\begin{cases}
R+8\lceil R/256\rceil+8, & \texttt{float} \\[2pt]
2R+8\lceil R/256\rceil+8, & \texttt{int32\_t}
\end{cases}
$$

HIGH_PRECISION 路径需满足上文[高精度切分预算](#高精度切分预算)中更大的契约。在 192 KB 统一缓冲预算与 `validCols/64 <= 255`（8 位 repeat 计数）限制下，各行数对应的最宽合法 `N`：

| 行数 | TREMS float | TREMS int32 | TREMS HP |
|---|---|---|---|
| 1 | 16192 | 12160 | 10880 |
| 2 | 9728 | 8128 | 7296 |
| 4 | 5440 | 4864 | 4352 |
| 8 | 2880 | 2688 | 2432 |
| 16 | 1472 | 1408 | 1280 |
| 32 | 704 | 704 | 640 |

经验公式：float 约 `N \lesssim 23500/(行数+1)`，int32 约 `N \lesssim 17500/(行数+1)`，staged-HP 上限见预算章节。更宽的 tile 需拆分为多次调用。两种精度模式均支持原地 `dst == src`。

## 约束

- **实现检查 （Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品）**:
    - `dst` 和 `src` 必须使用相同的元素类型。
    - 支持的元素类型：`float` 和 `int32_t`。
    - `dst` 和 `src` 必须是向量Tile。
    - `dst` 和 `src` 必须是行主序。
    - 运行时：`dst.GetValidRow() == src.GetValidRow() > 0` 且 `dst.GetValidCol() == src.GetValidCol() > 0`。
    - 原地操作（`dst` 与 `src` 分配到同一UB区域）在两种精度模式下均支持。
    - **tmp缓冲区要求**：
      - `tmp.GetValidCol() >= dstRowStride + maskF + 8`（float），或 `>= 2*dstRowStride + maskF + 8`（int32_t —— int 路径将被除数的 float 域中间量与商链保存在 `tmp`，输入 tile 只读），其中 `maskF = 8*ceil(dstRowStride/256)`
      - `tmp.GetValidRow() >= 1`（至少1行）
      - 数据类型必须与 `TileDataDst::DType` 匹配。
- **实现检查 (Ascend 950PR/Ascend 950DT)**:
    - `dst` 和 `src` 必须使用相同的元素类型。
    - 支持的元素类型：`float`、`int32_t`、`uint32_t`、`int64_t`、`uint64_t`、`half`、`int16_t` 和 `uint16_t`。
    - `dst` 和 `src` 必须是向量Tile。
    - 两个Tile的静态有效边界都必须满足 `ValidRow <= Rows` 且 `ValidCol <= Cols`。
    - 运行时：`dst.GetValidRow() == src.GetValidRow()` 且 `dst.GetValidCol() == src.GetValidCol()`。
    - 注意：tmp参数在Ascend 950PR/Ascend 950DT上被接受但不进行验证或使用。
- **除零**:
    - 在Ascend 950PR/Ascend 950DT上，`int64_t` 和 `uint64_t` 在 `scalar` 为零时分别返回 `-1` 和 `UINT64_MAX`（64 位全为 1）。
    - 行为由目标定义；CPU模拟器在调试构建中会断言。
- **有效区域**:
    - 该操作使用 `dst.GetValidRow()` / `dst.GetValidCol()` 作为迭代域。
- **对于 `int32_t` 输入（仅Atlas A2/A3 训练系列产品/Atlas A2/A3 推理系列产品）**：`src` 的元素和 `scalar` 必须在 `[-2^24, 2^24]` 范围内（即 `[-16777216, 16777216]`），以确保在计算过程中能精确转换为float32。
- **高精度算法**
    - 仅在Ascend 950PR/Ascend 950DT上有效，`PrecisionType`选项在Atlas A3 训练系列产品/Atlas A3 推理系列产品上将被忽略。

## 示例

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  Tile<TileType::Vec, float, 16, 16> tmp;
  TREMS(out, x, 3.0f, tmp);

  // 高精度：8x448 tile，tmp 按高精度切分预算取宽
  // (C = 448/64 = 7, tmpCols = 128*7 + 8*ceil(7/4) + 8 + 640*7 = 5400)
  using HpTmpT = Tile<TileType::Vec, float, 1, 5400>;
  Tile<TileType::Vec, float, 8, 448> hpOut, hpSrc;
  HpTmpT hpTmp;
  TREMS<RemSAlgorithm::HIGH_PRECISION>(hpOut, hpSrc, 7.5f, hpTmp);
}
```

## 附录：高精度算法数学规范（A2/A3）<a name="section_hp_appendix"></a>

本附录给出当前 A2/A3 `HIGH_PRECISION` 实现的数学规范。记 $\mathrm{RN}$ 为 `float32` 就近舍入（24 位尾数，$\mathit{minNormal} = 2^{-126}$，$\mathit{FLT\_MAX} = (2 - 2^{-23})\,2^{127}$），$\operatorname{ulp}(v)$ 为 $v$ 的最小位数单位。下文所有内积的操作数均为非负数。

### A.1 语义与定义域

对有限的 $y \ne 0$，结果为向下取整求余：

$$
r \;=\; x - \lfloor x/y \rfloor\, y, \qquad r\,\operatorname{sign}(y) \in [\,0,\ |y|\,), \qquad \operatorname{sign}(r) = \operatorname{sign}(y)\ \text{（当}\ r \ne 0\text{）},
$$

按完整 `float32` 精度求值 —— 只要精确值可表示（由 A.3，这总是成立）则结果精确。契约：$y \ne 0$ 且 $y \ne \mathrm{NaN}$。保证范围：$|y| \ge 1$ 或 $y = \pm\infty$。特殊值：

| $x$ | $y$ | 结果 |
|---|---|---|
| NaN | 任意 | NaN |
| $\pm\infty$ | 任意 | NaN |
| $\pm 0$ | $y \ne 0$ | $x$（保留 $x$ 的符号） |
| 有限值 | $\pm\infty$ | $\operatorname{sign}(x) = \operatorname{sign}(y)$ 时为 $x$，否则 $\operatorname{copysign}(\infty, y) = x + y$ |

### A.2 数字剥离

核心以每次 24 bit 一段的方式，用 $\mathit{state} = |x|$ 对 $\mathit{div} = |y|$ 做长除。迭代 $i = 11, \dots, 1$ 使用刻度梯 $e(i) = 24(i-1)$，以至多两次精确的 2 的幂缩放实现（两步式避免中间溢出；TREMS 在宿主常量中折算 $\mathrm{RN}(\mathrm{RN}(|y| 2^{a}) 2^{b})$，TREM 在向量单元上施加）：

$$
b_i \;=\; \mathit{div}\cdot 2^{e(i)}, \qquad
k_i \;=\; \begin{cases} \lfloor \mathit{state}/b_i \rfloor & \text{求余（就近取整）} \\ \operatorname{trunc}(\mathit{state}/b_i) & \text{fmod} \end{cases} \qquad
\mathit{state} \;\leftarrow\; \mathit{state} - k_i\, b_i .
$$

每一步在实现中的性质：

- **缩放除数。** $b_i$ 是 $\mathit{div}$ 的精确 2 的幂缩放（不溢出则在 `float32` 中精确）。非末次迭代将 $b_i$ 钳制到 $\mathit{FLT\_MAX}$（`vmins`），因此 $b_i > \mathit{state}$ 的梯级得到 $k_i = 0$，且即使未钳制的 $b_i$ 为 $+\infty$，乘积 $0 \cdot b_i$ 仍为 $0$。
- **商数字。** $k_i = \mathrm{roundToInt}(\mathrm{RN}(\mathit{state}/b_i))$，经饱和式 `f32 → s32 → f32` 转换：一切 $|k_i| \le 2^{24}$ 的整数精确往返；更大的商（仅在 $|y| < 1$、保证范围之外可达）在 $2^{31}$ 饱和，每个梯级把指数差钳掉 $\ge 31$ bit。相邻梯级间隔 24 个指数位，第 $i+1$ 级留下 $\mathit{state} < b_{i+1}\cdot(1 + O(2^{-23}))$，故随后的每个数字满足 $k_i \le 2^{24}$ 且精确。
- **减法更新。** 对 $i \ge 2$，$\mathit{state} \leftarrow \mathrm{RN}(\mathit{state} - \mathrm{RN}(k_i b_i))$：乘积与减法都在 *该量级* 上舍入、从不在 $|x|$ 的量级上舍入，每次迭代抵消剩余指数差的高约 24 bit，累计误差沿梯级几何衰减。第 $i = 1$ 次迭代改用 A.3 的补偿更新，随后施加守卫修正 $\mathit{state} < 0 \Rightarrow \mathit{state} \leftarrow \mathit{state} + b_1$，恢复 $\mathit{state} \in [0, b_1)$。

第 1 次迭代之后，state 即 $|x|$ 对 $|y|$（次正规区间为对 $|y|\,2^{24}$，见 A.5）的精确向下取整余数，仅携带末次迭代的单次舍入 —— 而 A.3 将其消除。

### A.3 末次迭代精确化（`HpFusedSub`）

最后一次更新需以**单次舍入**语义减去 $k \cdot b$：即 $\mathrm{RN}(\mathit{state} - k\,b)$，其中 $k\,b$ 精确成形。向量单元没有融合乘加，故乘积由若干各自装得进 24 位尾数的片段精确重组。记 $s = \mathit{state}$，$b = b_1$，数字 $k < 2^{24}$（整数，精确）。

**引理 1（Veltkamp 拆分，$C = 257 = 2^8 + 1$）。** 取 $t = \mathrm{RN}(bC)$，$q = \mathrm{RN}(t - b)$，$b_H = \mathrm{RN}(t - q)$，$b_L = \mathrm{RN}(b - b_H)$：

$$
b_H + b_L = b\ \text{精确}, \qquad b_H\ \text{含}\ \le 8\ \text{位有效数字}, \qquad b_L\ \text{含}\ \le 16\ \text{位},
$$

在 $|b| \le \mathit{FLT\_MAX}/257$ 时成立（$t$ 不溢出）。两次减法因 Sterbenz 型抵消而精确（$t - b$ 抵消 $t$ 的高位、留下拆分的低位；$b - b_H$ 为互补低位、可装入尾数）。

**引理 2（数字分解）。** 拆 $k = d_1 2^{16} + d_2 2^{8} + d_3$，$0 \le d_j < 2^8$。提取过程 —— 乘 $2^{-16}$、取整、相减；再在 $2^{-8}$ 上重复 —— 全程精确：每个中间量都是 $< 2^{25}$ 的整数，而所有此类差都是 $< 2^{24}$ 的整数，故可精确表示。

**引理 3（精确部分积）。** 六个乘积 $d_j \cdot b_H$ 与 $d_j \cdot b_L$（$j = 1,2,3$）各至多 $8 + 8 = 16$ 或 $8 + 16 = 24$ 位有效数字，`vmul` 精确算出；权重 $2^{16}$、$2^{8}$ 在乘后以精确 2 的幂缩放施加，且 $k\,b \le s \le \mathit{FLT\_MAX}$ 约束每个中间量不溢出。

**引理 4（TwoSum）。** 对任意 `float32` $a, b$：$s = \mathrm{RN}(a+b)$，$z = \mathrm{RN}(s-a)$，$e = \mathrm{RN}(b - z)$，则残差 $e$ 精确且 $s + e = a + b$ 精确（Knuth 六步展开）。六个精确部分积经此链汇入一对 $(\mathit{hi}, \mathit{lo})$：

$$
\mathit{hi} + \mathit{lo} \;=\; k \cdot b \quad \text{精确}。
$$

**引理 5（Sterbenz 减法）。** $r_1 = \mathrm{RN}(s - \mathit{hi})$ 精确：$k \ge 1$ 时 $\mathit{hi} \approx kb \in (s/2,\ s]$，即 $s/2 < \mathit{hi} \le s$ —— Sterbenz 区间 $y/2 \le x \le 2y$，减法精确；$k = 0$ 时 $\mathit{hi} = 0$，平凡精确。

**结果。** $r = \mathrm{RN}(r_1 - \mathit{lo})$：精确值 $r_1 - \mathit{lo} = s - kb$ 是两个 `float32` 值的向下取整余数，由经典二进制 `fmod` 可表示性（余数是 $s$ 抵消高位后的残留，不需要超过 24 位尾数）知其可表示；可表示值的 $\mathrm{RN}$ 为恒等，故 $r$ 是**精确**余数。

### A.4 有限性守卫与回退

与 $r$ 并行计算普通两步值 $f_b = \mathrm{RN}(s - \mathrm{RN}(k\,b))$，通道结果为

$$
\text{result} \;=\; \begin{cases} r, & |r| \le \mathit{FLT\_MAX} \\ f_b, & \text{其余（含 } r = \mathrm{NaN}\text{）} \end{cases}
$$

由一次打包比较选择（`|r| <= FLT_MAX`；NaN 比较为假）。这把两类通道引到 $f_b$：(i) $|b| > \mathit{FLT\_MAX}/257$，引理 1 的拆分不可用（部分积为 NaN/inf）；(ii) 极罕见的近 $\mathit{FLT\_MAX}$ 余数，其补偿形式溢出。因 $k\,b \le s \le \mathit{FLT\_MAX}$，$f_b$ 总是有限值，绝对误差 $\le \tfrac{1}{2}\operatorname{ulp}(k\,b) \le \tfrac{1}{2}\operatorname{ulp}(s)$。

### A.5 次正规除数区间

当 $|y| \le \mathit{minNormal}$（TREMS 在宿主判定，TREM 用打包掩码比较+选择逐元素判定）时，第 1 次迭代先同乘 $2^{24}$ 再取 $b_1 = |y|\,2^{24}$：除数提升精确（次正规的 2 的幂缩放），状态提升在 $|x|\,2^{24} \le \mathit{FLT\_MAX}$ 时精确。后处理仅对这些通道乘回 $2^{-24}$；结果为正规数时该缩放精确。本区间在 A.1 的保证范围之外，沿用 A5 的迭代算术。

### A.6 后处理

按序施加的选择链（向量形式；TREMS 标量形式在宿主侧解决依赖除数的分支）：

1. **符号恢复**：$x < 0 \Rightarrow \mathit{state} \leftarrow -\mathit{state}$。负次正规数可正确比较；$-0$ 读作 $0$，由第 4 步覆盖。
2. **次正规缩回**：提升过的通道上 $\mathit{state} \leftarrow \mathit{state}\cdot 2^{-24}$（A.5）。
3. **无穷除数**（$|y| = \pm\infty$）：A.1 的向下取整语义 —— 同号 $x$ 保持 $x$；异号 $x$ 取 $x + y = \operatorname{copysign}(\infty, y)$；$x = \pm\infty$ 变为 NaN（在同号选择之后重施加，否则该选择会把被除数留在那里）。符号探测要与除数相乘，而 $\pm 0 \cdot \pm\infty = \mathrm{NaN}$，故零通道保持 $x$。标量（TREMS）形式下无穷除数将整个迭代链短路（逐迭代除数广播与符号探测在 $\pm\infty$ 处均失效）；向量（TREM）形式下这些通道走完整条链，向下取整无穷选择在第 5 步之后执行，其后仅剩 NaN 重施加。
4. **零被除数**：定义域内除数下 $x = \pm 0 \Rightarrow x$（保留 $x$ 的符号）。
5. **余数符号修正**：$\mathit{state} \cdot y < 0 \Rightarrow \mathit{state} \leftarrow \mathit{state} + y$。标量形式对无穷 $y$ 跳过；向量形式照算（无穷除数通道上无害溢出），再被第 3 步的选择覆盖。

### A.7 执行组织

两种调度求值同一张 A.2–A.6 的逐通道数据流图：**逐向量**路径（一个 64 元素向量跑完全部 11 次迭代 + 后处理；用于小 tile、重叠 tile 与行尾）与 **staged** 路径（每遍对一行全部满向量推进一次迭代，状态暂存于 `dst`，末遍按 $\min(C, 8)$ 个向量分批补偿调用）。分批只重排*跨通道*的操作；单通道内的 RN 数据流不变，故两种调度 —— 以及 staged 各遍的任意分块 —— 产出按位一致的逐通道结果。
