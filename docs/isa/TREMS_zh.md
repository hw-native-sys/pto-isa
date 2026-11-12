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

存在两种执行组织方式（issue #2 的循环倒置）。逐向量路径一次一个 64 元素向量，在暂存 state 缓冲中跑完全部 11 次迭代 —— 小 tile、重叠 tile 和行尾的回退路径。staged 路径**每个满向量每遍只推进一次迭代**，逐向量状态暂存于 `dst`；在 A2/A3 上每一遍中的每条操作都是一条覆盖该行全部满向量的 repeat 指令：

```
seed:      dst[row] = |src[row]|                     (vabs, rep = fullPerRow)
第 11→1 遍: 对每一行: 在 dst[row] 上原地做 SolveScale迭代 (rep = fullPerRow)
post:      对每一行: HpPostS
行尾:      每行不足 64 元素的尾部走逐向量回退
```

当 `validRows * (validCols / 64) >= TREM_HP_STAGED_MIN_VECS`（48）且 `dst` tile 不与输入重叠时启用 staged；`validCols / 64 <= 255`（8 bit repeatTimes）始终适用 —— 更宽的 tile 请拆分成多次调用。

### 末次迭代精确化

最后一次剥离迭代以单次舍入语义计算 `state -= k * b`。A5 用融合 `vmula` 一步完成；A2/A3 没有融合乘加（`vmla` 先对乘积舍入），因此将 $k$（小于 $2^{24}$ 的整数）与 $b$ 拆成 12 位两半，计算四个精确部分积，用完整 TwoSum 链把 $k \cdot b$ 精确重组为 `(hi, lo)`，再分两步相减。第一步减法满足 Sterbenz 精确性，第二步仅舍入一次，得到真余数（真余数总是可表示的）。$|b| > \text{FLT\_MAX}/4097$ 的通道保留普通两步结果，其误差不超过被除数的半个 ulp（商超过 $2^{24}$ 时的容忍误差类）。这使得 $4580025 \bmod -7.25$ 之类得到精确结果（$-3.0$ 而非 $-3.25$）。

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
| `comp` | `+512\,\min(C,8)$ | 末次迭代的补偿减法（`HpFusedSub`），按 $\min(C,8)$ 个向量分批：12 位拆分、四个精确部分积与 $k \cdot b$ 的 TwoSum 重组 |

商/临时量共用是安全的：`k` 在读取它的 `vmul` 处即死亡，后处理的探测值在被 `vcmp` 消费后才被候选覆盖 —— 每次交接都以 `pipe_barrier` 分隔（原地 `dst == src0` 操作是 A2/A3 的既有习惯）。staged 路径的状态在 `dst` 中，`state` 绝不随 $C$ 增长。

### tmp 占用

$$
\mathit{tmpCols}(N)=
\begin{cases}
208+512, & \text{TREMS},\ C=1 \\[2pt]
128C+8\lceil C/4\rceil+8+512\min(C,8), & \text{TREMS},\ C\ge 2 \\[2pt]
272+512, & \text{TREM},\ C=1 \\[2pt]
192C+8\lceil C/4\rceil+8+512\min(C,8), & \text{TREM},\ C\ge 2
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
| $[1, 64]$ | 256 | 256 | 2880 | 3392 | 2% | 逐向量（1） |
| $[32, 64]$ | 8192 | 8192 | 2880 | 19264 | 10% | 逐向量（32） |
| $[12, 256]$ | 12288 | 12288 | 10304 | 34880 | 18% | staged（48） |
| $[8, 448]$ | 14336 | 14336 | 18016 | 46688 | 24% | staged（56） |
| $[2, 2048]$ | 16384 | 16384 | 33056 | 65824 | 33% | staged（64） |
| $[1, 8192]$ | 32768 | 32768 | 82976 | 148512 | 76% | staged（128） |
| $[8, 2496]$ | 79872 | 79872 | 36704 | 196448 | 100% | staged（312） |
| $[1, 11136]$ | 44544 | 44544 | 106912 | 196000 | 100% | staged（174） |

每个行数下最大的打包 $W$（TREMS；经验公式 $W \lesssim 22278/(R+1)$）：

| 行数 R | 1 | 2 | 3 | 4 | 7 | 8 | 12 | 16 | 32 |
|---|---|---|---|---|---|---|---|---|---|
| 最大 W | 11136 | 7424 | 5568 | 4480 | 2752 | 2496 | 1728 | 1280 | 640 |

TREM（$W \lesssim 14852/(R+1)$）：相同行数下最大 W 分别为 7424 / 4928 / 3712 / 2944 / 1856 / 1664 / 1152 / 832 / 448；单行合法最大宽度为 `[1, 7424]`（195520 B）。

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
| 1 | 16192 | 12160 | 11136 |
| 2 | 9728 | 8128 | 7424 |
| 4 | 5440 | 4864 | 4480 |
| 8 | 2880 | 2688 | 2496 |
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
  // (C = 448/64 = 7, tmpCols = 128*7 + 8*ceil(7/4) + 8 + 512*7 = 4504)
  using HpTmpT = Tile<TileType::Vec, float, 1, 4504>;
  Tile<TileType::Vec, float, 8, 448> hpOut, hpSrc;
  HpTmpT hpTmp;
  TREMS<RemSAlgorithm::HIGH_PRECISION>(hpOut, hpSrc, 7.5f, hpTmp);
}
```
