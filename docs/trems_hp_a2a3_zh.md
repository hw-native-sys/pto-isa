# TREMS / TREM 高精度求余（A2/A3）—— 算法、接口与切分预算

本文档描述 [issue #2](https://gitcode.com/bynshard/pto-isa/issues/2) 中
高精度 FP32 求余在 A2/A3（Ascend910B1）上的实现：标量除数版本
`TREMS<RemSAlgorithm::HIGH_PRECISION>` 与向量除数版本
`TREM<RemAlgorithm::HIGH_PRECISION>`。内容包括数值算法、调用接口、
`tmp` tile 的内存契约（哪些暂存区按数据生命周期刻意复用、为何安全），
以及在 192 KB A2/A3 UB 预算内允许的最大切分尺寸。

代码位置：`include/pto/npu/a2a3/custom/TFmodRemHp.hpp`（算法）、
`include/pto/npu/a2a3/custom/TRemHpStaged.hpp`（staged / 逐向量驱动），
分发位于 `include/pto/npu/a2a3/TRemS.hpp` 与 `TRem.hpp`。

---

## 1. 语义

`dst = rem(dst, src, scalar)` 计算**向下取整求余（floored remainder）**：

$$
\begin{aligned}
r &= x - \lfloor x / s \rfloor \cdot s && \text{（等价于精确算术后一次性舍入到 f32）} \\
r &\leftarrow r + s && \text{若 } r \text{ 与 } s \text{ 符号不同}
\end{aligned}
$$

特殊值遵循 IEEE 语义（即 numpy 的 `x % s`）：

| x | s | 结果 |
|---|---|------|
| ±inf | 有限值 | NaN |
| 有限值 | ±inf | x（若 x 也是 ±inf 则为 NaN） |
| ±0 | s ≠ 0 | ±0（保留 x 的符号） |
| NaN | 任意 | NaN |

这与 DEFAULT（快速）TREMS 路径不同：快速路径用一次 f32 除法计算
$\lfloor x/s \rfloor$，当 $|x/s| > 2^{24}$ 时，一次成形的乘积 $q \cdot s$
在 $x$ 的量级上舍入，随后的减法发生灾难性抵消，余数退化为
$\pm(1\ldots 2)\,\mathrm{ulp}(x)$（在 FLT_MAX 边界附近甚至翻转为
$\pm\infty$）。高精度路径把余数解析到完整 f32 精度，其参考
金标就是 `np.float32(x) % np.float32(s)`。

标量为 0 不做特殊处理（与 A5 实现一致，由迭代算术自然传播）；请使用非零
有限标量。

## 2. 算法 —— 以 $2^{24}$ 为基的逐位剥离（digit-peeling）长除法

数值上遵循 A5 实现（`a5/custom/TFmodRemHp.hpp`，与 AscendC
`fmod_3510` 同族）。

### 2.1 为什么是 11 次迭代

在 $\mathit{state} = |x|$、$\mathit{div} = |s|$ 上工作。f32 商的动态范围可达
$\sim 2^{276}$（FLT_MAX 除以最小次正规数）。算法从高位起按约 24 bit
一段剥离指数差：

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

$\lceil 276/24 \rceil = 11$ 覆盖全部范围（对较大的 `|s|`，梯次顶端项溢出为
inf；`vmins` 截断与饱和整数转换共同充当商的钳制，使这些迭代退化为空操作
而非污染状态）。

### 2.2 单次迭代

$$
\begin{aligned}
b &= \mathit{div} \cdot 2^{e(i)} \\[2pt]
k &= \begin{cases} \lfloor \mathit{state}/b \rfloor & \text{求余 remainder（\texttt{ROUND\_F}}) \\[2pt]
      \operatorname{trunc}(\mathit{state}/b) & \text{fmod（\texttt{ROUND\_Z}}) \end{cases} \\[2pt]
\mathit{state} &\leftarrow \mathit{state} - k \cdot b
\end{aligned}
$$

$b$ 是除数的精确 2 的幂缩放。$k$ 为整数，在 f32 中精确；int32 转换饱和，
极端指数差下相当于免费的商钳制。减法只在该量级的 chunk 上有 $\le 1$ ulp
的舍入 —— 从不在 $|x|$ 的量级上舍入 —— 从而抵消掉指数差的最高约 24 bit。

第 1 次迭代额外做：
- 若 $|s| \le \mathit{minNormal}$：先把 $\mathit{state}$ 与除数同乘 $2^{24}$
  （把次正规数抬升到正规范围；后处理再乘回 $2^{-24}$）；
- 最终不缩放的 $k = \lfloor \mathit{state}/|s| \rfloor$；
- $\mathit{state} \leftarrow \mathit{state} + [\,\mathit{state} < 0\,] \cdot b$
  （次正规尺度的符号修正；$[\,P\,]$ 为 Iverson 括号，$P$ 成立时为 1，否则为 0）。

后处理（`HpPostS`）：被除数符号恢复
（$\mathit{src0} < 0$ 时 $\mathit{state} \leftarrow -\mathit{state}$）、
次正规回缩放（$|s| \le \mathit{minNormal}$ 时
$\mathit{state} \leftarrow \mathit{state} \cdot 2^{-24}$）、第 1 节的异常修正，
以及余数符号修正
（$\mathit{state} \cdot s < 0 \ \Rightarrow\ \mathit{state} \leftarrow \mathit{state} + s$）。

对商本就落在 24 bit 内的情况（如 `x = 100.7, s = 7.5`），第 11..2 次迭代
计算 `k = 0`，仅第 1 次迭代产生结果；梯次是为巨大指数差准备的
（`x = 1e30`、FLT_MAX 被除数、次正规除数等）。

## 3. 接口

```cpp
#include <pto/npu/a2a3/TRemS.hpp>

TREMS<RemSAlgorithm::HIGH_PRECISION>(dstTile, srcTile, scalar, tmpTile);
TREM<RemAlgorithm::HIGH_PRECISION>(dstTile, src0Tile, src1Tile, tmpTile);  // 向量除数
```

- 元素类型：仅 `float`（`HIGH_PRECISION` + 非 float 触发 static_assert；
  int32 走 DEFAULT 路径）。
- Tile：`TileType::Vec`、行主序，dst 与 src 的有效形状一致。tile 的
  `Cols` 须满足 `Cols * sizeof(float) % 32 == 0`；非对齐的有效宽度用
  `validCol < tileCols` 表达。
- TREM 除数契约：src1 元素须非零且非 NaN（与 A5 实现同一域）。
  $\pm\infty$ 没问题（$x \bmod \pm\infty = x$）。
  **保证域是 $|y| \ge 1$（或 $\pm\infty$）**：余数符号修正探测
  $r \cdot y$，当 $|r \cdot y| < 2^{-149}$ 时下溢为 $\pm 0$ 从而漏掉修正
  —— 这是 A5 的固有限制，本实现按原样继承。更小的除数遵循 A5 迭代算术及该限制；
  NaN 除数虽会经迭代传播，但会与 `src0 == ±0` 修正竞争；±0 除数不在契约内。
- 分发（与 A5 结构一致；TREM 的 staged 额外要求 dst 与 src0、src1 均不重叠）：

| 条件 | 路径 |
|---|---|
| `validRows * (validCols / 64) >= TREM_HP_STAGED_MIN_VECS` 且 dst tile 不与输入 tile 重叠 | **staged**（循环倒置，第 4 节） |
| 否则（小 tile、重叠 tile、行尾） | **逐向量**回退：每个 64 元素向量在暂存 state 缓冲中跑完全部 11 次迭代 |

  `TREM_HP_STAGED_MIN_VECS` 默认 48，可在编译期覆盖
  （`-DTREM_HP_STAGED_MIN_VECS=...`）。
- staged 路径另要求 `validCols / 64 <= 255`（行批处理的 repeat 计数为
  8 bit）。更宽的行会触发断言；请切分 tile。
- `tmp` tile：1 行、`HpScratchRequiredCols(cap)` 列（TREM 为 V 变体，见
  第 5/8.1 节）。TREMS/TREM 在运行期校验。

### 3.1 用法示例

```cpp
#include <pto/pto-inst.hpp>
#include <pto/npu/a2a3/TRemS.hpp>
#include <pto/npu/a2a3/TRem.hpp>

// 标量除数，8x448 tile：dst = rem(x, 7.5f)，完整 f32 精度。
constexpr unsigned capVecs  = 448 / 64;                       // 每行满向量数
constexpr unsigned maskF    = ((capVecs * 8 + 31) / 32) * 8;  // 打包掩码，32B 单位
constexpr unsigned tmpCols  = 128 * capVecs + maskF + 8;      // TREMS 契约（第 5 节）

using DstTile = Tile<TileType::Vec, float, 8, 448, BLayout::RowMajor, -1, -1>;
using SrcTile = Tile<TileType::Vec, float, 8, 448, BLayout::RowMajor, -1, -1>;
using TmpTile = Tile<TileType::Vec, float, 1, tmpCols, BLayout::RowMajor, -1, -1>;

DstTile dst(8, 448);
SrcTile src(8, 448);
TmpTile tmp(1, tmpCols);
// ... TASSIGN 把 tile 分配进 UB、TLOAD 装载 src（见测试内核）...
TREMS<RemSAlgorithm::HIGH_PRECISION>(dst, src, 7.5f, tmp);

// 向量除数：同样形状，tmp 按 192C 变体取宽（第 8.1 节）
TREM<RemAlgorithm::HIGH_PRECISION>(dst, src0, src1, tmpV);
```

完整内核（TASSIGN/TLOAD/TSTORE 流程、跨步与带行尾的 tile）见
`tests/npu/a2a3/src/st/testcase/trems/trems_kernel.cpp` 与
`tests/npu/a2a3/src/st/testcase/trem/trem_kernel.cpp`。

## 4. staged 执行（issue #2 的重排，A2/A3 形态）

存在两种执行组织方式。逐向量路径一次一个 64 元素向量，在暂存 state
缓冲中跑完全部 11 次迭代。staged 路径把迭代循环倒置：**每个满向量每遍
只推进一次 SolveScale 迭代**，逐向量的状态暂存在 dst 各行中。A2/A3 上每一遍中的每条操作都是一条覆盖该行全部满向量的
repeat 指令：

```
seed:      dst[row] = |src[row]|                     （vabs，rep = fullPerRow）
第 11→1 遍： 对每一行：在 dst[row] 上原地做 SolveScale 迭代           （rep = fullPerRow）
post:      对每一行：在 dst[row] 上做 HpPostS
行尾:      每行不足 64 元素的尾部走逐向量回退
```


## 5. `tmp` tile 契约 —— 布局与安全别名

其中 $C = \max(\mathit{validCols}/64,\ 1)$ 为向量容量，
$\mathit{maskF}(C) = \lceil 8C/32 \rceil \cdot 8$ 个 float：

$$\mathit{HpScratchRequiredCols}(C) = 128C + \mathit{maskF}(C) + 8 + 64\,[\,C = 1\,]\ \text{个 float}$$

| 区域 | float 偏移 | 生命周期 | 说明 |
|---|---|---|---|
| `bTmp` | `[0, 64C)` | 单次迭代内 | 缩放后的除数，每次迭代由宿主侧标量 `vector_dup` —— **没有任何除数缓冲跨迭代存活** |
| `k` | `[64C, 128C)` | 见下 | 商与商后临时量共用该区域 |
| `cmpMask` | `[128C, +maskF)` | 从 `vcmp` 直到消费它的 `vsel` | 打包比较位掩码，32 字节对齐 |
| `addrBuf` | `+8` | 每次 `set_cmpmask` 前后 | 保存 `cmpMask` 的完整 64 位地址 |
| `state` | `+64`，仅 `C == 1` | 整个逐向量调用 | 逐向量回退的状态；staged 的状态在 dst 中，因此它绝不随 `C` 增长 |

两处别名决策及其无错的理由：

1. **商 / 临时量共用（`k`）**——`vmul(k, k, bTmp)` 读取 k 形成 `k·bTmp`
   乘积的那一刻，k 即死亡；临时量之后的所有用途（符号因子、选择操作数、
   探测值、修正候选）都以对该区域的一次全新写入开始。`dst == src0` 的原地
   向量操作是 A2/A3 的既有习惯（如 TPow 的 `vmul(dst, dst, exp)`）。后处理中
   探测值 $k = \mathit{state} \cdot \mathit{scalar}$ 先被 `vcmpvs_lt` 消费（中间隔 `pipe_barrier`），
   **然后**才被 `vadds(k, state, scalar)` 以修正候选覆盖 —— 每次交接都以
   barrier 分隔。
2. **不保留符号/常量缓冲**——A5 的 `GetSignBit` 需要基于 `vshrs` 的位提取
   外加 1.0/0.0 向量。A2/A3 拼法用 `vcmpvs_lt(src0, 0)` + `vmuls(k, state, -1)`
   + `vsel` 完成符号恢复，不需要 `k` 之外的存储；第 1 次迭代的修正用同一套
   掩码机构做 `k = (state < 0) ? bTmp : 0`。


## 6. 192 KB UB 预算下的最大切分

A2/A3（Ascend910B1）UB 上限：**196608 B**（编译期检查，
`error: UBUF memory (...) exceeds limit`）。对打包 tile
（src = dst = `R × W`，`validCols = W`）：

$$
\mathit{UBUF}(R, W) \;=\; \underbrace{8\,R\,W}_{\text{src + dst tile}} \;+\;
\underbrace{4\,\bigl(128C + \mathit{maskF}(C) + 8\bigr)}_{\text{tmp}} \;\le\; 196608\ \mathrm{B},
\qquad C = \max(W/64,\ 1),\quad C \le 255\ \text{（8 bit repeatTimes）}
$$

每个行数下最大的打包 `W`（64 的倍数）：

| 行数 R | 最大 W | 该 W 处 UBUF | C |
|---|---|---|---|
| 1 | 12160 | 196128 | 190 |
| 2 | 8128 | 196128 | 127 |
| 3 | 6080 | 195360 | 95 |
| 4 | 4864 | 195200 | 76 |
| 7 | 3008 | 192928 | 47 |
| 8 | 2688 | 193920 | 42 |
| 12 | 1856 | 193312 | 29 |
| 16 | 1408 | 191712 | 22 |
| 32 | 704 | 185984 | 11 |

（以上都高于 48 满向量的 staged 门限，因此走 staged。）测试套件中的示例：
`1×8192` → 132128 B，`2×2048` → 49440 B，`8×448` → 32352 B。

### 6.1 TREM 预算

TREM 带三个 tile（dst、src0、src1）与第 8.1 节的每向量 192C 暂存：

$$
\mathit{UBUF}(R, W) \;=\; \underbrace{12\,R\,W}_{\text{dst + src0 + src1 tile}} \;+\;
\underbrace{4\,\bigl(192C + \mathit{maskF}(C) + 8\bigr)}_{\text{tmp}} \;\le\; 196608\ \mathrm{B},
\qquad C = \max(W/64,\ 1),\quad C \le 255
$$

每个行数下最大的打包 `W`（64 的倍数）：

| 行数 R | 最大 W | 该 W 处 UBUF | C |
|---|---|---|---|
| 1 | 8128 | 196128 | 127 |
| 2 | 5440 | 196576 | 85 |
| 3 | 4032 | 194080 | 63 |
| 4 | 3264 | 196288 | 51 |
| 7 | 1984 | 190752 | 31 |
| 8 | 1792 | 193792 | 28 |
| 12 | 1216 | 189888 | 19 |
| 16 | 960 | 196000 | 15 |
| 32 | 448 | 177504 | 7 |

经验公式：$W \lesssim 24576/(R+1)$（TREMS）；TREM 因多一块输入 tile 低一档，
$W \lesssim 16384/(R+1)$。示例：TREM `1×8128` → 196128 B（单行合法最大宽度）、
`2×2048` → 74016 B、`8×448` → 48480 B。

一般原则：
- **宽度主导**：暂存只随 $W$ 增长（约 $8W$ 字节），tile 随 $RW$ 增长。
- 超出上表的 tile 必须**拆成多次 TREMS/TREM 调用**（按列分块）—— 每次调用
  处理一个 tile。
- DEFAULT（快速）路径的 tmp 契约小得多（`dstRowStride + maskF + 8` 个
  float），同一 tile 可能对 DEFAULT 合法而对 HIGH_PRECISION 非法；请按
  实际分发的精度模式核对预算。
- `validCol < tileCols`（行尾）不增加 tmp 开销；行尾无论门限如何都走逐向量
  路径。

### 6.2 以 tile 维度 $[M, N]$ 表示的各 tile 占用

对 $M$ 行 × $N$ 列的打包 tile，tmp 占用只取决于宽度 $N$（向量容量 $C$）；
数据 tile 取决于 $M \cdot N$：

$$
\mathit{tmpCols}(N)=
\begin{cases}
208, & \text{TREMS},\ C=1 \\[2pt]
128C+8\lceil C/4\rceil+8, & \text{TREMS},\ C\ge 2 \\[2pt]
272, & \text{TREM},\ C=1 \\[2pt]
192C+8\lceil C/4\rceil+8, & \text{TREM},\ C\ge 2
\end{cases}
\qquad
C=\max\Bigl(\Bigl\lfloor \frac{N}{64}\Bigr\rfloor,\ 1\Bigr)
$$

其中 $8\lceil C/4\rceil$ 为打包比较掩码（按 32 字节对齐填充，第 5 节）；
$C=1$ 的常量分支折入了 64 个 float 的逐向量 state 缓冲。

$$
\mathit{UBUF}(M,N) \;=\;
\underbrace{n_{\text{tiles}}\cdot 4MN}_{\text{数据 tile}} \;+\;
\underbrace{4\,\mathit{tmpCols}(N)}_{\text{tmp}}
\;\le\; 196608\ \mathrm{B},
\qquad n_{\text{tiles}} = 2\ \text{（TREMS）},\ 3\ \text{（TREM）}
$$

（对带填充/跨步的 tile：各数据 tile 按其物理的 $M \times \mathit{Cols}$
计大小，而 tmp 契约按**有效**宽度取；仅当一行只有单个满向量或纯行尾时
出现 $+64$ 个 float，即逐向量 state 缓冲。）

算例 —— 每个 tile 的字节大小：

TREMS：

| tile $[M, N]$ | dst | src | tmp | 总 UBUF | 占 196608 B | 分发（$M \cdot C$ 对 48） |
|---|---|---|---|---|---|---|
| $[1, 64]$ | 256 | 256 | 832 | 1344 | 1% | 逐向量（1） |
| $[32, 64]$ | 8192 | 8192 | 832 | 17216 | 9% | 逐向量（32） |
| $[12, 256]$ | 12288 | 12288 | 2112 | 26688 | 14% | staged（48） |
| $[8, 448]$ | 14336 | 14336 | 3680 | 32352 | 16% | staged（56） |
| $[2, 2048]$ | 16384 | 16384 | 16672 | 49440 | 25% | staged（64） |
| $[1, 8192]$ | 32768 | 32768 | 66592 | 132128 | 67% | staged（128） |
| $[8, 2688]$ | 86016 | 86016 | 21888 | 193920 | 99% | staged（336） |
| $[1, 12160]$ | 48640 | 48640 | 98848 | 196128 | 100% | staged（190） |

TREM：

| tile $[M, N]$ | dst | src0 | src1 | tmp | 总 UBUF | 占 196608 B | 分发 |
|---|---|---|---|---|---|---|---|
| $[1, 64]$ | 256 | 256 | 256 | 1088 | 1856 | 1% | 逐向量（1） |
| $[32, 64]$ | 8192 | 8192 | 8192 | 1088 | 25664 | 13% | 逐向量（32） |
| $[8, 448]$ | 14336 | 14336 | 14336 | 5472 | 48480 | 25% | staged（56） |
| $[2, 2048]$ | 16384 | 16384 | 16384 | 24864 | 74016 | 38% | staged（64） |
| $[1, 8128]$ | 32512 | 32512 | 32512 | 98592 | 196128 | 100% | staged（127） |

切分设计清单：
- 先定工作宽度 $N$ —— 它单独决定 tmp 大小；行数再由
  $M \le (196608 - 4\,\mathit{tmpCols}(N))\,/\,(4N \cdot
  \#\text{tiles})$ 约束。
- 若需要 staged 路径：$C \le 255$ 且 $M \cdot C \ge 48$。
- 超出预算的按列分块：每次调用处理一个 tile。

## 7. 测试

`tests/npu/a2a3/src/st/testcase/trems` —— 用例 50–59 覆盖 staged、逐向量
回退、48 向量门限、行尾、跨步 dst 与负标量，金标为
`np.float32(x) % np.float32(s)`，输入织入角点池（±0、±inf、NaN、次正规、
FLT_MAX）。比较器允许 $8\cdot\mathrm{ulp}$ 的抵消松弛（第 1 节所述 ±ulp(x) 伪影在高精度
路径中并不存在，但硬件除/乘法器与 IEEE-RN 有 ~2 ulp 偏差，且对巨大被除数
的后处理乘法在 `|s|` 的量级上舍入）。

---

## 8. TREM —— 向量除数变体

`TREM<RemAlgorithm::HIGH_PRECISION>` 运行同样的 11 次迭代逐位剥离（第 2 节）、
同样的 staged 循环倒置（第 4 节）与同样的 48 向量 / 不重叠 / ≤255 向量分发
规则。差异全部来自除数是张量：

1. **每遍重载除数。** 每一遍从原始 tile 重新读取 `|src1|`（`vabs` 到暂存的
   `div` 区域），因为缩放过的除数不能喂给下一遍。seed、迭代与后处理遍其他
   部分与第 4 节一致。
2. **逐元素的次正规区间。** 标量除数时"除数是否次正规"是宿主侧常量；向量
   除数时第 1 次迭代逐元素判定：在 $|\mathit{src1}| \le \mathit{minNormal}$
   的通道上把除数与状态同乘 $2^{24}$（打包掩码比较 + 两次选择），后处理再
   精确地对这些通道乘回 $2^{-24}$（同样逐元素）。
3. **异常修正变成逐元素选择**（A5 的 `SolveExceptionScenarios`）：
   `src1 = ±inf → src0`；随后（必须在其后，否则上一步会把被除数留在那里）
   `src0 = ±inf ∧ src1 = ±inf → NaN`；然后除数在保证域内时
   `src0 = ±0 → src0`。`x = ±inf` 配有限除数在梯次内部就会坍缩为 NaN
   （顶部溢出迭代的 `inf / inf`），无需显式选择。
4. **符号修正乘以除数行**
   （$\mathit{state} \cdot \mathit{src1} < 0 \ \Rightarrow\ \mathit{state} \leftarrow \mathit{state} + \mathit{src1}$）
   而非标量。

### 8.1 TREM `tmp` 契约 —— 比 TREMS 多一个区域，以及原因

$$\mathit{HpScratchRequiredColsV}(C) = 192C + \mathit{maskF}(C) + 8 + 64\,[\,C = 1\,]\ \text{个 float}$$

| 区域 | float 偏移 | 生命周期 |
|---|---|---|
| `div` | `[0, 64C)` | 整遍：每遍从 src1 重新 `vabs`，被每次迭代的除数缩放与第 1 次迭代的次正规分支读取 |
| `bTmp` | `[64C, 128C)` | 单次迭代内（缩放后的除数） |
| `k` | `[128C, 192C)` | 商，随后（乘积 `vmul` 之后）与临时量共用 —— 与第 5 节的 k/t 别名相同 |
| `cmpMask`、`addrBuf`、`state` | 同第 5 节 | |

TREMS 能把除数完全折掉（128C），因为标量可以在每迭代直接 `vector_dup`
进 `bTmp`。TREM 的除数是张量，未缩放的 `|src1|` 基准必须**跨该遍的全部
迭代存活**作为重新缩放的源 —— 它不能放进 `bTmp`（第 1 次迭代在 `bTmp`
被写入之后还需要未缩放的值），也不能放进 `k`（商在迭代中途覆盖它）。
若要去掉第三个区域，需要在每次迭代把除数重新 `vabs` 进 `bTmp`；本契约
选择保留第三个区域。

## 9. 测试（两个算子）

- TREMS `tests/npu/a2a3/src/st/testcase/trems` 用例 50–59：staged、逐向量
  回退、48 向量门限、行尾、跨步 dst 与负标量。
- TREM `tests/npu/a2a3/src/st/testcase/trem` 的 `case_hp_*`（10 个用例）：
  同样的覆盖外加上述单行合法最大宽度（`1×8128`，第 6.1 节 R=1 的最大值）；
  HP 除数角点池按第 3 节除数契约排除 ±0、NaN 及 `|y| < 1`，被除数池保留
  完整角点集（±0、±inf、NaN、次正规、FLT_MAX）。

两个套件的金标均为 `np.float32(x) % np.float32(除数)`，比较器带
$8\cdot\mathrm{ulp}$ 抵消松弛与"末位坍缩为零"规则（第 7 节的理由；当最后一次迭代的
$k \cdot \mathit{bTmp}$ 在半 ulp 平局下舍入到恰好等于状态时，暂存的余数清零，而真实求余落在零的
一个除数宽度之内 —— A5 呈现同样伪影）。
