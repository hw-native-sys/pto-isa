# 逐元素 Tile-Tile 指令集

逐元素 Tile-Tile 操作用两个或一个 tile 作为输入，在目标 tile 的 valid region 上逐元素执行运算。它们是 PTO tile 计算路径里最常见、也最基础的一类指令。

## 指令一览

| 操作 | 说明 | 类别 |
| --- | --- | --- |
| `pto.tadd` | 逐元素加法 | Binary |
| `pto.tabs` | 逐元素绝对值 | Unary |
| `pto.tand` | 逐元素按位与 | Binary |
| `pto.tor` | 逐元素按位或 | Binary |
| `pto.tsub` | 逐元素减法 | Binary |
| `pto.tmul` | 逐元素乘法 | Binary |
| `pto.tmin` | 逐元素最小值 | Binary |
| `pto.tmax` | 逐元素最大值 | Binary |
| `pto.tcmp` | 逐元素比较 | Binary |
| `pto.tdiv` | 逐元素除法 | Binary |
| `pto.tshl` / `pto.tshr` | 逐元素移位 | Binary |
| `pto.txor` | 逐元素按位异或 | Binary |
| `pto.tlog` / `pto.trecip` / `pto.texp` / `pto.tsqrt` / `pto.trsqrt` | 一元数学运算 | Unary |
| `pto.tpow` | 逐元素幂运算 | Binary |
| `pto.tprelu` / `pto.trelu` / `pto.tneg` / `pto.tnot` | 激活或一元变体 | Unary/Binary |
| `pto.taddc` / `pto.tsubc` | 三输入融合加减 | Ternary-like Binary |
| `pto.tcvt` | 逐元素类型转换 | Unary |
| `pto.tsel` | 条件选择 | Ternary |
| `pto.trem` / `pto.tfmod` | 余数 / 浮点模 | Binary |

## 机制

这组指令的共同点不是“都长得像算术”，而是**都以目标 tile 的 valid region 为迭代域**。无论源 tile 自己的 valid region 怎么声明，真正被遍历的坐标集合都由目标 tile 决定。

对目标 tile 中每个 `(r, c)`：

$$ \mathrm{dst}_{r,c} = f(\mathrm{src0}_{r,c}, \mathrm{src1}_{r,c}) $$

对于 `TSEL`：

$$ \mathrm{dst}_{r,c} = (\mathrm{cmp}_{r,c} \neq 0) ? \mathrm{src0}_{r,c} : \mathrm{src1}_{r,c} $$

## Valid Region 兼容性

所有逐元素 Tile-Tile 操作都遵循同一条规则：

- 迭代域总是目标 tile 的 valid region。
- 源 tile 会在同一 `(r, c)` 坐标被读取。
- 如果源 tile 在这个坐标超出了自己的 valid region，那么读到的值属于 implementation-defined。
- 除非某条指令页单独收窄这一点，否则程序不能依赖这些域外值。

这条规则非常重要。它意味着“源 tile 自己声明的有效区域较小”并不会自动把目标迭代域缩小，也不会自动补零或自动修复。

## `_c` 变体

当前这组指令里的 `_c` 变体并不是“饱和算术”的统一命名约定。以当前 canonical 叶子页和实现签名为准：

- `TADDC` 表达的是三输入逐元素加法：`src0 + src1 + src2`
- `TSUBC` 表达的是三输入逐元素运算：`src0 - src1 + src2`

因此，不能把 `_c` 后缀一概理解成 saturating / carry 变体。具体语义必须看各自 per-op 页面。

## 目标 Profile 支持

| 元素类型 | CPU | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| `f32 / f16 / bf16` | Yes | Yes | Yes |
| `i8/u8, i16/u16, i32/u32, i64/u64` | Yes | Yes | Yes |
| `f8e4m3 / f8e5m2` | No | No | Yes |

具体到单条指令，支持子集仍以 per-op 页面为准。

## 约束

- layout、shape 和 valid-region 状态都会影响合法性。
- 源与目标 tile 的物理 shape 必须兼容。
- `TCMP` 产生的是谓词 tile，不是普通数值 tile。
- `TCVT` 允许源和目标 dtype 不同，但必须落在文档化转换组内。
- 位移类要求第二操作数小于元素位宽。

## 不允许的情形

- 假设存在隐式广播、隐式 reshape 或 valid-region 自动修复。
- 依赖源 tile 域外 lane 的确定值。
- 把 `_c` 后缀机械理解成“饱和变体”。
- 使用超出元素位宽的 shift count。

## A2/A3 吞吐与时延

逐元素 tile-tile 操作在 A2/A3 上会落到 CCE 向量指令，由 `include/pto/costmodel/a2a3/` 下的模型负责估算。

### 周期模型

```text
total_cycles = startup + completion + repeats × per_repeat + (repeats - 1) × interval
```

其中 `repeats` 由 tile 布局、stride 和 valid region 共同决定。

### 常见常量

| 指标 | 常量 | 周期 | 适用范围 |
| --- | --- | --- | --- |
| 启动时延 | `A2A3_STARTUP_BINARY` | 14 | 二元算术 |
| 启动时延 | `A2A3_STARTUP_REDUCE` | 13 | 一元 / 超越函数 |
| 完成时延 | `A2A3_COMPL_FP_BINOP` | 19 | FP 加减、部分转换 |
| 完成时延 | `A2A3_COMPL_INT_BINOP` | 17 | INT 加减、比较一类 |
| 完成时延 | `A2A3_COMPL_INT_MUL` | 18 | INT 乘法 |
| 完成时延 | `A2A3_COMPL_FP_MUL` | 20 | FP 乘法 |
| 完成时延 | `A2A3_COMPL_FP32_EXP` | 26 | `texp` 等 |
| 完成时延 | `A2A3_COMPL_FP32_SQRT` | 27 | `tsqrt` 等 |
| 每次 repeat 吞吐 | `A2A3_RPT_1` | 1 | 一元 / 标量类 |
| 每次 repeat 吞吐 | `A2A3_RPT_2` | 2 | 二元算术 |
| 每次 repeat 吞吐 | `A2A3_RPT_4` | 4 | 部分 f16 超越函数 |
| 流水间隔 | `A2A3_INTERVAL` | 18 | 向量路径通用 |

### repeat 的影响

`TBinOp.hpp` / `TBinSOp.hpp` / `TUnaryOp.hpp` 会根据 tile 的 geometry 计算 `repeats`：

- 连续 fast path：`repeats = validRow × validCol / elementsPerRepeat`
- 非连续路径：会按 stride 拆分为更一般的 repeat 序列
- 小 shape 时还可能走 `Bin1LNormModeSmall` 之类的特化路径

### layout 的影响

| 布局 | stride 特征 | 成本影响 |
| --- | --- | --- |
| `RowMajor` | 源 / 目标 stride 连续 | 最容易走 fast path |
| `ColMajor` | stride 不连续 | 更容易走 general path |
| 混合布局 / 特殊布局 | 非线性 stride | 只能走 general path |

### 搬运带宽模型

| 路径 | 带宽（B/cycle） | 常量 |
| --- | --- | --- |
| GM → Vec Buffer (`TLOAD`) | 128 | `A2A3_BW_GM_VEC` |
| Vec → Vec (`TMOV`) | 128 | `A2A3_BW_VEC_VEC` |
| GM → Mat | 256 | `A2A3_BW_GM_MAT` |
| Mat → L0A | 256 | `A2A3_BW_MAT_LEFT` |
| Mat → L0B | 128 | `A2A3_BW_MAT_RIGHT` |
| Mat → Mat (`TEXTRACT`) | 32 | `A2A3_BW_MAT_MAT` |

搬运周期通常按 `ceil(bufferSize / bandwidth)` 估算。

## 相关页面

- [Tile 指令族](../instruction-families/tile-families_zh.md)
- [Tile 指令面](../instruction-families/tile-families_zh.md)
