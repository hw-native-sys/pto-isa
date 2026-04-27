# 不规则与复杂指令集

这一组操作容纳那些不适合放进“标准逐元素”“标准归约”“标准内存搬运”这三大框架里的 tile 指令。它们通常要么有专门的数据访问方式，要么有专门的算法语义，要么受 target profile 缩窄得更明显。

把这些操作单独归组，不是因为它们“不重要”，而是因为它们共享的不是算术骨架，而是“非标准合同”。

## 操作

| 操作 | 作用 | 类别 | Profile |
| --- | --- | --- | :---: |
| [pto.tprint](./ops/irregular-and-complex/tprint_zh.md) | 打印 tile 内容，便于调试 | Debug | All |
| [pto.tmrgsort](./ops/irregular-and-complex/tmrgsort_zh.md) | 对 tile 行做 merge sort | Sort | All |
| [pto.tsort32](./ops/irregular-and-complex/tsort32_zh.md) | 排序 32-bit 元素 | Sort | All |
| [pto.tgather](./ops/irregular-and-complex/tgather_zh.md) | 按索引 gather | Gather | All |
| [pto.tgatherb](./ops/irregular-and-complex/tgatherb_zh.md) | 批量 gather | Gather | All |
| [pto.tscatter](./ops/irregular-and-complex/tscatter_zh.md) | 按索引 scatter | Scatter | All |
| [pto.tci](./ops/irregular-and-complex/tci_zh.md) | 复杂索引生成或索引辅助 | Index | All |
| [pto.ttri](./ops/irregular-and-complex/ttri_zh.md) | 三角矩阵相关操作 | Matrix | All |
| [pto.tpartadd](./ops/irregular-and-complex/tpartadd_zh.md) | 局部加法规约 | Partial reduction | All |
| [pto.tpartmul](./ops/irregular-and-complex/tpartmul_zh.md) | 局部乘法规约 | Partial reduction | All |
| [pto.tpartmax](./ops/irregular-and-complex/tpartmax_zh.md) | 局部最大值规约 | Partial reduction | All |
| [pto.tpartmin](./ops/irregular-and-complex/tpartmin_zh.md) | 局部最小值规约 | Partial reduction | All |
| [pto.tquant](./ops/irregular-and-complex/tquant_zh.md) | 把 tile 值量化为低精度表示 | Quantize | A2/A3, A5 |
| [pto.tdequant](./ops/irregular-and-complex/tdequant_zh.md) | 把整数量化 tile 反量化为浮点值 | Quantize | A2/A3, A5 |
| [pto.trandom](./ops/irregular-and-complex/trandom.md) | 生成随机值到 tile 状态 | Generation | A5 |
| [pto.thistogram](./ops/irregular-and-complex/thistogram.md) | 从 tile 值累积直方图 bin | Statistics | A5 |

## 机制

### 排序

- `TSORT32` 面向 32-bit 元素排序；
- `TMRGSORT` 更强调对 tile 行的 merge sort 语义。

这类指令不只是“比较若干元素再交换位置”，而是把排序本身作为一条 tile 级合同暴露出来。

### Gather / Scatter

这组操作根据索引 tile 做非连续访问：

$$ \mathrm{dst}_i = \mathrm{src}_{\mathrm{index}_i} \quad \text{(gather)} $$

$$ \mathrm{dst}_{\mathrm{index}_i} = \mathrm{src}_i \quad \text{(scatter)} $$

它们和普通顺序搬运类指令的差别，在于地址次序由索引决定，而不是由 tile 的线性布局决定。

### Partial Reduction

`TPART*` 系列不是完整的行 / 列归约，而是先做一段局部规约，得到后续还要继续合并的中间 tile。它们的价值在于分阶段地表达更大规模的 reduction。

### 量化

`TQUANT` 把浮点 tile 数据转换为量化表示。`TDEQUANT` 使用按行广播的 scale 和 offset tile，把整数量化 tile 转回浮点值。二者都是 tile payload 变换，因此属于本 tile 指令族，而不是系统调度。

### 生成状态与统计状态

`TRANDOM` 和 `THISTOGRAM` 会产生 tile 可见的 payload 状态。它们的结果是 tile 数据，而不是调度效果，因此归入不规则 tile 操作。

## 目标 Profile 支持

| 元素类型 | CPU | A2/A3 | A5 |
| --- | :---: | :---: | :---: |
| f32 / f16 / bf16 | Yes | Yes | Yes |
| i8/u8、i16/u16、i32/u32、i64/u64 | Yes | Yes | Yes |
| INT4 / FP4 / NF4 等量化格式 | No | Yes | Yes |

## 约束

- 排序操作要求元素类型与具体排序变体兼容。
- 量化要求 scale 非零，且 zero-point 落在合法范围内。
- `TSCATTER` 要求索引非负且落在目标范围内。
- `TPART*` 的具体行为可能随 profile 缩窄。

## 不允许的情形

- 使用非法 scale 或越界 zero-point 做量化；
- scatter 到目标 tile 形状之外；
- 把某个 target 上偶然可接受的行为当成可移植合同。

## 相关页面

- [Tile 指令族](../instruction-families/tile-families_zh.md)
- [Tile 指令表面](../instruction-families/tile-families_zh.md)
