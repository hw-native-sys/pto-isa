# 布局与重排指令集

布局与重排类操作负责改变 tile 数据的组织方式、访问窗口和布局解释。它们大多数不改变有效元素的算术值，但会改变这些元素如何被排列、提取、插入、转置或送入后续算子。

这组操作的关键不在“是不是搬运”，而在“搬运后数据以什么布局被后续指令看见”。

## 操作

| 操作 | 作用 | 类别 |
| --- | --- | --- |
| [pto.tmov](./ops/layout-and-rearrangement/tmov_zh.md) | tile 搬运 / 拷贝 | Copy |
| [pto.tmov_fp](./ops/layout-and-rearrangement/tmov_zh.md) | fix pipe 搬运 / 填充式搬运 | Copy |
| [pto.treshape](./ops/layout-and-rearrangement/treshape_zh.md) | 改变 tile 的形状解释 | Transform |
| [pto.ttrans](./ops/layout-and-rearrangement/ttrans_zh.md) | 转置 tile | Transform |
| [pto.tconcat](./ops/layout-and-rearrangement/tconcat.md) | 沿指定维度拼接 tile 序列 | Transform |
| [pto.tpack](./ops/layout-and-rearrangement/tpack.md) | 把 tile 元素打包为目标布局或紧凑表示 | Pack |
| [pto.textract](./ops/layout-and-rearrangement/textract_zh.md) | 抽取子 tile | Extract |
| [pto.textract_fp](./ops/layout-and-rearrangement/textract_zh.md) | 通过 fix pipe 做抽取 / 填充 | Extract |
| [pto.tinsert](./ops/layout-and-rearrangement/tinsert_zh.md) | 插入子 tile | Insert |
| [pto.tinsert_fp](./ops/layout-and-rearrangement/tinsert_zh.md) | 通过 fix pipe 做插入 / 填充 | Insert |
| [pto.tfillpad](./ops/layout-and-rearrangement/tfillpad_zh.md) | 填充 padding 区域 | Fill |
| [pto.tfillpad_inplace](./ops/layout-and-rearrangement/tfillpad-inplace_zh.md) | 原地填充 padding | Fill |
| [pto.tfillpad_expand](./ops/layout-and-rearrangement/tfillpad-expand_zh.md) | 填充 padding 并扩大 valid region | Fill |
| [pto.timg2col](./ops/layout-and-rearrangement/timg2col_zh.md) | 图像块重排为列布局 | Transform |

说明：本家族里 `*_fp` 的 `_fp` 指的是 `fix pipe`，不是 floating point。

## 机制

### 搬运

`TMOV` 与 `TMOV_FP` 负责把一个 tile 的数据复制到另一个 tile。两者的差别不是“一个是浮点，一个不是”，而是是否走 fix pipe 及其附带的填充 / 边界处理合同。

### 形状与布局重解释

- `TRESHAPE` 在不改变元素总数的前提下改变 shape 解释；
- `TTRANS` 把行列对调；
- `TCONCAT` 按文档定义的维度连接 tile 元素序列；
- `TPACK` 把 tile 元素重新打包到目标布局或紧凑表示；
- `TIMG2COL` 把卷积窗口重排成列式布局，供后续卷积 lowering 或矩阵化计算使用。

### 抽取与插入

- `TEXTRACT` 从源 tile 中抽出一个窗口；
- `TINSERT` 把子 tile 插回目标 tile 的指定位置；
- 对应的 `*_fp` 变体在 fix pipe 上完成相同类型的边界与填充合同。

### Padding 填充

`TFILLPAD`、`TFILLPAD_INPLACE` 与 `TFILLPAD_EXPAND` 处理 valid region 之外的 padding 区域。它们看起来像“写入默认值”，但本质上是在为后续布局敏感或边界敏感的算子准备更稳定的输入形态。

## 为什么这组指令要单列

如果不把布局重排单独拿出来，很多 tile 算术页就会被迫重复解释“输入到底是原样、转置后，还是某个子窗口”。把这组指令集中到一个家族里，能把“数据如何被重新组织”这层语义单独讲清楚。

## 目标 Profile 支持

| 元素类型 | CPU | A2A3 | A5 |
| --- | :---: | :---: | :---: |
| f32 / f16 / bf16 | Yes | Yes | Yes |
| i8/u8、i16/u16、i32/u32、i64/u64 | Yes | Yes | Yes |
| f8e4m3 / f8e5m2 | No | No | Yes |

## 约束

- `TRESHAPE` 要求总元素数保持不变。
- `TEXTRACT` 的 offset 与子形状必须落在源 tile 声明范围内。
- `TINSERT` 插入后的区域必须落在目标 tile 声明范围内。
- `*_fp` 变体要求填充值与 tile 元素类型兼容。
- `TIMG2COL` 依赖 kernel、padding、stride 等配置，且会被具体 profile 收窄。

## 不允许的情形

- reshape 到不同总元素数的形状；
- 用越界 offset 做 extract / insert；
- 在不支持的 profile 上把 FP8 与 `TIMG2COL` 组合使用。

## 相关页面

- [Tile 指令族](../instruction-families/tile-families_zh.md)
- [Tile 指令表面](../instruction-families/tile-families_zh.md)
- [布局](../state-and-types/layout_zh.md)
