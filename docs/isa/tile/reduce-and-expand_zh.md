# 归约与扩展指令集

归约操作沿某个轴把二维 tile 折叠成一维结果，扩展操作则把一维 tile 沿某个轴广播回二维 tile。它们经常出现在 softmax、归一化、池化、广播算术和索引类后处理里。

## 指令一览

### 按行归约

| 操作 | 说明 |
| --- | --- |
| `pto.trowsum` | 对每一行按列求和 |
| `pto.trowprod` | 对每一行按列求积 |
| `pto.trowmax` | 对每一行按列取最大值 |
| `pto.trowmin` | 对每一行按列取最小值 |
| `pto.trowargmax` | 求每一行最大值所在列索引 |
| `pto.trowargmin` | 求每一行最小值所在列索引 |

### 按列归约

| 操作 | 说明 |
| --- | --- |
| `pto.tcolsum` | 对每一列按行求和 |
| `pto.tcolprod` | 对每一列按行求积 |
| `pto.tcolmax` | 对每一列按行取最大值 |
| `pto.tcolmin` | 对每一列按行取最小值 |
| `pto.tcolargmax` | 求每一列最大值所在行索引 |
| `pto.tcolargmin` | 求每一列最小值所在行索引 |

### 按行扩展

| 操作 | 说明 |
| --- | --- |
| `pto.trowexpand` | 把 `(R,1)` 扩成 `(R,C)` |
| `pto.trowexpandadd` | 行扩展后再逐元素加 |
| `pto.trowexpandsub` | 行扩展后再逐元素减 |
| `pto.trowexpandmul` | 行扩展后再逐元素乘 |
| `pto.trowexpanddiv` | 行扩展后再逐元素除 |
| `pto.trowexpandmax` | 行扩展后再逐元素取 max |
| `pto.trowexpandmin` | 行扩展后再逐元素取 min |
| `pto.trowexpandexpdif` | 行扩展后做指数差相关运算 |

### 按列扩展

| 操作 | 说明 |
| --- | --- |
| `pto.tcolexpand` | 把 `(1,C)` 扩成 `(R,C)` |
| `pto.tcolexpandadd` | 列扩展后再逐元素加 |
| `pto.tcolexpandsub` | 列扩展后再逐元素减 |
| `pto.tcolexpandmul` | 列扩展后再逐元素乘 |
| `pto.tcolexpanddiv` | 列扩展后再逐元素除 |
| `pto.tcolexpandmax` | 列扩展后再逐元素取 max |
| `pto.tcolexpandmin` | 列扩展后再逐元素取 min |
| `pto.tcolexpandexpdif` | 列扩展后做指数差相关运算 |

## 机制

### 归约

按行归约的基本形式：

$$ \mathrm{dst}_{r} = \bigoplus_{c=0}^{C-1} \mathrm{src}_{r,c} $$

按列归约的基本形式：

$$ \mathrm{dst}_{c} = \bigoplus_{r=0}^{R-1} \mathrm{src}_{r,c} $$

这里的 `⊕` 可以是求和、求积、取极值或取索引等。真正的实现通常会借助临时 tile、分阶段归约或 tree reduction，但对 ISA 作者来说，首先要关心的是**输出轴被压成 1，另一个轴保留下来**。

### 扩展

按行扩展的基本形式：

$$ \mathrm{dst}_{r,c} = \mathrm{src}_{r} $$

按列扩展的基本形式：

$$ \mathrm{dst}_{r,c} = \mathrm{src}_{c} $$

扩展类变体通常会先做广播，再与另一个 tile 做逐元素组合。

## 输出形状

| 操作 | 输入形状 | 输出形状 |
| --- | --- | --- |
| 行归约 | `(R, C)` | `(R, 1)` |
| 列归约 | `(R, C)` | `(1, C)` |
| 行扩展 | `(R, 1)` | `(R, C)` |
| 列扩展 | `(1, C)` | `(R, C)` |

## 约束

- 源 tile 的 valid region 决定归约域。
- `argmax / argmin` 变体输出的是索引 tile，不是数值 tile。
- 归约目标 tile 在被归约轴上的 extent 必须为 1。
- 扩展变体要求第二输入在被扩展轴长度上匹配。
- `expdif` 类变体依赖特殊的指数差语义，不能把它当普通广播算术处理。

## 不允许的情形

- 在长度为 0 的轴上归约。
- 对不支持的元素类型使用 `arg` 类变体。
- 用不匹配的轴长度去做 expand 变体。
- 假设归约会自动修复布局或 valid region。

## 相关页面

- [Tile 指令族](../instruction-families/tile-families_zh.md)
- [Tile 指令面](../instruction-families/tile-families_zh.md)
