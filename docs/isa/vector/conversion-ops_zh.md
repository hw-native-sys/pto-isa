# 转换操作指令集

转换操作指令集覆盖向量层的数值类型转换、索引生成和保留浮点类型的截断/舍入。位宽变化、舍入、饱和和 lane 分区都属于架构可见约束。

## 常见操作

- `pto.vci`
- `pto.vcvt`
- `pto.vtrc`
- `pto.vbitcast`
- `pto.pbitcast`

## 操作数模型

- `%input`：源向量寄存器
- `%result`：目标向量寄存器
- `round_mode`：舍入模式
- `sat`：饱和模式
- `part`：偶数/奇数 lane 分区模式

## 机制

### `pto.vci`

根据标量索引或 seed 生成索引向量。它不是普通数值转换，而是索引生成。

### `pto.vcvt`

在浮点与整数之间做类型转换，可带舍入、饱和和偶/奇 lane 放置。

### `pto.vtrc`

把浮点值按指定舍入模式变成“整数值的浮点数”，但不改变元素类型。

### `pto.vbitcast`

对 `!pto.vreg<...>` 值做按位重新解释，保持位模式不变（总位宽恒为 2048 bits），只改变元素类型与车道数。源与目标都必须是 `!pto.vreg<...>`，且 `N * bitwidth(T0) = M * bitwidth(T1) = 2048`。仅支持整型和浮点元素类型。详细说明见 [`pto.vbitcast`](./ops/conversion-ops/vbitcast_zh.md)。

### `pto.pbitcast`

对 `!pto.mask<...>` 值做按位重新解释，不改变底层谓词位，仅切换 mask 粒度视图（`b8` / `b16` / `b32`）。常用于生产者与消费者对同一谓词状态采用不同粒度的场景。详细说明见 [`pto.pbitcast`](./ops/conversion-ops/pbitcast_zh.md)。

## 舍入模式

| 模式 | 含义 |
| --- | --- |
| `ROUND_R` | 最近偶数 |
| `ROUND_A` | 远离零 |
| `ROUND_F` | 向负无穷 |
| `ROUND_C` | 向正无穷 |
| `ROUND_Z` | 向零 |
| `ROUND_O` | Round to odd |

## 饱和模式

| 模式 | 含义 |
| --- | --- |
| `RS_ENABLE` | 溢出时饱和 |
| `RS_DISABLE` | 不做饱和 |

## Part 模式

| 模式 | 含义 |
| --- | --- |
| `PART_EVEN` | 写入偶数 lane |
| `PART_ODD` | 写入奇数 lane |

## 约束

!!! warning "约束"
    - 只允许文档化的源/目标类型对
    - `PART_EVEN` / `PART_ODD` 只对位宽变化形式有意义
    - `vtrc` 不改变元素类型
    - profile 会缩窄部分转换对和低精度支持

## 不允许的情形

!!! danger "不允许的情形"
    - 假设所有宽度变化转换都自动支持 pack/unpack 组合
    - 在没有文档说明的情况下使用未支持的源/目标类型组合
    - 把 `vci` 当作普通数值转换指令

## 相关页面

- [类型系统](../state-and-types/type-system_zh.md)
- [向量指令族](README_zh.md)
