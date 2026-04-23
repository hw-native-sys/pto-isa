# 非 ISA 与支撑操作

这组操作提供 tile 序列、量化、内存释放和辅助变换等支撑语义。部分操作可以映射到多个核心 ISA 操作的组合，部分则依赖目标 profile 的专用支持。

## 操作概览

| 操作 | 说明 | 类别 |
| --- | --- | --- |
| `talias` | 在不复制数据的前提下创建 tile 别名视图 | Alias |
| `taxpy` | 融合乘加 `dst = src0 * scalar + src1` | Fused compute |
| `tconcat` | 按指定维度拼接 tile | Tile sequence |
| [pto.tdequant](./ops/non-isa-and-supporting-ops/tdequant_zh.md) | 把量化表示恢复成数值表示 | Quantize |
| `tfree` | 释放先前分配的 tile 或缓冲区 | Memory |
| `thistogram` | 统计 tile 元素直方图 | Statistics |
| `tpack` | 将多个 tile 打包进单一 tile buffer | Tile sequence |
| `tpop` | 计算谓词 mask 的 population count | Predicate |
| `tpush` | 计算谓词 mask 的 push count | Predicate |
| `trandom` | 用随机值填充 tile | Generation |

## 机制摘要

### Alias

`talias` 生成共享底层存储的新视图，不复制数据。别名视图与原 tile 可以拥有不同的 shape、layout 或 valid region，但共享同一块底层存储。

### Tile 序列

`tconcat` 和 `tpack` 用于在 tile 序列层组织数据。它们描述的是更高层的组合关系，而不是单条底层算术指令。

### 量化

`tdequant` 负责在量化表示和浮点表示之间转换。当前作者维护树中，[pto.tdequant](./ops/non-isa-and-supporting-ops/tdequant_zh.md) 位于支撑操作路径；`pto.tquant` 仍位于 tile 不规则与复杂路径。

### Memory

`tfree` 修改可复用的分配状态。它不会产生新的 tile 数据，但会影响后续资源使用。

## 约束

- 量化必须使用合法的 scale 和 zero-point。
- `tconcat` 需要沿拼接轴满足维度兼容关系。
- `tfree` 不能释放仍在使用中的对象。
- A5 专属操作只能在支持它们的 profile 上使用。

## 不允许的情形

- 在仍被后续操作引用的对象上调用 `tfree`
- 使用非法量化参数
- 假设所有 backend 都以同一条硬件指令实现 `taxpy`
- 在不支持的 profile 上使用 `thistogram`、`tpack` 或 `trandom`

## 相关页面

- [其他指令集](../instruction-surfaces/other-instructions_zh.md)
- [其他指令族](../instruction-families/other-families_zh.md)
