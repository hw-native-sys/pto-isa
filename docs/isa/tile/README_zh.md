# Tile 指令参考

`pto.t*` 是 PTO 指令集架构里以 tile 为中心的主干执行面。它覆盖 tile 数据的装载、逐元素计算、归约与扩展、布局重排、矩阵乘、显式同步，以及少量不规则专用操作。

这组文档按“先看家族页，再看单指令页”的方式组织。家族页负责解释共享机制、角色、约束和 profile 边界；`tile/ops/` 下的 leaf 页负责给出逐条指令的合同。

## 指令族

| 指令族 | 说明 | 典型指令 |
| --- | --- | --- |
| [同步与配置](./sync-and-config_zh.md) | 资源绑定、事件等待、tile 侧模式设置 | `TASSIGN`、`TSYNC` |
| [逐元素 Tile-Tile](./elementwise-tile-tile_zh.md) | tile 与 tile 的逐元素算术、比较和选择 | `TADD`、`TMUL`、`TSEL` |
| [Tile-标量与立即数](./tile-scalar-and-immediate_zh.md) | tile 与标量或立即数的组合运算 | `TADDS`、`TMULS` |
| [归约与扩展](./reduce-and-expand_zh.md) | 行 / 列归约和按轴扩展 | `TROWSUM`、`TROWEXPAND` |
| [内存与数据搬运](./memory-and-data-movement_zh.md) | GM 与 tile 间传输，以及 tile 侧 gather / scatter | `TLOAD`、`TSTORE` |
| [矩阵与矩阵-向量](./matrix-and-matrix-vector_zh.md) | cube 路径矩阵乘、GEMV 及其变体 | `TMATMUL`、`TGEMV` |
| [布局与重排](./layout-and-rearrangement_zh.md) | reshape、transpose、extract、insert、img2col | `TTRANS`、`TIMG2COL` |
| [不规则与复杂](./irregular-and-complex_zh.md) | 排序、量化、索引型搬运、部分归约等 | `TSORT32`、`TQUANT` |

## 常见 tile 角色

PTO 手册里的 tile 角色是架构抽象，不应和某个后端的单一物理实现混为一谈。读 tile 指令时，先分清角色，再看 dtype、shape、layout 和 valid region。

| 角色 | 含义 | 典型用途 |
| --- | --- | --- |
| `Vec` | 向量 tile buffer 抽象 | 逐元素、归约、搬运、重排 |
| `Left` | 左矩阵操作数 tile，对应 L0A 路径 | matmul / GEMV 左输入 |
| `Right` | 右矩阵操作数 tile，对应 L0B 路径 | matmul / GEMV 右输入 |
| `Acc` | 累加器 / 输出 tile | matmul / GEMV 结果 |
| `Bias` | 偏置 tile | `*_bias` 变体 |
| `ScaleLeft` / `ScaleRight` | MX block-scale 的左右 scale tile | `*_mx` 变体 |

## 阅读顺序

如果你刚开始查 PTO tile 指令，建议按这个顺序读：

1. 先看 [Tile 指令表面](../instruction-surfaces/tile-instructions_zh.md)，明确 tile 路径和标量 / 向量路径的边界。
2. 再看 [位置意图与合法性](../state-and-types/location-intent-and-legality_zh.md) 与 [布局](../state-and-types/layout_zh.md)，把角色和布局约束建立起来。
3. 然后进入对应家族页。
4. 最后再查具体 leaf 页。

## 共享约束

- tile 的 `dtype`、`shape`、`layout`、`role` 和 `valid region` 都可能进入合法性判断。
- 大多数逐元素与重排操作以目标 tile 的 valid region 作为迭代域。
- 矩阵乘类操作额外受 `Left` / `Right` / `Acc` / `Bias` / scale tile 角色约束。
- 某些高性能路径或特殊格式路径只在特定 profile 上开放，例如 A5 专属 MX block-scale。

## 相关页面

- [Tile 指令表面](../instruction-surfaces/tile-instructions_zh.md)
- [Tile 指令族](../instruction-families/tile-families_zh.md)
- [指令描述格式](../reference/format-of-instruction-descriptions_zh.md)
