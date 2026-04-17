# 同步与配置指令集

同步与配置类操作负责管理 tile 路径可见的状态：资源绑定、事件等待、tile 侧模式设置，以及基于现有 tile 创建逻辑视图。它们本身通常不产生算术载荷，但会决定后续 tile 指令怎样取数、怎样排序、怎样解释数据。

这类页最容易被写成“几条杂项命令列表”。实际上它们承担的是 tile 执行面的控制面，因此应先说明状态模型，再看具体指令。

## 操作

| 操作 | 作用 | 类别 |
| --- | --- | --- |
| [pto.tassign](./ops/sync-and-config/tassign_zh.md) | 把 tile 绑定到 tile-buffer 地址 | 资源绑定 |
| [pto.tsync](./ops/sync-and-config/tsync_zh.md) | 等待事件或插入屏障 | 同步 |
| [pto.tsettf32mode](./ops/sync-and-config/tsettf32mode_zh.md) | 设置 tile 路径的 TF32 行为 | 模式配置 |
| [pto.tset_img2col_rpt](./ops/sync-and-config/tset-img2col-rpt_zh.md) | 设置 img2col 重复次数 | 模式配置 |
| [pto.tset_img2col_padding](./ops/sync-and-config/tset-img2col-padding_zh.md) | 设置 img2col padding 形态 | 模式配置 |
| [pto.tsubview](./ops/sync-and-config/tsubview_zh.md) | 基于已有 tile 创建子视图 | 视图 |
| [pto.tget_scale_addr](./ops/sync-and-config/tget-scale-addr_zh.md) | 取得量化 / MX 路径使用的 scale 地址 | 辅助查询 |

兼容性说明：

- [tsethf32mode](./ops/sync-and-config/tsethf32mode_zh.md) 和 [tsetfmatrix](./ops/sync-and-config/tsetfmatrix_zh.md) 仍保留旧路径入口；
- 但它们的规范位置已经迁移到 [标量控制与配置](../scalar/control-and-configuration_zh.md)，因为它们配置的是标量可见状态，而不是 tile 载荷合同。

## 机制

### TASSIGN

`TASSIGN` 把 tile 句柄绑定到某个 tile-buffer 地址。Auto 模式下，这类地址可以由编译器 / 运行时管理；Manual 模式下，则需要程序员显式指定。

这条指令修改的是“tile 句柄与物理存储的关系”，不是 tile 内容本身。

### TSYNC

`TSYNC` 有两类常见用法：

1. 事件等待形式：`TSYNC(%e0, %e1)`
   等待前序操作产生的事件完成。
2. 屏障形式：`TSYNC<Op>()`
   对某一类 tile 操作建立顺序边界。

它的核心价值不在“暂停一下”，而在于把 producer-consumer 关系写成可验证的显式合同。

### TSET*

本页只保留 tile 侧配置，例如 TF32 或 img2col 相关模式。它们会影响后续同类 tile 指令，直到再次被覆盖。

### TSUBVIEW

`TSUBVIEW` 创建的是逻辑子视图，不会复制底层数据。源 tile 和子视图共享同一份底层存储，因此它的价值在于“改变解释与访问窗口”，而不是搬运数据。

### TGET_SCALE_ADDR

`TGET_SCALE_ADDR` 用于取得量化或 MX 路径使用的 scale 地址，便于后续配置或调度使用。它本身不改变 tile 载荷。

## 为什么这组指令要单列

很多 tile 算术页都会引用这组状态，但不会重复解释它们。把同步、资源绑定、模式配置和逻辑视图集中到一个家族里，读者才能先建立控制面，再去理解数据面。

## 约束

- `TASSIGN` 绑定的是地址。若两个没有别名关系的 tile 同时复用同一地址，其行为不受手册保证。
- 空参数 `TSYNC()` 是 no-op。
- `TSET*` 的配置会影响后续依赖该模式的同类指令，直到下一次同类设置覆盖它。
- `TSUBVIEW` 与源 tile 共享底层存储，超出子视图范围的访问不受手册保证。

## 不允许的情形

- 在没有别名语义的前提下，让两个活跃 tile 共享同一物理地址；
- 等待一个从未由前序操作产生的事件；
- 在相关依赖操作尚未完成时改写会影响它们的模式配置。

## 相关页面

- [Tile 指令族](../instruction-families/tile-families_zh.md)
- [顺序与同步](../machine-model/ordering-and-synchronization_zh.md)
- [标量控制与配置](../scalar/control-and-configuration_zh.md)
