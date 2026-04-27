# PTO 汇编与 ISA 文档约定（中文）

本页定义在 `docs/isa/` 指令参考与 `docs/assembly/` 汇编文档中复用的通用约定。

## Tiles 与形状

- **Tile**：PTO 指令的核心操作数类型。多数指令在 Tile 的**有效区域**上定义语义。
- **有效区域（valid region）**：Tile 的活跃子矩形。多数操作迭代 `tile.GetValidRow()` 与 `tile.GetValidCol()`。
- **布局（layout）**：Tile 布局由模板参数描述，例如外层 `BLayout` 与可选内层盒化布局 `SLayout`。

## GlobalTensor（GM）

- **GlobalTensor** 表示存放在全局内存（GM）中的张量视图。`TLOAD`/`TSTORE` 在 GM 与 Tile 之间搬运数据。

## 事件与同步

PTO 支持用事件建模操作间依赖：

- **Producer**：指令在完成时可以 *record* 一个事件 token。
- **Consumer**：指令可以依赖一个或多个先前记录的事件 token。

在 C++ 内建接口中，这通常体现为将 event 对象作为额外参数传入下一条 op。

## 汇编语法（PTO-AS）

指令文档中的 PTO-AS 示例采用类 MLIR 的约定：

- SSA 值名使用 `%name`。
- 类型使用 MLIR 风格拼写，例如 `!pto.tile<...>` 与 `!pto.memref<...>`。

完整语法与文法参见：`docs/isa/syntax-and-operands/assembly-model_zh.md`。
