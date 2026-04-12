<!-- Generated from `docs/isa/conventions_zh.md` -->

# PTO ISA 通用约定

本页定义 `docs/isa/` 指令参考文档中使用的通用术语与写法，并与 `include/pto/common/pto_instr.hpp` 中的 C++ 内建接口保持一致。

## 记号

- **Tile**：片上二维操作数对象（例如 `pto::Tile<...>`）。大量指令以 Tile 作为输入/输出，并通过 `GetValidRow()` / `GetValidCol()` 使用 Tile 的有效区域（valid region）。
- **GM（全局内存）**：通过 `pto::GlobalTensor<...>` 访问的片外内存视图。
- **标量 / 立即数**：主机侧标量值，或在 `*S` / `*C` 等变体中编码的立即数参数。

关于这些对象的 C++ 编程模型（类型、布局、枚举、约束等），可参考：

- Tile：`docs/coding/Tile_zh.md`
- GlobalTensor：`docs/coding/GlobalTensor_zh.md`
- 标量与枚举：`docs/coding/Scalar_zh.md`

## 形状与布局

- **行主序 / 列主序**：除非指令页明确声明支持多种布局，否则示例与参考实现默认假设为行主序 Tile。支持多布局的指令会在约束小节中列出具体要求。
- **有效区域（valid region）**：Tile 运行时计算域，通常写作 `(valid_row, valid_col)`，并通过 `GetValidRow()` / `GetValidCol()` 查询。

### 有效区域语义

在指令页中，当我们写“对有效区域内的每个元素 `(i, j)`”，含义为：

- 除非指令显式定义不同的迭代域，否则默认使用 `valid_row = dst.GetValidRow()`、`valid_col = dst.GetValidCol()`。
- 数学语义仅对 `0 <= i < valid_row` 且 `0 <= j < valid_col` 的 `dst[i, j]` 做出定义。
- 有效区域之外元素的值为**未指定**，除非指令页明确说明（不要假设一定清零或保持不变）。

对多输入指令（例如 `src0`、`src1`），除非约束小节有更严格的要求，文档默认输入 Tile 与迭代域在形状/有效区域上是兼容的。

## 数据类型

每条指令页会列出支持的数据类型（例如 `fp16`、`fp32`、`int8`、`int16`、`int32`、`uint8`、`uint16`、`uint32` 等）。
不同后端/目标对数据类型与布局支持可能不同，具体以对应实现与编译期检查为准。

## 事件与同步

- 某些指令序列需要建立内存与向量流水线之间的顺序关系。示例中出现的事件（例如 `set_flag(...)` / `wait_flag(...)`）用于表达后端需要满足的顺序约束。
- 在需要显式同步的场景，使用 `TSYNC` 建立阶段间的顺序关系。

事件模型可参考：`docs/coding/Event_zh.md`。
