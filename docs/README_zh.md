<p align="center">
  <img src="figures/pto_logo.svg" alt="PTO Tile Lib" width="200" />
</p>

# PTO ISA 指南

这里是 PTO Tile Lib 文档入口，介绍 PTO ISA（指令集架构）的命名/符号约定，以及如何查阅“每条指令一页”的参考手册。

## 命名与符号

- **Tile**：小张量的基础数据类型（例如 `MatTile`、`LeftTile`、`RightTile`、`BiasTile`、`AccumulationTile`、`VecTile`）。
- **GlobalTensor**：存放在全局内存（GM）中的张量；`TLOAD`/`TSTORE` 用于在 GM 与 Tile 之间搬运数据。
- **`%R`**：标量/立即数寄存器；例如 `cmpMode`、`rmode` 等字段属于指令修饰符（modifier）。
- **形状与对齐**：通过编译期约束与运行期断言共同约束；不合法的使用应尽快失败（fail fast）。

## 从哪里开始

- [虚拟 ISA 手册入口](PTO-Virtual-ISA-Manual_zh.md)
- [ISA 总览](PTOISA_zh.md)
- [指令索引](isa/README_zh.md)
- [PTO AS 索引](assembly/README_zh.md)
- [通用约定](isa/conventions_zh.md)
- [PTO 汇编语法（PTO-AS）](assembly/PTO-AS_zh.md)
- [虚拟 ISA / AS 指南](../manual/09-virtual-isa-and-ir_zh.md)
- [字节码 / 工具链指南](../manual/10-bytecode-and-toolchain_zh.md)
- [内存顺序 / 一致性指南](../manual/11-memory-ordering-and-consistency_zh.md)
- [后端画像 / 一致性指南](../manual/12-backend-profiles-and-conformance_zh.md)
- [入门指南（建议先跑 CPU 仿真）](getting-started_zh.md)
- [实现与扩展说明](coding/README_zh.md)
- [Kernel 示例（偏 NPU）](../kernels/README_zh.md)
- [文档工具（manifest/index/svg/一致性检查）](tools/)

## 文档组织

- `docs/isa/`：指令参考（每条指令一页，以及分类索引）
- `docs/assembly/`：PTO 汇编语法与规范（PTO-AS）
- `docs/coding/`：扩展 PTO Tile Lib 的开发者说明
- `docs/auto_mode/`: auto模式文档
