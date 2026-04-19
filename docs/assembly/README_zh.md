# PTO AS 文档导航

这里是 PTO AS 文档的主入口页，用于帮助读者按主题快速定位汇编相关文档，而不是逐个文件查找。

## 按任务选择

| 你的需求 | 从这里开始 |
|----------|----------|
| 理解 PTO-AS 语法与文法 | [PTO-AS 规范](PTO-AS_zh.md) |
| 了解操作分类与链接入口 | [PTO AS 操作参考](README_zh.md) |
| 理解命名与文档编写约定 | [PTO-AS 约定](conventions_zh.md) |
| 按类别查找操作 | 见下方文档分类 |

## 文档分类

### 1. PTO-AS 语法与核心规范

| 文档 | 内容 |
|------|------|
| [PTO-AS 规范](PTO-AS_zh.md) | 文本格式、SSA 风格命名、directives 与文法概览 |
| [PTO-AS 约定](conventions_zh.md) | 汇编语法约定与相关文档规则 |
| `PTO-AS.bnf` | PTO-AS 的 BNF 形式文法定义 |

### 2. PTO Tile 操作分类

| 文档 | 内容 |
|------|------|
| [逐元素操作](elementwise-ops_zh.md) | tile-tile 逐元素操作 |
| [Tile-标量操作](tile-scalar-ops_zh.md) | tile 与标量之间的算术、比较与激活操作 |
| [轴归约和扩展](axis-ops_zh.md) | 行/列归约与广播式扩展操作 |
| [内存操作](memory-ops_zh.md) | GM 与 tile 之间的数据搬运操作 |
| [矩阵乘法](matrix-ops_zh.md) | GEMM 与 GEMV 相关操作 |
| [数据移动和布局](data-movement-ops_zh.md) | 提取、插入、转置、reshape 与 padding 操作 |
| [复杂操作](complex-ops_zh.md) | 排序、gather/scatter、随机数、量化与工具类操作 |
| [手动资源绑定](manual-binding-ops_zh.md) | 赋值与硬件/资源配置类操作 |

### 3. 辅助 AS 与 MLIR 派生操作

| 文档 | 内容 |
|------|------|
| [辅助函数](nonisa-ops_zh.md) | 张量视图、tile 分配、索引与同步辅助构造 |
| [标量算术操作](scalar-arith-ops_zh.md) | 来自 MLIR `arith` 的标量算术操作 |
| [控制流操作](control-flow-ops_zh.md) | 来自 MLIR `scf` 的结构化控制流操作 |

### 4. 相关参考

| 文档 | 内容 |
|------|------|
| [ISA 指令参考](../isa/README_zh.md) | 逐条指令的规范语义 |
| [docs 文档入口](../README_zh.md) | PTO Tile Library 文档总导航页 |
| [Machine 文档](../machine/README_zh.md) | 抽象执行模型 |

## 相关入口

- [ISA 指令参考](../isa/README_zh.md)
- [docs 文档入口](../README_zh.md)
- [Machine 文档](../machine/README_zh.md)
