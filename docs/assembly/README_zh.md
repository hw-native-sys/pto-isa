# PTO AS 文档导航

PTO-AS 的规范语法页面是 PTO ISA 手册中的 [汇编写法与操作数](../isa/syntax-and-operands/assembly-model_zh.md)。手册导航与指令页面链接应指向该页面。

PTO AS 文档主要覆盖以下几类内容：

- PTO-AS 语法、文法与文本表示形式
- ISA 级 tile 操作与辅助 AS 构造
- 从 MLIR 复用的标量算术与控制流操作
- 汇编相关约定与配套参考资料

## 建议阅读路径

如果您第一次接触 PTO-AS，建议按以下顺序阅读：

1. [汇编写法与操作数](../isa/syntax-and-operands/assembly-model_zh.md)：PTO-AS 写法、操作数与语法
2. [PTO-AS 约定](conventions_zh.md)：理解命名与文档编写约定
3. 各类操作文档：按任务需要继续阅读对应分类页面

## 文档分类

### 1. PTO-AS 语法与核心规范

- [汇编写法与操作数](../isa/syntax-and-operands/assembly-model_zh.md)：PTO ISA 手册发布的规范语法页面
- [PTO-AS 约定](conventions_zh.md)：汇编语法约定与相关文档规则

### 2. PTO Tile 操作分类

- [逐元素操作](elementwise-ops_zh.md)：tile-tile 逐元素操作
- [Tile-标量操作](tile-scalar-ops_zh.md)：tile 与标量之间的算术、比较与激活操作
- [轴归约和扩展](axis-ops_zh.md)：行/列归约与广播式扩展操作
- [内存操作](memory-ops_zh.md)：GM 与 tile 之间的数据搬运操作
- [矩阵乘法](matrix-ops_zh.md)：GEMM 与 GEMV 相关操作
- [数据移动和布局](data-movement-ops_zh.md)：提取、插入、转置、reshape 与 padding 操作
- [复杂操作](complex-ops_zh.md)：排序、gather/scatter、随机数、量化与工具类操作
- [手动资源绑定](manual-binding-ops_zh.md)：赋值与硬件/资源配置类操作

### 3. 辅助 AS 与 MLIR 派生操作

- [辅助函数](nonisa-ops_zh.md)：张量视图、tile 分配、索引与同步辅助构造
- [标量算术操作](scalar-arith-ops_zh.md)：来自 MLIR `arith` 的标量算术操作
- [控制流操作](control-flow-ops_zh.md)：来自 MLIR `scf` 的结构化控制流操作

### 4. 相关参考

- [ISA 指令参考](../isa/scalar/ops/micro-instruction/README_zh.md)：逐条指令的规范语义
- [docs 文档入口](../isa/scalar/ops/micro-instruction/README_zh.md)：返回 PTO Tile Lib 文档总导航页

## 目录结构

关键条目如下：

```text
├── PTO-AS*                     # PTO-AS 语法与规范文档
├── conventions*                # 汇编约定文档
├── elementwise-ops*            # 逐元素操作参考
├── tile-scalar-ops*            # Tile-标量操作参考
├── axis-ops*                   # 轴归约与扩展参考
├── memory-ops*                 # 内存操作参考
├── matrix-ops*                 # 矩阵乘法参考
├── data-movement-ops*          # 数据移动与布局参考
├── complex-ops*                # 复杂操作参考
├── manual-binding-ops*         # 手动资源绑定参考
├── scalar-arith-ops*           # 标量算术参考
├── control-flow-ops*           # 控制流参考
└── nonisa-ops*                 # 辅助 AS 构造参考
```

## 相关入口

- [ISA 指令参考](../isa/scalar/ops/micro-instruction/README_zh.md)：查看 PTO 指令的规范语义
- [docs 文档入口](../isa/scalar/ops/micro-instruction/README_zh.md)：返回文档总导航页
- [Machine 文档](../machine/README_zh.md)：了解抽象执行模型
