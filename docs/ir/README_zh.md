# PTO IR 操作参考

本目录包含 PTO IR 操作的全面文档，涵盖 ISA 级别的 tile 操作和 PTO Level-1 及 Level-2 中间表示中使用的辅助 IR 构造。

---

## 概述

PTO AS 提供 **116 个 tile 操作**、**11 个辅助函数**、**47 个标量算术操作**和 **7 个控制流操作**。

每个操作都记录有：
- **IR Level 1 (SSA)**：静态单赋值形式
- **IR Level 2 (DPS)**：目标传递风格
- **数学语义**：形式化数学解释
- **约束**：类型、布局和运行时要求

---

## 辅助函数（11 个函数）

**文档**：[辅助函数](PTO-IR-ops_zh.md)

用于张量视图管理、tile 分配和同步的 IR 级构造：

- **张量视图**：`make_tensor_view`、`partition_view`
- **Tile 管理**：`alloc_tile`、`tgetval`、`tsetval`
- **索引**：`get_block_idx`、`get_subblock_idx`、`get_block_num`、`get_subblock_num`
- **指针运算**：`addptr`
- **同步**：`record_event`、`wait_event`、`barrier`、`PIPE_BARRIER`

---

## Tile 操作（116 个操作）

### 逐元素操作（Tile-Tile）- 28 个操作
**文档**：[逐元素操作](PTO-IR-elementwise-ops_zh.md)

- **算术**：`TADD`、`TSUB`、`TMUL`、`TDIV`、`TABS`、`TNEG`
- **位运算**：`TAND`、`TOR`、`TXOR`、`TNOT`、`TSHL`、`TSHR`
- **比较**：`TCMP`、`TMIN`、`TMAX`
- **数学**：`TLOG`、`TEXP`、`TSQRT`、`TRSQRT`、`TRECIP`
- **激活**：`TRELU`、`TPRELU`
- **类型转换**：`TCVT`
- **条件**：`TSEL`
- **复合**：`TADDC`、`TSUBC`
- **取模**：`TREM`、`TFMOD`

### Tile-标量操作 - 19 个操作
**文档**：[Tile-标量操作](PTO-IR-tile-scalar-ops_zh.md)

- **算术**：`TADDS`、`TSUBS`、`TMULS`、`TDIVS`、`TMINS`、`TMAXS`
- **位运算**：`TANDS`、`TORS`、`TXORS`、`TSHLS`、`TSHRS`
- **取模**：`TREMS`、`TFMODS`
- **广播**：`TEXPANDS`
- **比较**：`TCMPS`
- **条件**：`TSELS`
- **激活**：`TLRELU`
- **复合**：`TADDSC`、`TSUBSC`

### 轴归约和扩展 - 23 个操作
**文档**：[轴归约和扩展](PTO-IR-axis-ops_zh.md)

- **行归约**：`TROWSUM`、`TROWMAX`、`TROWMIN`
- **列归约**：`TCOLSUM`、`TCOLMAX`、`TCOLMIN`、`TCOLPROD`
- **行扩展**：`TROWEXPAND`、`TROWEXPANDADD`、`TROWEXPANDMUL`、`TROWEXPANDDIV`、`TROWEXPANDSUB`、`TROWEXPANDMAX`、`TROWEXPANDMIN`、`TROWEXPANDEXPDIF`
- **列扩展**：`TCOLEXPAND`、`TCOLEXPANDADD`、`TCOLEXPANDMUL`、`TCOLEXPANDDIV`、`TCOLEXPANDSUB`、`TCOLEXPANDMAX`、`TCOLEXPANDMIN`、`TCOLEXPANDEXPDIF`

### 内存操作 - 6 个操作
**文档**：[内存操作](PTO-IR-memory-ops_zh.md)

- **加载/存储**：`TLOAD`、`TSTORE`、`TSTORE_FP`、`TPREFETCH`
- **收集/分散**：`MGATHER`、`MSCATTER`

### 矩阵乘法 - 8 个操作
**文档**：[矩阵乘法](PTO-IR-matrix-ops_zh.md)

- **基础**：`TMATMUL`、`TMATMUL_ACC`、`TMATMUL_BIAS`
- **混合精度**：`TMATMUL_MX`
- **向量**：`TGEMV`、`TGEMV_ACC`、`TGEMV_BIAS`、`TGEMV_MX`

### 数据移动和布局 - 12 个操作
**文档**：[数据移动和布局](PTO-IR-data-movement-ops_zh.md)

- **提取/插入**：`TEXTRACT`、`TEXTRACT_FP`、`TINSERT`、`TINSERT_FP`
- **转换**：`TTRANS`、`TRESHAPE`、`TIMG2COL`
- **移动**：`TMOV`、`TMOV_FP`
- **填充**：`TFILLPAD`、`TFILLPAD_INPLACE`、`TFILLPAD_EXPAND`

### 复杂操作 - 13 个操作
**文档**：[复杂操作](PTO-IR-complex-ops_zh.md)

- **排序**：`TSORT32`、`TMRGSORT`
- **收集**：`TGATHER`、`TGATHERB`、`TSCATTER`
- **部分操作**：`TPARTADD`、`TPARTMUL`、`TPARTMAX`、`TPARTMIN`
- **实用工具**：`TCI`、`TTRI`、`TQUANT`、`TPRINT`

### 手动资源绑定 - 6 个操作
**文档**：[手动资源绑定](PTO-IR-manual-binding-ops_zh.md)

- **赋值**：`TASSIGN`
- **模式配置**：`TSETHF32MODE`、`TSETTF32MODE`、`TSETFMATRIX`
- **IMG2COL 配置**：`TSET_IMG2COL_RPT`、`TSET_IMG2COL_PADDING`

---

## 标量算术操作（47 个操作）

**文档**：[标量算术操作](PTO-IR-scalar-arith-ops_zh.md)

来自 MLIR `arith` 方言的标准标量操作（仅标量，无向量/张量）：

- **整数算术**：`addi`、`subi`、`muli`、`divsi`、`divui`、`remsi`、`remui`、`ceildivsi`、`ceildivui`、`floordivsi`
- **浮点算术**：`addf`、`subf`、`mulf`、`divf`、`remf`、`negf`
- **位运算**：`andi`、`ori`、`xori`
- **移位**：`shli`、`shrsi`、`shrui`
- **比较**：`cmpi`、`cmpf`
- **最小/最大**：`minsi`、`minui`、`maxsi`、`maxui`、`minimumf`、`maximumf`、`minnumf`、`maxnumf`
- **类型转换**：`extsi`、`extui`、`trunci`、`extf`、`truncf`、`sitofp`、`uitofp`、`fptosi`、`fptoui`、`bitcast`、`index_cast`、`index_castui`
- **特殊操作**：`select`、`constant`
- **扩展算术**：`addui_extended`、`mulsi_extended`、`mului_extended`

---

## 控制流操作（7 个操作）

**文档**：[控制流操作](PTO-IR-control-flow-ops_zh.md)

来自 MLIR `scf` 方言的结构化控制流操作：

- **循环**：`scf.for`、`scf.while`
- **条件**：`scf.if`、`scf.index_switch`
- **区域**：`scf.execute_region`
- **终止符**：`scf.yield`、`scf.condition`

---

## 相关资源

- **ISA 指令参考**：`../isa/README_zh.md` - 逐条指令的规范语义
- **PTO-AS 文法**：`../grammar/PTO-AS_zh.md` - 汇编语言语法和文法
