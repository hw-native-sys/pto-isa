# PTO ISA 文档导航

<p align="center">
  <img src="figures/pto_logo.svg" alt="PTO ISA" width="200" />
</p>

**PTO ISA**（Parallel Tile Operation Instruction Set Architecture，平行瓦片操作指令集架构）是昇腾 NPU 的稳定、跨代际的机器无关指令集。它位于高层前端（C/C++、Python、TileLang、PyPTO）与目标特定后端之间，在昇腾各代际（A2/A3/A5）间提供统一版本化的指令语言。

> **文档版本：** PTO ISA 1.0
> **适用目标：** CPU 模拟器 · A2/A3（Ascend 910B/910C） · A5（Ascend 950）

---

## 快速导航

使用本页面作为**阅读指南**，而非目录索引。手册按五个逻辑层次组织——从与你的目标匹配的层次开始阅读。

### 五层结构

| 层次 | 内容 | 受众 |
|------|------|------|
| **1. 基础** | 引言、编程模型、机器模型 | 所有人——从此开始 |
| **2. 语法与语义** | 汇编模型、操作数、类型系统、内存模型 | 内核作者、编译器开发者 |
| **3. 指令集概述** | 指令集总览与指令族契约 | 所有用户 |
| **4. 参考手册** | Tile、Vector、Scalar、通信参考 | 性能工程师、内核作者 |
| **5. 附录** | 格式指南、诊断、术语表、可移植性 | 所有人 |

### 按指令集导航

| 指令集 | 前缀 | 职责 | 数量 | 参考 |
|--------|------|------|------|------|
| **Tile** | `pto.t*` | Tile 级计算、数据搬运、布局变换、同步 | ~120 条 | [Tile 参考](isa/tile/README_zh.md) |
| **Vector** | `pto.v*` | 向量微指令、lane 级 mask、流水线控制 | ~99 条 | [Vector 参考](isa/vector/README_zh.md) |
| **标量与控制** | `pto.*` | 配置、控制流、DMA 设置、谓词操作 | ~60 条 | [Scalar 参考](isa/scalar/README_zh.md) |
| **通信** | `pto.*` | 多 NPU 集体通信与运行时支撑 | ~24 条 | [通信参考](isa/other/README_zh.md) |

### 按任务导航

| 你的任务 | 从这里开始 |
|----------|----------|
| 理解 PTO 在软件栈中的位置 | [PTO ISA 是什么？](isa/introduction/what-is-pto-visa_zh.md) |
| 编写矩阵乘法 kernel | [Tile → 矩阵运算](isa/tile/matrix-and-matrix-vector_zh.md) |
| 优化逐元素运算 | [Tile → 逐元素](isa/tile/elementwise-tile-tile_zh.md) |
| 实现卷积 kernel | [Tile → img2col](isa/tile/ops/layout-and-rearrangement/timg2col_zh.md) |
| 设置数据搬运（GM ↔ tile） | [Tile 内存操作](isa/tile/memory-and-data-movement_zh.md) |
| 手写向量 kernel | [Vector 指令](isa/vector/README_zh.md) |
| 使用 lane 级 mask 与谓词 | [Vector → 谓词操作](isa/vector/predicate-and-materialization_zh.md) |
| 实现多 NPU 集体通信 | [通信指令](isa/other/README_zh.md) |
| 排序、量化或直方图操作 | [非常规操作](isa/tile/irregular-and-complex_zh.md) |
| 让编译器管理同步 | [Auto vs Manual 模式](isa/programming-model/auto-vs-manual_zh.md) |
| 手动管理流水线同步 | [同步模型](isa/machine-model/ordering-and-synchronization_zh.md) |
| 查询 A5 vs A2/A3 支持的数据类型/特性 | [目标 Profile](isa/machine-model/execution-agents_zh.md) |
| 首次阅读单条指令页面 | [指令描述格式](isa/reference/format-of-instruction-descriptions_zh.md) |

---

## 新手上路

初次接触 PTO？按以下路径阅读：

1. **[PTO ISA 是什么？](isa/introduction/what-is-pto-visa_zh.md)** — 核心概念、设计理念、PTO 在软件栈中的位置
2. **[编程模型：Tile 与有效区域](isa/programming-model/tiles-and-valid-regions_zh.md)** — Tile 抽象——PTO 的核心编程对象
3. **[机器模型：执行代理与 Profile](isa/machine-model/execution-agents_zh.md)** — 执行层次、流水线、目标 Profile 与同步
4. **[指令集概述](isa/instruction-surfaces/README_zh.md)** — 四大指令集总览及选用指南
5. **[逐指令参考](isa/README_zh.md)** — 按类别组织的完整指令目录

---

## 什么是 PTO ISA？

PTO ISA 是昇腾 NPU 软件栈的稳定指令语言。它抽象了 A2/A3/A5 各代际间的硬件差异，同时保留了充足的性能调优控制能力。

```
高层语言
（C/C++、Python、TileLang、PyPTO、代码生成器）
        │
        ▼
   PTO 指令（.pto 文本）
        │
        ├──► ptoas ──► C++ ──► bisheng ──► 二进制   （Flow A：经 C++ 中间层）
        │
        └──► ptoas ──────────────────────► 二进制        （Flow B：直接汇编）

目标平台：CPU 模拟器 / A2A3（Ascend 910B / 910C）/ A5（Ascend 950）
```

### Tile 指令 vs Vector 指令：何时选哪个？

| 判断标准 | Tile 指令（`pto.t*`） | Vector 指令（`pto.v*`） |
|----------|----------------------|------------------------|
| **典型用途** | 密集张量代数、矩阵乘法、逐元素运算 | 细粒度向量流水线控制、lane 级 mask |
| **数据搬运** | `TLOAD`/`TSTORE`（隐式 tile↔UB） | `copy_gm_to_ubuf` + `vlds`/`vsts` + `copy_ubuf_to_gm` |
| **同步方式** | `TSYNC`、`set_flag`/`wait_flag` | `set_flag`/`wait_flag`（向量流水线）、`mem_bar` |
| **布局控制** | 通过 tile 布局参数（`RowMajor`、`ColMajor`、分形布局） | 通过 distribution mode（`NORM`、`BRC`、`DS` 等） |
| **谓词** | 无 lane 级 mask（有效区域是粗粒度的） | 每个操作都支持完整 lane 级谓词 mask |
| **目标可移植性** | 所有 Profile（CPU、A2/A3、A5） | A5 硬件支持；CPU/A2/A3 为仿真 |
| **抽象层级** | 高层 tile 语义、有效区域 | 低层向量寄存器、显式 UB 暂存 |

> **经验法则：** 张量运算优先使用 tile 指令。只有在需要 lane 级 mask、自定义数据布局或 tile 指令无法表达的性能微调时，才降级到向量指令。

---

## 核心概念

阅读逐指令页面之前，以下概念必不可少。

### Tile

**Tile** 是带有架构可见 shape、layout 和有效区域元数据的受限多维数组片段。Tile 是 PTO 中的主要编程对象。

```cpp
Tile<Vec, float, 16, 16> a;  // 16×16 f32 tile，位于向量 tile buffer（UB）
Tile<Left, f16, 64, 64> b;   // 64×64 f16 左操作数（L0A）
Tile<Acc, i32, 128, 128> c; // 128×128 i32 累加器（L0C）
```

[了解更多 →](isa/programming-model/tiles-and-valid-regions_zh.md)

### 有效区域（Valid Region）

**有效区域** `(Rv, Cv)` 是 tile 声明形状中含有有效数据的子集。操作在目标 tile 的有效区域内迭代；源 tile 有效区域外的值为实现定义。

### TileType（位置意图）

**TileType** 决定 tile 由哪种硬件 buffer 支撑：

| TileType | 硬件 Buffer | 容量 | 典型用途 |
|----------|------------|------|----------|
| `Vec` | Unified Buffer（UB） | 256 KB | 通用逐元素运算 |
| `Left` | L0A | 64 KB | 矩阵乘法 A 操作数 |
| `Right` | L0B | 64 KB | 矩阵乘法 B 操作数 |
| `Acc` | L0C | 256 KB | 矩阵乘法累加器/输出 |
| `Mat` | L1 | 512 KB | 2D 矩阵操作数 |

### GlobalTensor

**GlobalTensor** 是片外设备内存（`__gm__` 地址空间）的视图。GM 与 tile buffer 之间的所有数据搬运均通过显式的 `TLOAD`/`TSTORE` 或 DMA 操作完成。

[了解更多 →](isa/programming-model/globaltensor-and-data-movement_zh.md)

### Auto vs Manual 模式

| 模式 | 资源绑定 | 同步 | 数据搬运 | 管理方 |
|------|---------|------|----------|--------|
| **Auto** | 编译器插入 `TASSIGN` | 编译器插入 `TSYNC` | 编译器插入 `TLOAD`/`TSTORE` | 编译器 |
| **Manual** | 作者显式写 `TASSIGN` | 作者显式写 `TSYNC` | 作者显式写 `TLOAD`/`TSTORE` | 你 |

[Auto vs Manual →](isa/programming-model/auto-vs-manual_zh.md)

### 目标 Profile

PTO ISA 由具体的**目标 Profile** 实例化，为特定后端限定可接受的子集。

| 特性 | CPU 模拟器 | A2/A3 Profile | A5 Profile |
|------|:---------:|:-------------:|:----------:|
| Tile 指令（`pto.t*`） | 完整 | 完整 | 完整 |
| Vector 指令（`pto.v*`） | 仿真 | 仿真 | 硬件完整支持 |
| 矩阵乘法 / CUBE 运算 | 软件回退 | 硬件 | 硬件 |
| 向量宽度（f32 / f16,bf16 / i8） | 可配置 | 64 / 128 / 256 | 64 / 128 / 256 |
| FP8 类型（`f8e4m3`、`f8e5m2`） | — | — | 支持 |
| 分形布局（NZ/ZN/FR/RN） | 仿真 | 仿真 | 完整 |
| 分块级集体通信 | — | 支持 | 支持 |

---

## 指令集导航地图

PTO 将其指令分为四个命名指令集。每个指令集有**契约页面**（共享规则）和**逐指令页面**（单条指令说明）。

### Tile 指令集 — `pto.t*`

```
Tile 指令集
├── 同步与配置             → tassign、tsync、tsetf32mode、tsetfmatrix、tset_img2col_*、tsubview、tget_scale_addr
├── 逐元素 Tile-Tile       → tadd、tsub、tmul、tdiv、tmin、tmax、tcmp、tcvt、tsel、tlog、trecip、texp、tsqrt、trsqrt、trem、tfmod、tabs、tand、tor、txor、tnot、tneg、tprelu、taddc、tsubc、tshl、tshr
├── Tile-标量与立即数       → tadds、tsubs、tmuls、tdi等等vs、tcmps、tsels、texpands、tfmods、trems、tands、tors、txors、tshls、tshrs、tlrelu、taddsc、tsubsc
├── 归约与扩展             → trowsum、tcolsum、trowprod、tcolprod、tcolmax、tcolmin、trowmax、trowmin、tcolargmax、tcolargmin、trowargmax、trowargmin
│                             → trowexpand、trowexpandadd、trowexpanddiv、trowexpandmul、trowexpandsub、trowexpandmax、trowexpandmin、trowexpandexpdif
│                             → tcolexpand、tcolexpandadd、tcolexpanddiv、tcolexpandmul、tcolexpandsub、tcolexpandmax、tcolexpandmin、tcolexpandexpdif
├── 内存与数据搬运         → tload、tprefetch、tstore、tstore_fp、mgather、mscatter
├── 矩阵与矩阵-向量         → tgemv、tgemv_mx、tgemv_acc、tgemv_bias、tmatmul、tmatmul_mx、tmatmul_acc、tmatmul_bias
├── 布局与重排             → tmov、tmov_fp、ttrans、textract、textract_fp、tinsert、tinsert_fp、timg2col、tfillpad、tfillpad_inplace、tfillpad_expand、treshape
└── 非常规与复杂操作       → tprint、tmrgsort、tsort32、tgather、tgatherb、tscatter、tci、ttri、tpartadd、tpartmul、tpartmax、tpartmin、tquant
```

[Tile 指令族契约 →](isa/instruction-families/tile-families_zh.md)

### Vector 指令集 — `pto.v*`

```
Vector 指令集
├── 向量加载存储             → vlds、vldas、vldus、vldx2、vsld、vsldb、vgather2、vgatherb、vgather2_bc
│                             → vsts、vstx2、vsst、vsstb、vscatter、vsta、vstas、vstar、vstu、vstus、vstur
├── 谓词与物化              → vbr、vdup
├── 一元向量运算            → vabs、vneg、vexp、vln、vsqrt、vrsqrt、vrec、vrelu、vnot、vbcnt、vcls、vmov
├── 二元向量运算            → vadd、vsub、vmul、vdiv、vmax、vmin、vand、vor、vxor、vshl、vshr、vaddc、vsubc
├── 向量-标量运算           → vadds、vsubs、vmuls、vmaxs、vmins、vands、vors、vxors、vshls、vshrs、vlrelu、vaddcs、vsubcs
├── 类型转换                → vci、vcvt、vtrc
├── 归约指令                → vcadd、vcmax、vcmin、vcgadd、vcgmax、vcgmin、vcpadd
├── 比较与选择              → vcmp、vcmps、vsel、vselr、vselrv2
├── 数据重排                → vintlv、vdintlv、vslide、vshift、vsqz、vusqz、vperm、vpack、vsunpack、vzunpack、vintlvv2、vdintlvv2
└── SFU 与 DSA             → vprelu、vexpdiff、vaddrelu、vsubrelu、vaxpy、vaddreluconv、vmulconv、vmull、vmula、vtranspose、vsort32、vbitsort、vmrgsort
```

[Vector 指令族契约 →](isa/instruction-families/vector-families_zh.md)

### 标量与控制指令集 — `pto.*`

```
标量与控制指令集
├── 控制与配置              → nop、barrier、yield；tsetf32mode、tsethf32mode、tsetfmatrix
├── 流水线同步             → set_flag、wait_flag、wait_flag_dev、pipe_barrier、mem_bar、get_buf、rls_buf
│                             → set_cross_core、set_intra_block、wait_intra_core
├── DMA 拷贝               → set_loop_size_outtoub、set_loop1/2_stride_outtoub
│                             → set_loop_size_ubtoout、set_loop1/2_stride_ubtoout
│                             → copy_gm_to_ubuf、copy_ubuf_to_gm、copy_ubuf_to_ubuf
├── 谓词加载存储            → pld、plds、pldi、psts、pst、psti、pstu
├── 谓词生成                → pset_b8/b16/b32、pge_b8/b16/b32、plt_b8/b16/b32
│                             → pand、por、pxor、pnot、psel、ppack、punpack
│                             → pdintlv_b8、pintlv_b16
├── 共享标量算术            → 跨指令集共享的标量算术运算
├── 共享结构化控制流        → 标量结构化控制流
└── 微指令                  → BlockDim 查询、指针操作、向量作用域、对齐状态
    [微指令汇总 →](isa/vector/micro-instruction-summary.md)
```

[标量指令族契约 →](isa/instruction-families/scalar-and-control-families_zh.md)

### 通信指令集 — `pto.*`

```
通信指令集
├── 集体操作                → tbroadcast、tget、tget_async、tput、tput_async
│                             → tscatter、tgather、treduce、ttest、twait、tnotify
└── 非 ISA 支撑操作          → talias、taxpy、tconcat、tdequant、tfree、thistogram
                              → tpack、tpop、tpush、trandom
```

[通信指令族契约 →](isa/instruction-families/other-families_zh.md)

---

## 编译流程

### Flow A：高层编译（ptoas → C++ → bisheng → 二进制）

高层前端发出 `.pto` 文本文件。`ptoas` 解析、验证并降级为调用 `pto-isa` C++ 库的 C++ 代码。后端编译器（bisheng）再生成最终二进制。

**适用人群：** 编译器开发者、库作者、高层框架集成商。`.pto` 格式可移植、可缓存。

### Flow B：直接汇编（ptoas → 二进制）

`ptoas` 直接汇编为目标二进制，跳过 C++ 中间步骤。

**适用人群：** 需要直接控制最终指令流的性能工程师，或将 `ptoas` 作为纯汇编器使用的工具链。

[了解更多编译流程 →](isa/introduction/what-is-pto-visa_zh.md#two-compilation-flows)

---

## 关键参考

| 参考资料 | 内容 |
|----------|------|
| **[PTO-AS 规范](assembly/PTO-AS_zh.md)** | `.pto` 文本文件的汇编语法与文法 |
| **[Tile 编程模型](coding/Tile_zh.md)** | Tile shape、mask 与数据组织 |
| **[事件与同步](coding/Event_zh.md)** | set/wait flag 与流水线同步 |
| **[性能优化](coding/opt_zh.md)** | 瓶颈分析与调优指导 |
| **[Auto Mode 概述](auto_mode/Auto_Mode_Overview_zh.md)** | 编译器驱动的资源管理与同步插入 |
| **[微指令汇总](isa/vector/micro-instruction-summary.md)** | 标量微指令：BlockDim、指针操作、向量作用域 |
| **[可移植性与目标 Profile](isa/reference/portability-and-target-profiles_zh.md)** | 各目标支持哪些特性 |
| **[术语表](isa/reference/glossary_zh.md)** | 术语定义参考 |
| **[规范来源](isa/reference/source-of-truth_zh.md)** | 哪些文件定义权威语义 |
| **[构建文档](mkdocs/README_zh.md)** | 本地生成文档站点 |

---

## 参与贡献

本文档源自 [github.com/PTO-ISA/pto-isa](https://github.com/PTO-ISA/pto-isa) 的权威 PTO ISA 规范。通过仓库 Issues 反馈问题，通过 Pull Request 提交更改。

---

*PTO ISA 是昇腾软件栈的一部分。版权所有 © Huawei Technologies Co., Ltd.*
