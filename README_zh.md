<p align="center">
  <img src="docs/figures/pto_logo.svg" alt="PTO Tile Lib" width="220" />
</p>

# PTO Tile Library

PTO（Parallel Tile Operation）是昇腾 CANN 定义的一套面向 tile 编程的虚拟 ISA。本仓库提供 PTO Tile 指令的实现、示例、测试与文档，帮助开发者在不同昇腾代际之间更平滑地迁移和优化算子。

[![License](https://img.shields.io/badge/License-CANN%20Open%20Software%20License%202.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Ascend%20A2%20%7C%20A3%20%7C%20A5%20%7C%20CPU-green.svg)](#-平台支持)
[![Docs](https://img.shields.io/badge/文档-blue.svg)](docs/README_zh.md)

## 最新动态

- **2025-12-27**：PTO Tile Library 正式开源发布。
- **2026-01-30**：新增归约类指令与 MX 指令。
- **2026-02-28**：新增卷积类指令、量化类指令、核间通信类指令。
- **2026-03-30**：支持昇腾 A5 芯片，新增异步通信指令与 CostModel 性能仿真。
- **2026-04-02**：本地工程链路进一步完善，补充 pre-commit 检查、文档构建校验与 CPU-SIM 验证能力。

## 项目定位

PTO ISA 基于昇腾底层硬件与软件抽象，定义 90+ 条标准 tile 指令，用更高层的 tile 编程模型桥接不同代际之间的实现差异。它的目标不是隐藏底层能力，而是在提升抽象层级的同时保留性能调优空间。

- **统一跨代 tile 抽象**：降低不同 Ascend 代际之间的迁移成本。
- **兼顾可移植性与性能**：在固定 tile shape 下保证正确工作，同时保留 tile size、tile shape、指令顺序等调优空间。
- **面向框架、算子与工具链**：作为上层框架、算子实现和编译工具链的共同接口。
- **持续可扩展**：当前已定义 90+ 条标准操作，并持续补充实现与生态集成。

PTO 指令现已集成到以下框架中：

- [PyPTO](https://gitcode.com/cann/pypto/) — PTO 上层 Python 编程框架
- [TileLang Ascend](https://github.com/tile-ai/tilelang-ascend/) — TileLang 昇腾后端
- 更多语言与前端支持持续完善中

## 核心特性

- **统一的 Tile ISA 抽象**：用标准 PTO 指令描述 tile 级计算与数据流。
- **跨代迁移与性能调优兼顾**：既提升可移植性，也保留充足的底层控制能力。
- **Auto / Manual 双模式开发路径**：先快速验证逻辑，再逐步深入优化。Auto Mode 目前可用于 CPU 仿真。
- **CPU Simulator 支持**：无需 Ascend 硬件即可在 CPU 上进行功能验证与开发调试。
- **覆盖关键编程要素**：tile shape、tile mask、事件同步、固定功能单元与流水线建模。
- **文档、测试、示例齐全**：ISA 文档、开发文档、测试脚本和性能案例全面覆盖。

## 适用人群

PTO Tile Library 主要面向以下开发者：

- 直接对接昇腾硬件的框架或编译器后端开发者
- 需要跨平台迁移与复用实现的高性能算子开发者
- 需要显式控制 tile、buffer 与流水线的性能优化工程师

## Tile 指令 vs Vector 指令

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

## 快速开始

### 环境准备

- **CPU 路径**：需要 Python、CMake 和支持 C++20 的编译器，适合跨平台快速验证。
- **NPU 路径**：需要 Linux 环境与 Ascend CANN toolkit，适合在昇腾硬件或模拟器上运行。
- 更详细的环境部署说明请参见：[快速开始指南](docs/getting-started_zh.md)

### 编译与运行

```bash
# CPU Simulator（推荐第一步）
python3 tests/run_cpu.py --clean --verbose

# 运行 GEMM demo
python3 tests/run_cpu.py --demo gemm --verbose

# 运行 Flash Attention demo
python3 tests/run_cpu.py --demo flash_attn --verbose

# 运行单个 ST 用例
python3 tests/script/run_st.py -r sim -v a3 -t tadd -g TADDTest.case_float_64x64_64x64

# 一键构建并运行推荐测试
./build.sh --run_all --a3 --sim
```

更完整的构建、测试和脚本说明请参见：[快速开始指南](docs/getting-started_zh.md)、[测试说明](tests/README_zh.md)

### 推荐样例

- [Auto Mode Add 示例](demos/auto_mode/baseline/add/README_zh.md)：了解 PTO 指令如何组织 tile 级计算与数据搬运
- [GEMM 性能示例](kernels/manual/a2a3/gemm_performance/README_zh.md)：理解 tile 级算子优化与性能调参
- [Flash Attention 示例](kernels/manual/common/flash_atten/README_zh.md)：理解复杂算子与性能调优

### 推荐上手路径

1. 从简单示例开始，理解 PTO 指令如何组织 tile 级计算与数据搬运。
2. 在 CPU 仿真中验证功能与正确性，建立对指令语义和结果的直观认知。
3. 将代码移植到昇腾硬件上验证正确性并采集性能数据。参见 [msprof 工具](https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/Profiling/atlasprofiling_16_0010.html)
4. 定位性能瓶颈（CUBE Bound / MTE Bound / Vector Bound），开始优化与调参。参见 [性能优化](docs/coding/opt_zh.md)

本仓库展示了标准 tile 操作如何通过模板参数映射到不同流水线实现：

- [Tile 编程模型](docs/coding/Tile_zh.md)：理解静态 tile shape、动态 tile mask 与数据组织方式
- [事件与同步](docs/coding/Event_zh.md)：理解 set/wait flag 与流水线同步
- [Auto Mode](docs/auto_mode/Auto_Mode_Overview_zh.md)：编译器自动管理资源绑定与同步插入
- [通用约定](docs/isa/conventions_zh.md)：理解 PTO 编程中的通用规则与约束
- [PTO 指令列表](docs/isa/README_zh.md)：按分类查看 PTO ISA 已定义的标准操作

## 文档导航

### ISA 与编程模型

| 文档 | 内容 |
|------|------|
| [ISA 总览](docs/README_zh.md) | PTO ISA 文档入口与阅读导航 |
| [PTO 指令列表](docs/isa/README_zh.md) | 按指令分类查看 PTO 标准操作 |
| [Tile 编程模型](docs/coding/Tile_zh.md) | tile 的形状、mask 与编程模型 |
| [事件与同步](docs/coding/Event_zh.md) | 事件记录、等待与同步方式 |
| [通用约定](docs/isa/conventions_zh.md) | 命名、约束与通用规则 |
| [Auto Mode](docs/auto_mode/Auto_Mode_Overview_zh.md) | 编译器驱动的资源管理与同步 |

### 开发与优化

| 文档 | 内容 |
|------|------|
| [开发文档索引](docs/coding/README_zh.md) | 扩展 PTO Tile Library 的开发文档入口 |
| [性能优化](docs/coding/opt_zh.md) | 性能分析与调优建议 |
| [文档构建说明](docs/mkdocs/README_zh.md) | MkDocs 文档的本地构建方式 |

## 性能参考

### GEMM

- 参考实现：`kernels/manual/a2a3/gemm_performance/`
- 详细分析与调参说明：[高性能 GEMM 算子示例](kernels/manual/a2a3/gemm_performance/README_zh.md)

![GEMM 性能参考（Ascend A3，24 核）](docs/figures/performance/gemm_performance_a3.svg)

### Flash Attention

- 参考实现：`kernels/manual/common/flash_atten/`
- 详细分析与调参说明：[Flash Attention 算子实现](kernels/manual/common/flash_atten/README_zh.md)
- S0：query 序列长度（Q/O 的行数）
- S1：key/value 序列长度（K/V 的行数）

![Flash Attention 归一化 TFLOPS（A2/A3）](docs/figures/performance/fa_normalized_tflops_a2a3.svg)

## 平台支持

- Ascend A2（Ascend 910B）
- Ascend A3（Ascend 910C）
- Ascend A5（Ascend 950）
- CPU（x86_64 / AArch64）

更多细节请参考 [include/README_zh.md](include/README_zh.md)。

## 路线图

| 功能 | 描述 | 范围 |
| --- | --- | --- |
| PTO Auto Mode | BiSheng 编译器支持：自动分配 tile buffer 并插入同步。 | 编译器 / 工具链 |
| PTO Tile Fusion | BiSheng 编译器支持：自动融合 tile 操作。 | 编译器 / 工具链 |
| PTO-AS | PTO ISA 的字节码支持。 | 编译器 / 工具链 |
| **卷积扩展** | PTO ISA 对卷积 kernel 的支持。 | ISA 扩展 |
| **集合通信扩展** | PTO ISA 对集合通信 kernel 的支持。 | ISA 扩展 |
| **系统调度扩展** | PTO ISA 对 SPMD/MPMD 编程的调度支持。 | ISA 扩展 |

## 目录结构

```
PTO Tile Library
├── include/                     # PTO 对外头文件与接口
│   └── pto/                    # 公共类型、ISA 接口、CPU/NPU 实现
│       ├── common/             # 平台无关的 Tile 与指令基础设施
│       ├── cpu/               # CPU 侧仿真支持
│       └── npu/               # NPU 侧实现（按 SoC 代际划分）
│           ├── a2a3/          # Ascend A2/A3 系列
│           └── a5/            # Ascend A5 系列
├── kernels/                    # kernel 与算子实现
│   ├── manual/                 # 手工优化实现与性能示例
│   │   ├── a2a3/             # A2/A3 平台 kernels（GEMM、Conv2D、TopK）
│   │   ├── a5/               # A5 平台 kernels（Flash Attention、MXFP4/8 Matmul）
│   │   └── common/            # 跨平台 kernels（Flash Attention）
│   └── custom/                 # 自定义算子脚手架
├── demos/                      # Auto Mode、baseline 与 torch_jit 示例
├── docs/                       # ISA、编程模型、快速开始与文档站点源文件
│   ├── isa/                   # 指令参考与分类索引
│   ├── coding/                # 开发与性能优化文档
│   ├── assembly/              # PTO-AS 汇编语法与规范
│   ├── auto_mode/             # Auto Mode 文档
│   └── mkdocs/                # MkDocs 文档构建配置与源文件
├── tests/                      # CPU / NPU 测试、脚本与测试入口
│   ├── cpu/                   # CPU 仿真测试
│   ├── npu/                   # 按 SoC 拆分的 NPU 测试
│   └── script/                # 测试构建与运行脚本
├── scripts/                    # 构建、安装与发布脚本
├── cmake/                      # CMake 公共配置与打包逻辑
├── build.sh                    # 一键构建与运行入口脚本
└── CMakeLists.txt              # 顶层 CMake 配置
```

## 相关信息

- [贡献指南](CONTRIBUTING_zh.md)：参与项目开发与提交流程
- [安全与漏洞披露](SECURITY_zh.md)：安全问题反馈流程
- [版本说明](ReleaseNote_zh.md)：版本更新与发布记录
- [许可证](LICENSE)：CANN Open Software License Agreement Version 2.0
- [PyPTO](https://gitcode.com/cann/pypto/)：PTO 生态中的上层 Python 编程框架
- [PTOAS](https://gitcode.com/cann/PTOAS/)：面向 PTO 工作流的汇编器与编译后端
- [pto-dsl](https://gitcode.com/cann/pto-dsl/)：面向 PTO 的 Python 前端与 JIT 工作流探索

## 联系我们

- **问题反馈**：通过仓库 Issues 提交问题
- **功能建议**：通过仓库 Issues 或讨论区反馈需求
- **贡献代码**：通过 Pull Request 参与项目贡献
