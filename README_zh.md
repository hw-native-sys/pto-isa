<p align="center">
  <img src="docs/figures/pto_logo.svg" alt="PTO Tile Lib" width="220" />
</p>

# PTO Tile Library

PTO（Parallel Tile Operation）是昇腾 CANN 定义的一套面向 tile 编程的虚拟 ISA。本仓库提供 PTO Tile 指令的实现、示例、测试与文档，帮助开发者在不同昇腾代际之间更平滑地迁移和优化算子。

[![License](https://img.shields.io/badge/License-CANN%20Open%20Software%20License%202.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Ascend%20A2%20%7C%20A3%20%7C%20A5%20%7C%20CPU-green.svg)](#️-平台支持)
[![Docs](https://img.shields.io/badge/Docs-文档-blue.svg)](docs/README_zh.md)

## 📰 新闻

- 🎉 **2025-12-27**：PTO Tile Library 正式开源发布。
- ✨ **2026-01-30**：新增合轴类指令、MX 指令。
- 🚀 **2026-02-28**：新增卷积类指令、量化类指令、核间通信类指令。
- 🔥 **2026-03-30**：支持昇腾 A5 芯片，新增异步通信指令、CostModel 性能仿真。
- 🛠️ **2026-04-02**：本地工程链路进一步完善，补充了 pre-commit 检查、文档构建校验与 CPU-SIM 验证能力。

## 🎯 项目定位

PTO ISA 基于昇腾底层硬件与软件抽象，定义 90+ 条标准 tile 指令，用更高层的 tile 编程模型桥接不同代际之间的实现差异。它的目标不是隐藏底层能力，而是在提升抽象层级的同时保留性能调优空间。

- **统一跨代 tile 抽象**：降低不同 Ascend 代际之间的迁移成本。
- **兼顾可移植性与性能**：在固定 tile shape 下保证正确工作，同时保留 tile size、tile shape、指令顺序等调优能力。
- **面向框架、算子与工具链**：可作为上层框架、算子实现和编译工具链的共同接口。
- **支持持续扩展**：当前已定义 90+ 条标准操作，并持续补充实现与生态集成。

除计算与数据搬运指令外，PTO ISA 还提供了面向 NPU 间数据传输与同步的**通信扩展指令集**，覆盖点对点通信、信号同步和集合通信三类能力。

这些通信原语延续了与计算指令一致的 tile 级抽象和跨平台设计，并可驱动昇腾上的多种数据搬移硬件引擎，帮助用户构建计算与通信深度融合的 kernel。通信 ISA 入口见 [docs/isa/comm/README_zh.md](docs/isa/comm/README_zh.md)。

目前，PTO 指令已集成到以下框架中：

- [PyPTO](https://gitcode.com/cann/pypto/)
- [TileLang Ascend](https://github.com/tile-ai/tilelang-ascend/)
- 更多语言与前端持续完善中

## ✨ 核心特性

- **统一的 Tile ISA 抽象**：用标准 PTO 指令描述 tile 级计算与数据流。
- **跨代迁移与性能调优兼顾**：既提升可移植性，也保留足够的底层控制能力。
- **Auto / Manual 双模式开发路径**：先快速验证逻辑，再逐步深入优化实现。当前 Auto Mode 主要可用于 CPU 仿真。
- **CPU Simulator 支持**：支持在 CPU 上进行功能验证与开发调试。
- **覆盖关键编程要素**：支持 tile shape、tile mask、事件同步、固定功能单元与流水线建模。
- **文档、测试、示例齐全**：提供 ISA 文档、开发文档、测试脚本和性能案例。

## 👥 适用人群

PTO Tile Lib 主要面向以下开发者：

- 直接对接昇腾硬件的框架或编译器后端开发者
- 需要跨平台迁移与复用实现的高性能算子开发者
- 需要显式控制 tile、buffer 与 pipeline 的性能优化工程师

## 🚀 快速开始

### 环境准备

- **CPU 路径**：需要 Python、CMake 和支持 C++20 的编译器，适合跨平台快速验证。
- **NPU 路径**：需要 Linux 环境与 Ascend CANN toolkit，适合在昇腾硬件或模拟器上运行。
- 更详细的环境部署说明请参见：[快速开始指南](docs/getting-started_zh.md)

### 编译与运行

```bash
# CPU Simulator（建议第一步）
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

- [Auto Mode Add 示例](demos/auto_mode/baseline/add/README_zh.md)：适合第一次了解 PTO 指令组织方式
- [GEMM 性能示例](kernels/manual/a2a3/gemm_performance/README_zh.md)：适合理解 tile 级算子优化
- [Flash Attention 示例](kernels/manual/common/flash_atten/README_zh.md)：适合理解复杂算子与性能调优

### 推荐上手路径

1. 从简单示例开始，理解 PTO 指令如何组织 tile 级计算与数据搬运。
2. 在 CPU 仿真中验证功能与正确性，建立对指令语义和结果的直观认知。
3. 将代码移植到昇腾硬件上验证正确性并采集性能数据。参见 [msprof 工具](https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/Profiling/atlasprofiling_16_0010.html)
4. 定位性能瓶颈（CUBE Bound / MTE Bound / Vector Bound），开始优化与调参。参见 [性能优化](docs/coding/opt_zh.md)

本仓库也展示了标准 tile 操作如何通过模板参数映射到不同流水线实现：

- [Tile 编程模型](docs/coding/Tile_zh.md)：理解静态 tile shape、动态 tile mask 与数据组织方式
- [事件与同步](docs/coding/Event_zh.md)：理解 set/wait flag 与流水线同步
- [通用约定](docs/isa/conventions_zh.md)：理解 PTO 编程中的通用规则与约束
- [PTO 指令列表](docs/isa/README_zh.md)：查看 PTO ISA 已定义的标准操作

## 🗂️ 文档导航

### ISA 与编程模型

- [ISA 总览](docs/README_zh.md)：PTO ISA 文档入口与阅读导航
- [PTO 指令列表](docs/isa/README_zh.md)：按指令分类查看 PTO 标准操作
- [Tile 编程模型](docs/coding/Tile_zh.md)：理解 tile 的形状、mask 与编程模型
- [事件与同步](docs/coding/Event_zh.md)：理解事件记录、等待与同步方式
- [通用约定](docs/isa/conventions_zh.md)：查看命名、约束与通用规则

### 开发与优化

- [开发文档索引](docs/coding/README_zh.md)：查看扩展 PTO Tile Lib 的开发文档
- [性能优化](docs/coding/opt_zh.md)：查看性能分析与调优建议
- [文档构建说明](docs/mkdocs/README_zh.md)：查看 MkDocs 文档的本地构建方式

## 📊 示例与性能参考

### GEMM

- 参考实现：`kernels/manual/a2a3/gemm_performance/`
- 详细分析与调参说明：[高性能 GEMM 算子示例](kernels/manual/a2a3/gemm_performance/README_zh.md)

![GEMM 性能参考（Ascend A3，24 核）](docs/figures/performance/gemm_performance_a3.svg)

### Flash Attention

- 算子实现与调优说明：[A2/A3 版本](kernels/manual/common/flash_atten/README_zh.md)，[A5 版本](kernels/manual/a5/flash_atten/README_zh.md)
- A5 构建说明（A5 性能数据待补充）：[Flash Attention Performance Kernel (A5)](kernels/manual/a5/flash_atten/README_zh.md)
- S0：query 序列长度（Q/O 的行数）
- S1：key/value 序列长度（K/V 的行数）

Ascend 910B2 多核对比，基线为 `torch_npu`：

| 序列长度 | PTO 时间 (us) | torch_npu 时间 (us) | PTO TFLOPS | torch_npu TFLOPS | PTO 加速比 |
| --- | --- | --- | --- | --- | --- |
| 1024 | 20.960 | 58.461 | 25.61 | 9.18 | 2.79x |
| 2048 | 32.461 | 70.801 | 66.16 | 30.33 | 2.18x |
| 4096 | 88.902 | 118.302 | 96.62 | 72.61 | 1.33x |
| 8192 | 292.626 | 353.147 | 117.42 | 97.30 | 1.21x |
| 16384 | 909.058 | 1118.462 | 151.19 | 122.88 | 1.23x |
| 32768 | 3262.645 | 3646.173 | 168.50 | 150.78 | 1.12x |

![Flash Attention 910B2 PTO vs torch_npu](docs/figures/performance/fa_910b2_pto_vs_torch_npu.png)

### 通信指令带宽

- 参考实现：`kernels/manual/a2a3/tget_bandwidth/`
- 详细分析与构建运行说明：[TGET / TGET_ASYNC 带宽对比示例](kernels/manual/a2a3/tget_bandwidth/README_zh.md)

该示例在 Ascend A2/A3 上测量点对点远程读带宽，对比 `TGET`（同步，经 UB 中转）与 `TGET_ASYNC`（异步，经 DMA 引擎直接传输）的表现。

### GEMM AllReduce 通算融合

- 参考实现：`kernels/manual/a2a3/gemm_ar/`
- 详细分析与调参说明：[高性能 GEMM AllReduce 融合算子示例](kernels/manual/a2a3/gemm_ar/README_zh.md)

该示例展示了如何在同一个算子流水线中融合 PTO 通信原语与计算 kernel，实现 GEMM 与 AllReduce 的重叠执行。

## 🖥️ 平台支持

- Ascend A2（Ascend 910B）
- Ascend A3（Ascend 910C）
- Ascend A5（Ascend 950）
- CPU（x86_64 / AArch64）

更多细节请参考 [include/README_zh.md](include/README_zh.md)。

## 🛣️ 路线图

未来计划发布的特性：

| 功能 | 描述 | 范围 | 进度 / 完成时间 |
| --- | --- | --- | --- |
| **PTO Auto Mode** | BiSheng 编译器支持：自动分配 tile buffer 并插入同步。 | 编译器 / 工具链 | 持续演进 |
| **PTO Tile Fusion** | BiSheng 编译器支持：自动融合 tile 操作。 | 编译器 / 工具链 | 持续演进 |
| **PTO-AS** | PTO ISA 的字节码（Byte Code）支持。 | 编译器 / 工具链 | 持续演进 |
| **卷积扩展** | PTO ISA 对卷积 kernel 的支持。 | ISA 扩展 | 持续演进 |
| **集合通信扩展** | 新增支持 Ccu 及 Roce 异步通信指令，新增 TPREFECTH（AIV 直驱）通信指令。 | 通信 ISA 扩展 | 2026 Q2 |
| **系统调度扩展** | PTO ISA 对 SPMD/MPMD 编程的调度支持。 | ISA 扩展 | 规划中 |
| **微指令** | 支持用户通过微指令表达高性能算子，同时提供基础的高性能微指令库。 | ISA 扩展 / 算子开发 | 2026 Q2 |
| **基础指令** | 进一步优化 A5 指令性能，新增 Pooling 相关基础指令，并增强卷积、量化、Fixpipe 类指令能力。 | ISA 扩展 | 2026 Q2 |
| **CostModel** | 支持 A5 指令的 CostModel 性能仿真。 | 工具链 / 性能建模 | 2026 Q2 |
| **CPU-SIM** | CPU-SIM 随指令增强同步构建。 | CPU 仿真 | 2026 Q2 |

## 🗃️ 目录结构

关键目录如下：

```text
├── include/                     # PTO 对外头文件与接口
│   └── pto/                     # 公共类型、ISA 接口、CPU/NPU 实现
├── kernels/                     # kernel 与算子实现
│   ├── manual/                  # 手工优化实现与性能示例
│   └── custom/                  # 自定义算子示例
├── docs/                        # ISA、编程模型、快速开始与文档站点源文件
│   ├── isa/                     # 指令参考与分类索引
│   ├── coding/                  # 开发与性能优化文档
│   ├── assembly/                # PTO-AS 汇编语法与规范
│   └── mkdocs/                  # MkDocs 文档构建配置与源文件
├── demos/                       # Auto Mode、baseline 与 torch_jit 示例
├── tests/                       # CPU / NPU 测试、脚本与测试入口
│   ├── cpu/                     # CPU 仿真测试
│   ├── npu/                     # 按 SoC 拆分的 NPU 测试
│   └── script/                  # 测试构建与运行脚本
├── scripts/                     # 构建、安装与发布脚本
├── cmake/                       # CMake 公共配置与打包逻辑
├── build.sh                     # 一键构建与运行入口脚本
└── CMakeLists.txt               # 顶层 CMake 配置
```

## ℹ️ 相关信息

- [贡献指南](CONTRIBUTING_zh.md)：参与项目开发与提交流程
- [安全与漏洞披露](SECURITY_zh.md)：安全问题反馈流程
- [版本说明](ReleaseNote_zh.md)：版本更新与发布记录
- [许可证](LICENSE)：CANN Open Software License Agreement Version 2.0
- [PyPTO](https://gitcode.com/cann/pypto/)：PTO 生态中的上层编程框架
- [PTOAS](https://github.com/PTO-ISA/PTOAS/)：面向 PTO 工作流的汇编器与编译后端
- [pto-dsl](https://github.com/PTO-ISA/pto-dsl/)：面向 PTO 的 Python 前端与 JIT 工作流探索

## 📬 联系我们

- **问题反馈**：通过仓库 Issues 提交问题
- **功能建议**：通过仓库 Issues 或讨论区反馈需求
- **贡献代码**：通过 Pull Request 参与项目贡献
