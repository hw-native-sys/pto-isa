<p align="center">
  <img src="figures/pto_logo.svg" alt="PTO Tile Lib" width="200" />
</p>

# PTO ISA 文档导航

这里是 PTO Tile Lib 的文档入口页，用于帮助读者按主题快速定位文档，而不是逐个目录查找。

PTO 相关文档主要覆盖以下几类内容：

- ISA 基础概念与整体阅读路径
- 指令索引与逐条指令参考
- PTO 汇编语法与 PTO-AS 规范
- Tile 编程模型、事件同步与性能优化
- 快速开始、测试运行与文档构建说明

## 建议阅读路径

如果您第一次接触 PTO Tile Lib，建议按以下顺序阅读：

1. [快速开始指南](getting-started_zh.md)：先完成环境准备并运行 CPU Simulator
2. [ISA 总览](PTOISA_zh.md)：建立对 PTO ISA 的整体认识
3. [PTO 指令列表](isa/README_zh.md)：按类别浏览已定义的标准操作
4. [Tile 编程模型](coding/Tile_zh.md)：理解 tile shape、tile mask 与数据组织方式
5. [事件与同步](coding/Event_zh.md)：理解 set/wait flag 与流水线同步
6. [性能优化](coding/opt_zh.md)：理解常见瓶颈与调优方向

## 文档分类

### 1. ISA 与指令参考

- [虚拟 ISA 手册入口](PTO-Virtual-ISA-Manual_zh.md)：PTO ISA 手册总入口
- [ISA 总览](PTOISA_zh.md)：介绍 PTO ISA 的背景、目标与整体结构
- [PTO 指令列表](isa/README_zh.md)：按类别组织的 PTO 标准操作索引
- [通用约定](isa/conventions_zh.md)：命名、约束、使用规范等通用规则

### 2. PTO 汇编与表示形式

- [PTO 汇编索引](assembly/README_zh.md)：PTO-AS 文档入口
- [PTO 汇编语法（PTO-AS）](assembly/PTO-AS_zh.md)：PTO 汇编语法与规范说明

### 3. 编程模型与开发文档

- [开发文档索引](coding/README_zh.md)：扩展 PTO Tile Lib 的开发文档入口
- [Tile 编程模型](coding/Tile_zh.md)：介绍 tile shape、tile mask 与数据布局
- [事件与同步](coding/Event_zh.md)：介绍事件记录、等待与同步机制
- [性能优化](coding/opt_zh.md)：介绍性能分析与调优建议
- [CostModel 后端说明（打桩 stub / 拟合 fit）](costmodel-backends_zh.md)：后端拆分、拟合路径与运行方式

### 4. 入门、测试与文档构建

- [快速开始指南](getting-started_zh.md)：环境准备、CPU / NPU 运行说明
- [测试说明](../tests/README_zh.md)：测试入口、测试脚本与常用命令
- [文档构建说明](mkdocs/README_zh.md)：MkDocs 文档本地构建说明

### 5. 其他相关文档

- [Machine 文档](machine/README_zh.md)：抽象机器模型与相关说明

## 目录结构

关键目录如下：

```text
├── isa/                        # PTO 指令参考与分类索引
├── assembly/                   # PTO 汇编语法与 PTO-AS 规范
├── coding/                     # 编程模型、开发与性能优化文档
├── auto_mode/                  # Auto Mode 相关文档
├── machine/                    # 抽象机器模型相关文档
├── mkdocs/                     # 文档站点构建配置与脚本
├── figures/                    # 文档中使用的图片与图示资源
├── README*                     # 文档入口页
├── PTOISA*                     # ISA 总览文档
└── getting-started*            # 快速开始指南
```

## 相关入口

- [根目录 README_zh](../README_zh.md)：项目总览、快速开始与仓库入口
- [kernels 目录说明](../kernels/README_zh.md)：kernel 与算子实现入口
- [include 目录说明](../include/README_zh.md)：头文件与接口说明
- [tests 目录说明](../tests/README_zh.md)：测试与运行入口
