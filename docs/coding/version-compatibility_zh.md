# 版本兼容性说明

本文档基于当前 PTO Tile Lib 仓库中的实际内容，说明版本与平台兼容性的基本原则。

本文档有意避免写入仓库中未正式定义的信息，例如未经声明的版本生命周期、无法核实的历史废弃 API，或仓库中未给出依据的框架兼容矩阵。

## 1. 兼容性范围

PTO Tile Lib 当前面向以下执行环境：

- **Ascend A2**
- **Ascend A3**
- **Ascend A5**
- **CPU 仿真**（`x86_64` / `AArch64`）

仓库提供了：

- `include/pto/` 下统一的 C++ intrinsic 接口
- 面向 CPU 仿真和受支持 NPU 目标的后端实现
- `docs/isa/` 下按指令组织的 ISA 文档

项目级的平台说明可参考 `README.md`；按指令划分的后端实现覆盖情况可参考 `include/README.md`。

## 2. 在 PTO Tile Lib 中，“兼容性”具体指什么

在 PTO Tile Lib 中，兼容性通常需要从以下三个维度理解。

### 2.1 API 兼容性

公共编程接口由 C++ intrinsic 头文件定义，主要包括：

- `include/pto/common/pto_instr.hpp`
- `include/pto/` 下的相关公共头文件

各条指令的语义与使用约束由以下文档说明：

- `docs/isa/*.md`
- `docs/isa/comm/*.md`

因此，当需要判断代码写法是否兼容时，应以上述头文件和 ISA 文档为准。

### 2.2 后端兼容性

并不是每条指令都会在每个后端上实现。

例如，某条 PTO 指令可能出现以下情况：

- 在 CPU 仿真后端可用
- 在 A2/A3 上可用，但在 A5 上尚未实现
- 已经进入 ISA 文档，但在部分后端中仍标记为 `TODO`

因此，后端兼容性应按指令逐项确认，而不能简单地假设“某个平台完全支持全部 ISA”。

当前最权威的后端覆盖情况表位于 `include/README.md`。

### 2.3 程序可移植性

一段在 PTO 源码层面合法的 kernel，在迁移到不同后端时，仍可能需要针对以下方面进行检查：

- 指令是否已实现
- Tile 布局约束
- valid region 约束
- 性能调优参数
- 手动放置与同步策略

在实际开发中，更稳妥的流程通常是：先在 CPU 仿真上验证逻辑，再在目标 Ascend 平台上验证功能与性能。

## 3. 当前支持的平台

当前仓库文档明确覆盖以下主要目标：

| 目标 | 说明 |
| --- | --- |
| **Ascend A2** | NPU 后端目标 |
| **Ascend A3** | NPU 后端目标 |
| **Ascend A5** | NPU 后端目标 |
| **CPU 仿真** | 用于功能开发与调试 |

平台命名和项目概览可参考 `README.md`。

如果需要查看 CPU / Costmodel / A2 / A3 / A5 / Kirin 的逐指令支持状态，请参考 `include/README.md`。

## 4. 构建与环境相关说明

根据当前仓库文档与构建脚本，可以确认以下实践性要求：

- **需要 C++20 或更高版本**。
- **CPU 仿真**是推荐的第一步功能验证路径。
- **NPU 执行**需要 Ascend CANN 环境。
- 某些 CPU 仿真场景对编译器还有额外要求；例如仓库说明中明确提到：**CPU 仿真中的 bfloat16 支持需要 GCC >= 14**。

本仓库中常见的入口命令例如：

```bash
# CPU 仿真
python3 tests/run_cpu.py --clean --verbose

# NPU / simulator 侧 ST 执行
python3 tests/script/run_st.py -r sim -v a3 -t tadd -g TADDTest.case_float_64x64_64x64
```

具体命令应以仓库内现有文档和测试脚本为准。

## 5. API 与指令兼容性的判断方法

### 5.1 以 intrinsic 声明和 ISA 文档为准

判断某个 API 用法是否合法时，应优先查看：

- `include/pto/common/pto_instr.hpp` 中的 intrinsic 声明
- `docs/isa/` 下对应的指令文档

例如：

- `TASSIGN` 用于手动地址绑定
- `TLOAD` / `TSTORE` 用于 GM 与 Tile 之间的数据移动
- `TSYNC` 与事件相关接口用于显式顺序控制

### 5.2 Event 兼容性与后端相关

PTO Tile Lib 支持显式事件模型，但其具体行为与后端有关：

- 在 device build 下，使用类型化的 `Event<SrcOp, DstOp>` 对象表达依赖
- 在 CPU 仿真下，同步行为会被简化，部分事件路径表现为 no-op

Event 的详细模型说明请参考 [Event 编程模型](Event_zh.md)。

### 5.3 Auto mode 与 Manual mode

兼容性判断还应结合编程模式理解：

- **PTO-Auto** 强调直接描述数据流
- **PTO-Manual** 强调显式 Tile 放置和显式顺序控制

例如，`TASSIGN(tile, addr)` 在某些 auto mode 配置下可能是 no-op；而在 manual mode 中，它属于手动放置流程的一部分。

可参考：

- [PTO ISA 快速上手](tutorial_zh.md)
- [Tile 编程模型](Tile_zh.md)
- [TASSIGN 指令](../isa/TASSIGN.md)

## 6. 推荐的兼容性检查流程

在开发或迁移 PTO kernel 时，建议采用以下流程：

1. **基于公共 PTO intrinsic 接口编写代码**。
2. **在 `docs/isa/` 中核对指令语义和约束**。
3. **优先在 CPU 仿真上验证功能正确性**。
4. **在 `include/README.md` 中确认目标后端是否支持相关指令**。
5. **在目标 Ascend 平台上运行并进行调优**。

相比于假设“所有已文档化的指令都在所有平台上统一实现”，这种方式更符合当前仓库的实际情况。

## 7. 本文档不做的承诺

本文档**不**定义以下内容：

- PTO Tile Lib 的正式语义化版本策略
- LTS / 支持周期承诺
- 针对未公开历史版本的迁移兼容保证
- 仓库中未明确给出依据的外部框架兼容矩阵

如果未来仓库引入正式版本策略或发布政策，建议在 release note 或专门的策略文档中单独说明。
