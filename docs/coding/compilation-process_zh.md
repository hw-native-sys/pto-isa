# 编译流程说明

本文档从源码组织、公共 intrinsics、backend 选择和仓库构建入口几个角度，说明 PTO Tile Lib 的构建与编译流程。

本文档重点描述开发者可见的工作流，不将未公开定义的编译器内部阶段扩展为规范接口说明。

## 1. 概述

PTO kernel 以 C++ 形式编写，并通过 `TLOAD`、`TADD`、`TMATMUL`、`TSYNC`、`TSTORE` 等 PTO intrinsic 表达计算与数据移动。

常用的公共入口头文件是：

```cpp
#include <pto/pto-inst.hpp>
```

intrinsic 层主要由 [PTO 公共头文件](../../include/pto/README.md) 下的头文件提供，其中最核心的是 `../../include/pto/common/pto_instr.hpp`。

## 2. 构建与编译特征

PTO Tile Lib 采用 **C++ intrinsic 接口**。

从公共 API 角度看，该库主要采用 **header-based / template-based** 的使用方式。
同一份 PTO 源码可以在不同 build 配置下对接不同 backend。
CPU 仿真是推荐的首选功能验证路径，NPU 执行则依赖 Ascend CANN 环境。
代码库要求使用 **C++20 或更高版本**。

项目级构建说明可参考 [项目概览](../../README.md) 和 [快速开始](../getting-started.md)。

## 3. 构建流程

从开发者视角看，构建流程可以概括为：

```text
PTO C++ 源码
  -> C++ 预处理 / 编译
  -> PTO intrinsic 头文件选择对应 backend 实现
  -> 构建系统编译测试、kernel 或 demo
  -> 生成二进制或测试产物
```

该描述采用开发者视角，用于概括源代码到构建产物之间的主要关系。

当前文档不将某个完整的专有编译器流水线表述为公开契约，例如“frontend -> PTO intrinsic expansion -> middle-end IR -> backend lowering”这样的固定内部阶段顺序。相关过程可能存在于工具链中，但不作为本文档中的规范接口说明。

## 4. 公共 intrinsic 层与 backend 选择

公共 intrinsic 入口位于 `../../include/pto/common/pto_instr.hpp`。

该头文件暴露了以下一类接口：

- `TASSIGN`
- `TSYNC`
- `TLOAD`
- `TSTORE`
- `TADD`、`TMUL`、`TEXP` 等向量类指令
- `TMATMUL` 等矩阵类指令

同时，这个头文件也会根据构建条件包含不同的 backend 实现头文件。

因此，从开发者角度理解编译过程时，更准确的方式是：

1. 使用 PTO intrinsics 编写 C++ 代码
2. 按照仓库的构建配置进行编译
3. 由所选 backend 提供具体实现路径

## 5. 本仓库中实际使用的构建工具

当前仓库可以明确依赖以下工具：

- **CMake**
- **Python**（用于脚本和测试）
- **支持 C++20 的编译器**

本仓库中常见的命令例如：

```bash
# CPU 仿真
python3 tests/run_cpu.py --clean --verbose

# 在 CPU 仿真上运行 demo
python3 tests/run_cpu.py --demo gemm --verbose

# 在 simulator backend 上运行 ST
python3 tests/script/run_st.py -r sim -v a3 -t tadd -g TADDTest.case_float_64x64_64x64
```

在本仓库内进行构建时，建议优先采用已有脚本和文档中的命令，而不是自行假设一套独立的构建流程。

## 6. CPU 仿真路径与 NPU 路径

### 6.1 CPU 仿真路径

CPU 仿真路径主要用于功能开发和正确性验证。

在该路径下：

- PTO intrinsic 仍以 C++ 源码形式直接出现
- backend 行为由 CPU 仿真实现建模
- 某些仅设备端有效的同步细节会被简化，或者表现为 no-op

相关文档：

- [CPU 仿真](cpu_sim.md)
- [快速开始教程](tutorial.md)
- [事件与同步](Event.md)

### 6.2 NPU 路径

NPU 路径面向 Ascend 硬件或 simulator 侧执行。

在该路径下：

- 会使用面向 NPU 的 backend 实现
- 设备端约束会更直接地影响代码合法性
- 指令是否可用需要结合 backend 支持表逐项确认

相关参考：

- [后端实现状态](../../include/README.md)
- [PTO ISA 参考](../isa/README.md)

## 7. 编译相关检查项

当 PTO kernel 编译失败或行为不符合预期时，最可靠的检查路径是：

1. **头文件级 API 用法**
   - intrinsic 的使用方式是否符合 `../../include/pto/common/pto_instr.hpp` 中的声明？

2. **ISA 约束**
   - `docs/isa/` 中对应指令是否允许当前 tile 类型、布局和操作数组合？

3. **Tile 与 GlobalTensor 定义**
   - tile shape、valid region、layout 是否合法？
   - `GlobalTensor` 的 shape / stride 声明是否正确？

4. **backend 支持情况**
   - 目标指令在所选 backend 上是否已实现？可参考 [后端实现状态](../../include/README.md)。

5. **构建环境**
   - 编译器、Python 环境、CANN 环境是否满足要求？

## 8. 关于构建示例的说明

一些在通用 AI 工具或网络示例中常见的片段，例如通用的 `find_package(PTO REQUIRED)`、假设存在的 `PTO::pto` 链接目标等，并**不能**直接视为本仓库已经正式定义的标准集成方式。

补充文档或扩展 PTO Tile Lib 时，应以仓库内构建脚本、顶层 `CMakeLists.txt` 以及现有测试和 demo 的构建方式为主要参考。

## 9. 说明

PTO Tile Lib 的编译流程可概括为：

- PTO 代码以 C++ 和公共 intrinsics 形式编写；
- 构建系统根据配置选择对应的 backend 实现；
- CPU 仿真是推荐的首选验证路径；
- backend 支持情况与指令合法性在开发过程中显式检查。

文档重点说明公共编程接口和使用模型；除非在专门的工具链文档中另行定义，编译器内部阶段仍属于实现细节。
