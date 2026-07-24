# FA PTO PyTorch 移植示例

## 概述

本示例演示了如何使用 PTO 实现 Flash Attention 内核，并通过 `torch_npu` 将其作为自定义 PyTorch 算子对外暴露。示例展示了在 Ascend AI 处理器上实现高性能自定义内核集成，并具备自动 tile 适配能力。

## 支持的 AI 处理器

- A2/A3/A5

## 1. 环境准备

创建虚拟环境并安装依赖：

```bash
python -m venv virEnv
source virEnv/bin/activate
python3 -m pip install -r requirements.txt
```

确保已配置 Ascend Toolkit 和 PTO 库：

```bash
export ASCEND_HOME_PATH=[YOUR_ASCEND_PATH/SYSTEM_ASCEND_PATH]
source [YOUR_ASCEND_PATH/SYSTEM_ASCEND_PATH]/latest/bin/setenv.bash
export PTO_LIB_PATH=[YOUR_PATH]/pto-isa
```

## 2. 构建 Wheel 包

项目支持通过 `SOC_VERSION` 环境变量为不同的 SOC 版本进行构建。构建系统会根据目标 SOC 自动配置正确的优化宏（例如 `PTO_NPU_ARCH_A2A3` 与 `PTO_NPU_ARCH_A5`）。

**默认构建（A2 / A3）：**
```bash
python3 setup.py bdist_wheel
```

**为特定 SOC 构建（例如 A5）：**
```bash
# A5 示例
SOC_VERSION=ascend910_9599 python3 setup.py bdist_wheel
```

## 3. 安装 Wheel 包

```bash
pip install dist/*.whl --force-reinstall
```

## 4. 运行测试

运行验证脚本，将内核结果与黄金参考值进行比较。测试涵盖多种序列长度（1k 至 32k）并验证动态 tile 逻辑。

```bash
cd test
python3 test.py
```

## 特性

- **动态 Tiling**：根据输入序列长度自动选择最佳 tile 大小（128 或 256）。
- **跨架构支持**：通过构建时配置，统一的代码库同时支持 A2/A3 和 A5 架构。
