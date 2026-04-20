# Custom Operators

本目录包含 **PTO 自定义算子开发示例**，展示如何从零开始实现自定义算子。

## 按任务选择

| 你的目标 | 从这里开始 |
|----------|----------|
| 第一次学习 PTO | [快速入门](../../docs/getting-started_zh.md) |
| 编写第一个算子 | [上手教程](../../docs/coding/tutorial_zh.md) |
| Add 算子示例 | [demos/baseline/add/README_zh.md](../../demos/baseline/add/README_zh.md) |
| 算子融合 | [fused_add_relu_mul/README_zh.md](fused_add_relu_mul/README_zh.md) |

## 示例列表

| 算子 | 说明 | 关键技术 |
|------|------|----------|
| [fused_add_relu_mul](./fused_add_relu_mul/README_zh.md) | 融合 Add + ReLU + Mul 为单个 kernel | 算子融合、Tile 级流水、2-3x 性能提升 |

## 如何运行

每个子目录都是一个独立示例，包含各自的构建/运行说明。请从这里开始：

- [fused_add_relu_mul/README_zh.md](fused_add_relu_mul/README_zh.md)

## 开发自定义算子步骤

参考 `fused_add_relu_mul/` 示例，按以下步骤开发：

1. **创建目录**：`mkdir -p kernels/custom/my_operator`
2. **实现 kernel**：`my_operator_kernel.cpp`
3. **编写测试**：`main.cpp`
4. **配置构建**：`CMakeLists.txt`
5. **运行验证**：`./run.sh --sim`

## 详细开发指南

| 文档 | 内容 |
|------|------|
| [算子融合技术](../../docs/coding/operator-fusion_zh.md) | 融合多个算子的技术 |
| [性能优化指南](../../docs/coding/opt_zh.md) | 性能瓶颈分析与调优 |

## 相关文档

| 文档 | 内容 |
|------|------|
| [kernels/README_zh.md](../README_zh.md) | kernels 总入口 |
| [demos/README_zh.md](../../demos/README_zh.md) | 端到端示例 |
| [docs/isa/README_zh.md](../../docs/isa/README_zh.md) | ISA 指令参考 |
