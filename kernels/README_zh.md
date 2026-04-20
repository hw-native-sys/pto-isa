# Kernels

本目录包含与 PTO Tile Library 配套的 kernel / 算子实现。

每个子目录都是一个**自包含的小工程**（kernel + host + 脚本），通常包含自己的 `README.md`、`CMakeLists.txt` 与 `run.sh`，便于独立发现与运行。

## 按任务选择

| 你的目标 | 从这里开始 |
|----------|----------|
| 学习 PTO 编程 | [docs/coding/tutorial_zh.md](../docs/coding/tutorial_zh.md) |
| 高性能 GEMM | [manual/a2a3/gemm_performance/README_zh.md](manual/a2a3/gemm_performance/README_zh.md) |
| Flash Attention | [manual/common/flash_atten/README_zh.md](manual/common/flash_atten/README_zh.md) |
| Conv2D forward | [manual/a2a3/conv2d_forward/README_zh.md](manual/a2a3/conv2d_forward/README_zh.md) |
| MXFP8 / MXFP4 Matmul | [manual/a5/matmul_mxfp8_performance/README_zh.md](manual/a5/matmul_mxfp8_performance/README_zh.md) |
| 自定义算子脚手架 | [custom/README_zh.md](custom/README_zh.md) |

## 目录结构

```
kernels/
├── manual/                    # 手工调优（手写、面向性能）的 NPU kernels
│   ├── a2a3/                 # Ascend A2/A3 平台
│   │   ├── gemm_performance/ # 高性能 GEMM — 流水线优化、双缓冲、地址规划
│   │   ├── conv2d_forward/   # Conv2D 前向 kernel — img2col + GEMM
│   │   ├── topk/            # TopK kernel — 排序与选取
│   │   └── allgather_gemm/  # 多卡 GEMM — AllGather 通信与 GEMM 融合
│   │
│   ├── a5/                  # Ascend A5 平台
│   │   ├── flash_atten/     # Flash Attention — A5 专用优化版本
│   │   ├── matmul_mxfp8_performance/  # MXFP8 矩阵乘法
│   │   ├── matmul_mxfp4_performance/  # MXFP4 矩阵乘法
│   │   ├── allgather_gemm/  # 多卡 GEMM（A5）
│   │   └── engram_simt/     # Engram SIMT 示例
│   │
│   └── common/               # 跨平台 kernels（适用于 A2/A3/A5）
│       └── flash_atten/      # Flash Attention — 跨平台通用版本
│
└── custom/                   # 自定义算子脚手架与示例
    └── fused_add_relu_mul/  # 融合算子示例：Add + ReLU + Mul
```

## Manual kernels 的特点

与 `demos/` 中的示例不同，`manual/` 下的 kernels 面向**生产级性能调优**：

- **显式管理** tile buffer 地址分配（`TASSIGN`）
- **显式管理** 流水线同步（`set_flag`/`wait_flag`）
- **双缓冲 / 多缓冲** 重叠数据搬运与计算
- **地址对齐与规划** 最大化 UB / L0 利用率
- 针对特定 SoC（A2/A3 vs A5）的微架构特性调优

## 相关文档

| 文档 | 内容 |
|------|------|
| [kernels/README_zh.md](./README_zh.md) | 中文版入口 |
| [demos/](../demos/README_zh.md) | 端到端示例（包含 CPU 版本） |
| [docs/coding/opt_zh.md](../docs/coding/opt_zh.md) | 性能优化与瓶颈分析 |
| [docs/isa/README_zh.md](../docs/isa/README_zh.md) | PTO ISA 指令参考 |

## 新增 kernel 注意事项

新增 kernel 工程时，建议配套：

1. 一个简短的 `README_zh.md`（中文）和 `README.md`（英文），说明平台依赖与运行方式
2. 一个 `run.sh` 或等效脚本，方便统一发现与运行
3. 在 `CMakeLists.txt` 中使用 `pto_add_kernel(<target_name>)` 模板
