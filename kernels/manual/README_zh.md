# 手工调优 kernels

本目录包含**手工调优（手写、面向性能）**的 kernel 示例：在支持的昇腾 NPU 上使用显式 buffer 管理、同步与流水线控制，以获得最佳性能。

如果你刚接触 PTO 编程，建议先从 ISA 与教程入手：

- 编程教程：[docs/coding/tutorial_zh.md](../../docs/coding/tutorial_zh.md)
- 优化笔记：[docs/coding/opt_zh.md](../../docs/coding/opt_zh.md)
- PTO ISA 参考：[docs/isa/README_zh.md](../../docs/isa/README_zh.md)

## 按平台选择

| 平台 | 目录 | 典型 Kernel |
|------|------|------------|
| Ascend A2/A3 | `a2a3/` | GEMM、Conv2D、TopK、AllGather-GEMM |
| Ascend A5 | `a5/` | Flash Attention、MXFP4/8 Matmul、AllGather-GEMM |
| 跨平台 | `common/` | Flash Attention（A2/A3/A5 通用） |

## 目录索引

### A2/A3 Kernels（`a2a3/`）

| Kernel | 说明 | 关键技术 |
|--------|------|----------|
| [GEMM Performance](a2a3/gemm_performance/) | 高性能矩阵乘法 | 双缓冲、L0A/L0B/L0C 分块、UB 暂存 |
| [Conv2D Forward](a2a3/conv2d_forward/) | Conv2D 前向（img2col 方式） | img2col + GEMM 融合、分形布局 |
| [TopK](a2a3/topk/) | Top-K 元素选取 | 基于排序的选取、tile 级归约 |
| [AllGather-GEMM](a2a3/allgather_gemm/) | 多卡 GEMM 与 AllGather | 集合通信与 GEMM 融合 |

### A5 Kernels（`a5/`）

| Kernel | 说明 | 关键技术 |
|--------|------|----------|
| [Flash Attention](a5/flash_atten/) | Flash Attention 算法 | 动态分块、在线 softmax、A5 专用优化 |
| [MXFP8 Matmul](a5/matmul_mxfp8_performance/) | MXFP8 精度矩阵乘法 | MXFP8 反量化、FP8 算子、A5 硬件支持 |
| [MXFP4 Matmul](a5/matmul_mxfp4_performance/) | MXFP4 精度矩阵乘法 | MXFP4 反量化、FP4 算子 |
| [AllGather-GEMM](a5/allgather_gemm/) | 多卡 GEMM（A5） | 集合通信与 GEMM 融合 |
| [Engram SIMT](a5/engram_simt/) | Engram SIMT 示例 | A5 上的 SIMT 风格编程 |

### 跨平台 Kernels（`common/`）

| Kernel | 说明 | 平台 |
|--------|------|------|
| [Flash Attention](common/flash_atten/) | Flash Attention 算法 | A2/A3/A5 |

## 如何运行

每个子目录都是独立示例，包含各自的构建/运行说明。参见各目录中的 `README_zh.md`。

## 相关文档

- [kernels/README_zh.md](../README_zh.md) — 父目录入口
- [docs/coding/opt_zh.md](../../docs/coding/opt_zh.md) — 性能瓶颈分析与调优
- [docs/isa/README_zh.md](../../docs/isa/README_zh.md) — PTO ISA 指令参考
