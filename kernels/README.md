# Kernels

本目录包含与 PTO Tile Lib 配套的 kernel / operator 实现。

多数子目录都是**自包含的小工程**（kernel + host + 脚本），通常会包含自己的 `README.md`、`CMakeLists.txt` 与 `run.sh`，便于独立发现与运行。

## 从哪里开始

- 手工调优（manual）的 NPU kernels：[manual](manual/README.md)
- 自定义算子脚手架：[custom](custom/README.md)
- 端到端 demo（包含 CPU）：[demos](../demos/README.md)

## 目录结构

- `manual/`：手工调优 kernels（显式管理 buffer / 同步 / 流水线，偏 NPU）
  - `manual/a2a3/`：A2/A3 平台 kernels
    - `manual/a2a3/gemm_performance/`：高性能 GEMM 示例
    - `manual/a2a3/conv2d_forward/`：Conv2D 前向 kernel 示例
    - `manual/a2a3/topk/`：TopK kernel 示例
  - `manual/a5/`：A5 平台 kernels
    - `manual/a5/flash_atten/`：A5 平台 Flash-Attention kernel
    - `manual/a5/matmul_mxfp4_performance/`：MXFP4 矩阵乘法示例
    - `manual/a5/matmul_mxfp8_performance/`：MXFP8 矩阵乘法示例
  - `manual/common/`：跨平台 kernels
    - `manual/common/flash_atten/`：Flash-Attention kernel（A2/A3/A5）
- `custom/`：自定义 kernel / operator 扩展的示例与脚手架

## 备注

- 公共接口在 `include/`；测试在 `tests/`。
- 新增 kernel 工程时，建议配套一个简短的 `README.md` 和一个 `run.sh`，方便统一发现与运行。
