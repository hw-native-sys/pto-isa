# Kernels

本目录包含与 PTO Tile Lib 配套的 kernel / operator 实现。

多数子目录都是**自包含的小工程**（kernel + host + 脚本），通常会包含自己的 `README.md`、`CMakeLists.txt` 与 `run.sh`，便于独立发现与运行。

## 从哪里开始

- 手工调优（manual）的 NPU kernels：[manual/README.md](manual/README_zh.md)
- 自定义算子脚手架：[custom/README.md](custom/README.md)
- 端到端 demo（包含 CPU）：[demos/](../demos/README_zh.md)

## 目录结构

- `manual/`：手工调优 kernels（显式管理 buffer / 同步 / 流水线，偏 NPU）
  - `manual/a2a3/`：A2/A3 平台 kernels
    - `manual/a2a3/gemm_performance/`：高性能 GEMM 示例
    - `manual/a2a3/flash_atten/`：手写 Flash-Attention 示例
- `custom/`：自定义 kernel / operator 扩展的示例与脚手架

## 备注

- 公共接口在 `include/`；测试在 `tests/`。
- 新增 kernel 工程时，建议配套一个简短的 `README.md` 和一个 `run.sh`，方便统一发现与运行。
