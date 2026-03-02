# tests/

PTO Tile Lib 的测试与示例，覆盖 CPU 仿真与 NPU（`sim` 和板上 `npu` 两种模式）。

## 目录结构

- `script/`：推荐的入口脚本
  - `run_st.py`：构建并运行 NPU ST（`-r sim|npu -v a3|a5 -t <testcase> -g <gtest_filter>`）
  - `build_st.py`：仅构建 NPU ST
  - `all_cpu_tests.py`：批量构建并运行 CPU ST 套件
  - `README.md`：脚本使用说明
- `cpu/`：CPU 侧 ST 测试（gtest + CMake）
  - `cpu/st/`：CPU ST 工程与 testcase 数据生成脚本
- `npu/`：按 SoC 拆分的 NPU 侧 ST 测试
  - `npu/a2a3/src/st/`：A2/A3 ST
  - `npu/a5/src/st/`：A5 ST
- `common/`：共享测试资源（如存在）

## 建议阅读顺序

- 入门指南（建议先 CPU，再 NPU）：[docs/getting-started.md](../docs/getting-started.md)
