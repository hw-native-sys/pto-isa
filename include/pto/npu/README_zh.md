# include/pto/npu/

NPU 侧 PTO 指令实现。不同 SoC 代际会对应不同的优化实现与流水线细节。

## 目录结构

- `a2a3/`：Ascend A2/A3 实现（例如 `TAdd.hpp`、`TMatmul.hpp`、`TLoad.hpp`）
- `a5/`：Ascend A5 实现（例如 `TAdd.hpp`、`TMatmul.hpp`、`TLoad.hpp`）
- `linx/`：[Linx 原生 device API 及对应 CPU-SIM](linx/README_zh.md)。

## 选择 SoC 版本

SoC 选择由构建系统与测试脚本控制：

- `tests/script/run_st.py` / `tests/script/build_st.py`：通过 `-v a3|a5` 选择
- `tests/npu/<soc>/src/st/CMakeLists.txt`：按 SoC 构建对应的 ST 目标与依赖

端到端流程建议从 `docs/getting-started.md` 开始。

Linx device 由原生编译器的 `__linx` / `PTO_NPU_ARCH_LINX` 选择；CPU 使用
`__CPU_SIM` 与 `PTO_CPU_SIM_TARGET_LINX`。两种后端保留相同 Linx 指令调用形式。
