# include/pto/npu/

NPU-side PTO instruction implementations. Different SoC generations have different optimized implementations and pipeline details.

## Layout

- `a2a3/`: Ascend A2/A3 implementations (e.g., `TAdd.hpp`, `TMatmul.hpp`, `TLoad.hpp`)
- `a5/`: Ascend A5 implementations (e.g., `TAdd.hpp`, `TMatmul.hpp`, `TLoad.hpp`)
- `linx/`: [restricted experimental Linx device API and CPU-SIM companion](linx/README.md)

## Selecting the SoC Version

Linx is selected by the native compiler target (`__linx` / `PTO_NPU_ARCH_LINX`). Its CPU companion uses `__CPU_SIM` with `PTO_CPU_SIM_TARGET_LINX`. It retains the Linx API rather than adapting Ascend call signatures.

Ascend SoC selection is controlled by the build system and test scripts:

- `tests/script/run_st.py` / `tests/script/build_st.py`: select via `-v a3|a5`
- `tests/npu/<soc>/src/st/CMakeLists.txt`: builds the corresponding ST targets and dependencies per SoC

For an end-to-end walkthrough, start with `docs/getting-started.md`.
