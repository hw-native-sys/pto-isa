# tests/

Tests and examples for PTO Tile Lib, covering both CPU simulation and NPU (including `sim` and on-board `npu` modes).

## Layout

- `script/`: Recommended entry scripts
  - `run_st.py`: Build and run NPU ST (`-r sim|npu -v a3|a5 -t <testcase> -g <gtest_filter>`)
  - `build_st.py`: Build NPU ST only
  - `all_cpu_tests.py`: Build and run CPU ST suites in batch
  - `README.md`: Script usage
- `cpu/`: CPU-side ST tests (gtest + CMake)
  - `cpu/st/`: CPU ST projects and testcase data generation scripts
- `npu/`: NPU-side ST tests split by SoC
  - `npu/a2a3/src/st/`: A2/A3 ST
  - `npu/a5/src/st/`: A5 ST
- `common/`: Shared test resources (if present)

## Suggested Reading

- Getting started (recommended: CPU first, then NPU): [docs/getting-started.md](../docs/getting-started.md)
