# Repo Context for AI Agents (PTO Tile Lib)

This document is a fast, practical orientation for agents working in this repo: what it is, where the key entrypoints live, and the shortest paths to build/run in **CPU**, **NPU simulator (`sim`)**, and **on-board NPU (`npu`)** modes.

## What This Repo Is

- **PTO Tile Library**: C++ headers + implementations for the PTO (Parallel Tile Operation) virtual ISA defined by Ascend CANN.
- Supports multiple backends:
  - **CPU simulation** (cross-platform, no Ascend driver/CANN required).
  - **Ascend NPU** backends split by SoC generation:
    - **A2/A3 family**: `include/pto/npu/a2a3/` (selected via `-v a3` in test scripts).
    - **A5**: `include/pto/npu/a5/`.
- Primary include for upper-layer code: `#include <pto/pto-inst.hpp>` (unified entry header).

## Repo Map (Where To Look First)

- Project overview + common commands: `README.md`
- Detailed setup (CPU first, then NPU): `docs/getting-started.md`
- ISA docs and navigation:
  - `docs/README.md` (ISA guide entry)
  - `docs/isa/` (per-instruction reference)
- Public API headers and backend status table: `include/README.md`
- Core public headers / backend split: `include/pto/README.md`
- Build/package entrypoint: `build.sh`, top-level `CMakeLists.txt`, `cmake/`
- Tests entrypoints:
  - CPU simulator tests: `tests/run_cpu.py`, `tests/run_cpu_tests.sh`
  - NPU ST build/run: `tests/script/run_st.py`, `tests/run_st.sh`
  - Test layout overview: `tests/README.md`
- Demos: `demos/` (CPU demos used by `tests/run_cpu.py --demo ...`)

## Run: CPU Simulator (Recommended First)

CPU simulation is meant to be the “works everywhere” correctness path.

From repo root:

```bash
python3 tests/run_cpu.py --clean --verbose
```

Useful variants:

```bash
python3 tests/run_cpu.py --testcase tadd
python3 tests/run_cpu.py --testcase tadd --gtest_filter 'TADDTest.*'
python3 tests/run_cpu.py --demo gemm --verbose
python3 tests/run_cpu.py --demo flash_attn --verbose
```

Notes:

- CPU ST uses CMake and GoogleTest; it may download GTest if not installed system-wide.
- Compiler requirement is at least **C++20** (see `tests/cpu/st/CMakeLists.txt`).
- For enabling bfloat16 support in CPU-SIM, GCC>=14 is required

## Run: NPU ST (Ascend) — `sim` and `npu`

NPU ST is built/run via `tests/script/run_st.py`:

```bash
python3 tests/script/run_st.py -r [sim|npu] -v [a3|a5] [-a] -t <testcase> -g <gtest_filter>
```

Key points:

- `-a` compiles the test case in auto mode instead of manual mode.
- `-v a3` selects the **A2/A3** implementation under `include/pto/npu/a2a3/` (the test script maps it to a SoC string like `Ascend910B1`).
- `-r sim` uses the Ascend simulator libraries under `$ASCEND_HOME_PATH/tools/simulator/<SOC>/lib` and `runtime/lib64/stub`.
- `-r npu` runs on real hardware.

Examples (single case):

```bash
python3 tests/script/run_st.py -r sim -v a3 -t tadd -g TADDTest.case_float_64x64_64x64
python3 tests/script/run_st.py -r npu -v a3 -t tadd -g TADDTest.case_float_64x64_64x64
```

Recommended suites (wrapper script):

```bash
chmod +x ./tests/run_st.sh
./tests/run_st.sh a3 sim simple
./tests/run_st.sh a3 npu simple
```

## Environment: Ascend CANN / Toolkit

NPU ST requires a working Ascend environment. Typical setup (choose the correct install path):

```bash
source /usr/local/Ascend/cann/bin/setenv.bash
# or
source $HOME/Ascend/ascend-toolkit/latest/bin/setenv.bash
```

`tests/script/run_st.py` expects `ASCEND_HOME_PATH` to be set (usually done by `setenv.bash`).

## Common Pitfalls (And How This Repo Handles Them)

- **GTest ABI mismatch on Linux**: some systems have `libgtest*.a` built with `_GLIBCXX_USE_CXX11_ABI=0`.
  - CPU and NPU ST CMake projects support `PTO_GLIBCXX_USE_CXX11_ABI=auto|0|1` and auto-detect when possible.
- **`sim` open-files limit**: simulator runs may require a higher `ulimit -n` (see `docs/getting-started.md` and `build.sh`).

