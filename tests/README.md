# tests/

Tests and examples for PTO Tile Lib, covering both CPU simulation and NPU (including `sim` and on-board `npu` modes).

## Test Entry Points

Common test entry points:

- Full CPU Simulator run: `python3 tests/run_cpu.py --clean --verbose`
- GEMM demo: `python3 tests/run_cpu.py --demo gemm --verbose`
- Flash Attention demo: `python3 tests/run_cpu.py --demo flash_attn --verbose`
- Single ST testcase: `python3 tests/script/run_st.py -r [sim|npu] -v [a3|a5] -t [TEST_CASE] -g [GTEST_FILTER_CASE]`
- One-click scripts: `./tests/run_st.sh`, `./tests/run_cpu_tests.sh`

## Layout

- `script/`: Recommended entry scripts
  - `run_st.py`: Build and run NPU ST
  - `build_st.py`: Build NPU ST only
  - `all_cpu_tests.py`: Build and run CPU ST suites in batch
  - `cpu_bfloat16.py`: CPU bfloat16 test script
  - `README.md`: Script usage
- `cpu/`: CPU-side ST tests (gtest + CMake)
  - `st/`: CPU compute ST projects and testcase data generation scripts
  - `comm/st/`: CPU communication ST
- `npu/`: NPU-side ST tests split by SoC
  - `a2a3/src/st/`: A2/A3 compute ST
  - `a2a3/src/common/`: A2/A3 shared test resources
  - `a2a3/comm/st/`: A2/A3 communication ST
  - `a5/src/st/`: A5 compute ST
  - `a5/src/common/`: A5 shared test resources
  - `a5/comm/st/`: A5 communication ST
  - `kirin9030/src/st/`: Kirin9030 compute ST
  - `kirin9030/src/common/`: Kirin9030 shared test resources
- `costmodel/`: Cost model tests
  - `st/`: Cost model ST (operator cost measurement)
  - `st_fit/`: Cost model fitting tests
- `common/`: Shared test resources
- `run_cpu.py`: CPU simulator full run entry point
- `run_cpu_tests.sh`: CPU ST one-click script
- `run_st.sh`: NPU ST one-click script
- `run_comm_test.sh`: Communication ST one-click script (see below)
- `run_costmodel.py`: Cost model run script
- `run_costmodel_tests.sh`: Cost model one-click script
- `run_pipeline.sh`: Pipeline test script
- `validate_op_coverage.py`: Operator coverage validation script
- `validate_testcase_names.py`: Testcase name validation script

## Communication Tests (Comm ST)

Communication tests verify multi-device PTO communication primitives (Put / Get / Broadcast / Gather / Scatter / Reduce / Notify / Wait / Test), built on MPI + HCCL.

### Prerequisites: MPI Installation

Communication tests require an MPI environment (MPICH or OpenMPI). Two components are needed at runtime:

1. **`mpirun`**: launches multi-process execution
2. **`libmpi.so`**: loaded at runtime via `dlopen`

#### Install MPICH (Recommended)

```bash
# Ubuntu / Debian
sudo apt install mpich libmpich-dev

# CentOS / RHEL / EulerOS
sudo yum install mpich mpich-devel
# May need to load a module or add to PATH manually:
export PATH=/usr/lib64/mpich/bin:$PATH
```


#### Build MPICH from Source (No Root)

```bash
wget https://www.mpich.org/static/downloads/4.2.3/mpich-4.2.3.tar.gz
tar xzf mpich-4.2.3.tar.gz && cd mpich-4.2.3
./configure --prefix=$HOME/mpich --disable-fortran
make -j$(nproc) && make install
export MPI_HOME=$HOME/mpich
export PATH=$MPI_HOME/bin:$PATH
```

#### Environment Variables

| Variable | Description |
|----------|-------------|
| `MPI_HOME` | MPI installation root; the script searches `$MPI_HOME/bin/mpirun` |
| `MPI_LIB_PATH` | Direct path to `libmpi.so` (overrides default search) |

If `mpirun` is already on `PATH` and `libmpi.so` is in a standard library path, these variables are not required.

#### Verify Installation

```bash
mpirun --version
mpirun -n 2 echo "MPI OK"
```

### Sync vs Async Instruction Tests

Communication tests are split into **sync instructions** (e.g. `tput`, `tget`) and **async instructions** (e.g. `tput_async`, `tget_async`):

| Type | Test Case Examples | CANN Version Required |
|------|-------------------|----------------------|
| Sync | `tput`, `tget`, `treduce`, `tbroadcast`, etc. | CANN 8.x and above |
| Async | `tput_async`, `tget_async` | **CANN 9.0 and above** |

Async instructions depend on the SDMA opapi interface (e.g. `aclnnShmemSdmaStarsQuery`) introduced in CANN 9.0. They will fail on lower CANN versions due to missing symbols. Therefore, `run_comm_test.sh` **excludes async tests by default** — use the `-a` flag to enable them.

### Quick Start

```bash
# Run all tests with 8 NPUs (default A2/A3, excludes async)
./run_comm_test.sh

# Include async instruction tests (requires CANN 9.0+)
./run_comm_test.sh -a

# Run only the async tput testcase
./run_comm_test.sh -t tput_async

# A5 SoC, 2 NPUs
./run_comm_test.sh -v a5 -n 2

# Run only the tput testcase
./run_comm_test.sh -t tput

# Enable debug logging
./run_comm_test.sh -d -t tput
```

You can also run directly via `run_st.py`, which automatically splits execution by rank count:

```bash
# Auto-split tput_async across 2/4/8 ranks
python3 tests/script/run_st.py -r npu -v a3 -t comm/tput_async

# Limit to 2 ranks
python3 tests/script/run_st.py -r npu -v a3 -t comm/tput_async -n 2
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-n` | Number of available NPUs: 2, 4, or 8 | 8 |
| `-v` | SoC version: `a3` (Ascend910B) or `a5` (Ascend950) | a3 |
| `-t` | Run specific testcase(s) (repeatable), e.g. `tput`, `treduce` | all |
| `-a` | Include async instruction tests (`*_async`), requires CANN 9.0+ | off |
| `-d` | Enable debug mode with verbose init/sync logging | off |

### How It Works

The script automatically runs each testcase at each applicable rank count (2 / 4 / 8, up to `-n`), using GTest filters to select only the tests matching the current rank count. For example, with `-n 4` it first runs default tests at 2 ranks, then tests with the `4Ranks` suffix at 4 ranks, skipping 8-rank tests.

## Suggested Reading

- Getting started (recommended: CPU first, then NPU): [docs/getting-started.md](../docs/getting-started.md)
