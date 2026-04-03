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
- `gpu/`: NVIDIA GPU backend tests
  - `gpu/st/`: standalone CUDA smoke / correctness tests for the GPU backend
- `npu/`: NPU-side ST tests split by SoC
  - `npu/a2a3/src/st/`: A2/A3 compute ST
  - `npu/a2a3/comm/st/`: A2/A3 communication ST
  - `npu/a5/src/st/`: A5 compute ST
  - `npu/a5/comm/st/`: A5 communication ST
- `common/`: Shared test resources (if present)
- `run_comm_test.sh`: One-click script for communication ST (see below)

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

### Quick Start

```bash
# Run all tests with 8 NPUs (default A2/A3)
./run_comm_test.sh

# A5 SoC, 2 NPUs
./run_comm_test.sh -v a5 -n 2

# Run only the tput testcase
./run_comm_test.sh -t tput

# Enable debug logging
./run_comm_test.sh -d -t tput
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-n` | Number of available NPUs: 2, 4, or 8 | 8 |
| `-v` | SoC version: `a3` (Ascend910B) or `a5` (Ascend910_9599) | a3 |
| `-t` | Run specific testcase(s) (repeatable), e.g. `tput`, `treduce` | all |
| `-d` | Enable debug mode with verbose init/sync logging | off |

### How It Works

The script automatically runs each testcase at each applicable rank count (2 / 4 / 8, up to `-n`), using GTest filters to select only the tests matching the current rank count. For example, with `-n 4` it first runs default tests at 2 ranks, then tests with the `4Ranks` suffix at 4 ranks, skipping 8-rank tests.

## Suggested Reading

- Getting started (recommended: CPU first, then NPU): [docs/getting-started.md](../docs/getting-started.md)
