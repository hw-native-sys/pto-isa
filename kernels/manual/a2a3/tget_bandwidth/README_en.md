# TGET / TGET_ASYNC Bandwidth Comparison Example

## Overview

This example compares the point-to-point communication bandwidth of **TGET** (synchronous remote read) and **TGET_ASYNC** (asynchronous SDMA remote read), sweeping transfer sizes from 4 KB to 4 MB, and measuring both host-side bandwidth (GB/s) and device-side average execution cycles.

- **TGET** performs remote reads via UB (Unified Buffer) staging: `Remote GM → UB → Local GM`. Bandwidth saturates at approximately 4 GB/s due to UB throughput limits.
- **TGET_ASYNC** performs remote reads via the SDMA engine: `Remote GM → SDMA → Local GM`. By bypassing the UB bottleneck, it reaches approximately 13–14 GB/s at 4 MB.

## Supported AI Processors

- A2/A3

## Directory Layout

```
kernels/manual/a2a3/tget_bandwidth/
├── scripts/
│   └── plot_bw_compare.py           # Generate bandwidth comparison plot
├── CMakeLists.txt                   # Build configuration
├── tget_bandwidth_kernel.cpp        # Kernel implementation (AICORE + host orchestration)
├── tget_bandwidth_kernel.h          # Kernel header
├── main.cpp                         # Host-side entry point (MPI initialization)
├── run.sh                           # Convenience script
├── README.md                     # Chinese version
└── README.md                        # This file
```

## Operator Description

### Data Flow

**TGET (synchronous)**:
```
Peer NPU GM ──TGET──▶ Local UB ──TSTORE──▶ Local GM
```

**TGET_ASYNC (asynchronous)**:
```
Peer NPU GM ──SDMA──▶ Local GM   (direct transfer, no UB staging)
```

The current SDMA asynchronous completion protocol uses a kernel-local post ID. Each Post appends an 8-byte flag
SQE after the data SQEs. The session tracks the cumulative number of queues used and appends each Post's Flag to all
of them, so waiting the latest Event also covers every earlier Post in that session. Wait/Test only polls the
corresponding post ID and no longer submits an SQE during Wait. Each channel group has an independent 64-slot
Payload arena (8 bytes per slot, 512 bytes total). Post Done Records reuse the group's `recv_workspace` with a 64-byte
stride. The host-side `SdmaWorkspaceManager` manages the STARS
context and Payload storage for all groups, so kernel callers do not need to allocate a separate control region.

### Test Procedure

1. Each rank prepares send data in HCCL shared memory (`PrepareSendBufferKernel`)
2. The root rank runs TGET and TGET_ASYNC for each transfer size
3. Host-side timing measures bandwidth; device-side `SYS_CNT` measures cycles
4. Received data is verified for correctness

Both timed paths measure only the transfer from the peer's symmetric `sendShmem` to the root's symmetric
`recvShmem`. The received buffer is copied to the ordinary verification GM buffer once after timing, so CopyOut is
not included in either the host-side bandwidth or device-cycle result.

### Specification

| Item           | Value |
| -------------- | ----- |
| Data type      | `float` |
| NPU count      | 2 (point-to-point) |
| Transfer sizes | 4 KB, 16 KB, 64 KB, 256 KB, 1 MB, 4 MB |
| Metrics        | host bandwidth (GB/s), device average cycles |

## Measured Performance (Reference)

The following measurements were collected on Ascend A2/A3 (float type, 2-NPU point-to-point).

| Transfer Size | TGET BW (GB/s) | TGET_ASYNC BW (GB/s) | TGET Device Avg Cycles | TGET_ASYNC Device Avg Cycles |
| ------------- | --------------: | --------------------: | ---------------------: | ---------------------------: |
| 4 KB          | 0.20            | 0.18                  | 36.28                  | 136.51                       |
| 16 KB         | 0.77            | 0.64                  | 149.98                 | 161.40                       |
| 64 KB         | 1.50            | 2.44                  | 579.62                 | 277.25                       |
| 256 KB        | 2.40            | 7.90                  | 2409.72                | 745.54                       |
| 1 MB          | 2.63            | 8.98                  | 9317.93                | 2624.24                      |
| 4 MB          | 4.85            | 10.49                 | 37924.12               | 10273.68                     |

### Analysis

- **TGET** bandwidth gradually increases with transfer size but saturates at approximately **4 GB/s** — the throughput ceiling of the UB staging path.
- **TGET_ASYNC** significantly outperforms TGET for large transfers (≥256 KB), reaching approximately **10.5 GB/s**
  host-observed bandwidth at 4 MB.
- For very small transfers (4 KB), TGET_ASYNC is slightly slower than TGET due to SDMA launch overhead.

### Bandwidth Comparison Plot

Run the plotting script to generate the comparison chart:

```bash
python3 scripts/plot_bw_compare.py
```

## Build and Run

### Prerequisites

- CANN Toolkit >= 8.5.0 (TGET synchronous instruction); >= 9.0.0 (TGET_ASYNC asynchronous instruction)
- **MPICH** (recommended; the host side loads `libmpi.so` via `comm_mpi.h` with MPICH-compatible communicator handles)
- 2 or more Ascend NPUs

#### MPI installation (MPICH recommended)

```bash
# Ubuntu / Debian
sudo apt install mpich libmpich-dev

# Or install under $HOME without root — see tests/README.md "Build MPICH from Source"
export PATH=$HOME/mpich/bin:$PATH
export MPI_LIB_PATH=$HOME/mpich/lib/libmpi.so
```

`run.sh` searches common MPICH install paths and sets `MPI_LIB_PATH`. Override with `MPI_SEARCH_DIRS` (space-separated list of `bin/` directories).

> **Note**: This example does **not** support OpenMPI. `comm_mpi.h` hardcodes MPICH `MPI_COMM_WORLD` handle values; OpenMPI uses a different communicator representation and runtime MPI calls may fail. `--allow-run-as-root` is OpenMPI-specific and is not supported by MPICH.

### Steps

1. Configure your Ascend CANN environment:

```bash
source ${ASCEND_HOME_PATH}/bin/setenv.bash
# or source <workspace>/set_env_new.sh
```

2. Run the example (2-NPU by default). `run.sh` switches to its own directory automatically; invoke it from the repo root or from this directory:

```bash
# Option A: cd into this example first
cd ${git_clone_path}/kernels/manual/a2a3/tget_bandwidth
bash run.sh -r npu -v a3

# Option B: from the pto-isa-main repo root
bash kernels/manual/a2a3/tget_bandwidth/run.sh -r npu -v a3
```

Use `-n` to specify the number of ranks (default is 2):

```bash
bash run.sh -r npu -v a3 -n 2
```

`-v a3` matches the ST test scripts and maps internally to `SOC_VERSION=Ascend910B1` for the A2/A3 platform.

On success, the output looks like:

```text
================ TGET/TGET_ASYNC Bandwidth Sweep ================
peer_rank=1 dtype=float tile_elems=1024
[BW] instr=TGET bytes=4096 iters=1000 ...
[BW] instr=TGET_ASYNC bytes=4096 iters=1000 ...
...
test success
```

### Configurable SDMA Device Baseline

Use `device_baseline` mode to configure the transfer size, SQE size, queue count, and Post count. Set only the
parameters that need to differ from their defaults:

```bash
export TGET_BENCH_MODE=device_baseline
export TGET_DEVICE_BASELINE_BYTES=131072
export TGET_DEVICE_BASELINE_BLOCK_DIVISOR=1
export TGET_DEVICE_BASELINE_QUEUE_NUM=1
export TGET_DEVICE_BASELINE_POST_COUNT=1
export TGET_DEVICE_BASELINE_OUTER_ITERS=20
export TGET_DEVICE_BASELINE_INNER_ITERS=300

bash run.sh -r npu -v a3 -n 2
```

`BLOCK_DIVISOR=1` means `block_bytes=total_bytes`, so each Post contains one data SQE. The timed region measures
only the symmetric-memory transfer and Post completion. Verification runs after timing.

Constraints:

- `QUEUE_NUM <= 48`.
- At most 64 unwaited Payload slots may be retained per channel group.
- The data and completion SQEs generated by one Post on a queue must fit in that SQ.
- The default mode waits only for the latest Event while still verifying every Post; enable
  `TGET_DEVICE_BASELINE_WAIT_EACH_EVENT=1` to wait for each Event.

## Changelog

| Date       | Change |
| ---------- | ------ |
| 2026-07-16 | Use fixed post ID Events; add queue, channel-group, and multi-Post tests |
| 2026-06-01 | Align docs and `run.sh` with MPICH; remove OpenMPI-only `mpirun` flag |
| 2026-04-02 | Migrated from ST test to standalone performance example |
