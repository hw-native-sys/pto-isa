# tests/

Tests and examples for PTO Tile Library, covering both CPU simulation and NPU (including `sim` and on-board `npu` modes).

## Test Entry Points

| Scenario | Command |
|---------|---------|
| Full CPU Simulator run | `python3 tests/run_cpu.py --clean --verbose` |
| GEMM demo | `python3 tests/run_cpu.py --demo gemm --verbose` |
| Flash Attention demo | `python3 tests/run_cpu.py --demo flash_attn --verbose` |
| Single ST testcase (NPU) | `python3 tests/script/run_st.py -r [sim\|npu] -v [a3\|a5] -t [TEST_CASE] -g [GTEST_FILTER_CASE]` |
| One-click scripts | `./build.sh --run_all --a3 --sim` |

## Layout

```
tests/
├── script/                     # Test entry scripts (recommended entry point)
│   ├── run_st.py              # Build and run NPU ST
│   ├── build_st.py            # Build NPU ST only
│   ├── all_cpu_tests.py       # Batch build and run CPU ST suites
│   └── README.md              # Script usage guide
│
├── cpu/                        # CPU-side ST tests (gtest + CMake)
│   └── st/                    # CPU ST projects and testcase data generation scripts
│
├── npu/                        # NPU-side ST tests split by SoC
│   ├── a2a3/
│   │   ├── src/st/           # A2/A3 compute ST
│   │   └── comm/st/           # A2/A3 communication ST
│   └── a5/
│       ├── src/st/            # A5 compute ST
│       └── comm/st/           # A5 communication ST
│
├── run_st.sh                   # NPU ST one-click run script
└── run_comm_test.sh           # Communication ST one-click run script
```

## Synchronous and Asynchronous Communication Tests

Communication tests verify multi-device PTO communication primitives (Put / Get / Broadcast / Gather / Scatter / Reduce / Notify / Wait / Test), built on MPI + HCCL.

Communication tests are divided into **synchronous** and **asynchronous** instruction categories:

| Type | Test Examples | CANN Version Required |
|------|--------------|----------------------|
| Synchronous | `tput`, `tget`, `treduce`, `tbroadcast`, etc. | CANN 8.x+ |
| Asynchronous | `tput_async`, `tget_async` | **CANN 9.0+** |

> Asynchronous instructions depend on SDMA opapi interfaces introduced in CANN 9.0 (e.g., `aclnnShmemSdmaStarsQuery`). They will fail on older CANN versions due to missing symbols. `run_comm_test.sh` **does not include async tests by default**; use `-a` to opt in.

### Quick Start

```bash
# 8-NPU full test (default A2/A3, no async tests)
./run_comm_test.sh

# Include async tests (requires CANN 9.0+)
./run_comm_test.sh -a

# A5 SoC, 2 NPUs
./run_comm_test.sh -v a5 -n 2

# Enable debug logging
./run_comm_test.sh -d -t tput
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-n` | Number of available NPUs: 2, 4, or 8 | 8 |
| `-v` | SoC version: `a3` (Ascend910B) or `a5` (Ascend910_9599) | a3 |
| `-t` | Run specific testcase(s) (repeatable) | all |
| `-a` | Include async instruction tests (requires CANN 9.0+) | off |
| `-d` | Enable debug mode | off |

## Suggested Reading Order

| Order | Document |
|-------|---------|
| 1 (learn first) | [docs/getting-started.md](../docs/getting-started.md) |
| 2 | [docs/coding/tutorial.md](../docs/coding/tutorial.md) |
| 3 | This page |

For more complete environment setup and dependency details, see:
- [docs/getting-started.md](../docs/getting-started.md)
- [tests/README_zh.md](./README_zh.md) — 中文版
