# Python DSL Flash Attention Example

## Overview

This example demonstrates a high-performance Flash Attention implementation written with the PTO Python DSL (`ptodsl`). It is a Python-DSL port and parity experiment for the manual kernel in `kernels/manual/common/flash_atten`, and it follows the same four-stage software pipeline:

```text
compute_qk (Cube) -> compute_p (Vector) -> compute_pv (Cube) -> compute_gu (Vector)
```

The implementation also references the Huawei CSL PTO DSL AOT Flash Attention 140 TFLOPS example:

```text
https://github.com/huawei-csl/pto-dsl/tree/main/examples/aot/flash_attention/140tflops
```

The case is useful for validating that the Python DSL can express a production-style Flash Attention pipeline, including Cube/Vector cooperation, runtime S1 looping, software FIFO staging through global memory, correctness checks, and performance comparison against `torch_npu.npu_fused_infer_attention_score`.

## Supported Platform

- Ascend A3-class target (`--pto-arch=a3`, `--npu-arch=dav-2201` in `compile.sh`)
- CANN environment with `bisheng`
- PTO assembler `ptoas`
- Python environment with `ptodsl`, `torch`, and `torch_npu`

## External Dependencies

This example is not self-contained. It depends on the PTO Python DSL package
provided by the external `huawei-csl/pto-dsl` repository:

```text
https://github.com/huawei-csl/pto-dsl
```

The version used to validate this example is:

```text
pto-dsl package version: 0.1.2
upstream branch: main
commit: 6755794cfc145c8ffe4fae92483aa20148e57327
commit date: 2026-05-18 11:18:56 +0200
```

`ptodsl` is used in two places:

- `kernels/fa_builder.py` imports `ptodsl` to construct the PTO MLIR module.
- `run.py` imports `ptodsl` benchmark and NPU device helpers.

Install a compatible `pto-dsl` package before building or running this example.
For example, to install the current upstream main branch:

```bash
python3 -m pip install --user --upgrade git+https://github.com/huawei-csl/pto-dsl.git
```

You can verify which `ptodsl` package Python will use with:

```bash
python3 -c "import ptodsl, inspect; from ptodsl import to_ir_module; print(ptodsl.__file__); print(inspect.signature(to_ir_module))"
```

The `PTO_LIB_PATH` environment variable used by `compile.sh` is separate from
the Python `ptodsl` package. `PTO_LIB_PATH` must point to this PTO-ISA tile
library checkout so that `bisheng` can find headers under `include/`.

The full runtime/build dependency set is:

- `ptodsl` from `https://github.com/huawei-csl/pto-dsl`
- `torch` and `torch_npu`
- Ascend CANN runtime/toolkit with `bisheng`
- `ptoas` compatible with the MLIR emitted by the installed `ptodsl`
- this repository's PTO C++ headers, supplied through `PTO_LIB_PATH`

## Directory Layout

```text
kernels/python/flash_atten/
├── caller.cpp              # Host shim exported as call_kernel for ctypes
├── compile.sh              # Generates MLIR/C++ and builds build_artifacts/fa.so
├── kernels/
│   ├── fa_builder.py       # PTO Python DSL Flash Attention kernel builder
│   └── ptodsl_compat.py    # Local compatibility layer for ptodsl 0.1.2
├── scripts/
│   └── patch_vec_barriers.py
│                           # Generated-C++ PIPE_V barrier patch helper
└── run.py                  # Build, run, verify, and benchmark entry point
```

Generated files are placed under `build_artifacts/`:

```text
build_artifacts/fa.mlir     # MLIR emitted by fa_builder.py
build_artifacts/fa.cpp      # C++ emitted by ptoas
build_artifacts/fa_patched.cpp
                           # Optional C++ after compile.sh barrier patching
build_artifacts/fa.so       # Shared library loaded by run.py
build_artifacts/fa_summary_*.tsv
```

## Kernel Scope

Current shape and feature constraints are intentionally aligned with the manual parity target:

- `HEAD = 128`
- `S0 = 128` per Q block
- `TILE_S1 = 256` by default; `FA_S1_TILE=512` is available for experiments
- `CUBE_S1 = 128`
- `QK_PRELOAD = 3` in the DSL-tuned runtime path (`FA_QK_PRELOAD=4` is available for experiments; `MANUAL_QK_PRELOAD = 4` is kept only as the parity target metadata)
- non-causal attention only
- total Q rows are configured by `FA_Q_ROWS` and must be a multiple of `128`
- total KV rows are supplied at runtime by `run.py`; each S1 length must be at least `S1_TILE * QK_PRELOAD` and divisible by the selected `S1_TILE`

The generated shared library is specialized for the current `FA_Q_ROWS`, while S1 is handled at runtime.

## Build and Run

1. Configure the Ascend CANN environment.

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

2. Enter the example directory and set the PTO include path.

```bash
cd ${git_clone_path}/kernels/python/flash_atten
export PTO_LIB_PATH=${git_clone_path}
```

If `ptoas` or `bisheng` are not in `PATH`, set them explicitly:

```bash
export PTOAS=/path/to/ptoas
export BISHENG=/path/to/bisheng
```

3. Run one default benchmark case.

```bash
python3 run.py --case case1
```

4. Run the full default benchmark suite.

```bash
python3 run.py
```

The default suite runs `case1` to `case8` and recompiles the kernel for each `FA_Q_ROWS` value. It uses `TILE_S1=256` for `case1` to `case4`, and `TILE_S1=512` for `case5` to `case8` (`S1 >= 16384`).

| Case | Q rows (S0 total) | KV rows (S1) |
| --- | ---: | ---: |
| `case1` | 1024 | 1024 |
| `case2` | 2048 | 2048 |
| `case3` | 4096 | 4096 |
| `case4` | 8192 | 8192 |
| `case5` | 16384 | 16384 |
| `case6` | 32768 | 32768 |
| `case7` | 65536 | 65536 |
| `case8` | 131072 | 131072 |

## Custom Cases

Run a custom shape by setting `FA_Q_ROWS` and `FA_BENCH_LENGTHS`:

```bash
FA_Q_ROWS=1024 FA_BENCH_LENGTHS=1024 python3 run.py
```

Run several S1 lengths for one compiled Q shape:

```bash
FA_Q_ROWS=2048 FA_BENCH_LENGTHS=1024,2048,4096 python3 run.py
```

Control benchmark iterations:

```bash
FA_Q_ROWS=1024 FA_BENCH_LENGTHS=1024 FA_BENCH_WARMUP=20 FA_BENCH_ITERS=200 python3 run.py
```

Reuse an existing `build_artifacts/fa.so` when it was already compiled for the same `FA_Q_ROWS`:

```bash
FA_Q_ROWS=1024 bash compile.sh
FA_Q_ROWS=1024 FA_BENCH_LENGTHS=1024 python3 run.py --no-build
```

## Removing Redundant PIPE_V Barriers

`compile.sh` can patch selected `pipe_barrier(PIPE_V);` statements after `ptoas` emits C++ and before `bisheng` builds the shared library. Prefer op-pattern based removal over fixed generated-C++ line numbers:

The stable `gu` pattern is enabled by default for `compile.sh` and `run.py`. Additional patterns can be supplied explicitly:

```bash
python3 run.py --case case1
python3 run.py --remove-vec-barrier-patterns gu,softmax-sum-add
```

The same option can be passed through the environment:

```bash
FA_REMOVE_VEC_BARRIER_PATTERNS=gu,softmax-sum-add python3 run.py
```

Pass `--remove-vec-barrier-patterns none` to disable the default `gu` patch for comparison runs.

Supported patterns:

| Pattern | Aliases | Effect | Notes |
| --- | --- | --- | --- |
| `gu` | `trowexpandmul-tadd` | Removes the vector barrier in `TROWEXPANDMUL -> pipe_barrier(PIPE_V) -> wait_flag(PIPE_MTE2, PIPE_V) -> TADD`. | Stable strategy. It targets redundant V-pipe barriers in `compute_gu` where the MTE wait already provides spacing. |
| `softmax-exp-sum` | `texp-trowsum` | Attempts to remove the barrier in `TEXP -> pipe_barrier(PIPE_V) -> TROWSUM`. | Guarded by direct tile-dependency checks. If `TROWSUM` reads a tile written by `TEXP`, the candidate is kept and reported as skipped. |
| `softmax-sum-add` | `trowsum-tadd` | Removes barriers in `TROWSUM -> pipe_barrier(PIPE_V) -> TADD` when there is no direct tile dependency. | Experimental strategy. It can be combined with `gu`; small and mid-sized cases may improve, while large cases are usually close to neutral. |

The build log reports how many barriers were actually removed, for example:

```text
Patched generated C++ -> .../fa_patched.cpp (removed 74 PIPE_V barriers; lines=0, patterns=74)
Skipped PIPE_V barrier pattern candidates: softmax-exp-sum:direct-tile-dependency=2
```

For line-number based experiments, `--remove-vec-barriers line1,line2,...` and `FA_REMOVE_VEC_BARRIERS` are also available. This mode depends on the current generated C++ line numbers and is not recommended as the default workflow.

Performance validation should cover the full default case set. `ptodsl` event timing can occasionally report invalid 0.x us timings for the `ctypes` custom kernel launch path; use the conservative synchronized timing mode when validating barrier-removal performance:

```bash
python3 run.py --timing sync
```

## Output and Correctness

`run.py` prints latency, throughput, speedup, and max error for each shape. It compares the DSL kernel with:

- a host FP32 PyTorch reference when `Q_ROWS * S1` is small enough
- `torch_npu.npu_fused_infer_attention_score` for all benchmark sizes

Throughput is reported as TFLOP/s using matmul, scale, and softmax operation counts, following the 140 TFLOPS reference script convention.

A summary TSV is generated automatically for the default suite. You can choose the output path with `FA_SUMMARY_TSV`:

```bash
FA_SUMMARY_TSV=/tmp/fa_summary.tsv python3 run.py --case case1
```

## Notes

- `compile.sh` defaults `PTO_LIB_PATH` to `/sources/pto-isa`; set `PTO_LIB_PATH=${git_clone_path}` when working from this repository.
- `--no-build` is only suitable for one selected case because `fa.so` is rebuilt per `FA_Q_ROWS`.
- Large sequence lengths can skip the host FP32 reference to avoid allocating a very large QK matrix; correctness is then checked against the NPU fused reference with a looser tolerance.
