# Python DSL Flash Attention Example

## Overview

This example demonstrates a high-performance Flash Attention implementation written with the PTO Python DSL (`ptodsl`). Its goal is to provide a performance-optimal Python-DSL Flash Attention kernel. The kernel is built around a four-stage software pipeline:

```text
compute_qk (Cube) -> compute_p (Vector) -> compute_pv (Cube) -> compute_gu (Vector)
```

It shows that the Python DSL can express a fully performance-tuned, production-style Flash Attention pipeline, including Cube/Vector cooperation, runtime S1 looping, software FIFO staging through global memory, correctness checks, and a performance comparison against `torch_npu.npu_fused_infer_attention_score`.

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

### Verified environment versions (for reproducible runs)

The performance and correctness results were last reproduced with the exact
versions below. Pin these for an accurate reproduction; mixing other versions
usually requires a matching `ptoas` / header pair (see the note that follows the
table).

| Dependency | Version (pin) | Notes |
| --- | --- | --- |
| PTO-ISA repo (this repo; headers via `PTO_LIB_PATH`) | commit `b9122ec586b5a7b4b686ca7498874a1c94b3573c` (2026-06-12) | plus the Flash-Attention working-tree changes under `kernels/python/flash_atten/` and the GlobalData header fix in `include/pto/common/pto_instr.hpp` (see below) |
| `pto-dsl` Python package (`ptodsl`) | `0.1.2`, commit `6755794cfc145c8ffe4fae92483aa20148e57327` (2026-05-18), branch `main` | the only `ptodsl` runtime/build dependency |
| `ptoas` | `0.45` | must match the `ptodsl`-emitted MLIR and these C++ headers |
| CANN toolkit (`Ascend-cann-toolkit`) | `9.0.0` (inner `V100R001C10SPC001B250`) | provides `bisheng` |
| `bisheng` | clang `15.0.5` (bundled with CANN 9.0.0) | |
| `torch` | `2.9.0` | |
| `torch_npu` | `2.9.0` | |
| Python | `3.10.19` | |
| Ascend driver | `26.0.rc1` (ascendhal `7.35.23`) | |
| Target NPU | Ascend A3-class (`--pto-arch=a3`, `--npu-arch=dav-2201`) | |

**`ptoas` / header pairing.** `ptoas` codegen and these C++ headers must be
compatible. With `ptoas 0.45`, the public GlobalData FIFO ops on the NPU path
(`TALLOC` / `TPUSH` / `TPOP` / `TFREE`) must pass explicit template arguments
`<Pipe, GlobalData, Split>` to their `*_IMPL` calls in
`include/pto/common/pto_instr.hpp`; without this 4-line fix (which mirrors the
existing `__CPU_SIM` branch) `bisheng` fails with `no member named 'cons'`. The
fix is required to build this example with `ptoas >= 0.45`.

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

Current shape and feature constraints. The kernel is deliberately specialized for a single shape family so the implementation can be tuned for peak performance:

- `HEAD = 128`
- `S0 = 128` per Q block
- `S1_TILE = 256` by default; `FA_S1_TILE=512` is available for experiments
- `CUBE_S1 = 128`
- `QK_PRELOAD = 3` is the performance-tuned default for the DSL runtime path (`FA_QK_PRELOAD=4` is available for experiments)
- `EXP_RING = QK_PRELOAD` is a hard invariant (`FA_EXP_RING` must equal `FA_QK_PRELOAD`): softmax and `compute_gu` reuse matching rescale slots, so they must share the same ring depth
- `KV_SPLIT = 1` by default (`FA_KV_SPLIT` may be `1`, `2`, or `4`). When `> 1`, each Q block's KV (S1) is split across `KV_SPLIT` work-units so that `NUM_Q_BLOCKS * KV_SPLIT` units fill the cube-core waves, and a second grid launch (`call_reduce`) flash-combines the per-unit partials into the final normalized `O`. The default suite auto-selects this per shape and per device (see [Build and Run](#build-and-run))
- non-causal attention only
- total Q rows are configured by `FA_Q_ROWS` and must be a multiple of `128`
- total KV rows are supplied at runtime by `run.py`; each S1 length must be divisible by the selected `S1_TILE` and (at `KV_SPLIT = 1`) yield at least `QK_PRELOAD` tiles. When `KV_SPLIT > 1`, the tile count must additionally be divisible by `KV_SPLIT`, and each chunk must still hold at least `QK_PRELOAD` tiles (i.e. `S1 / S1_TILE / KV_SPLIT >= QK_PRELOAD`)

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

The default suite runs `case1` to `case8` and recompiles the kernel for each `FA_Q_ROWS` value. Two knobs are selected per case:

- `S1_TILE`: `256` for `S1 < 32768` (`case1`–`case5`) and `512` for `S1 >= 32768` (`case6`–`case8`).
- `KV_SPLIT`: auto-selected per shape **and per device** from the cube-core wave utilization. On the reference 24-cube-core A3 target this enables `KV_SPLIT=2` only for `case5` and keeps `KV_SPLIT=1` everywhere else; a device with a different cube-core count may select differently. Override with `FA_KV_SPLIT`.

| Case | Q rows (S0 total) | KV rows (S1) | `S1_TILE` | `KV_SPLIT`\* |
| --- | ---: | ---: | ---: | ---: |
| `case1` | 1024 | 1024 | 256 | 1 |
| `case2` | 2048 | 2048 | 256 | 1 |
| `case3` | 4096 | 4096 | 256 | 1 |
| `case4` | 8192 | 8192 | 256 | 1 |
| `case5` | 16384 | 16384 | 256 | 2 |
| `case6` | 32768 | 32768 | 512 | 1 |
| `case7` | 65536 | 65536 | 512 | 1 |
| `case8` | 131072 | 131072 | 512 | 1 |

\*The `KV_SPLIT` column is for the reference 24-cube-core A3 target; it is chosen automatically from the cube-core count, so other devices may differ.

### Split-KV (`KV_SPLIT`) wave-quantization fix

When a Q block does not produce enough cube work-units to fill all cube-core waves (low wave utilization), `KV_SPLIT > 1` splits each Q block's KV range into `KV_SPLIT` work-units, raising `TOTAL_UNITS = NUM_Q_BLOCKS * KV_SPLIT`. Each unit computes an unnormalized partial (`O_acc`, running max, running sum); a second grid launch (`call_reduce`, gated by `-DFA_KV_SPLIT` in `compile.sh` and the `#if FA_KV_SPLIT > 1` block in `caller.cpp`) flash-combines the partials per Q block into the final normalized `O`. `KV_SPLIT=1` is the exact baseline (no split, no second launch). The split only pays off when it materially raises utilization and each chunk still satisfies the `QK_PRELOAD` prologue floor, which is why the suite gate enables it only for `case5` on the reference target.

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

Override the split-KV factor (custom single runs default to `KV_SPLIT=1`; only the default suite auto-selects it):

```bash
FA_Q_ROWS=16384 FA_BENCH_LENGTHS=16384 FA_S1_TILE=256 FA_KV_SPLIT=2 python3 run.py
```

Select the benchmark timing mode through the environment (default `event`; `sync` is the conservative host-timed mode, equivalent to `--timing sync`):

```bash
FA_Q_ROWS=1024 FA_BENCH_LENGTHS=1024 FA_BENCH_TIMING=sync python3 run.py
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

Throughput is reported as TFLOP/s computed from matmul, scale, and softmax operation counts (the same operation-count convention as the upstream reference `run.py`). **TFLOP/s is a measured value that depends on the input shape and data, not a fixed target** — absolute numbers vary between shapes and runs, so use the per-shape latency and the speedup against `torch_npu.npu_fused_infer_attention_score` for comparison.

A summary TSV is generated automatically for the default suite. You can choose the output path with `FA_SUMMARY_TSV`:

```bash
FA_SUMMARY_TSV=/tmp/fa_summary.tsv python3 run.py --case case1
```

## Expected Performance

> The numbers below are the **target** band for the pinned reference environment
> in the [version table](#verified-environment-versions-for-reproducible-runs)
> above (Ascend A3-class, 24 cube cores; `ptoas 0.45`; `torch_npu 2.9.0`). As
> noted in [Output and Correctness](#output-and-correctness), absolute latency and
> TFLOP/s vary with shape, data, and device — treat these as the reference band to
> reproduce, not a hard SLA.

With the `TPipe::SyncPeriod` setting in `include/pto/npu/a2a3/TPush.hpp` optimized
(see [issue #172](https://github.com/hw-native-sys/pto-isa/issues/172) for
details), the default suite is expected to reach the band below on the reference
target. Small cases use `event` timing (device time); large cases use `sync`
timing to avoid the occasional `event` glitch on very large `ctypes` launches (see
[Removing Redundant PIPE_V Barriers](#removing-redundant-pipe_v-barriers)).

| Case | S0 = S1 | DSL latency (µs) | `torch_npu` (µs) | Speedup | Timing |
| --- | ---: | ---: | ---: | :---: | :---: |
| `case1` | 1024 | 21 | 54 | 2.55× | event |
| `case2` | 2048 | 38 | 73 | 1.91× | event |
| `case3` | 4096 | 105 | 139 | 1.32× | event |
| `case4` | 8192 | 239 | 291 | 1.22× | event |
| `case5` | 16384 | 1001 | 985 | 0.98× | sync |
| `case6` | 32768 | 3185 | 3184 | 1.00× | sync |
| `case7` | 65536 | 12106 | 12158 | 1.00× | sync |
| `case8` | 131072 | 48062 | 48135 | 1.00× | sync |

The expected shape is a large lead at small sizes (where `torch_npu` carries a
fixed host-dispatch cost), narrowing as the size grows, and reaching parity
(~185 TFLOP/s) at the largest sizes.

## Notes

- `compile.sh` defaults `PTO_LIB_PATH` to `/sources/pto-isa`; set `PTO_LIB_PATH=${git_clone_path}` when working from this repository.
- `--no-build` is only suitable for one selected case because `fa.so` is rebuilt per `FA_Q_ROWS`.
- Large sequence lengths can skip the host FP32 reference to avoid allocating a very large QK matrix; correctness is then checked against the NPU fused reference with a looser tolerance.
