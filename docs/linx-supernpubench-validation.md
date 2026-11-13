# SuperNPUBench one-level-arch validation

[中文](linx-supernpubench-validation_zh.md)

The CPU comparison covers the 544-entry portfolio in [cases.json](../tests/cpu/linx_supernpubench/cases.json): 444 active build configurations, 75 supplementary source configurations and 25 additional numerical configurations. These are configuration records, not 544 distinct operators. Of these, 78 have independent numerical checks; the others only establish successful execution.

CPU passes **462/544** records and covers **all 428/428 native successes**, with no missing native pass. The 58 native numerical passes also pass the independent golden on CPU.

| Outcome | CPU-SIM | Native Linx |
| --- | ---: | ---: |
| Numerical pass | 71 | 58 |
| Execution only | 391 | 370 |
| Build/API failure | 76 | 60 |
| Compiler failure | 0 | 30 |
| Execution failure | 1 | 16 |
| Execution timeout | 1 | 0 |
| Numerical failure | 4 | 10 |

## Versions and reproduction

| Component | Version |
| --- | --- |
| SuperNPUBench | `f6359b207dac65eca7332b4d527d6ccfca95f722`; `d34cea27e3d03228c2d897f83417ce1e7cde84e1` has the same benchmark sources |
| Linx-TileOP-API | `74d7326c521aeb1b21c52602b67c55b42b3563a3` |
| SuperScalarModel checkout | `34c110137a5c039f0283118749450e7966556118`; executable hash is recorded with the results |
| Compiler | GCC 15.2.1 on aarch64; Linx Clang 15.0.4 with matching musl libraries |
| Build | Fresh objects, `-O2`, no fast math, `-ffp-contract=off`, 1 or 4 PEs as recorded per case |

Install NumPy and PyTorch for the upstream input generators. The runner uses the repository Makefiles and the checked-in case parameters:

```bash
python3 tests/cpu/test_linx_supernpubench.py \
  --benchmark /path/to/SuperNPUBench \
  --upstream /path/to/Linx-TileOP-API \
  --toolchain /path/to/linx_blockisa_llvm_musl \
  --model /path/to/SuperScalarModel \
  --musl-libs /path/to/musl-only-libs \
  --compiler /path/to/g++-15 --routes cpu,upstream \
  --jobs 16 --output-dir /tmp/linx-supernpubench
```

Use a fresh output directory. `--ids S0063,C0408` selects specific records; `--routes cpu` only needs the benchmark and host compiler. `--sanitize` enables ASan/UBSan for host builds. Each build/run has a 240-second timeout by default, adjustable with `--timeout`. A nonzero portfolio exit means a native success is missing on CPU; documented failures on both routes are retained rather than hidden. Inspect `counts` even when the command returns zero.

`results.json` contains all outcomes and exact argv; `provenance.json` records source/header/adapter hashes and model identity. Individual logs and comparisons are under `results/<ID>/<route>/`. Numeric inputs are generated once, copied to both routes and SHA-256 checked. Numeric acceptance uses the upstream NumPy/PyTorch golden and its tolerance, not agreement with CPU output. Model completion requires guest `R2 = 0` and no error/assertion diagnostics; a host process exit code of zero alone is insufficient.

## CPU launch and source adapters

Native builds use the original benchmark sources and pinned native headers. CPU builds use an isolated source copy with an explicit [adapter](../tests/cpu/linx_supernpubench/host_adapter.py); the exact diff is emitted as `host/SuperNPUBench/benchmark/host-adapter.patch`. This is adapted source validation, not a claim that GNU C extensions, Linx inline assembly or physical pointers compile and execute unchanged on a host.

- The host launcher preserves C++ `main` return semantics by renaming its object symbol after compilation, and launches the recorded number of PEs in a shared Core context. RES_CHECK publication/completion flags use host atomics.
- GNU C range initializers become equivalent pre-main array initialization. Two driver-local `sync` variables are renamed to avoid POSIX `sync` lookup ambiguity.
- Hash, histogram/top-k and QLI inline assembly emitters map to CPU Tile operations; only QLI trace hints are discarded. Empty Shared-register output constraints in dynamic convolution map to explicit publication/acquisition of two Core slots.
- The dynamic MoE combine driver has a barrier after reset/input initialization. Its original unsynchronized resets can race with PE0's window-state update on concurrent host threads.
- The launcher supplies zero-initialized GM storage for `[0, 1 MiB)` and `[0x4000800000, 0x4001800000)`. QLI scalar physical pointers use explicit translation. These model-memory assumptions are visible in the launcher and are not implicit permissions for arbitrary null or physical host dereferences. In particular, QLI dynamic execution with these buffers is not a numerical validation of an external dataset.

Kernels and goldens are not replaced with host reference algorithms. The documented emitter and driver adaptations are part of the validation setup. Missing embedded QLI/sparse-MLA input objects remain build failures.

## Backend behavior exercised

BF16 scalar conversions preserve two-byte storage, round ties to even and retain signed zero/NaNs. CUBE valid extents use physical capacity; reduction-prefix views inherit dynamic parent extents. RowMajor TileArray assembly places fragments into the rectangular parent while CUBE fragments preserve their physical CELL order. GPR masks retain the native column/lane ordering and inactive gather lanes do not access memory.

`runCore` supports 1–4 PEs, reusable barriers, error propagation and explicitly mapped GM. Shared slot generations preserve delayed consumers; copying a handle aliases the same acquired generation. The runtime regression exercises repeated publications, missing producers, dynamic views, mapped peer reads and aliased selection. This does not establish every possible alias, memory-domain or bounded hardware FIFO behavior. TPUSH/TPOP are absent from the native API. See the [runtime contract](../include/pto/npu/linx/README.md#build-and-launch).

## Remaining benchmark failures

The [machine-readable matrix](../tests/cpu/linx_supernpubench/validation-20261112.json) records every status and the first relevant native diagnostic; IDs map to exact directories and parameters in the manifest. The native route reports build/API constraints, compiler descriptor failures, execution failures and numerical mismatches separately. A failing configuration is not automatically a CPU backend defect.

An isolated diagnostic copy changing the two packed GM descriptor column counts from `kStoredGK` to logical `gK` passes P0008/P0009 on both routes with unchanged inputs and goldens. The original-source failures remain in the matrix.

Confirmed sample/contract issues include packed four-bit GM strides expressed in byte-carrier counts rather than logical elements, row-index gather used with a byte-offset MGATHER API, unsynchronized MoE startup and descriptor reinterpret-casts, missing fixture objects, and hash tails omitted when per-PE query counts are not multiples of 32. An extra hash self-check (139 queries, four PEs) reports 11 wrong outputs on both routes despite the execution-only portfolio record completing successfully.

For scalar/vector fixp quantization, samples put FP19 scale in descriptor bits `[31:13]`, while the tested model decodes the low 19 bits. The mismatch explains native near-zero/scalar-offset outputs; CPU follows the packed descriptor contract. Conv2d uses ColMajor GM views with default CUBE load/store selectors; the layout contract needs review. Other descriptor/model failures require API/compiler/model triage and are not all attributed to benchmark source.
