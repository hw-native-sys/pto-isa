# Sparse FlashMLA with BF16 compute and FP4 cache

Standalone A5 PTO implementation of sparse MLA attention: Q, decoded K/V
and P use BF16, both matrix products accumulate in FP32, and the KV cache
uses packed FP4 with FP8 scales. The API is in [flash_mla_api.h](flash_mla_api.h).

## Kernel semantics

For each batch item and query position, SparseMLA attends to a supplied list
of cache tokens. All query heads share this selection and the cached latent
vectors; each head has its own query vector. A decoded cache vector serves as
both the key and the value.

Let $b$, $t$ and $h$ denote the batch, query position and head, with feature
dimension $D=512$. Let $I_{b,t,j}$ be the physical cache-token index at list
position $j$, and $L_b$ the active prefix length, defaulting to `topK`.
The valid list positions are

$$
\mathcal{J}_{b,t}=\{j\mid 0\le j<L_b,\ I_{b,t,j}\ne -1\}.
$$

Each selected cache vector $c_i$ is decoded using its per-16-feature scales:

$$
c_{i,d}=\operatorname{BF16}\!\left(
\operatorname{decode}_{\mathrm{E2M1}}(f_{i,d})\,
\operatorname{decode}_{\mathrm{E4M3}}(s_{i,\lfloor d/16\rfloor})
\right),\qquad 0\le d<512.
$$

Here $f$ contains the FP4 codes and $s$ the FP8 scale codes. The decoded product
is exactly representable in BF16. Gathered rows satisfy
$K_{j,:}=V_{j,:}=c_{I_{b,t,j},:}$ for $j\in\mathcal{J}_{b,t}$. Repeated cache
indices remain separate list entries and contribute once per occurrence.

For a fixed $(b,t)$ with a nonempty selection, the attention equations are

$$
S=\tau Q_{b,t}K^{\mathsf T},\qquad
Z_h=\sum_{j\in\mathcal{J}_{b,t}}\exp(S_{h,j}),\qquad
P_{h,j}=\frac{\exp(S_{h,j})}{Z_h+\exp(a_h)},
$$

$$
O_{b,t}=PV,\qquad
\operatorname{LSE}_{b,h,t}=\log Z_h.
$$

The attention scale $\tau$ is `softmaxScale`, defaulting to $1/\sqrt{512}$.
The optional per-head sink logit $a_h$ is shared across batch and query
positions. It acts as an additional attention entry with logit $a_h$ and a
zero value vector: $\exp(a_h)$ contributes to the denominator. LSE covers
the selected cache tokens. With the sink absent, $a_h=-\infty$
and $P=\operatorname{softmax}(S)$ along the selected-token dimension.
A sink of $+\infty$ gives zero output.

Optional extra-cache entries append to the selected list and use the same
softmax denominator. Their indices address the extra cache's own allocation.
For an empty combined selection, the API returns $O=0$ and
$\operatorname{LSE}=+\infty$ for every head.

These equations define the mathematical target. The kernel forms its PV
operand as `BF16(exp(score - held_basis))`; FP32 row sums use the exponentials
before that cast. Matrix products accumulate in FP32, and the final output
is rounded to BF16. The [numerical reference](golden/reference.py) supplies
both mathematical and device-arithmetic models.

## Inputs, outputs and numerical contract

| Tensor | Type and layout |
| --- | --- |
| Q | BF16 `[B,Q,H,512]`, H = 64 or 128 |
| Cache / optional extra cache | Bytes `[pages,page_size,288]` |
| Sparse indices | INT32 `[B,Q,topK]`, addressing physical cache tokens |
| Optional prefix lengths | INT32 `[B]`, bounded by the corresponding topK |
| Optional attention sink | FP32 `[H]` |
| Output | BF16 `[B,Q,H,512]` |
| LSE | FP32 `[B,H,Q]`, natural logarithms |

Each cache record contains 256 packed E2M1 bytes followed by 32 E4M3 scale
bytes. The low nibble contains the even feature; each scale covers 16
features. All 512 features, including RoPE, use this format. Finite signed
and zero scales are supported. Dequantization forms the exact BF16
`E2M1 * E4M3` product.

Callers must validate each index in the active per-batch prefix: `-1` masks
an entry; other entries must address a valid physical cache token in
`[0, pages * page_size)`. Entries beyond the prefix are ignored. The fixture
runner enforces this contract. Duplicate indices retain their multiplicity.
Extra-cache selections follow the same rules for their cache and prefix,
follow the main-cache stream and share its softmax denominator.

Conditional rescaling holds a row basis until the maximum increase exceeds
`ln(256)`. Each AIV decides over its 32 rows; either AIV can trigger export
of the prior PV segment before the current PV. The final segment is always
exported. BF16 P rounding and FP32 accumulation order can differ from full
rescaling. Finite Q/cache data and finite intermediate scores are required.
See the [recurrence](docs/design.md#conditional-rescaling).

## Build and run

Requires the Ascend950 CANN toolchain, Python NumPy and `ml_dtypes`.
Host code uses C++20 and the device frontend uses C++17.
The PTO headers must provide native control `TPipe` support: `CtrlPipeConfig`,
OR-merged V2C requests and C2V replies with configurable synchronization.

Select your installed CANN toolkit with `ASCEND_HOME_PATH`; the scripts source
its `set_env.sh`:

```bash
export ASCEND_HOME_PATH=/path/to/your/cann-installation
```

When unset, the scripts check `cann` and `ascend-toolkit/latest` under
`$HOME/Ascend`, then `/usr/local/Ascend`. For other installation paths, set
`ASCEND_HOME_PATH` explicitly. Use a fresh build directory when switching
toolkit installations so CMake selects the matching compiler and libraries.

Build, generate golden data and launch in one command. From the repository root:

```bash
cd kernels/manual/a5/sparseFlashMLA
bash build_and_run.sh --run-mode sim
```

The default case uses B=1, Q=2 and topK 1024: two jobs sharing one cache pool
on one simulator core. Use
`--case NAME` to select another validation fixture and `--tile-n 64` for N64.
The same tile size is passed to compilation and golden generation. For hardware:

```bash
bash build_and_run.sh --run-mode npu --soc-version Ascend950PR_9599 \
  --build-dir ./build/npu
```

Substitute the installed hardware's Ascend950 SoC version. The combined script
generates its fixture under the selected build directory and forwards launch
and log options to `run.sh`. Use `--validate-only` for compilation, golden
generation and host fixture validation.

Supply shape options to generate a custom case:

```bash
bash build_and_run.sh --run-mode npu --build-dir ./build/npu \
  --query-length 672 --top-k 1024 --kv-length 16384 --page-size 64
```

Shape options select `--case custom` automatically and write the fixture to
`BUILD_DIR/golden/custom`. The custom generator uses seeded random Q/cache
data and sparse indices, including masked entries. Named validation fixtures
keep their prescribed inputs; combining `--case NAME` with shape options
requires `NAME=custom`.

| Custom-case option | Default |
| --- | --- |
| `--batch-size` / `--batch` | `1` |
| `--query-length` / `--sq` | `1` |
| `--num-heads` / `--heads` | `64` (`64` or `128` supported) |
| `--top-k` / `--topk` | `129` |
| `--kv-length` | `129` logical tokens per pool; each pool rounds up to whole pages |
| `--kv-pool` | `per-batch`: independent pool per batch; `shared` uses one pool for all batches |
| `--page-size` | `32` |
| `--seed` | `17` |
| `--softmax-scale` | `1 / sqrt(512)` |
| `--rescale-rate` | Unset: random inputs; set `0`–`100` for controlled rescale inputs |

Feature dimension is fixed at 512. `--tile-n` continues to select N64 or
N128 for both compilation and reference generation.

With `--kv-pool per-batch`, each batch item owns a separate physical KV cache
region. Its query positions share that region, with a sparse index list per
query. This models independent-cache decode with Q=1 and B independent
sequences. For a prefill-style shared-pool workload, use B=1 and increase Q.
`--kv-pool shared` allows all batches to address one pool. Optional extra
caches follow the same pool mode.

The generator stores absolute token indices, adding each batch's page-aligned
pool offset to valid indices and preserving `-1` masks. The kernel API is
unchanged. Main-cache storage in bytes is
`pool_count * ceil(kv_length / page_size) * page_size * 288`, where
`pool_count` is B for `per-batch` and 1 for `shared`. Q does not multiply cache
storage. Pool layout is recorded in `fixture.json`.

For an independent-cache decode case:

```bash
bash build_and_run.sh --run-mode npu --build-dir ./build/npu \
  --batch-size 28 --query-length 1 --kv-pool per-batch \
  --top-k 1024 --kv-length 16384 --page-size 64
```

Golden generation prints the reference-predicted rescale rate and saves it
under `rescale_stats` in `fixture.json`, using the existing device-reference
pass. Each eligible decision is a tile after the first, counted per 64-head
job; either AIV can trigger its conditional PV flush. The final mandatory
flush is excluded. Cases with no eligible decisions report `N/A`.

To select a controlled rate, add `--rescale-rate PERCENT` to the custom case:

```bash
bash build_and_run.sh --run-mode npu --build-dir ./build/npu \
  --query-length 672 --top-k 1024 --kv-length 16384 --page-size 64 \
  --rescale-rate 50
```

This constructs synthetic Q/cache values with seeded trigger positions and
verifies every job's decisions after BF16/FP4 quantization. The requested
event count is rounded to the nearest integer, with ties rounded up; both
requested and achieved rates are recorded. The target applies to `--tile-n`,
so the same bytes can have a different rate under N64 and N128.

Controlled generation supports H64/H128 and partial final tiles. It requires
positive `--top-k`, `--kv-length >= --top-k`, a positive finite softmax scale,
and at most 126 triggers per job. Scale settings that cannot produce finite
controlled QK scores are rejected. Prefix-length overrides, extra cache and
an invalid first tile are available in random/validation cases. Controlled
cases use all valid indices and support both cache-pool modes. For a fixed
shape, pool mode and seed, physical indices stay unchanged across rate settings.
These reference rates describe generated inputs; they are not onboard counter
measurements.

To run the steps separately:

```bash
cd kernels/manual/a5/sparseFlashMLA
OPENBLAS_NUM_THREADS=1 python3 -m unittest discover -s host_tests -v
python3 golden/gen_validation.py --case topk1024_jobs2 --tile-n 128
bash compile.sh --tile-n 128
bash run.sh --core-limit 1
```

For a custom shape, generate the fixture with `gen_data.py` and pass its
directory to the runner. For example, this small case can run in simulation:

```bash
python3 golden/gen_data.py --output build/golden/custom \
  --query-length 2 --top-k 128 --kv-length 512 --page-size 32 --tile-n 128
bash compile.sh --run-mode sim --tile-n 128
bash run.sh --run-mode sim --golden-dir build/golden/custom --core-limit 1
```

`run.sh` and `flash_mla_bf16_bench --golden-dir DIR` read the shape from the
fixture metadata. To change the shape, regenerate the fixture; no kernel
rebuild is needed unless the tile size or build options change.

Default artifacts live under this kernel's `build/` directory, regardless of
the caller's working directory: golden fixtures in `build/golden/`, run logs
in `build/runs/`, and the private simulator cache in `build/simulator/`.
The runner selects `build/golden/topk1024_jobs2` by default. Use `--build-dir`,
`--golden-dir` or `--run-dir` to override the corresponding paths. Run logs and
the simulator cache follow the selected build directory. The build and fixture
must use the same token tile size.

| Build option | Default | Values / purpose |
| --- | --- | --- |
| `--tile-n` | `128` | `64` or `128` tokens |
| `--run-mode` | `sim` | `sim` or `npu` |
| `--soc-version` | `Ascend950PR_9599` | Ascend950 target |
| `--build-dir` | `build/` | Build output directory |
| `--jobs` | `8` | Parallel build jobs |

Use `bash build_and_run.sh --help`, `bash compile.sh --help` and
`bash run.sh --help` for the complete CLI.

`run.sh` uses a private simulator configuration with one physical AIC and
two AIVs by default. `--sim-cores N` selects the simulator core count;
`--core-limit N` bounds the launch. `FLASH_MLA_SIM_LIB` selects an existing
simulator library/configuration directory. The runner checks that the launch
fits its configured physical cores. Direct callers of `LaunchFlashMLABf16`
must bound `launchCount` to the available physical AIC count.

When using the separate scripts for hardware, pass matching `--run-mode npu`
and `--soc-version` arguments to both compilation and launch.

## Profiling and run artifacts

Each `run.sh` invocation creates a fresh `build/runs/run.*` directory, or
uses an empty directory supplied with `--run-dir DIR`. It records
`stdout.log`, `stderr.log`, `exit_status.txt` and `manifest.json`; the
manifest captures binary, source, input and simulator configuration hashes.
Simulator instruction and timing logs remain available after a successful
run. Memory dumps are removed only after success; use `--keep-memory-logs`
when investigating memory ownership. Failed runs retain all logs.
Cleanup prints compact totals; `cleanup_summary.json` retains the full details,
including the list of processes whose open files could not be inspected.

Summarize one or more completed runs with:

```bash
python3 tools/summarize_run.py build/runs/run.RUN_ID \
  --output build/summary.json
```

Compare maximum per-subcore **kernel elapsed ticks** with stamps disabled.
Host `Total tick`, local VF work sums and simulator wall time are separate
measurements. Match source/build identities, input bytes, physical core
count, rescale decisions and exit status before attributing a difference.
Busy counters include producer waits; use transfer bytes and elapsed ticks to assess bandwidth.

Diagnostic stamps are defined in [flash_mla_profile.h](flash_mla_profile.h).
`FLASH_MLA_MARK_STAMP_COMPUTE_PIPE=1` and
`FLASH_MLA_MARK_STAMP_DATA_PIPE=1` can be passed through `CXXFLAGS` when
configuring a separate **NPU build** (`--run-mode npu`). Simulator builds
reject either stamp option at compile time because stamps can deadlock the
simulator; this also applies to kernel-only builds and `--validate-only`.
Data stamps add same-pipe barriers and affect timing.
The [stamp map](docs/design.md#profiling-stamps) explains stage parts
and ownership across persistent job boundaries.

### Onboard benchmark

NPU builds also produce `flash_mla_bf16_bench`, which uses the same fixture
validation and kernel library as `flash_mla_bf16_test`. It runs warmup and
measurement launches, synchronizes each launch, and checks the final output
and LSE against both references. Outputs are poisoned before every launch.
The benchmark is excluded from simulator builds.

The default onboard example uses random inputs with batch 1, query length 672,
64 heads, dimension 512, top-k 1024 and N128 tiles, sharing one cache pool.
Golden generation reports the achieved reference rescale rate. The controlled
`topk1024_jobs672` validation fixture remains available through
`golden/gen_validation.py`.

| Benchmark option | Default | Meaning |
| --- | --- | --- |
| `--golden-dir DIR` | Required | Fixture directory matching the compiled tile size |
| `--samples N` | `5` | Measurement launches; must be positive |
| `--warmup N` | `2` | Initial launches excluded from reported measurements; zero is allowed |
| `--flush-bytes N` | `8000000000` | Bytes written to a separate device buffer before each launch; zero disables this step |
| `--core-limit N` | `0` | Maximum AIC count; zero uses the available count, bounded by the job count |
| `--validate-only` | Off | Validate fixtures without initializing the device or launching kernels |

The flush is an attempt to evict cached data before each launch. Its effect
depends on the device and memory path; verify it with cache metrics when
needed. The requested allocation must succeed; reduce `--flush-bytes`
explicitly if memory is limited. A zero-flush run measures repeated use of
the same buffers and may benefit from cache reuse. Compare runs using the
same flush size, inputs, core limit and toolkit. Cache reuse within a launch
also depends on whether jobs share a cache pool.

Start with stamp-free `CXXFLAGS` and a build directory configured without
stamps. From this kernel directory, build the NPU binary and generate a
benchmark fixture:

```bash
bash compile.sh --run-mode npu --build-dir ./build/npu
python3 golden/gen_data.py --output ./build/npu/golden/benchmark \
  --query-length 672 --top-k 1024 --kv-length 1024 --page-size 37
source tools/setup_cann.sh
export LD_LIBRARY_PATH="$PWD/build/npu/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
build/npu/flash_mla_bf16_bench \
  --golden-dir build/npu/golden/benchmark --validate-only
```

Collect device task durations with `msprof`. Use a fresh directory per run
so that output captures and profiler reports belong to one invocation:

```bash
kernel_dir="$PWD"
mkdir -p build/npu/profiles
profile_dir=$(mktemp -d "$PWD/build/npu/profiles/run.XXXXXX")
(
  set -o pipefail
  cd "$profile_dir"
  msprof --task-time=on --output=./profile \
    --application="\"${kernel_dir}/build/npu/flash_mla_bf16_bench\" --golden-dir \"${kernel_dir}/build/npu/golden/benchmark\" --warmup 2 --samples 5 --flush-bytes 8000000000" \
    2>&1 | tee stdout.log
)
```

Confirm `Implementation correctness: PASS` in the log before interpreting
timing. Select the `op_summary_*.csv` for this process and device beneath
the profile directory, then summarize it:

```bash
python3 tools/summarize_profile.py /path/to/op_summary_FILE.csv --warmup 2 --samples 5
```

The summary selects `flashMlaBf16Kernel` tasks by name, requires exactly
seven matching launches for these options, sorts by device start time and
excludes the first two. It reports all five durations plus mean, median,
minimum and maximum. `Task Duration(us)` measures the kernel task; host
fixture loading, cache flushing and output validation are separate work.
The tool leaves all measurement samples in the summary.

For pipe or cache diagnosis, use a separate `msprof op` run with the exact
kernel name from `Op Name`. Consult the installed `msprof op --help` for
supported metrics and launch selection; `PipeUtilization,Default` and
`L2Cache,Default` are useful groups when supported. Detailed profiling may
replay kernels and perform its own warmups, changing cache state. Use the
full-process task durations above for the benchmark comparison, and the
separate detailed run to investigate stalls. Keep stamps disabled for
uninstrumented timing.

## Validation

The host suite checks cache encoding, the numerical reference, fixture formats,
and conditional-rescale results against independent closed-form expectations.
It covers masks, tails, empty jobs, extra cache, sinks, negative scales and
decoded values beyond the FP8 range. Simulator and hardware runs validate
device synchronization and buffer reuse.
Generate the compact boundary fixtures with:

```bash
python3 golden/gen_skip_validation.py
```

These fixtures are written to `build/golden/skip_validation/` by default.
Use `--tile-n 64` for N64 device references or `--case singleton` for a
single-job fixture. `--case extra_cache` selects one token from each cache
stream and checks their shared softmax state.

## Pipeline and memory ownership

The default uses N128 token tiles and paired gathers with 32-token slices.
Independent operand credits, QK/PV unit flags and three L1 KV banks coordinate
the persistent schedule across logical jobs.

The [instruction interface guide](docs/instructions.md) describes the custom
tile-based metadata, gather, softmax and global-update entries, and the
`TPUSH`/`TPOP` data/control handoffs used by the AIC/AIV schedule.
Each custom instruction has its own header under `custom/`. Gather, store and
compile-time scalar fill use the namespaced PTO names `custom::MGATHER`,
`custom::TSTORE` and `custom::TEXPANDS`. Loads and push/pop dispatch use the
standard PTO entries.

The [design reference](docs/design.md#pipeline-and-memory-ownership) describes
the persistent schedule, L1/UB allocation, local and cross-core flag maps,
regional KV releases, query reuse and output completion credits. At the
default geometry, application storage is 480 KiB L1 and 250.875 KiB UB per AIV.
