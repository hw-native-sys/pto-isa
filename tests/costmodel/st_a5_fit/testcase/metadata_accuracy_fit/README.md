# A5 observable-operand metadata models

These tests exercise unchanged public PTO calls, not a direct formula-only entry
point. The accepted geometry routes embed 282 independent measured inputs: 30 padding,
39 TPARTADD, 39 TPARTMUL, 65 TPARTMAX, 65 TPARTMIN, 12 TSEL and 32 TTRI.
Another 28 TTRI historical-shape/new-configuration checks are counted separately,
not presented as reconstructed original inputs. The tolerance is 10% per input. Passing this suite
does not establish coverage of unspecified historical cases or every point in a
continuous shape range.

## Measurement and fit protocol

Each input has ten correct-output hardware launches at each of 256 and 1024 PTO
iterations. For each pure-AIV record, vector cycles are
`aiv_vec_time * aiv_total_cycles / aiv_time`. Discard the first launch, take the
remaining nine-launch median separately at each iteration count, then divide the
difference by 768. This estimates incremental vector-pipeline work, not whole
kernel latency. No fixed clock frequency is used.

Only training inputs select coefficients (OLS or relative-error WLS). Each model
has full column rank: five for general/padding paths, three for equal/row-empty
paths and seven for triangle. Models were frozen before reading their independent validation inputs.
No coefficient was adjusted using these tests. The failed padding-v1 holdouts
were retained as diagnostics; padding-v2 acceptance uses new independent inputs.

| API | New dtype coverage | Independent inputs | Maximum error |
| --- | --- | ---: | ---: |
| TFILLPAD | float | 6 | <4.96% |
| TFILLPAD_EXPAND | float, bfloat16, int16, int32 | 24 | <4.96% |
| TPARTADD | float, int16, int32 | 39 | 4.36245% |
| TPARTMUL | float, int16, int32 | 39 | 4.36256% |
| TPARTMAX | float, int16, int32, int8, uint8 | 65 | 4.36249% |
| TPARTMIN | float, int16, int32, int8, uint8 | 65 | 4.36274% |
| TSEL | uint16, uint32 | 12 | 1.76998% |
| TTRI | float, half, int16, int32 | 32 | 3.40988% |

## Observable input contract

- Vec, RowMajor, NoneBox; physical row stride equals physical column count and
  physical row size is a multiple of 32 bytes. All operands have runtime valid
  dimensions (`ValidRow = ValidCol = -1`); destinations are fully valid.
- Explicit TASSIGN addresses are 256-byte aligned; allocations are disjoint and
  contained in the 256-KiB UB. Host lazy-allocation pointers are not device
  addresses. Reading metadata never dereferences a device pointer.
- Padding is zero-padding, with positive contained source valid extents;
  destination rows 1..256 and row size 64..2048 bytes. TFILLPAD uses equal source
  and destination physical shapes. EXPAND permits a smaller physical source.
- Partial operations use equal physical shapes. One source's valid rectangle
  covers the entire destination; the other is contained. General paths allow
  at most 2728 rows and 87296 bytes per tile, subject to all UB checks above.
  Equal and row-empty fast paths are limited to rows 1..64 and row size
  64..1024 bytes. A row-empty source has zero valid rows and full valid columns;
  arbitrary empty-column geometry is not accepted.
- TSEL uses fully valid data with rows1..12 and cols32..96 in steps of32;
  packed uint8 mask physical/valid columns32, matching rows. The A5 uint16/uint32
  implementations do not use the temporary tile. The unchanged vector formula
  is selected from the data dtype and shape, not from the packed-mask operand.
- TTRI observes template direction and scalar diagonal, clamped as in the source.
  Full valid shape, half rows<=512 and rowbytes<=16384; float/int16/int32
  rows<=4095 and rowbytes<=261888, all subject to the same UB capacity/address
  checks. The rank7 continuous piecewise first-pass model and second-pass work
  model use an empirically selected knee, not an asserted ISA throughput rule.
  Frozen coefficients passed the separate28 extent checks at max3.41553%.
- Coefficients are separate per API/dtype/path, stored as two ordinary CSV
  coefficient blocks, not per-case constants. Metadata features count overlap
  vector repeats, copied repeats and relevant row/tail work. Unknown keys,
  unsupported layout, static-valid paths, overlap and out-of-domain inputs are
  rejected by the new route. Existing half routes remain separate and unchanged.
- Public BF16 coverage requires a distinct BF16 host type. The costmodel-only
  Clang `__bf16` fallback provides this when the standard library lacks
  `std::bfloat16_t`; it does not change CPU_SIM or NPU compilation. A compiler
  retaining the historical half alias cannot claim genuine BF16 public coverage.

These bounds describe the guarded formula domain, not exhaustive accuracy
certification of every geometry. Large-capacity training/holdout samples and
both source orders are included. Previously exposed failing TTRI-v3 holdouts are
diagnostics only; new v4 holdouts were shape-isolated and read after model freeze.

## Evidence provenance

Collection root: `remote_work/a5_metadata_fit_20261118` (2026-11-18). Raw attempts,
counter CSVs, correctness results, binaries and build hashes are retained outside
the source tree. `training_complete.tgz` SHA256:
`aaa5fbaca7c6df182a0ed9232c4f77be9a9be00932fdb54c4890645d23444580`.

| Evidence directory | Frozen model SHA256 | Independent report SHA256 |
| --- | --- | --- |
| padding_v2_analysis | `96dca6a8c1954d13d972f0ca99f89e387976121069dd08faa834474cc2495fd2` | `b0c8a02e52d8e01e4f2b0e6ec1bf17905a0b442119cb0599d73be688bf94ee84` |
| partadd_analysis | `3c0c07ee6ccab1184c888de801e467884ba1bc39d50238e97cb259bde975ede3` | `2ffeec90dc2171d622d85b8685fde6ce53d1a65900651b017bb2369c5aebdb05` |
| partmul_analysis | `bd4425c20cbb979b04d952ddc7eb45cf3407fd28863afa5a6a52b472a992f406` | `bce59ec07dd9ef22485e5dd7b7d9c79a4b17e3caf8d2e73917dcaba22665f93e` |
| partmax_analysis | `fccd48c94d328244151a795d8757a0e530af7d91b868f2c030977ecd82275c89` | `2daed9b780619a606600fecd22f1202c34795c4822de27f4b5981c005f14018f` |
| partmin_analysis | `afa702aba713a20482aaaba459792809eca89a1f99a11d48fc6ddbcc7aa8725f` | `8aa9b415c951b07fd9ec7b50a81f4b9366d2ed57f1d565788fdf4b5340779095` |
| triangle_v4_analysis | `78164cc6eeb0113d77212c50d57b59d23754ef822d65c66a7b543c5224585e7c` | `e22baf07e28b7df81b47adffe186ef884547b4b250c710a6e556c7d56c0e9233` |

The geometry model exports have57 API/dtype/path groups (114 coefficient-block CSV rows),
with TSEL retaining existing frozen coefficients.
The tests contain both measured cycles and the complete public geometry needed
to reproduce each prediction without accessing the remote evidence directory.

## Comparison implementation boundary

The native TCMP/TCMPS headers have been restored to the upstream implementation.
Its runtime-mode and constant-folded-mode vector loops can differ even for the
same API values. The costmodel cannot infer NPU compiler constant propagation
from host values or host-side `__builtin_constant_p`.

The costmodel exposes an explicit CPU-only setting; public PTO signatures and
NPU headers are unchanged:

```cpp
using pto::perf_sim::ComparisonCodegen;
using pto::perf_sim::ScopedComparisonCodegen;
{
    ScopedComparisonCodegen codegen(ComparisonCodegen::ConstantFolded);
    pto::TCMP(dst, lhs, rhs, pto::CmpMode::EQ);
    pto::TCMPS(dst, lhs, 2.0f, pto::CmpMode::EQ);
}
```

The declaration concerns the target binary, not whether the caller wrote a
literal or a variable. Prediction needs no NPU hardware or compiler. Scopes are
thread-local, nestable and restore the previous setting on exit (including
exception unwinding); do not reset the simulation context inside a live scope.
Each comparison tag snapshots the setting, so later configuration changes do
not reinterpret an already captured instruction. Direct formula clients set
`input.metadata.comparison_codegen` along with the complete comparison geometry.

`ConstantFolded` currently reuses only the **pre-hoist constant-EQ** calibration:
float/int32 TCMP and immediate-2 TCMPS, full runtime-valid source, rows1..64,
cols32..256 in steps of32, Vec RowMajor/NoneBox, uint8 destination with the same
physical and valid shape as the source (the original benchmark's overallocated
mask layout, although the comparison writes packed bits). Compact mask strides
are not covered by this old calibration. Explicit 256-byte UB
addresses, capacity and non-overlapping output/read spans are checked. Other
modes, Tile-rhs TCMPS, unknown host addresses and other dtypes reject. Existing
pre-hoist measurements are regression evidence, not a new independent holdout.

`RuntimeDispatch` is represented separately but **has no accepted upstream-native
curve yet**, and returns unsupported instead of borrowing the constant curve.
`Unknown` is the default and rejects the new packed-mask route; historical limited
legacy routes remain for compatibility. An explicit setting never falls back to
those unidentified legacy routes. The public last-cycle value0 denotes rejection,
not a zero-cycle comparison. This change does not restore full six-mode coverage.

The 34 post-hoist CSV curves and their 172 public accuracy checks were withdrawn:
their measurements apply to the reverted mode-specialized native implementation,
not to the current upstream one. They do not count toward current accuracy coverage.

The raw measurements remain archived outside the source tree:
`a5_metadata_compare_fit_20261118_complete.tgz` SHA256
`83480d05aa88965d5e3fad29226a570b9a5e460d77f66223decef6586625ef72`,
and `a5_metadata_compare_tile_v2_20261118_complete.tgz`.
No raw evidence was deleted or relabelled as upstream measurements.

A future runtime/multi-mode fit needs independently verified native code-generation
metadata and separate upstream-native
training/holdout measurements, and public-route validation. Merely renaming the
post-hoist keys or selecting a path from the mode value is not valid adaptation.
