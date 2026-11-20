# A5 default-input performance profiles

These profiles cover performance inputs whose original tensor values were not
recorded. They do **not** reconstruct the original input, and are not guarantees
for arbitrary index distributions. The NPU implementation and public PTO call
signatures are unchanged.

## Explicit assumptions

| Public call | Default performance profile | Data / index types |
| --- | --- | --- |
| Indexed `TGATHER(dst, src, index, tmp)` | All indices zero; source element zero is broadcast | 16-bit data with 16/32-bit integer indices; 32-bit data with 32-bit integer indices |
| Indexed `TSCATTER(dst, src, index)` | All indices zero; high destination contention; includes implementation's destination clear | 8/16-bit data with 16-bit integer indices; 32-bit data with 32-bit integer indices |
| `TGATHERB(dst, src, offset)` | Identity offsets to consecutive 32-byte source blocks | 8/16/32-bit data, `uint32_t` byte offsets |
| `TLOAD(dst, src)` or explicit `NormalFirstVictim` | Provisional warm same-address, 4KiB-aligned GM, contiguous ND 1x1x1x16x16 to full Vec 16x16 at UB zero | half / float / int32; known accuracy failure |

Gather/scatter have no API default index value and no reduction axis. Zero indices
are the user-authorized semantic convention, not an optional argument added to
the API. `TGATHERB` offsets are likewise a convention. `TLOAD` keeps its existing
default cache hint. `_2`/`_3` names with lost input details map to the same default
contract, not additional independent validation samples. The three 16x16 TLOAD
cases are integrated by explicit user request on 2026-11-19, despite failing the
10% training limit. They expose `WarmSameAddressApproximate`, not an accuracy-pass
claim. Other geometries and dtypes retain their existing TLOAD routing; unsupported
cache hints or geometry within this provisional route are rejected.

All profiles require explicit valid geometry: Vec RowMajor/NoneBox, dense physical
row strides, full runtime valid shape, explicit aligned UB addresses and safe UB
capacity. The measured indexed layout puts the index at zero, followed by source
and destination. Gather uses a one-block source. Its separately checked same-base
source/destination alias is restricted to 16-bit data/indices and the bounded
geometry accepted by the backend. Unknown addresses, static-valid specialization,
unmeasured layouts, unsupported dtypes or geometry are rejected. This is not a
claim that every Cartesian combination of listed shapes and dtypes is supported.

## Inspect or disable assumptions

Public predictions expose the selected assumption in
`pto::perf_sim::GetCostModelCtx().last_assumed_input_profile`. Direct backend
results expose `CostModelResult::assumed_input_profile`.

Set `GetCostModelCtx().assume_default_inputs = false` to reject these public-path
assumption-based predictions. Direct callers can set
`CostModelInput::default_input.enabled = false`. Rejected predictions return zero
with the existing unsupported diagnostic; they do not silently use another
profile. No tensor data or low UB address is dereferenced to infer index values.

## Measurement and fitting

Measurements use the actual PTO implementation on A5 with pure AIV,
Block=1/MixBlock=0, complete output validation on every launch, and 256/1024
iterations with ten launches each. Convert each launch independently using its
measured `aiv_total_cycles / aiv_time`; take the last nine launches' median at
each iteration count and divide their difference by 768. Indexed operations use
`aiv_vec_time`. The provisional TLOAD model uses `aiv_mte2_time`. This isolates steady-state pipeline
increments, not kernel latency, cold-cache latency or a fixed-frequency estimate.

Curves are fitted only to explicitly partitioned training configurations.
Observable work includes row count, 256-byte index/data repeats, destination-clear
repeats, full/tail row loops and byte volume. Piecewise/envelope parameters are
empirical coefficients over the accepted domain, not universal instruction
latencies or claimed hardware bank rules. Every selected curve must have full
parameter rank. Accuracy-qualified integer predictions must pass an independent
input set at 10% maximum relative error; provisional TLOAD is explicitly exempt
from that qualification, not from reporting its failure. Case IDs, card IDs and held-out values are not model
features; regression-test measurement constants are not fitted parameters.

The original missing-input list has 175 rows / 162 unique default contracts.
This input-profile count is distinct from public API-name coverage. BF16 device
fixtures use BF16 storage; CPU placeholder aliasing does not establish BF16
arithmetic accuracy. These fixtures only exercise checked bit movement.

## Validated coverage

All 175 original second-class CSV rows (162 unique default contracts) now have
prediction support under these defaults. **172/175 (98.29%) remain accuracy-qualified;
3/175 TLOAD rows are supported provisionally, not qualified.** API-name coverage
is a separate metric: these extend existing APIs, not 175 new instruction names.

### Provisional TLOAD accuracy exception

The fixed 16x16 model has one identifiable coefficient per storage width, fitted
only to training data by minimax relative error `2*min*max/(min+max)`:

| Storage / dtype | Coefficient | Integer cycles | Training observations | Maximum training error |
| --- | ---: | ---: | ---: | ---: |
| 16-bit / half | 153.46615813412387 | 153 | 6 | 11.76055% |
| 32-bit / float and int32 | 176.97873708736967 | 177 | 12 | 10.05648% |

Each width uses three training GM offsets (0, 8192, 16384), with two card replicas
per dtype. There is no card-ID parameter. Independent addresses 4096/12288 were
not used for selection and remain unevaluated; there is no independent-shape
validation or <=10% guarantee for these two provisional curves.

Measured cycles still vary at identical shape, default cache hint and observable
4KiB alignment. The collected observables do not explain that variation, so this
one-parameter profile misses the accuracy threshold. Physical memory mapping,
cache and memory-system state are possible unmeasured contributors, **not proven
root causes**; this does not diagnose a faulty card. Restricting shape/alignment
does not establish cold-cache, streaming, or end-to-end accuracy. The known-gap
test records failure explicitly and does not relax the 256-point vector test.

The 15 final curves use 247 training configurations. All 256 non-training
observations pass 10%: 159 original-list configurations and 97 held-out/regression
configurations. Maximum error is 9.523628%; family maxima are TGATHER 7.142079%,
TSCATTER 8.467562%, TGATHERB 9.523628%. The older failed holdout sets were retained
as regression evidence, not relabeled as fresh independent validation. New
vector v2 validation covered 26 shapes; the final TGATHERB revision then passed
12 new independent shapes, maximum 9.166852%, after its coefficients were frozen.

The final TGATHERB model is the simpler four-parameter row/repeat envelope
`max(a*rows*ceil(cols*bytes/256)+b, c*rows*ceil(cols*bytes/256)+d)`.
Additional address/tail candidates were investigated but were not selected.
Its training maximum is 8.54%, within the requested 10% limit (the earlier 8%
selection margin was stricter than the requested acceptance criterion).
Unchanged TGATHER/TSCATTER families retain their previously frozen parameters.

`evidence_summary.json` records frozen parameter provenance and sample counts;
`../default_input_accuracy_fit/main.cpp` contains all 256 measured regression
points. Full raw counters, attempts, build hashes, manifests, failed candidates,
and fitting scripts are retained in the task evidence archive, not represented
as unsupported claims of general index-distribution or end-to-end accuracy.
