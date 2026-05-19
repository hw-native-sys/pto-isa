# Vector Instruction Set: Reduction Instructions

`pto.v*` reduction instruction sets are defined here. Lane grouping, result placement, and inactive-lane rules are part of the visible vector contract and are not left to backend folklore.

> **Category:** Vector reduction operations
> **Pipeline:** PIPE_V (Vector Core)

Operations that reduce a vector to a scalar or per-group result.

## Common Operand Model

- `%input` is the source vector register value.
- `%mask` is the predicate operand `Pg`; inactive lanes do not participate.
- `%result` is the destination vector register value.
- Reduction results are written into the low-significance portion of the
  destination vector and the remaining destination bits are zero-filled.

---

## Execution Model: vecscope

Reduction operations execute inside a `pto.vecscope { ... }` region. Cross-lane reductions (`vcadd`/`vcmax`/`vcmin`) are issued to `PIPE_V` and perform tree-structured reduction in a single instruction. VLane-group reductions (`vcgadd`/`vcgmax`/`vccgmin`) operate within each 32-byte VLane independently.

**Typical pattern for row-wise sum (Softmax denominator):**

```mlir
pto.vecscope {
  %active = pto.pset_b32 "PAT_ALL" : !pto.mask<G>
  scf.for %row = %c0 to %row_count step %c1 {
    %vec = pto.vlds %ub_q[%row] : !pto.ptr -> !pto.vreg<64xf32>
    %row_sum_raw = pto.vcadd %vec, %active : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
    // row_sum_raw[0] contains the sum
    pto.vsts %row_sum_raw, %ub_sum[%row], %one_mask {dist = "1PT"} : ...
  }
}
```

**Cross-lane reduction mechanism:**
- `vcadd`: Tree reduction — pairs of lanes are added recursively until lane 0 holds the total
- `vcmax`/`vcmin`: Tree reduction with value+index packing in lane 0
- A `PIPE_V` barrier (`pto.barrier #pto.pipe`) is needed after group reductions when chaining with subsequent vector ops

---

## A5 Latency and Throughput (Ascend910_9599)

> All values are **popped→retire** cycle counts on the cycle-accurate simulator.

### Latency Summary Table

| PTO op | A5 RV (CA) | f32 | f16 | i32 | i16 |
|--------|-------------|-----|------|-----|------|
| `pto.vcadd` | `RV_VCADD` | 19 | 21 | 19 | 17 |
| `pto.vcmax` / `vcmin` | `RV_VCMIN` | 19 | 21 | 19 | 17 |
| `pto.vcpadd` | `RV_VCPADD` | 19 | 21 | — | — |
| `pto.vcgadd` | `RV_VCGADD` | 19 | 21 | 19 | 17 |
| `pto.vcgmax` / `vcgmin` | `RV_VCGMAX` | 19 | 21 | 19 | 17 |

### A2/A3 Latency and Throughput

| Metric | Constant | Value (cycles) | Applies To |
|--------|-----------|---------------|------------|
| Startup latency | `A2A3_STARTUP_REDUCE` | 13 | all reduction ops |
| Completion: FP group reduce (f16) | `A2A3_COMPL_FP_CGOP` | 21 | `vcgadd`/`vcgmax`/`vcgmin` (f16) |
| Completion: FP reduce (f32) | `A2A3_COMPL_FP_BINOP` | 19 | `vcadd`/`vcmax`/`vcmin` (f32) |
| Completion: INT reduce (i16) | `A2A3_COMPL_INT_BINOP` | 17 | all INT16 reductions |
| Completion: INT reduce (i32/f32) | `A2A3_COMPL_FP_BINOP` | 19 | all INT32/FP32 reductions |
| Per-repeat throughput | `A2A3_RPT_1` | 1 | INT16 group reductions |
| Per-repeat throughput | `A2A3_RPT_2` | 2 | INT32/FP32/FP16 reductions |
| Pipeline interval | `A2A3_INTERVAL` | 18 | all vector ops |

**Cycle model (A2/A3):** `total_cycles = startup + completion + repeats × per_repeat + (repeats - 1) × interval`

---

## Full Vector Reductions

### `pto.vcadd`

- **syntax:** `%result = pto.vcadd %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i16-i64, f16, f32
- **semantics:** Sum all elements. Result in lane 0, others zeroed.

```c
T sum = 0;
for (int i = 0; i < N; i++)
    sum += src[i];
dst[0] = sum;
for (int i = 1; i < N; i++)
    dst[i] = 0;
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result` contains the reduction result in its low element(s).
- **constraints and limitations:** Some narrow integer forms may widen the
  internal accumulation or result placement. If all predicate bits are zero, the
  result is zero.

---

### `pto.vcmax`

- **syntax:** `%result = pto.vcmax %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i16-i32, f16, f32
- **semantics:** Find max element with argmax. Result value + index in lane 0.

```c
T mx = -INF; int idx = 0;
for (int i = 0; i < N; i++)
    if (src[i] > mx) { mx = src[i]; idx = i; }
dst_val[0] = mx;
dst_idx[0] = idx;
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result` carries the reduction result in the low destination
  positions.
- **constraints and limitations:** This instruction set computes both the extremum and
  location information, but the exact packing of that information into the
  destination vector depends on the chosen form. If all predicate bits are zero,
  the result follows the zero-filled convention.

---

### `pto.vcmin`

- **syntax:** `%result = pto.vcmin %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i16-i32, f16, f32
- **semantics:** Find min element with argmin. Result value + index in lane 0.

```c
T mn = INF; int idx = 0;
for (int i = 0; i < N; i++)
    if (src[i] < mn) { mn = src[i]; idx = i; }
dst_val[0] = mn;
dst_idx[0] = idx;
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result` carries the reduction result in the low destination
  positions.
- **constraints and limitations:** As with `pto.vcmax`, the exact value/index
  packing depends on the chosen form and MUST be preserved consistently.

---

## Per-VLane (Group) Reductions

The vector register is organized as **8 VLanes** of 32 bytes each. Group reductions operate within each VLane independently.

```
vreg layout (f32 example, 64 elements total):
VLane 0: [0..7]   VLane 1: [8..15]  VLane 2: [16..23] VLane 3: [24..31]
VLane 4: [32..39] VLane 5: [40..47] VLane 6: [48..55] VLane 7: [56..63]
```

### `pto.vcgadd`

- **syntax:** `%result = pto.vcgadd %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i16-i32, f16, f32
- **semantics:** Sum within each VLane. 8 results at indices 0, 8, 16, 24, 32, 40, 48, 56 (for f32).

```c
int K = N / 8;  // elements per VLane
for (int g = 0; g < 8; g++) {
    T sum = 0;
    for (int i = 0; i < K; i++)
        sum += src[g*K + i];
    dst[g*K] = sum;
    for (int i = 1; i < K; i++)
        dst[g*K + i] = 0;
}
// For f32: results at dst[0], dst[8], dst[16], dst[24], dst[32], dst[40], dst[48], dst[56]
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result` contains one sum per 32-byte VLane group, written
  contiguously into the low slot of each group.
- **constraints and limitations:** This is a per-32-byte VLane-group reduction.
  Inactive lanes are treated as zero.

---

### `pto.vcgmax`

- **syntax:** `%result = pto.vcgmax %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i16-i32, f16, f32
- **semantics:** Max within each VLane.

```c
int K = N / 8;
for (int g = 0; g < 8; g++) {
    T mx = -INF;
    for (int i = 0; i < K; i++)
        if (src[g*K + i] > mx) mx = src[g*K + i];
    dst[g*K] = mx;
    for (int i = 1; i < K; i++)
        dst[g*K + i] = 0;
}
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result` contains one maximum per 32-byte VLane group.
- **constraints and limitations:** Grouping is by hardware 32-byte VLane, not by
  arbitrary software subvector.

---

### `pto.vcgmin`

- **syntax:** `%result = pto.vcgmin %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i16-i32, f16, f32
- **semantics:** Min within each VLane.

```c
int K = N / 8;
for (int g = 0; g < 8; g++) {
    T mn = INF;
    for (int i = 0; i < K; i++)
        if (src[g*K + i] < mn) mn = src[g*K + i];
    dst[g*K] = mn;
    for (int i = 1; i < K; i++)
        dst[g*K + i] = 0;
}
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result` contains one minimum per 32-byte VLane group.
- **constraints and limitations:** Grouping is by hardware 32-byte VLane, not by
  arbitrary software subvector.

---

## Prefix Operations

### `pto.vcpadd`

- **syntax:** `%result = pto.vcpadd %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** f16, f32
- **semantics:** Inclusive prefix sum (scan).

```c
dst[0] = src[0];
for (int i = 1; i < N; i++)
    dst[i] = dst[i-1] + src[i];
```

**Example:**
```c
// input:  [1, 2, 3, 4, 5, ...]
// output: [1, 3, 6, 10, 15, ...]
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result` is the inclusive prefix-sum vector.
- **constraints and limitations:** Only floating-point element types are
  documented on the current A5 instruction set here.

---

## Typical Usage

```mlir
// Softmax: find max for numerical stability
%max_vec = pto.vcmax %logits, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
// max is in lane 0, broadcast it
%max_broadcast = pto.vlds %ub_tmp[%c0] {dist = "BRC_B32"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

// Row-wise sum using vcgadd (for 8-row tile)
%row_sums = pto.vcgadd %tile, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
// Results at indices 0, 8, 16, 24, 32, 40, 48, 56

// Full vector sum for normalization
%total = pto.vcadd %values, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
// total[0] contains the sum

// Prefix sum for cumulative distribution
%cdf = pto.vcpadd %pdf, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
```
