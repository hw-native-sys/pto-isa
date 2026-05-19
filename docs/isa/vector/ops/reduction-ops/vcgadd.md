# pto.vcgadd

`pto.vcgadd` is part of the [Reduction Instructions](../../reduction-ops.md) instruction set.

## Summary

Lane-group reduction that sums within each 32-byte VLane group, writing one result per group to its lane-0 position and zeroing all other lanes in each group.

## Mechanism

Reduces elements within each 32-byte VLane group independently, using a tree-reduction strategy implemented by the hardware. The result of each group's reduction is written to lane 0 of that group; all other lanes in each group are zeroed.

For a 128-element vector with 32-byte VLane groups (4-byte `f32` → 8 elements per VLane):

$$ \mathrm{dst}_{g \times 8} = \sum_{i=0}^{7} \mathrm{src}_{g \times 8 + i} $$

for each VLane group `g` in `0 .. 7`. All other lanes `(g × 8 + 1 .. g × 8 + 7)` are zeroed.

Inactive lanes are treated as zero.

## Syntax

### PTO Assembly Form

```text
vcgadd %dst, %src, %mask : !pto.vreg<NxT>
```

### AS Level 1 (SSA)

```mlir
%result = pto.vcgadd %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>
```

Supported element types on A5: `i16-i32`, `f16`, `f32`.

## Inputs

| Operand | Role | Description |
|---------|------|-------------|
| `%input` | Source vector | Vector register holding the values to reduce; lanes are reduced within their VLane group |
| `%mask` | Predicate mask | Selects which lanes participate in the reduction; inactive lanes contribute zero |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%result` | `!pto.vreg<NxT>` | Result vector: lane 0 of each VLane group holds the group sum; all other lanes in each group are zeroed |

For a 128-element `f32` vector: results appear at indices `0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120` (one per 32-byte VLane group). Note that for a 128-element f32 vector, there are 16 VLane groups, but the hardware may produce results at a different stride depending on the microarchitecture.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

!!! warning "Constraints"
    - **VLane granularity**: Reduction operates within 32-byte VLane groups; results are written to lane 0 of each group.
    - **Inactive lanes**: Inactive lanes are treated as zero.
    - **All lanes inactive in a group**: If all predicate bits are zero within a group, that group's lane 0 result is zero.
    - **Type support**: Only `i16-i32`, `f16`, `f32` are supported on A5.

## Exceptions

!!! danger "Exceptions"
    - The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
    - Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - Documented A5 coverage: `i16-i32`, `f16`, `f32`.
    - A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
    - Code that depends on an instruction-set-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Performance

### A5 Latency and Throughput

Vector lane-group reduction latency and throughput are target-specific. Lane-group reductions (`vcgadd`) typically have lower latency than full-vector reductions (`vcadd`) because each VLane group reduces independently and in parallel.

---

## Examples

### C — Scalar Pseudocode

```c
int K = N / 8;  // elements per VLane group (8 for f32)
for (int g = 0; g < 8; g++) {
    T sum = 0;
    for (int i = 0; i < K; i++)
        sum += src[g*K + i];
    dst[g*K] = sum;
    for (int i = 1; i < K; i++)
        dst[g*K + i] = 0;
}
// For f32 128-element vector: results at dst[0], dst[8], dst[16], dst[24],
// dst[32], dst[40], dst[48], dst[56], dst[64], dst[72], dst[80], dst[88],
// dst[96], dst[104], dst[112], dst[120]
```

### MLIR — SSA Form

```mlir
// Lane-group sum reduction: one result per 32-byte VLane
%result = pto.vcgadd %input, %mask : !pto.vreg<128xf32>, !pto.mask<b32> -> !pto.vreg<128xf32>
```

### MLIR — DPS Form

```mlir
pto.vcgadd ins(%input, %mask : !pto.vreg<128xf32>, !pto.mask<b32>)
            outs(%result : !pto.vreg<128xf32>)
```

### Typical Usage — Softmax Row Sum

```mlir
// Compute row-wise softmax: step 1 — compute exp and lane-group sum
%exp = pto.vexpdif %row, %c0 : !pto.vreg<128xf32>, f32 -> !pto.vreg<128xf32>
%mask = pto vidu %c128 : i1 -> !pto.mask<G>
%sum = pto.vcgadd %exp, %mask : !pto.vreg<128xf32>, !pto.mask<b32> -> !pto.vreg<128xf32>
// %sum[0,8,16,...] holds per-VLane exp sums for normalization
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Reduction Instructions](../../reduction-ops.md)
- Previous op in instruction set: [pto.vcmin](./vcmin.md)
- Next op in instruction set: [pto.vcgmax](./vcgmax.md)
- Related reduction: [pto.vcadd](./vcadd.md) — full-vector reduction
- Related reduction: [pto.vcmax](./vcmax.md) — full-vector max
- Related reduction: [pto.vcgmax](./vcgmax.md) — lane-group max
