# pto.vcadd

`pto.vcadd` is part of the [Reduction Instructions](../../reduction-ops.md) instruction set.

## Summary

Full-vector reduction that sums all active lanes into a single scalar result, written to lane 0 with all other lanes zeroed.

## Mechanism

Reduces all active lanes of the source vector to a scalar sum, using a tree-reduction strategy implemented by the hardware. The result is broadcast to lane 0 of the output vector; all other lanes are zeroed.

For each active lane `i` in `0 .. N-1`:

$$ \mathrm{dst}_{0} = \sum_{i=0}^{N-1} \mathrm{src}_{i} $$

Inactive lanes are treated as zero. If all predicate bits are zero, the result is zero.

## Syntax

### PTO Assembly Form

```text
vcadd %dst, %src, %mask : !pto.vreg<NxT>
```

### AS Level 1 (SSA)

```mlir
%result = pto.vcadd %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>
```

Supported element types on A5: `i16-i64`, `f16`, `f32`.

## Inputs

| Operand | Role | Description |
|---------|------|-------------|
| `%input` | Source vector | Vector register holding the values to reduce; read at each active lane `i` |
| `%mask` | Predicate mask | Selects which lanes participate in the reduction; inactive lanes contribute zero |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%result` | `!pto.vreg<NxT>` | Result vector: lane 0 holds the scalar sum; all other lanes are zeroed |

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

!!! warning "Constraints"
    - **Narrow integer widening**: Some narrow integer forms (e.g., `i8`, `i16`) may use an internal wider accumulator; the final result is still returned in the declared result type.
    - **All lanes inactive**: If all predicate bits are zero, `dst[0]` is zero and all other lanes are zero.
    - **Mask granularity**: The mask has one bit per lane; partial-masking at sub-lane granularity is not supported.

## Exceptions

!!! danger "Exceptions"
    - The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
    - Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - Documented A5 coverage: `i16-i64`, `f16`, `f32`.
    - A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
    - Code that depends on an instruction-set-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Performance

### A5 Latency and Throughput

Vector reduction latency and throughput are target-specific. Consult the target profile's performance model for cycle-accurate estimates. Reduction operations typically have higher latency than elementwise vector ops due to the tree-reduction sequence.

---

## Examples

### C — Scalar Pseudocode

```c
T sum = 0;
for (int i = 0; i < N; i++)
    sum += src[i];
dst[0] = sum;
for (int i = 1; i < N; i++)
    dst[i] = 0;
```

### MLIR — SSA Form

```mlir
// Full-vector sum reduction: result in lane 0
%result = pto.vcadd %input, %mask : !pto.vreg<128xf32>, !pto.mask<b32> -> !pto.vreg<128xf32>
```

### MLIR — DPS Form

```mlir
pto.vcadd ins(%input, %mask : !pto.vreg<128xf32>, !pto.mask<b32>)
          outs(%result : !pto.vreg<128xf32>)
```

### Typical Usage

```mlir
// Compute the sum of a 128-element f32 vector tile
%mask = pto vidu %c128 : i1 -> !pto.mask<G>
%sum = pto.vcadd %vec, %mask : !pto.vreg<128xf32>, !pto.mask<b32> -> !pto.vreg<128xf32>
// %sum[0] contains the total; %sum[1..127] are zero
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Reduction Instructions](../../reduction-ops.md)
- Next op in instruction set: [pto.vcmax](./vcmax.md)
- Related reduction: [pto.vcgadd](./vcgadd.md) — lane-group reduction
- Related reduction: [pto.vcmax](./vcmax.md) — full-vector max
