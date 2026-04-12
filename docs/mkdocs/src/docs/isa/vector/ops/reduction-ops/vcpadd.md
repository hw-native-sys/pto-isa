<!-- Generated from `docs/isa/vector/ops/reduction-ops/vcpadd.md` -->

# pto.vcpadd

Standalone reference page for `pto.vcpadd`. This page belongs to the [Reduction Ops](../../reduction-ops.md) family in the PTO ISA manual.

## Summary

Inclusive prefix sum (scan).

## Mechanism

`pto.vcpadd` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the family operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%result = pto.vcpadd %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

Documented A5 types or forms: `f16, f32`.

## Inputs

`%input` is the source vector and `%mask` selects participating
  lanes.

## Expected Outputs

`%result` is the inclusive prefix-sum vector.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

Only floating-point element types are
  documented on the current A5 surface here.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected family or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- Documented A5 coverage: `f16, f32`.
- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```c
dst[0] = src[0];
for (int i = 1; i < N; i++)
    dst[i] = dst[i-1] + src[i];
```

```c
// input:  [1, 2, 3, 4, 5, ...]
// output: [1, 3, 6, 10, 15, ...]
```

```mlir
// Softmax: find max for numerical stability
%max_vec = pto.vcmax %logits, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
// max is in lane 0, broadcast it
%max_broadcast = pto.vlds %ub_tmp[%c0] {dist = "BRC_B32"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

// Row-wise sum using vcgadd (for 8-row tile)
%row_sums = pto.vcgadd %tile, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
// Results at indices 0, 8, 16, 24, 32, 40, 48, 56

// Full vector sum for normalization
%total = pto.vcadd %values, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
// total[0] contains the sum

// Prefix sum for cumulative distribution
%cdf = pto.vcpadd %pdf, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

## Detailed Notes

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

## Typical Usage

```mlir
// Softmax: find max for numerical stability
%max_vec = pto.vcmax %logits, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
// max is in lane 0, broadcast it
%max_broadcast = pto.vlds %ub_tmp[%c0] {dist = "BRC_B32"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

// Row-wise sum using vcgadd (for 8-row tile)
%row_sums = pto.vcgadd %tile, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
// Results at indices 0, 8, 16, 24, 32, 40, 48, 56

// Full vector sum for normalization
%total = pto.vcadd %values, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
// total[0] contains the sum

// Prefix sum for cumulative distribution
%cdf = pto.vcpadd %pdf, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

## Related Ops / Family Links

- Family overview: [Reduction Ops](../../reduction-ops.md)
- Previous op in family: [pto.vcgmin](./vcgmin.md)
