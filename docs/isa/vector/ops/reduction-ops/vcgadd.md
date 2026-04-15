# pto.vcgadd

`pto.vcgadd` is part of the [Reduction Instructions](../../reduction-ops.md) instruction set.

## Summary

Sum within each VLane. 8 results at indices 0, 8, 16, 24, 32, 40, 48, 56 (for f32).

## Mechanism

`pto.vcgadd` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the instruction set operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%result = pto.vcgadd %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

Documented A5 types or forms: `i16-i32, f16, f32`.

## Inputs

`%input` is the source vector and `%mask` selects participating
  lanes.

## Expected Outputs

`%result` contains one sum per 32-byte VLane group, written
  contiguously into the low slot of each group.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

This is a per-32-byte VLane-group reduction.
  Inactive lanes are treated as zero.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- Documented A5 coverage: `i16-i32, f16, f32`.
- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on an instruction-set-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Performance

### Timing Disclosure

The current public VPTO timing material for PTO micro instructions remains limited.
For `pto.vcgadd`, those public sources describe the instruction semantics, operand legality, and pipeline placement, but they do **not** publish a numeric latency or steady-state throughput.

| Metric | Status | Source Basis |
|--------|--------|--------------|
| A5 latency | Not publicly published | Current public VPTO timing material |
| Steady-state throughput | Not publicly published | Current public VPTO timing material |

If software scheduling or performance modeling depends on the exact cost of `pto.vcgadd`, treat that cost as target-profile-specific and measure it on the concrete backend rather than inferring a manual constant.

## Examples

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

## Detailed Notes

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

## Related Ops / Instruction Set Links

- Instruction set overview: [Reduction Instructions](../../reduction-ops.md)
- Previous op in instruction set: [pto.vcmin](./vcmin.md)
- Next op in instruction set: [pto.vcgmax](./vcgmax.md)
