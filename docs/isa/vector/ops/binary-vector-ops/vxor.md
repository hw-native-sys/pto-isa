# pto.vxor

`pto.vxor` is part of the [Binary Vector Instructions](../../binary-vector-ops.md) instruction set.

## Summary

`%result` is the lane-wise bitwise XOR.

## Mechanism

`pto.vxor` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the instruction set operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%result = pto.vxor %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>
```

Documented A5 types or forms: `all integer types`.

## Inputs

`%lhs`, `%rhs`, and `%mask` as above.

## Expected Outputs

`%result` is the lane-wise bitwise XOR.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

Integer element types only.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- Documented A5 coverage: `all integer types`.
- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on an instruction-set-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Performance

### Timing Disclosure

The current public VPTO timing material for PTO micro instructions remains limited.
For `pto.vxor`, those public sources describe the instruction semantics, operand legality, and pipeline placement, but they do **not** publish a numeric latency or steady-state throughput.

| Metric | Status | Source Basis |
|--------|--------|--------------|
| A5 latency | Not publicly published | Current public VPTO timing material |
| Steady-state throughput | Not publicly published | Current public VPTO timing material |

If software scheduling or performance modeling depends on the exact cost of `pto.vxor`, treat that cost as target-profile-specific and measure it on the concrete backend rather than inferring a manual constant.

## Examples

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] ^ src1[i];
```

## Detailed Notes

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] ^ src1[i];
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Binary Vector Instructions](../../binary-vector-ops.md)
- Previous op in instruction set: [pto.vor](./vor.md)
- Next op in instruction set: [pto.vshl](./vshl.md)
