# pto.vdintlv

`pto.vdintlv` is part of the [Data Rearrangement](../../data-rearrangement.md) instruction set.

## Summary

Deinterleave elements into even/odd.

## Mechanism

`pto.vdintlv` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the instruction set operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%low, %high = pto.vdintlv %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>, !pto.vreg<NxT>
```

## Inputs

`%lhs` and `%rhs` represent the interleaved source stream in the
  current PTO ISA vector instructions representation.

## Expected Outputs

`%low` and `%high` are the separated destination vectors.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

The two outputs form the even/odd
  deinterleave result pair, and their ordering MUST be preserved.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on an instruction-set-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Performance

### Timing Disclosure

The current public VPTO timing material for PTO micro instructions remains limited.
For `pto.vdintlv`, those public sources describe the instruction semantics, operand legality, and pipeline placement, but they do **not** publish a numeric latency or steady-state throughput.

| Metric | Status | Source Basis |
|--------|--------|--------------|
| A5 latency | Not publicly published | Current public VPTO timing material |
| Steady-state throughput | Not publicly published | Current public VPTO timing material |

If software scheduling or performance modeling depends on the exact cost of `pto.vdintlv`, treat that cost as target-profile-specific and measure it on the concrete backend rather than inferring a manual constant.

## Examples

```c
// Deinterleave: separate even/odd elements
// low  = {src0[0], src0[2], src0[4], ...}  // even
// high = {src0[1], src0[3], src0[5], ...}  // odd
```

## Detailed Notes

```c
// Deinterleave: separate even/odd elements
// low  = {src0[0], src0[2], src0[4], ...}  // even
// high = {src0[1], src0[3], src0[5], ...}  // odd
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Data Rearrangement](../../data-rearrangement.md)
- Previous op in instruction set: [pto.vintlv](./vintlv.md)
- Next op in instruction set: [pto.vslide](./vslide.md)
