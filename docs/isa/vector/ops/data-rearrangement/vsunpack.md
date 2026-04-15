# pto.vsunpack

`pto.vsunpack` is part of the [Data Rearrangement](../../data-rearrangement.md) instruction set.

## Summary

Sign-extending unpack — narrow to wide (half).

## Mechanism

`pto.vsunpack` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the instruction set operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%result = pto.vsunpack %src, %part : !pto.vreg<NxT_narrow>, index -> !pto.vreg<N/2xT_wide>
```

## Inputs

`%src` is the packed narrow vector and `%part` selects which half
  is unpacked.

## Expected Outputs

`%result` is the widened vector.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

This is the sign-extending unpack instruction set.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on an instruction-set-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Performance

### Timing Disclosure

The current public VPTO timing material for PTO micro instructions remains limited.
For `pto.vsunpack`, those public sources describe the instruction semantics, operand legality, and pipeline placement, but they do **not** publish a numeric latency or steady-state throughput.

| Metric | Status | Source Basis |
|--------|--------|--------------|
| A5 latency | Not publicly published | Current public VPTO timing material |
| Steady-state throughput | Not publicly published | Current public VPTO timing material |

If software scheduling or performance modeling depends on the exact cost of `pto.vsunpack`, treat that cost as target-profile-specific and measure it on the concrete backend rather than inferring a manual constant.

## Examples

```c
// e.g., vreg<128xi16> → vreg<64xi32> (one half)
for (int i = 0; i < N/2; i++)
    dst[i] = sign_extend(src[part_offset + i]);
```

## Detailed Notes

```c
// e.g., vreg<128xi16> → vreg<64xi32> (one half)
for (int i = 0; i < N/2; i++)
    dst[i] = sign_extend(src[part_offset + i]);
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Data Rearrangement](../../data-rearrangement.md)
- Previous op in instruction set: [pto.vpack](./vpack.md)
- Next op in instruction set: [pto.vzunpack](./vzunpack.md)
