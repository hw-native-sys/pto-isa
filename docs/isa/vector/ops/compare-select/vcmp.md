# pto.vcmp

`pto.vcmp` is part of the [Compare And Select](../../compare-select.md) instruction set.

## Summary

Element-wise comparison, output predicate mask.

## Mechanism

`pto.vcmp` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the instruction set operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%result = pto.vcmp %src0, %src1, %seed, "CMP_MODE" : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.mask
```

## Inputs

`%src0`, `%src1`, and `%seed`; `CMP_MODE` selects the comparison
  predicate.

## Expected Outputs

`%result` is the generated predicate mask.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

Only lanes enabled by `%seed` participate.
  Integer and floating-point comparisons follow their own element-type-specific
  comparison rules.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on an instruction-set-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Performance

### Timing Disclosure

The current public VPTO timing material for PTO micro instructions remains limited.
For `pto.vcmp`, those public sources describe the instruction semantics, operand legality, and pipeline placement, but they do **not** publish a numeric latency or steady-state throughput.

| Metric | Status | Source Basis |
|--------|--------|--------------|
| A5 latency | Not publicly published | Current public VPTO timing material |
| Steady-state throughput | Not publicly published | Current public VPTO timing material |

If software scheduling or performance modeling depends on the exact cost of `pto.vcmp`, treat that cost as target-profile-specific and measure it on the concrete backend rather than inferring a manual constant.

## Examples

```c
for (int i = 0; i < N; i++)
    if (seed[i])
        dst[i] = (src0[i] CMP src1[i]) ? 1 : 0;
```

```mlir
%all_active = pto.pset_b32 "PAT_ALL" : !pto.mask
%lt_mask = pto.vcmp %a, %b, %all_active, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.mask
// lt_mask[i] = 1 if a[i] < b[i]
```

## Detailed Notes

```c
for (int i = 0; i < N; i++)
    if (seed[i])
        dst[i] = (src0[i] CMP src1[i]) ? 1 : 0;
```

**Compare modes:**

| Mode | Operation |
|------|-----------|
| `eq` | Equal (==) |
| `ne` | Not equal (!=) |
| `lt` | Less than (<) |
| `le` | Less than or equal (<=) |
| `gt` | Greater than (>) |
| `ge` | Greater than or equal (>=) |

**Example:**
```mlir
%all_active = pto.pset_b32 "PAT_ALL" : !pto.mask
%lt_mask = pto.vcmp %a, %b, %all_active, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.mask
// lt_mask[i] = 1 if a[i] < b[i]
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Compare And Select](../../compare-select.md)
- Next op in instruction set: [pto.vcmps](./vcmps.md)
