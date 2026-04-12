<!-- Generated from `docs/isa/vector/ops/predicate-and-materialization/vbr.md` -->

# pto.vbr

Standalone reference page for `pto.vbr`. This page belongs to the [Predicate And Materialization](../../predicate-and-materialization.md) family in the PTO ISA manual.

## Summary

Broadcast scalar to all vector lanes.

## Mechanism

`pto.vbr` materializes scalar or selected-lane state into a vector register. The architectural result is a new vector-register value, so the operation stays in the `pto.v*` surface even when its input is scalar.

## Syntax

```mlir
%result = pto.vbr %value : T -> !pto.vreg<NxT>
```

## Inputs

`%value` is the scalar source.

## Expected Outputs

`%result` is a vector whose active lanes all carry `%value`.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

Supported forms are `b8`, `b16`, and `b32`. For `b8`, only the low 8 bits of
  the scalar source are consumed.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected family or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```c
for (int i = 0; i < N; i++)
    dst[i] = value;
```

```mlir
%one = pto.vbr %c1_f32 : f32 -> !pto.vreg<64xf32>
```

## Detailed Notes

```c
for (int i = 0; i < N; i++)
    dst[i] = value;
```

**Example:**
```mlir
%one = pto.vbr %c1_f32 : f32 -> !pto.vreg<64xf32>
```

## Related Ops / Family Links

- Family overview: [Predicate And Materialization](../../predicate-and-materialization.md)
- Next op in family: [pto.vdup](./vdup.md)
