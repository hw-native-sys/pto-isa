<!-- Generated from `docs/isa/vector/ops/predicate-and-materialization/vdup.md` -->

# pto.vdup

Standalone reference page for `pto.vdup`. This page belongs to the [Predicate And Materialization](../../predicate-and-materialization.md) family in the PTO ISA manual.

## Summary

Duplicate scalar or vector element to all lanes.

## Mechanism

`pto.vdup` materializes scalar or selected-lane state into a vector register. The architectural result is a new vector-register value, so the operation stays in the `pto.v*` surface even when its input is scalar.

## Syntax

```mlir
%result = pto.vdup %input {position = "POSITION"} : T|!pto.vreg<NxT> -> !pto.vreg<NxT>
```

## Inputs

`%input` supplies the scalar or source-lane value selected by `position`.

## Expected Outputs

`%result` is the duplicated vector.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

`position` selects which source element or scalar position is duplicated. The
  current PTO ISA vector surface representation models that selector as an attribute rather than a
  separate operand.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected family or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```c
for (int i = 0; i < N; i++)
    dst[i] = input_scalar_or_element;
```

## Detailed Notes

```c
for (int i = 0; i < N; i++)
    dst[i] = input_scalar_or_element;
```

## Predicate Generation

## Related Ops / Family Links

- Family overview: [Predicate And Materialization](../../predicate-and-materialization.md)
- Previous op in family: [pto.vbr](./vbr.md)
