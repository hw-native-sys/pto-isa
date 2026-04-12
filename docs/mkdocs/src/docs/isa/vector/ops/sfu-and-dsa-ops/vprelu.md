<!-- Generated from `docs/isa/vector/ops/sfu-and-dsa-ops/vprelu.md` -->

# pto.vprelu

Standalone reference page for `pto.vprelu`. This page belongs to the [SFU And DSA Ops](../../sfu-and-dsa-ops.md) family in the PTO ISA manual.

## Summary

Parametric ReLU with per-element alpha vector.

## Mechanism

`pto.vprelu` is a specialized `pto.v*` operation. It exposes fused, widening, or domain-specific hardware behavior through one stable virtual mnemonic so the family can be reasoned about at the ISA level.

## Syntax

```mlir
%result = pto.vprelu %input, %alpha : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>
```

Documented A5 types or forms: `f16, f32`.

## Inputs

`%input` is the activation vector and `%alpha` is the per-element
  slope vector.

## Expected Outputs

`%result` is the parametric-ReLU vector.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

Floating-point element types only on the
  current A5 surface.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected family or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- Documented A5 coverage: `f16, f32`.
- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] >= 0) ? src[i] : alpha[i] * src[i];
```

## Detailed Notes

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] >= 0) ? src[i] : alpha[i] * src[i];
```

## Related Ops / Family Links

- Family overview: [SFU And DSA Ops](../../sfu-and-dsa-ops.md)
- Next op in family: [pto.vexpdiff](./vexpdiff.md)
