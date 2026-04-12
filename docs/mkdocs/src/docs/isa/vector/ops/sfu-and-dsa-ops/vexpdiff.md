<!-- Generated from `docs/isa/vector/ops/sfu-and-dsa-ops/vexpdiff.md` -->

# pto.vexpdiff

Standalone reference page for `pto.vexpdiff`. This page belongs to the [SFU And DSA Ops](../../sfu-and-dsa-ops.md) family in the PTO ISA manual.

## Summary

Fused exp(x - max) for numerically stable softmax.

## Mechanism

`pto.vexpdiff` is a specialized `pto.v*` operation. It exposes fused, widening, or domain-specific hardware behavior through one stable virtual mnemonic so the family can be reasoned about at the ISA level.

## Syntax

```mlir
%result = pto.vexpdiff %input, %max : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>
```

Documented A5 types or forms: `f16, f32`.

## Inputs

`%input` is the source vector and `%max` is the broadcasted
  subtraction term.

## Expected Outputs

`%result` is the fused `exp(input - max)` vector.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

Floating-point element types only.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected family or target profile.
- Target-defined numeric exceptional behavior, such as divide-by-zero or out-of-domain inputs, remains subject to the selected backend profile unless this page narrows it further.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- Documented A5 coverage: `f16, f32`.
- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```c
for (int i = 0; i < N; i++)
    dst[i] = expf(src[i] - max[i]);
```

## Detailed Notes

```c
for (int i = 0; i < N; i++)
    dst[i] = expf(src[i] - max[i]);
```

**Use case:** Softmax numerator computation with numerical stability.

## Fused Compute+Convert Ops

## Related Ops / Family Links

- Family overview: [SFU And DSA Ops](../../sfu-and-dsa-ops.md)
- Previous op in family: [pto.vprelu](./vprelu.md)
- Next op in family: [pto.vaddrelu](./vaddrelu.md)
