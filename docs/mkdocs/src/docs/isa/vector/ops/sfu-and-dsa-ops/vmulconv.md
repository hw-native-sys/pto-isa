<!-- Generated from `docs/isa/vector/ops/sfu-and-dsa-ops/vmulconv.md` -->

# pto.vmulconv

Standalone reference page for `pto.vmulconv`. This page belongs to the [SFU And DSA Ops](../../sfu-and-dsa-ops.md) family in the PTO ISA manual.

## Summary

Fused mul + type conversion (HW fusion).

## Mechanism

`pto.vmulconv` is a specialized `pto.v*` operation. It exposes fused, widening, or domain-specific hardware behavior through one stable virtual mnemonic so the family can be reasoned about at the ISA level.

## Syntax

```mlir
%result = pto.vmulconv %lhs, %rhs : !pto.vreg<NxT0>, !pto.vreg<NxT0> -> !pto.vreg<MxT1>
```

## Inputs

`%lhs` and `%rhs` are the source vectors.

## Expected Outputs

`%result` is the fused mul/convert result.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

Only backend-supported source/destination
  type pairs are legal.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected family or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```c
// f16→i8 variant:
for (int i = 0; i < 128; i++)
    dst_i8[i] = f16_to_i8(src0_f16[i] * src1_f16[i]);
```

## Detailed Notes

```c
// f16→i8 variant:
for (int i = 0; i < 128; i++)
    dst_i8[i] = f16_to_i8(src0_f16[i] * src1_f16[i]);
```

## Extended Arithmetic

## Related Ops / Family Links

- Family overview: [SFU And DSA Ops](../../sfu-and-dsa-ops.md)
- Previous op in family: [pto.vaddreluconv](./vaddreluconv.md)
- Next op in family: [pto.vmull](./vmull.md)
