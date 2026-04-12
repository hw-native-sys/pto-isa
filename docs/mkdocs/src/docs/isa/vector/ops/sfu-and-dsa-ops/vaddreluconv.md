<!-- Generated from `docs/isa/vector/ops/sfu-and-dsa-ops/vaddreluconv.md` -->

# pto.vaddreluconv

Standalone reference page for `pto.vaddreluconv`. This page belongs to the [SFU And DSA Ops](../../sfu-and-dsa-ops.md) family in the PTO ISA manual.

## Summary

Fused add + ReLU + type conversion (HW fusion).

## Mechanism

`pto.vaddreluconv` is a specialized `pto.v*` operation. It exposes fused, widening, or domain-specific hardware behavior through one stable virtual mnemonic so the family can be reasoned about at the ISA level.

## Syntax

```mlir
%result = pto.vaddreluconv %lhs, %rhs : !pto.vreg<NxT0>, !pto.vreg<NxT0> -> !pto.vreg<MxT1>
```

## Inputs

`%lhs` and `%rhs` are the source vectors.

## Expected Outputs

`%result` is the fused add/ReLU/convert result.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

Only backend-supported source/destination
  type pairs are legal. Rounding, saturation, and packing rules follow the
  semantics of this fused operation, not an arbitrary sequence of standalone
  ops.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected family or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```c
// f32→f16 variant:
for (int i = 0; i < 64; i++)
    dst_f16[i] = f32_to_f16(max(src0_f32[i] + src1_f32[i], 0));

// f16→i8 variant:
for (int i = 0; i < 128; i++)
    dst_i8[i] = f16_to_i8(max(src0_f16[i] + src1_f16[i], 0));
```

## Detailed Notes

```c
// f32→f16 variant:
for (int i = 0; i < 64; i++)
    dst_f16[i] = f32_to_f16(max(src0_f32[i] + src1_f32[i], 0));

// f16→i8 variant:
for (int i = 0; i < 128; i++)
    dst_i8[i] = f16_to_i8(max(src0_f16[i] + src1_f16[i], 0));
```

## Related Ops / Family Links

- Family overview: [SFU And DSA Ops](../../sfu-and-dsa-ops.md)
- Previous op in family: [pto.vaxpy](./vaxpy.md)
- Next op in family: [pto.vmulconv](./vmulconv.md)
