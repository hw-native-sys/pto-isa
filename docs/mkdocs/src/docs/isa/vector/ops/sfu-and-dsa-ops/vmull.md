<!-- Generated from `docs/isa/vector/ops/sfu-and-dsa-ops/vmull.md` -->

# pto.vmull

Standalone reference page for `pto.vmull`. This page belongs to the [SFU And DSA Ops](../../sfu-and-dsa-ops.md) family in the PTO ISA manual.

## Summary

Widening multiply with high/low results.

## Mechanism

`pto.vmull` is a specialized `pto.v*` operation. It exposes fused, widening, or domain-specific hardware behavior through one stable virtual mnemonic so the family can be reasoned about at the ISA level.

## Syntax

```mlir
%low, %high = pto.vmull %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>, !pto.vreg<NxT>
```

Documented A5 types or forms: `i32/u32 (native 32×32→64 widening multiply)`.

## Inputs

`%lhs` and `%rhs` are the source vectors and `%mask` selects
  active lanes.

## Expected Outputs

`%low` and `%high` expose the widened-product low/high parts.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

The current documented A5 form is the native
  widening 32x32->64 integer multiply family.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected family or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- Documented A5 coverage: `i32/u32 (native 32×32→64 widening multiply)`.
- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```c
for (int i = 0; i < 64; i++) {
    int64_t r = (int64_t)src0_i32[i] * (int64_t)src1_i32[i];
    dst_lo[i] = (int32_t)(r & 0xFFFFFFFF);
    dst_hi[i] = (int32_t)(r >> 32);
}
```

## Detailed Notes

```c
for (int i = 0; i < 64; i++) {
    int64_t r = (int64_t)src0_i32[i] * (int64_t)src1_i32[i];
    dst_lo[i] = (int32_t)(r & 0xFFFFFFFF);
    dst_hi[i] = (int32_t)(r >> 32);
}
```

## Related Ops / Family Links

- Family overview: [SFU And DSA Ops](../../sfu-and-dsa-ops.md)
- Previous op in family: [pto.vmulconv](./vmulconv.md)
- Next op in family: [pto.vmula](./vmula.md)
