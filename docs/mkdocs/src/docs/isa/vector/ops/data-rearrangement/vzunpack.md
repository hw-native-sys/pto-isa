<!-- Generated from `docs/isa/vector/ops/data-rearrangement/vzunpack.md` -->

# pto.vzunpack

Standalone reference page for `pto.vzunpack`. This page belongs to the [Data Rearrangement](../../data-rearrangement.md) family in the PTO ISA manual.

## Summary

Zero-extending unpack — narrow to wide (half).

## Mechanism

`pto.vzunpack` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the family operand model, and returns its results in vector-register or mask form.

## Syntax

```mlir
%result = pto.vzunpack %src, %part : !pto.vreg<NxT_narrow>, index -> !pto.vreg<N/2xT_wide>
```

## Inputs

`%src` is the packed narrow vector and `%part` selects which half
  is unpacked.

## Expected Outputs

`%result` is the widened vector.

## Side Effects

This operation has no architectural side effect beyond producing its SSA results. It does not implicitly reserve buffers, signal events, or establish memory fences unless the form says so.

## Constraints

This is the zero-extending unpack family.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected family or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on a family-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Examples

```c
for (int i = 0; i < N/2; i++)
    dst[i] = zero_extend(src[part_offset + i]);
```

```mlir
// AoS → SoA conversion using deinterleave
%even, %odd = pto.vdintlv %interleaved0, %interleaved1
    : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

// Filter: keep only elements passing condition
%pass_mask = pto.vcmps %values, %threshold, %all, "gt"
    : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.mask
%compacted = pto.vsqz %values, %pass_mask
    : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// Sliding window sum
%prev_window = pto.vslide %curr, %prev, %c1
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, i16 -> !pto.vreg<64xf32>
%window_sum = pto.vadd %curr, %prev_window, %all
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// Type narrowing via pack
%packed_i16 = pto.vpack %wide0_i32, %wide1_i32, %c0
    : !pto.vreg<64xi32>, !pto.vreg<64xi32>, index -> !pto.vreg<128xi16>
```

## Detailed Notes

```c
for (int i = 0; i < N/2; i++)
    dst[i] = zero_extend(src[part_offset + i]);
```

## Typical Usage

```mlir
// AoS → SoA conversion using deinterleave
%even, %odd = pto.vdintlv %interleaved0, %interleaved1
    : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

// Filter: keep only elements passing condition
%pass_mask = pto.vcmps %values, %threshold, %all, "gt"
    : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.mask
%compacted = pto.vsqz %values, %pass_mask
    : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// Sliding window sum
%prev_window = pto.vslide %curr, %prev, %c1
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, i16 -> !pto.vreg<64xf32>
%window_sum = pto.vadd %curr, %prev_window, %all
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// Type narrowing via pack
%packed_i16 = pto.vpack %wide0_i32, %wide1_i32, %c0
    : !pto.vreg<64xi32>, !pto.vreg<64xi32>, index -> !pto.vreg<128xi16>
```

## V2 Interleave Forms

## Related Ops / Family Links

- Family overview: [Data Rearrangement](../../data-rearrangement.md)
- Previous op in family: [pto.vsunpack](./vsunpack.md)
- Next op in family: [pto.vintlvv2](./vintlvv2.md)
