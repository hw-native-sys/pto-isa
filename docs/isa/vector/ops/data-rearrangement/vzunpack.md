# pto.vzunpack

`pto.vzunpack` is part of the [Data Rearrangement](../../data-rearrangement.md) instruction set.

## Summary

Zero-extending unpack — narrow to wide (half).

## Mechanism

`pto.vzunpack` is a `pto.v*` compute operation. It applies its semantics to active lanes, obeys the instruction set operand model, and returns its results in vector-register or mask form.

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

This is the zero-extending unpack instruction set.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported element types, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- A5 is the most detailed concrete profile in the current manual; CPU simulation and A2/A3-class targets may support narrower subsets or emulate the behavior while preserving the visible PTO contract.
- Code that depends on an instruction-set-specific type list, distribution mode, or fused form should treat that dependency as target-profile-specific unless the PTO manual states cross-target portability explicitly.

## Performance

### Timing Disclosure

The timing sources currently used for PTO micro-instruction pages are `~/visa.txt` and `PTOAS/docs/vpto-spec.md` on the latest fetched `feature_vpto_backend` branch.
For `pto.vzunpack`, those public sources describe the instruction semantics, operand legality, and pipeline placement, but they do **not** publish a numeric latency or steady-state throughput.

| Metric | Status | Source Basis |
|--------|--------|--------------|
| A5 latency | Not publicly published | `visa.txt`, `PTOAS/docs/vpto-spec.md` |
| Steady-state throughput | Not publicly published | `visa.txt`, `PTOAS/docs/vpto-spec.md` |

If software scheduling or performance modeling depends on the exact cost of `pto.vzunpack`, treat that cost as target-profile-specific and measure it on the concrete backend rather than inferring a manual constant.

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

## Related Ops / Instruction Set Links

- Instruction set overview: [Data Rearrangement](../../data-rearrangement.md)
- Previous op in instruction set: [pto.vsunpack](./vsunpack.md)
- Next op in instruction set: [pto.vintlvv2](./vintlvv2.md)
