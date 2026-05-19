# pto.vbitcast

`pto.vbitcast` is part of the [Conversion Ops](../../conversion-ops.md) instruction set.

## Summary

`pto.vbitcast` performs a bitwise reinterpretation of a `!pto.vreg<...>` value without changing the underlying bit pattern. The total bit width is preserved (always 2048 bits for a VPTO `vreg`), so only the element type and lane count interpretation change.

Unlike [pto.vcvt](./vcvt.md), `pto.vbitcast` does not round, saturate, or rescale any value — every bit of the source register is copied unchanged into the result register.

## Mechanism

The op is a pure type cast at the vector-register level. No payload bytes are modified; only the surrounding VPTO IR's interpretation of the register changes. This makes type punning between integer and floating-point families explicit in SSA form, instead of being inferred from hidden hardware state.

## Syntax

```mlir
%result = pto.vbitcast %input : !pto.vreg<NxT0> -> !pto.vreg<MxT1>
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%input` | `!pto.vreg<NxT0>` | Source vector register. |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%result` | `!pto.vreg<MxT1>` | Destination vector register with the same bit pattern, reinterpreted as `MxT1`. |

## Side Effects

`pto.vbitcast` has no architectural side effects beyond producing its SSA result. It does not implicitly reserve buffers, signal events, or establish memory fences.

## Constraints

!!! warning "Constraints"
    - Both source and result must be `!pto.vreg<...>` types.
    - Source and result vectors must have the same total bit width (currently 2048 bits): `N * bitwidth(T0) = M * bitwidth(T1) = 2048`.
    - Only integer and floating-point element types are supported.

**Element-bit-width equality examples:**

- `f32<64>` → `i32<64>` (both 32-bit elements, total 2048 bits)
- `f16<128>` → `i16<128>` (both 16-bit elements, total 2048 bits)
- `bf16<128>` → `ui16<128>` (both 16-bit elements, total 2048 bits)
- `si32<64>` → `ui32<64>` (both 32-bit elements, total 2048 bits)
- `f32<64>` → `i16<128>` (32-bit and 16-bit elements, total 2048 bits)

The verifier rejects shapes for which the source and destination total bit widths differ.

## Comparison with `pto.vcvt`

| Aspect | `pto.vcvt` | `pto.vbitcast` |
|--------|------------|----------------|
| Bit pattern | May change (rounding, saturation, sign extension) | Preserved exactly |
| Lane count | May change with documented type-pair rules | May change as long as total bit width stays 2048 |
| Rounding / saturation attributes | Supported (`rnd`, `sat`, `part`) | None |
| Predicate operand | Required (`%mask`) | None — bitcast is unconditional |

## Examples

### Reinterpret float as integer for bit manipulation

```mlir
// Prepare a vector of float values
%fvec = pto.vlds %ub[%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

// Reinterpret as integer for bitwise operations
%ivec = pto.vbitcast %fvec : !pto.vreg<64xf32> -> !pto.vreg<64xi32>

// Extract sign bit (bit 31)
%sign_bits = pto.vand %ivec, %sign_mask, %mask
    : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask<b32> -> !pto.vreg<64xi32>

// Reinterpret back to float
%fvec_without_sign = pto.vbitcast %sign_bits : !pto.vreg<64xi32> -> !pto.vreg<64xf32>
```

### Type punning between signed and unsigned integer

```mlir
%signed = pto.vlds %ub[%lane] : !pto.ptr<si32, ub> -> !pto.vreg<64xsi32>
%unsigned = pto.vbitcast %signed : !pto.vreg<64xsi32> -> !pto.vreg<64xui32>
// Bits are identical; interpretation changes from signed to unsigned
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Conversion Ops](../../conversion-ops.md)
- Value-changing conversion: [pto.vcvt](./vcvt.md)
- Mask-side bitcast: [pto.pbitcast](./pbitcast.md)
