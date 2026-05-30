# pto.pdintlv_b8

`pto.pdintlv_b8` is part of the [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md) instruction set.

## Summary

Deinterleave two predicate sources and materialize the lower and higher result halves as two predicate outputs.

## Mechanism

The installed 3510 Bisheng CCE header exposes `pdintlv_b8` as a four-operand, two-result helper:

- `void pdintlv_b8(vector_bool &dst0, vector_bool &dst1, vector_bool src0, vector_bool src1);`

The public call surface therefore models `pto.pdintlv_b8` as a paired-result operation. `dst0` receives the lower deinterleaved half and `dst1` receives the upper deinterleaved half produced from `src0` and `src1`.

## Syntax

### PTO Assembly Form

```mlir
%dst0, %dst1 = pto.pdintlv_b8 %src0, %src1 : !pto.mask<b8>, !pto.mask<b8> -> !pto.mask<b8>, !pto.mask<b8>
```

### AS Level 1 (SSA)

```mlir
%dst0, %dst1 = pto.pdintlv_b8 %src0, %src1 : !pto.mask<b8>, !pto.mask<b8> -> !pto.mask<b8>, !pto.mask<b8>
```

### AS Level 2 (DPS)

```mlir
pto.pdintlv_b8 ins(%src0, %src1 : !pto.mask<G>, !pto.mask<G>) outs(%dst0, %dst1 : !pto.mask<G>, !pto.mask<G>)
```

## C++ Intrinsic

```cpp
vector_bool dst0;
vector_bool dst1;
vector_bool src0;
vector_bool src1;
pdintlv_b8(dst0, dst1, src0, src1);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%src0` | `!pto.mask<G>` | First predicate source |
| `%src1` | `!pto.mask<G>` | Second predicate source |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%dst0` | `!pto.mask<G>` | Lower result half returned by the deinterleave helper |
| `%dst1` | `!pto.mask<G>` | Upper result half returned by the deinterleave helper |

## Side Effects

None.

## Constraints

!!! warning "Constraints"
    - The installed public CCE helper for `pdintlv_b8` returns two predicate results, not a single input-to-two-output split from one predicate operand.
    - Source and destination predicate widths must match the `_b8` variant selected by the instruction.

## Exceptions

!!! danger "Exceptions"
    - Illegal if the selected target profile does not support the requested predicate-deinterleave form.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    | Aspect | CPU Sim | A2/A3 | A5 |
    |--------|:-------:|:------:|:--:|
    | Predicate deinterleave helper | Simulated | Supported | Supported |
    | Public two-result CCE surface | Emulated | Supported | Supported |

## Examples

### C++ usage

```cpp
vector_bool dst0;
vector_bool dst1;
vector_bool src0;
vector_bool src1;
pdintlv_b8(dst0, dst1, src0, src1);
```

### SSA form

```mlir
%dst0, %dst1 = pto.pdintlv_b8 %src0, %src1 : !pto.mask<b8>, !pto.mask<b8> -> !pto.mask<b8>, !pto.mask<b8>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md)
- Other variants: [pto.pdintlv_b16](./pdintlv-b16.md), [pto.pdintlv_b32](./pdintlv-b32.md)
- Inverse: [pto.pintlv_b8](./pintlv-b8.md)
- Previous op in instruction set: [pto.psel](./psel.md)
- Next op in instruction set: [pto.pdintlv_b16](./pdintlv-b16.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
