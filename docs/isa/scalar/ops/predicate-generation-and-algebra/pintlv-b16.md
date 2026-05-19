# pto.pintlv_b16

`pto.pintlv_b16` is part of the [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md) instruction set.

## Summary

Interleave two predicate sources and materialize the lower and higher result halves as two predicate outputs.

## Mechanism

The installed 3510 Bisheng CCE header exposes `pintlv_b16` as a four-operand, two-result helper:

- `void pintlv_b16(vector_bool &dst0, vector_bool &dst1, vector_bool src0, vector_bool src1);`

The public call surface therefore models `pto.pintlv_b16` as a paired-result operation. `dst0` receives the lower interleaved half and `dst1` receives the upper interleaved half produced from `src0` and `src1`.

## Syntax

### PTO Assembly Form

```mlir
%dst0, %dst1 = pto.pintlv_b16 %src0, %src1 : !pto.mask<b16>, !pto.mask<b16> -> !pto.mask<b16>, !pto.mask<b16>
```

### AS Level 1 (SSA)

```mlir
%dst0, %dst1 = pto.pintlv_b16 %src0, %src1 : !pto.mask<b16>, !pto.mask<b16> -> !pto.mask<b16>, !pto.mask<b16>
```

### AS Level 2 (DPS)

```mlir
pto.pintlv_b16 ins(%src0, %src1 : !pto.mask<G>, !pto.mask<G>) outs(%dst0, %dst1 : !pto.mask<G>, !pto.mask<G>)
```

## C++ Intrinsic

```cpp
vector_bool dst0;
vector_bool dst1;
vector_bool src0;
vector_bool src1;
pintlv_b16(dst0, dst1, src0, src1);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%src0` | `!pto.mask<G>` | First predicate source |
| `%src1` | `!pto.mask<G>` | Second predicate source |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%dst0` | `!pto.mask<G>` | Lower result half returned by the interleave helper |
| `%dst1` | `!pto.mask<G>` | Upper result half returned by the interleave helper |

## Side Effects

None.

## Constraints

!!! warning "Constraints"
    - The installed public CCE helper for `pintlv_b16` returns two predicate results, not a single concatenated predicate value.
    - Source and destination predicate widths must match the `_b16` variant selected by the instruction.

## Exceptions

!!! danger "Exceptions"
    - Illegal if the selected target profile does not support the requested predicate-interleave form.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    | Aspect | CPU Sim | A2/A3 | A5 |
    |--------|:-------:|:------:|:--:|
    | Predicate interleave helper | Simulated | Supported | Supported |
    | Public two-result CCE surface | Emulated | Supported | Supported |

## Examples

### C++ usage

```cpp
vector_bool dst0;
vector_bool dst1;
vector_bool src0;
vector_bool src1;
pintlv_b16(dst0, dst1, src0, src1);
```

### SSA form

```mlir
%dst0, %dst1 = pto.pintlv_b16 %src0, %src1 : !pto.mask<b16>, !pto.mask<b16> -> !pto.mask<b16>, !pto.mask<b16>
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md)
- Other variants: [pto.pintlv_b8](./pintlv-b8.md), [pto.pintlv_b32](./pintlv-b32.md)
- Inverse: [pto.pdintlv_b16](./pdintlv-b16.md)
- Previous op in instruction set: [pto.pintlv_b8](./pintlv-b8.md)
- Next op in instruction set: [pto.pintlv_b32](./pintlv-b32.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
