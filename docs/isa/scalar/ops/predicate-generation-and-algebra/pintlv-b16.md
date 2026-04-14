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
%dst0, %dst1 = pto.pintlv_b16 %src0, %src1 : !pto.mask, !pto.mask -> !pto.mask, !pto.mask
```

### AS Level 1 (SSA)

```mlir
%dst0, %dst1 = pto.pintlv_b16 %src0, %src1 : !pto.mask, !pto.mask -> !pto.mask, !pto.mask
```

### AS Level 2 (DPS)

```mlir
pto.pintlv_b16 ins(%src0, %src1 : !pto.mask, !pto.mask) outs(%dst0, %dst1 : !pto.mask, !pto.mask)
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
| `%src0` | `!pto.mask` | First predicate source |
| `%src1` | `!pto.mask` | Second predicate source |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%dst0` | `!pto.mask` | Lower result half returned by the interleave helper |
| `%dst1` | `!pto.mask` | Upper result half returned by the interleave helper |

## Side Effects

None.

## Constraints

- The installed public CCE helper for `pintlv_b16` returns two predicate results, not a single concatenated predicate value.
- Source and destination predicate widths must match the `_b16` variant selected by the instruction.

## Exceptions

- Illegal if the selected target profile does not support the requested predicate-interleave form.

## Target-Profile Restrictions

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
%dst0, %dst1 = pto.pintlv_b16 %src0, %src1 : !pto.mask, !pto.mask -> !pto.mask, !pto.mask
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md)
- Previous op in instruction set: [pto.pdintlv_b8](./pdintlv-b8.md)
- Next op in instruction set: (none - last in instruction set)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
