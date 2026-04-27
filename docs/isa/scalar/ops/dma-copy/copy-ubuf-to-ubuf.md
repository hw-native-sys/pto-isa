# pto.copy_ubuf_to_ubuf

`pto.copy_ubuf_to_ubuf` is part of the [DMA Copy](../../dma-copy.md) instruction set.

## Summary

Execute a DMA transfer between two Unified Buffer regions.

## Mechanism

The DMA engine reads `%n_burst` rows from `%source` and writes them to `%dest`. `%len_burst` controls the contiguous byte count copied per row, while `%src_stride` and `%dst_stride` specify the row-to-row start offsets for the copy. This form is useful when the producer and consumer both operate in UB space but a DMA-style row copy is still preferred over vector payload instructions.

## Syntax

### PTO Assembly Form

```text
copy_ubuf_to_ubuf %source, %dest, %sid, %n_burst, %len_burst, %src_stride, %dst_stride
```

### AS Level 1 (SSA)

```mlir
pto.copy_ubuf_to_ubuf %source, %dest, %sid, %n_burst, %len_burst, %src_stride, %dst_stride : !pto.ptr<T, ub>, !pto.ptr<T, ub>, i64, i64, i64, i64, i64
```

## Inputs

| Operand | Type | Description |
| --- | --- | --- |
| %source | `!pto.ptr<T, ub>` | UB source pointer |
| %dest | `!pto.ptr<T, ub>` | UB destination pointer |
| %sid | `i64` | DMA stream identifier |
| %n_burst | `i64` | Number of burst rows to transfer |
| %len_burst | `i64` | Contiguous byte count transferred per row |
| %src_stride | `i64` | UB source row-to-row start offset in bytes |
| %dst_stride | `i64` | UB destination row-to-row start offset in bytes |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| None | `—` | This form does not return SSA values; it writes data into Unified Buffer memory. |

## Side Effects

Reads UB-visible storage, writes UB-visible storage, and consumes the active UB DMA state for the selected target profile.

## Constraints

- Source and destination regions MUST both satisfy the UB alignment rules of the selected target profile.
- `%len_burst` MUST fit within both the source and destination row stride.
- If source and destination regions alias, portable code MUST provide ordering that avoids undefined behavior. On A2/A3 and A5: if source and destination regions alias, the copy may proceed in either forward or backward direction and the exact order is not guaranteed; the programmer must ensure the copy is sequenced to avoid data hazard. On CPU simulator: aliasing is resolved by copying to a temporary buffer first.

## Exceptions

- The verifier rejects illegal operand shapes, unsupported pipe or event identifiers, and attribute combinations that are not valid for the selected instruction set or target profile.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- CPU simulation preserves the visible copy contract but may not expose all DMA overlap hazards.
- A2/A3 and A5 may narrow supported element sizes, row widths, or overlap behavior.

## Examples

```mlir
pto.copy_ubuf_to_ubuf %source, %dest, %sid, %n_burst, %len_burst, %src_stride, %dst_stride : !pto.ptr<f32, ub>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64
```

## Related Ops / Instruction Set Links

- Instruction set overview: [DMA Copy](../../dma-copy.md)
- Previous op in instruction set: [pto.copy_ubuf_to_gm](./copy-ubuf-to-gm.md)
- Next op in instruction set: (none)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
