# pto.plt_b16

`pto.plt_b16` is part of the [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md) instruction set.

## Summary

Generate a 16-bit tail predicate from a runtime element count and update the scalar through a post-update reference.

## Mechanism

The installed 3510 Bisheng CCE header exposes `plt_b16` as `vector_bool plt_b16(uint32_t &scalar, __cce_simd::POST_UPDATE)`. The call returns a predicate mask and writes the updated scalar value back through the reference argument.

In practice, this is the public CCE helper used for remainder-mask generation: the returned mask enables the currently active lanes, and the scalar reference carries the post-update state into the next step.

## Syntax

### PTO Assembly Form

```mlir
%mask, %scalar_out = pto.plt_b16 %scalar_in {post_update} : i32 -> !pto.mask, i32
```

### AS Level 1 (SSA)

```mlir
%mask, %scalar_out = pto.plt_b16 %scalar_in {post_update} : i32 -> !pto.mask, i32
```

### AS Level 2 (DPS)

```mlir
pto.plt_b16 ins(%scalar_in : i32) outs(%mask, %scalar_out : !pto.mask, i32)
```

## C++ Intrinsic

```cpp
uint32_t scalar = elementCount;
vector_bool mask = plt_b16(scalar, __cce_simd::POST_UPDATE);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `%scalar_in` | `i32` | Runtime element count carried into the predicate generator |
| `post_update` | attribute | Indicates the public CCE post-update form that writes `scalar_out` back through the reference |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%mask` | `!pto.mask` | 16-bit predicate generated from the current scalar value |
| `%scalar_out` | `i32` | Scalar value after the intrinsic's post-update step |

## Side Effects

- The public CCE call updates the scalar reference argument in place.

## Constraints

- The installed public CCE surface for `plt_b16` uses a `uint32_t &` scalar plus `__cce_simd::POST_UPDATE`.
- Programs that chain multiple `plt_b16` calls must thread the updated scalar value forward explicitly.
- The returned predicate width is fixed by the `_b16` suffix.

## Exceptions

- Illegal if the selected target profile does not support the requested predicate width.
- Illegal if the post-update scalar state is consumed in a way that breaks the required chaining discipline.

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| Tail-predicate helper | Simulated | Supported | Supported |
| Public post-update scalar form | Emulated | Supported | Supported |

## Examples

### C++ usage

```cpp
uint32_t scalar = elementCount;
vector_bool mask = plt_b16(scalar, __cce_simd::POST_UPDATE);
```

### SSA form

```mlir
%mask, %scalar_out = pto.plt_b16 %scalar_in {post_update} : i32 -> !pto.mask, i32
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md)
- Previous op in instruction set: [pto.plt_b8](./plt-b8.md)
- Next op in instruction set: [pto.plt_b32](./plt-b32.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
