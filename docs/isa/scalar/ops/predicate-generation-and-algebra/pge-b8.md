# pto.pge_b8

`pto.pge_b8` is part of the [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md) instruction set.

## Summary

Construct a 8-bit predicate mask from a documented `Pat*` token.

## Mechanism

The installed 3510 Bisheng CCE header exposes `pge_b8` as a pattern-token helper, not as a runtime scalar-threshold compare. The public call surface is `vector_bool pge_b8(T dist)` where `T` is one of the documented `__cce_simd::Pat*` marker types.

This page therefore models `pto.pge_b8` as pattern-based predicate materialization: the chosen token determines which lanes are active in the returned predicate register.

## Syntax

### PTO Assembly Form

```mlir
%mask = pto.pge_b8 "PAT_VL4" : !pto.mask
```

### AS Level 1 (SSA)

```mlir
%mask = pto.pge_b8 "PAT_VL4" : !pto.mask
```

### AS Level 2 (DPS)

```mlir
pto.pge_b8 "PAT_VL4" outs(%mask : !pto.mask)
```

## C++ Intrinsic

```cpp
vector_bool mask = pge_b8(__cce_simd::PAT_VL4);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `"PAT_*"` | string attribute | Predicate-pattern token such as `PAT_ALL`, `PAT_ALLF`, `PAT_VL*`, `PAT_M3`, `PAT_M4`, `PAT_H`, or `PAT_Q` |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `%mask` | `!pto.mask` | 8-bit predicate generated from the selected pattern token |

## Side Effects

None.

## Constraints

- The installed public CCE API accepts documented `Pat*` marker types only; there is no public runtime-scalar overload for `pge_b8` in the shipped 3510 header.
- Programs must use a pattern token that is valid for the selected target profile.
- This operation materializes a predicate register only; it does not update any scalar input in place.

## Exceptions

- Illegal if the pattern token is not supported by the selected target profile.
- Illegal if the result is consumed in a predicate-width context that the selected backend does not support.

## Target-Profile Restrictions

| Aspect | CPU Sim | A2/A3 | A5 |
|--------|:-------:|:------:|:--:|
| Pattern-token predicate generation | Simulated | Supported | Supported |
| Public CCE surface | Emulated | Supported | Supported |

## Examples

### C++ usage

```cpp
vector_bool mask = pge_b8(__cce_simd::PAT_VL4);
```

### SSA form

```mlir
%mask = pto.pge_b8 "PAT_VL4" : !pto.mask
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Predicate Generation And Algebra](../../predicate-generation-and-algebra.md)
- Previous op in instruction set: [pto.pset_b32](./pset-b32.md)
- Next op in instruction set: [pto.pge_b16](./pge-b16.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
