<!-- Generated from `docs/isa/scalar/shared-arith.md` -->

# Scalar And Control Families: Shared Scalar Arithmetic

PTO source programs use the shared MLIR `arith` surface for scalar math around tile and vector regions. These ops are part of the documented PTO source surface, but they are not PTO mnemonics themselves.

## Summary

Shared scalar arithmetic provides constants, scalar math, comparisons, casts, and selects that feed PTO payload regions. It exists so PTO does not need to invent a separate scalar arithmetic ISA for bookkeeping that MLIR already models well.

## Mechanism

`arith` values stay in ordinary scalar SSA form. They are used to:

- materialize constants and loop bounds
- compute offsets, dynamic shapes, and tail counts
- build predicates for `scf.if` or `scf.while`
- adapt scalar widths and types around PTO boundaries

When the program needs tile or vector payload math, it must switch back to the PTO instruction surfaces. `arith` is the scalar shell, not a substitute for `pto.t*` or `pto.v*`.

## Inputs

Shared scalar arithmetic consumes scalar values of these broad kinds:

- `index`
- integer values
- floating-point values
- boolean-like predicates produced by comparison operations

## Expected Outputs

It produces scalar SSA values that are later consumed by:

- loop bounds and control decisions
- tile-valid-region calculations
- pointer or offset calculations
- scalar operands to PTO instructions

## Side Effects

`arith` operations are value-producing only. They do not allocate buffers, trigger DMA, change vector masks, or establish synchronization by themselves.

## Constraints

- Shared scalar arithmetic **MUST** remain scalar. It does not define vector-register or tile-payload behavior.
- PTO pages **MUST** document `arith` as part of the supported source surface when a kernel author needs scalar setup around PTO regions.
- Type conversions or comparisons that affect later PTO legality **MUST** be stated explicitly rather than implied.

## Exceptions

The following are **ILLEGAL**:

- using `arith` to stand in for payload vector math
- leaving signedness, width change, or `index` conversion behavior implicit at a PTO boundary
- assuming backend-specific scalar widths beyond what the program spells explicitly

## Target-Profile Restrictions

The `arith` contract is largely target-neutral. Backend restrictions appear only when an `arith` result is later consumed by a target-restricted PTO instruction or by a target-specific lowering path.

## Examples

### Scalar Setup Around A PTO Region

```mlir
%c0 = arith.constant 0 : index
%c64 = arith.constant 64 : index
%tile_offset = arith.muli %tile_idx, %c64 : index
%is_tail = arith.cmpi slt, %remaining, %c64 : index
```

### Branch Predicate For Structured Control

```mlir
%needs_tail = arith.cmpi slt, %valid_cols, %tile_cols : index
%active_cols = arith.select %needs_tail, %valid_cols, %tile_cols : index
```

## Related Ops And Family Links

- [Shared Structured Control Flow](./shared-scf.md)
- [Scalar And Control Families: Control And Configuration](./control-and-configuration.md)
- [Programming Model: Tiles And Valid Regions](../programming-model/tiles-and-valid-regions.md)
