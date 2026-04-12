<!-- Generated from `docs/isa/tile/ops/sync-and-config/tsetfmatrix.md` -->

# pto.tsetfmatrix

Standalone reference page for `pto.tsetfmatrix`. This page belongs to the [Sync And Config](../../sync-and-config.md) family in the PTO ISA manual.

## Summary

Set FMATRIX register(s) for IMG2COL-like ops.

## Mechanism

Set the FMATRIX register(s) used by IMG2COL-like operations from an `Img2colTileConfig` (target/implementation-defined). It is part of the tile synchronization or configuration shell, so the visible effect is ordering or state setup rather than arithmetic payload transformation.

Unless otherwise specified, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

### AS Level 1 (SSA)

```text
pto.tsetfmatrix %cfg : !pto.fmatrix_config -> ()
```

### AS Level 2 (DPS)

```text
pto.tsetfmatrix ins(%cfg : !pto.fmatrix_config) outs()
```

### IR Level 1 (SSA)

```text
pto.tsetfmatrix %cfg : !pto.fmatrix_config -> ()
```

### IR Level 2 (DPS)

```text
pto.tsetfmatrix ins(%cfg : !pto.fmatrix_config) outs()
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent TSETFMATRIX(ConvTileData &src, WaitEvents&... events);
```

## Inputs

- `src` is the ConvTileData (IMG2COL configuration tile).
- `FmatrixMode` (optional): FMATRIX register to target.

## Expected Outputs

This form is defined primarily by its ordering or configuration effect. It does not introduce a new payload tile beyond any explicit state object named by the syntax.

## Side Effects

This operation may establish a synchronization edge, bind or configure architectural tile state, or update implementation-defined configuration that later tile instructions consume.

## Constraints

Type/layout/location/shape legality is backend-dependent; treat implementation-specific notes as normative for that backend.

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend surface.
- Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

- `pto.tsetfmatrix` preserves PTO-visible semantics across CPU simulation, A2/A3-class targets, and A5-class targets, but concrete support subsets may differ by profile.

- Portable code must rely only on the documented type, layout, shape, and mode combinations that the selected target profile guarantees.

## Examples

See related examples in `docs/isa/` and `docs/coding/tutorials/`.

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
pto.tsetfmatrix %cfg : !pto.fmatrix_config -> ()
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
pto.tsetfmatrix %cfg : !pto.fmatrix_config -> ()
```

### PTO Assembly Form

```text
pto.tsetfmatrix %cfg : !pto.fmatrix_config -> ()
# AS Level 2 (DPS)
pto.tsetfmatrix ins(%cfg : !pto.fmatrix_config) outs()
```

## Related Ops / Family Links

- Family overview: [Sync And Config](../../sync-and-config.md)
- Previous op in family: [pto.tsettf32mode](./tsettf32mode.md)
- Next op in family: [pto.tset_img2col_rpt](./tset-img2col-rpt.md)
