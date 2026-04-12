<!-- Generated from `docs/isa/tile/ops/sync-and-config/tset-img2col-rpt.md` -->

# pto.tset_img2col_rpt

Standalone reference page for `pto.tset_img2col_rpt`. This page belongs to the [Sync And Config](../../sync-and-config.md) family in the PTO ISA manual.

## Summary

Set IMG2COL repeat metadata from an IMG2COL configuration tile.

## Mechanism

Set IMG2COL repeat metadata from an IMG2COL configuration tile (implementation-defined). It is part of the tile synchronization or configuration shell, so the visible effect is ordering or state setup rather than arithmetic payload transformation.

No direct tensor arithmetic is produced by this instruction. It updates IMG2COL control state used by subsequent data-movement operations.

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Schematic form:

```text
tset_img2col_rpt %cfg
```

### AS Level 1 (SSA)

```text
pto.tset_img2col_rpt %cfg : !pto.fmatrix_config -> ()
```

### AS Level 2 (DPS)

```text
pto.tset_img2col_rpt ins(%cfg : !pto.fmatrix_config) outs()
```

### IR Level 1 (SSA)

```text
pto.tset_img2col_rpt %cfg
```

### IR Level 2 (DPS)

```text
pto.tset_img2col_rpt ins(%cfg) outs()
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename ConvTileData, typename... WaitEvents>
PTO_INST RecordEvent TSET_IMG2COL_RPT(ConvTileData &src, WaitEvents &... events);

template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent TSET_IMG2COL_RPT(ConvTileData &src, WaitEvents &... events);
```

For `MEMORY_BASE` targets, an overload without `SetFmatrixMode` is also provided.

## Inputs

- `src` is the ConvTileData (IMG2COL configuration tile) containing repeat metadata.

## Expected Outputs

This form is defined primarily by its ordering or configuration effect. It does not introduce a new payload tile beyond any explicit state object named by the syntax.

## Side Effects

This operation may establish a synchronization edge, bind or configure architectural tile state, or update implementation-defined configuration that later tile instructions consume.

## Constraints

- This instruction is backend-specific and available only for backends that expose IMG2COL configuration state.

- `src` must be a valid IMG2COL configuration tile type accepted by the backend implementation.

- The exact register/metadata fields updated by this instruction are implementation-defined.

- Use this instruction before dependent `TIMG2COL` operations in the same execution stream.

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend surface.
- Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

- `pto.tset_img2col_rpt` preserves PTO-visible semantics across CPU simulation, A2/A3-class targets, and A5-class targets, but concrete support subsets may differ by profile.

- Portable code must rely only on the documented type, layout, shape, and mode combinations that the selected target profile guarantees.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_set_img2col_rpt(Img2colTileConfig<uint64_t>& cfg) {
  TSET_IMG2COL_RPT(cfg);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
pto.tset_img2col_rpt %cfg : !pto.fmatrix_config -> ()
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
pto.tset_img2col_rpt %cfg : !pto.fmatrix_config -> ()
```

### PTO Assembly Form

```text
pto.tset_img2col_rpt %cfg : !pto.fmatrix_config -> ()
# AS Level 2 (DPS)
pto.tset_img2col_rpt ins(%cfg : !pto.fmatrix_config) outs()
```

## Related Ops / Family Links

- Family overview: [Sync And Config](../../sync-and-config.md)
- Previous op in family: [pto.tsetfmatrix](./tsetfmatrix.md)
- Next op in family: [pto.tset_img2col_padding](./tset-img2col-padding.md)
