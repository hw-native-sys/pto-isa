# pto.set_img2col_rpt

`pto.set_img2col_rpt` is part of the [Sync And Config](../../sync-and-config.md) instruction set.

## Summary

Set IMG2COL repeat metadata from an IMG2COL configuration tile. On A2/A3 and A5, this instruction programs the FMATRIX repeat-count registers (the RPT field in the FMATRIX_CTRL_0 register) which control how many times the IMG2COL DMA engine repeats the same row/patch data before advancing. On the CPU simulator, this is a functional no-op.

No direct tensor arithmetic is produced by this instruction. It updates IMG2COL control state used by subsequent data-movement operations.

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Schematic form:

```text
pto.set_img2col_rpt %cfg
```

### AS Level 1 (SSA)

```text
pto.set_img2col_rpt %cfg : !pto.fmatrix_config -> ()
```

### AS Level 2 (DPS)

```text
pto.set_img2col_rpt ins(%cfg : !pto.fmatrix_config) outs()
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename ConvTileData, typename... WaitEvents>
PTO_INST RecordEvent SET_IMG2COL_RPT(ConvTileData &src, WaitEvents &... events);

template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent SET_IMG2COL_RPT(ConvTileData &src, WaitEvents &... events);
```

For `MEMORY_BASE` targets, an overload without `SetFmatrixMode` is also provided.

## Inputs

| Operand | Description |
|---------|-------------|
| `src` | ConvTileData (IMG2COL configuration tile) containing repeat metadata |

## Expected Outputs

This form is defined primarily by its ordering or configuration effect. It does not introduce a new payload tile.

## Side Effects

- **A2/A3 and A5**: Updates the FMATRIX repeat-control register. Consumed by the next `pto.timg2col` DMA operation in the same execution stream.
- **CPU simulator**: No architectural state is affected.

## Constraints

- This instruction is backend-specific and available only for backends that expose IMG2COL configuration state.
- `src` must be a valid IMG2COL configuration tile type accepted by the backend implementation.
- On A2/A3 and A5, this instruction writes to the FMATRIX RPT field; the exact bit-width of the repeat field (e.g., 4-bit vs 8-bit encoding) is target-specific.
- Use this instruction before dependent `pto.timg2col` operations in the same execution stream.

## Cases That Are Not Allowed

- Calling `pto.set_img2col_rpt` on backends that do not expose FMATRIX configuration state.
- Using an invalid IMG2COL configuration tile type.

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_set_img2col_rpt(Img2colTileConfig<uint64_t>& cfg) {
  SET_IMG2COL_RPT(cfg);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
pto.set_img2col_rpt %cfg : !pto.fmatrix_config -> ()
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
pto.set_img2col_rpt %cfg : !pto.fmatrix_config -> ()
```

## See Also

- Instruction set overview: [Sync And Config](../../sync-and-config.md)
- Previous op: [pto.settf32mode](./settf32mode.md)
- Next op: [pto.set_img2col_padding](./set-img2col-padding.md)
