# pto.set_img2col_padding

`pto.set_img2col_padding` is part of the [Sync And Config](../../sync-and-config.md) instruction set.

## Summary

Set IMG2COL padding metadata from an IMG2COL configuration tile. On A2/A3 and A5, this instruction programs the FMATRIX padding registers (PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT fields in the FMATRIX_PAD register) which control the zero-padding applied around each image patch before the convolution window slides. On the CPU simulator, this is a functional no-op.

No direct tensor arithmetic is produced by this instruction. It updates IMG2COL padding control state consumed by subsequent data-movement operations.

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Schematic form:

```text
pto.set_img2col_padding %cfg
```

### AS Level 1 (SSA)

```text
pto.set_img2col_padding %cfg : !pto.fmatrix_config -> ()
```

### AS Level 2 (DPS)

```text
pto.set_img2col_padding ins(%cfg : !pto.fmatrix_config) outs()
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename ConvTileData, typename... WaitEvents>
PTO_INST RecordEvent SET_IMG2COL_PADDING(ConvTileData &src, WaitEvents &... events);

template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent SET_IMG2COL_PADDING(ConvTileData &src, WaitEvents &... events);
```

For `MEMORY_BASE` targets, an overload without `SetFmatrixMode` is also provided.

## Inputs

| Operand | Description |
|---------|-------------|
| `src` | ConvTileData (IMG2COL configuration tile) containing padding metadata |

## Expected Outputs

This form is defined primarily by its ordering or configuration effect. It does not introduce a new payload tile.

## Side Effects

- **A2/A3 and A5**: Updates the FMATRIX padding-control register. Consumed by the next `pto.timg2col` DMA operation in the same execution stream.
- **CPU simulator**: No architectural state is affected.

## Constraints

- This instruction is backend-specific and available only for backends that expose IMG2COL configuration state.
- `src` must be a valid IMG2COL configuration tile type accepted by the backend implementation.
- On A2/A3 and A5, this instruction writes to the FMATRIX PAD fields; the exact bit-width of each padding field (e.g., 4-bit per-side encoding) is target-specific.
- Use this instruction before dependent `pto.timg2col` operations in the same execution stream.

## Cases That Are Not Allowed

- Calling `pto.set_img2col_padding` on backends that do not expose FMATRIX configuration state.
- Using an invalid IMG2COL configuration tile type.

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_set_img2col_padding(Img2colTileConfig<uint64_t>& cfg) {
  SET_IMG2COL_PADDING(cfg);
}
```

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
pto.set_img2col_padding %cfg : !pto.fmatrix_config -> ()
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
pto.set_img2col_padding %cfg : !pto.fmatrix_config -> ()
```

## See Also

- Instruction set overview: [Sync And Config](../../sync-and-config.md)
- Previous op: [pto.set_img2col_rpt](./set-img2col-rpt.md)
- Next op: [pto.subview](./subview.md)
