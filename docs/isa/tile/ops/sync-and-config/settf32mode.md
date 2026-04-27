# pto.settf32mode

`pto.settf32mode` is part of the [Sync And Config](../../sync-and-config.md) instruction set.

## Summary

Configure TF32 transform mode for FP32 matrix-multiplication and convolution paths. This programs the CCE (Cube Compute Engine) TF32 mode register.

On A2/A3, TF32 is not supported and `pto.settf32mode` is a no-op; the enable flag is ignored and no mode state is established. On A5, when `enable = true`, the backend configures the CCE to use TF32 mantissa truncation (7-bit mantissa) for subsequent FP32 matrix-multiplication and convolution paths; the `mode` field selects the rounding behavior. On the CPU simulator, `pto.settf32mode` is a functional no-op that preserves the mode state in software.

This instruction controls backend-specific TF32 transformation behavior used by supported compute paths. It is part of the tile synchronization or configuration shell, so the visible effect is ordering or state setup rather than arithmetic payload transformation.

No direct tensor arithmetic is produced by this instruction. It updates target mode state used by subsequent instructions.

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Schematic form:

```text
pto.settf32mode {enable = true, mode = ...}
```

### IR Level 1 (SSA)

```text
pto.settf32mode {enable = true, mode = ...}
```

### IR Level 2 (DPS)

```text
pto.settf32mode ins({enable = true, mode = ...}) outs()
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <bool isEnable, RoundMode tf32TransMode = RoundMode::CAST_ROUND, typename... WaitEvents>
PTO_INST RecordEvent SETTF32MODE(WaitEvents &... events);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `enable` | `bool` | Enables (`true`) or disables (`false`) the TF32 transform mode |
| `mode` | `RoundMode` | TF32 rounding mode; `CAST_ROUND` rounds to nearest even; other modes are reserved |

## Expected Outputs

This form is defined primarily by its ordering or configuration effect. It does not introduce a new payload tile.

## Side Effects

- **A2/A3**: No-op. TF32 is not supported; no architectural state is affected.
- **A5**: Configures the CCE TF32 mode register. Affects the precision of subsequent FP32 matrix/convolution compute.
- **CPU simulator**: Updates software-simulation mode state; does not affect IEEE 754 rounding.

## Constraints

- On A2/A3, calling `pto.settf32mode` is legal but has no effect.
- On A5, `enable` must be `true` or `false` and `mode` must be a supported `RoundMode` (only `CAST_ROUND` is currently defined).
- On CPU simulator, any combination of enable and mode is accepted without error.
- This instruction has control-state side effects and should be ordered appropriately relative to dependent compute instructions.

## Cases That Are Not Allowed

- Relying on TF32 precision behavior on A2/A3 (not supported).
- Using modes other than `CAST_ROUND` (reserved/undefined).

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_enable_tf32() {
  SETTF32MODE<true, RoundMode::CAST_ROUND>();
}
```

## See Also

- Instruction set overview: [Sync And Config](../../sync-and-config.md)
- Previous op: [pto.tassign](./tassign.md)
- Next op: [pto.set_img2col_rpt](./set-img2col-rpt.md)
