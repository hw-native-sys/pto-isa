# pto.tsetfmatrix

`pto.tsetfmatrix` is documented in the [Scalar And Control Instruction Set: Control And Configuration](../../control-and-configuration.md) lane even though the historical API name keeps the `t` prefix.

## Summary

Program the FMATRIX configuration registers used by IMG2COL-like paths.

## Mechanism

`pto.tsetfmatrix` writes mode/configuration state derived from an `Img2colTileConfig`-style operand. It does not transform tile payload data directly; instead it prepares scalar-visible configuration that later IMG2COL-family or matrix-preparation paths consume. The name is historical; the architectural role is control/configuration rather than tile arithmetic.

In the current implementation, the configuration packs:

- input feature-map width
- input feature-map height
- the four padding bytes

into the FMATRIX register state selected by `SetFmatrixMode`.

## Syntax

### PTO Assembly Form

```text
tsetfmatrix %cfg : !pto.fmatrix_config -> ()
```

### AS Level 1 (SSA)

```text
pto.tsetfmatrix %cfg : !pto.fmatrix_config -> ()
```

## Inputs

| Operand | Type | Description |
|---|---|---|
| `%cfg` | `!pto.fmatrix_config` | Configuration object carrying feature-map geometry and padding information |
| `FmatrixMode` | `SetFmatrixMode` | Selects whether the configuration is written to the A-side or B-side FMATRIX register set |

## Expected Outputs

| Result | Type | Description |
|---|---|---|
| None | `—` | This form does not return an SSA payload value; it updates scalar-visible FMATRIX register state |

## Side Effects

Updates the selected FMATRIX register set for later IMG2COL-like operations.

## Constraints

- `%cfg` must describe a valid IMG2COL configuration for the selected target profile.
- The operation must be ordered before the consumer that reads the programmed FMATRIX state.
- `*_MANUAL` variants are the meaningful modes for the explicit setter path; auto modes are generally handled by the consuming operation.

## Exceptions

- Unsupported target-profile modes or illegal configuration tuples are rejected by the selected backend or verifier.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- The current manual documents this as a scalar/control configuration operation rather than a tile instruction.
- A2A3 and A5 may accept different subsets of FMATRIX configuration behavior.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent TSETFMATRIX(ConvTileData &src, WaitEvents&... events);
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Scalar And Control Instruction Set: Control And Configuration](../../control-and-configuration.md)
- Tile compatibility stub: [old tile path](../../../tile/ops/sync-and-config/tsetfmatrix.md)
- Related tile-side consumers: [pto.tset_img2col_rpt](../../../tile/ops/sync-and-config/tset-img2col-rpt.md), [pto.tset_img2col_padding](../../../tile/ops/sync-and-config/tset-img2col-padding.md), [pto.timg2col](../../../tile/ops/layout-and-rearrangement/timg2col.md)
