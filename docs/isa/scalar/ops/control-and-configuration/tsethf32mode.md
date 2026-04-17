# pto.tsethf32mode

`pto.tsethf32mode` is documented in the [Scalar And Control Instruction Set: Control And Configuration](../../control-and-configuration.md) lane even though the historical API name keeps the `t` prefix.

## Summary

Configure HF32 transform mode for later compute paths.

## Mechanism

`pto.tsethf32mode` does not mutate a tile payload. It updates scalar-visible mode state that later compute instructions consult when they execute HF32-capable hardware paths. The name is historical; the architectural role is control/configuration rather than tile arithmetic.

## Syntax

### PTO Assembly Form

```text
tsethf32mode {enable = true, mode = ...}
```

### AS Level 1 (SSA)

```text
pto.tsethf32mode {enable = true, mode = ...}
```

## Inputs

| Operand | Type | Description |
|---|---|---|
| `enable` | `bool` | Enables or disables the HF32 transform mode |
| `mode` | `RoundMode` | Selects the HF32 rounding mode consumed by later compute paths |

## Expected Outputs

| Result | Type | Description |
|---|---|---|
| None | `—` | This form does not return an SSA payload value; it updates scalar-visible mode state |

## Side Effects

Updates implementation-defined mode state used by later HF32-capable compute instructions.

## Constraints

- The exact mode values and downstream hardware behavior are target-defined.
- This operation must be ordered before the compute instructions that depend on the configured HF32 mode.
- Programs must not treat this operation as tile-payload transformation; it only changes mode state.

## Exceptions

- Unsupported target-profile modes or illegal configuration tuples are rejected by the selected backend or verifier.
- Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

- The current manual documents this as a scalar/control configuration operation rather than a tile instruction.
- A2A3 and A5 may accept different subsets of the HF32 control surface.

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <bool isEnable, RoundMode hf32TransMode = RoundMode::CAST_ROUND, typename... WaitEvents>
PTO_INST RecordEvent TSETHF32MODE(WaitEvents &... events);
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Scalar And Control Instruction Set: Control And Configuration](../../control-and-configuration.md)
- Tile compatibility stub: [old tile path](../../../tile/ops/sync-and-config/tsethf32mode.md)
- Related mode op: [pto.tsettf32mode](../../../tile/ops/sync-and-config/tsettf32mode.md)
