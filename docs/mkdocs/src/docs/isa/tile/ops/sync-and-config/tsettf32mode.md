<!-- Generated from `docs/isa/tile/ops/sync-and-config/tsettf32mode.md` -->

# pto.tsettf32mode

Standalone reference page for `pto.tsettf32mode`. This page belongs to the [Sync And Config](../../sync-and-config.md) family in the PTO ISA manual.

## Summary

Configure TF32 transform mode (implementation-defined).

## Mechanism

Configure TF32 transform mode (implementation-defined).

This instruction controls backend-specific TF32 transformation behavior used by supported compute paths. It is part of the tile synchronization or configuration shell, so the visible effect is ordering or state setup rather than arithmetic payload transformation.

No direct tensor arithmetic is produced by this instruction. It updates target mode state used by subsequent instructions.

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Schematic form:

```text
tsettf32mode {enable = true, mode = ...}
```

### IR Level 1 (SSA)

```text
pto.tsettf32mode {enable = true, mode = ...}
```

### IR Level 2 (DPS)

```text
pto.tsettf32mode ins({enable = true, mode = ...}) outs()
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <bool isEnable, RoundMode tf32TransMode = RoundMode::CAST_ROUND, typename... WaitEvents>
PTO_INST RecordEvent TSETTF32MODE(WaitEvents &... events);
```

## Inputs

- `enable` (bool): enables or disables the TF32 transform mode.
- `mode` (RoundMode): specifies the TF32 rounding mode.

## Expected Outputs

This form is defined primarily by its ordering or configuration effect. It does not introduce a new payload tile beyond any explicit state object named by the syntax.

## Side Effects

This operation may establish a synchronization edge, bind or configure architectural tile state, or update implementation-defined configuration that later tile instructions consume.

## Constraints

- Available only when the corresponding backend capability macro is enabled.

- Exact mode values and hardware behavior are target-defined.

- This instruction has control-state side effects and should be ordered appropriately relative to dependent compute instructions.

## Exceptions

- Illegal operand tuples, unsupported types, invalid layout combinations, or unsupported target-profile modes are rejected by the verifier or by the selected backend surface.
- Programs must not rely on behavior outside the documented legal domain of this operation, even if one backend currently accepts it.

## Target-Profile Restrictions

- `pto.tsettf32mode` preserves PTO-visible semantics across CPU simulation, A2/A3-class targets, and A5-class targets, but concrete support subsets may differ by profile.

- Portable code must rely only on the documented type, layout, shape, and mode combinations that the selected target profile guarantees.

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_enable_tf32() {
  TSETTF32MODE<true, RoundMode::CAST_ROUND>();
}
```

## Related Ops / Family Links

- Family overview: [Sync And Config](../../sync-and-config.md)
- Previous op in family: [pto.tsethf32mode](./tsethf32mode.md)
- Next op in family: [pto.tsetfmatrix](./tsetfmatrix.md)
