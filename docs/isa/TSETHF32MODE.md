# TSETHF32MODE


## Tile Operation Diagram

![TSETHF32MODE tile operation](../figures/isa/TSETHF32MODE.svg)

## Introduction

Configure HF32 transform mode (implementation-defined).

This instruction controls backend-specific HF32 transformation behavior used by supported compute paths.

## Math Interpretation

No direct tensor arithmetic is produced by this instruction. It updates target mode state used by subsequent instructions.

## Assembly Syntax

PTO-AS form: see [docs/grammar/PTO-AS.md](../grammar/PTO-AS.md).

Schematic form:

```text
tsethf32mode {enable = true, mode = ...}
```

### IR Level 1 (SSA)

```text
pto.tsethf32mode {enable = true, mode = ...}
```

### IR Level 2 (DPS)

```text
pto.tsethf32mode ins({enable = true, mode = ...}) outs()
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <bool isEnable, RoundMode hf32TransMode = RoundMode::CAST_ROUND, typename... WaitEvents>
PTO_INST RecordEvent TSETHF32MODE(WaitEvents &... events);
```

## Constraints

- Available only when the corresponding backend capability macro is enabled.
- Exact mode values and hardware behavior are target-defined.
- This instruction has control-state side effects and should be ordered appropriately relative to dependent compute instructions.

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_enable_hf32() {
  TSETHF32MODE<true, RoundMode::CAST_ROUND>();
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
pto.tsethf32mode {enable = true, mode = ...}
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
pto.tsethf32mode {enable = true, mode = ...}
```

### PTO Assembly Form

```text
pto.tsethf32mode {enable = true, mode = ...}
# IR Level 2 (DPS)
pto.tsethf32mode ins({enable = true, mode = ...}) outs()
```

