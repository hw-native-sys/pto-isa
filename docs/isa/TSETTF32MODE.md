# TSETTF32MODE


## Tile Operation Diagram

![TSETTF32MODE tile operation](../figures/isa/TSETTF32MODE.svg)

## Introduction

Configure TF32 transform mode (implementation-defined).

This instruction controls backend-specific TF32 transformation behavior used by supported compute paths.

## Math Interpretation

No direct tensor arithmetic is produced by this instruction. It updates target mode state used by subsequent instructions.

## Assembly Syntax

PTO-AS form: see [docs/grammar/PTO-AS.md](../grammar/PTO-AS.md).

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

## Constraints

- Available only when the corresponding backend capability macro is enabled.
- Exact mode values and hardware behavior are target-defined.
- This instruction has control-state side effects and should be ordered appropriately relative to dependent compute instructions.

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_enable_tf32() {
  TSETTF32MODE<true, RoundMode::CAST_ROUND>();
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
pto.tsettf32mode {enable = true, mode = ...}
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
pto.tsettf32mode {enable = true, mode = ...}
```

### PTO Assembly Form

```text
pto.tsettf32mode {enable = true, mode = ...}
# IR Level 2 (DPS)
pto.tsettf32mode ins({enable = true, mode = ...}) outs()
```

