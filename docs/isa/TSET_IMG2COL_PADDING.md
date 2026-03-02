# TSET_IMG2COL_PADDING

## Tile Operation Diagram

![TSET_IMG2COL_PADDING tile operation](../figures/isa/TSET_IMG2COL_PADDING.svg)

## Introduction

Set IMG2COL padding metadata from an IMG2COL configuration tile (implementation-defined).

## Math Interpretation

No direct tensor arithmetic is produced by this instruction. It updates IMG2COL padding control state consumed by subsequent data-movement operations.

## Assembly Syntax

PTO-AS form: see [docs/grammar/PTO-AS.md](../grammar/PTO-AS.md).

Schematic form:

```text
tset_img2col_padding %cfg
```

### IR Level 1 (SSA)

```text
pto.tset_img2col_padding %cfg : !pto.fmatrix_config -> ()
```

### IR Level 2 (DPS)

```text
pto.tset_img2col_padding ins(%cfg : !pto.fmatrix_config) outs()
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent TSET_IMG2COL_PADDING(ConvTileData &src, WaitEvents &... events);
```

For `MEMORY_BASE` targets, an overload without `SetFmatrixMode` is also provided.

## Constraints

- This instruction is backend-specific and available only for backends that expose IMG2COL configuration state.
- `src` must be a valid IMG2COL configuration tile type accepted by the backend implementation.
- The exact padding fields updated by this instruction are implementation-defined.
- Use this instruction before dependent `TIMG2COL` operations in the same execution stream.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_set_img2col_padding(Img2colTileConfig<uint64_t>& cfg) {
  TSET_IMG2COL_PADDING(cfg);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
pto.tset_img2col_padding %cfg : !pto.fmatrix_config -> ()
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
pto.tset_img2col_padding %cfg : !pto.fmatrix_config -> ()
```

### PTO Assembly Form

```text
pto.tset_img2col_padding %cfg : !pto.fmatrix_config -> ()
# IR Level 2 (DPS)
pto.tset_img2col_padding ins(%cfg : !pto.fmatrix_config) outs()
```

