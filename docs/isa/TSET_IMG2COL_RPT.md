# TSET_IMG2COL_RPT

## Tile Operation Diagram

![TSET_IMG2COL_RPT tile operation](../figures/isa/TSET_IMG2COL_RPT.svg)

## Introduction

Set IMG2COL repeat metadata from an IMG2COL configuration tile (implementation-defined).

## Math Interpretation

No direct tensor arithmetic is produced by this instruction. It updates IMG2COL control state used by subsequent data-movement operations.

## Assembly Syntax

PTO-AS form: see [PTO-AS Specification](../assembly/PTO-AS.md).

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

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename ConvTileData, typename... WaitEvents>
PTO_INST RecordEvent TSET_IMG2COL_RPT(ConvTileData &src, WaitEvents &... events);

template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent TSET_IMG2COL_RPT(ConvTileData &src, WaitEvents &... events);
```

For `MEMORY_BASE` targets, an overload without `SetFmatrixMode` is also provided.

## Constraints

- This instruction is backend-specific and available only for backends that expose IMG2COL configuration state.
- `src` must be a valid IMG2COL configuration tile type accepted by the backend implementation.
- The exact register/metadata fields updated by this instruction are implementation-defined.
- Use this instruction before dependent `TIMG2COL` operations in the same execution stream.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_set_img2col_rpt(Img2colTileConfig<uint64_t>& cfg) {
  TSET_IMG2COL_RPT(cfg);
}
```

## ASM Form Examples

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

