# PTO AS Operations Reference

This directory contains comprehensive documentation for PTO AS operations, covering ISA-level tile operations and auxiliary IR constructs used in PTO Level-1 and Level-2 intermediate representations.

---

## Overview

PTO AS provides **116 tile operations**, **11 auxiliary functions**, **47 scalar arithmetic operations**, and **7 control flow operations**.

Each operation is documented with:
- **AS Level 1 (SSA)**: Static Single Assignment form
- **AS Level 2 (DPS)**: Destination-Passing Style
- **Mathematical Semantics**: Formal mathematical interpretation
- **Constraints**: Type, layout, and runtime requirements

---

## Auxiliary Functions (11 functions)

**Document**: [Auxiliary Functions](nonisa-ops.md)

IR-level constructs for tensor view management, tile allocation, and synchronization:

- **Tensor View**: `make_tensor_view`, `partition_view`
- **Tile Management**: `alloc_tile`, `tgetval`, `tsetval`
- **Indexing**: `get_block_idx`, `get_subblock_idx`, `get_block_num`, `get_subblock_num`
- **Pointer Arithmetic**: `addptr`
- **Synchronization**: `record_event`, `wait_event`, `barrier`, `PIPE_BARRIER`
  For the current PTO-DSL kernel-authoring flow, prefer sync-free source plus
  `ptoas --enable-insert-sync`; treat `record_event` and `wait_event` as
  low-level IR or assembly constructs.

---

## Tile Operations (116 operations)

### Elementwise (Tile-Tile) - 28 operations
**Document**: [Elementwise Operations](elementwise-ops.md)

- **Arithmetic**: `TADD`, `TSUB`, `TMUL`, `TDIV`, `TABS`, `TNEG`
- **Bitwise**: `TAND`, `TOR`, `TXOR`, `TNOT`, `TSHL`, `TSHR`
- **Comparison**: `TCMP`, `TMIN`, `TMAX`
- **Mathematical**: `TLOG`, `TEXP`, `TSQRT`, `TRSQRT`, `TRECIP`
- **Activation**: `TRELU`, `TPRELU`
- **Type Conversion**: `TCVT`
- **Conditional**: `TSEL`
- **Compound**: `TADDC`, `TSUBC`
- **Modulo**: `TREM`, `TFMOD`

### Tile-Scalar Operations - 19 operations
**Document**: [Tile-Scalar Operations](tile-scalar-ops.md)

- **Arithmetic**: `TADDS`, `TSUBS`, `TMULS`, `TDIVS`, `TMINS`, `TMAXS`
- **Bitwise**: `TANDS`, `TORS`, `TXORS`, `TSHLS`, `TSHRS`
- **Modulo**: `TREMS`, `TFMODS`
- **Broadcast**: `TEXPANDS`
- **Comparison**: `TCMPS`
- **Conditional**: `TSELS`
- **Activation**: `TLRELU`
- **Compound**: `TADDSC`, `TSUBSC`

### Axis Reduction and Expansion - 23 operations
**Document**: [Axis Reduction and Expansion](axis-ops.md)

- **Row Reduction**: `TROWSUM`, `TROWMAX`, `TROWMIN`
- **Column Reduction**: `TCOLSUM`, `TCOLMAX`, `TCOLMIN`, `TCOLPROD`
- **Row Expansion**: `TROWEXPAND`, `TROWEXPANDADD`, `TROWEXPANDMUL`, `TROWEXPANDDIV`, `TROWEXPANDSUB`, `TROWEXPANDMAX`, `TROWEXPANDMIN`, `TROWEXPANDEXPDIF`
- **Column Expansion**: `TCOLEXPAND`, `TCOLEXPANDADD`, `TCOLEXPANDMUL`, `TCOLEXPANDDIV`, `TCOLEXPANDSUB`, `TCOLEXPANDMAX`, `TCOLEXPANDMIN`, `TCOLEXPANDEXPDIF`

### Memory Operations - 6 operations
**Document**: [Memory Operations](memory-ops.md)

- **Load/Store**: `TLOAD`, `TSTORE`, `TSTORE_FP`, `TPREFETCH`
- **Gather/Scatter**: `MGATHER`, `MSCATTER`

### Matrix Multiplication - 8 operations
**Document**: [Matrix Multiplication](matrix-ops.md)

- **Basic**: `TMATMUL`, `TMATMUL_ACC`, `TMATMUL_BIAS`
- **Mixed Precision**: `TMATMUL_MX`
- **Vector**: `TGEMV`, `TGEMV_ACC`, `TGEMV_BIAS`, `TGEMV_MX`

### Data Movement and Layout - 12 operations
**Document**: [Data Movement and Layout](data-movement-ops.md)

- **Extract/Insert**: `TEXTRACT`, `TEXTRACT_FP`, `TINSERT`, `TINSERT_FP`
- **Transform**: `TTRANS`, `TRESHAPE`, `TIMG2COL`
- **Move**: `TMOV`, `TMOV_FP`
- **Padding**: `TFILLPAD`, `TFILLPAD_INPLACE`, `TFILLPAD_EXPAND`

### Complex Operations - 13 operations
**Document**: [Complex Operations](complex-ops.md)

- **Sorting**: `TSORT32`, `TMRGSORT`
- **Gathering**: `TGATHER`, `TGATHERB`, `TSCATTER`
- **Partial Operations**: `TPARTADD`, `TPARTMUL`, `TPARTMAX`, `TPARTMIN`
- **Utility**: `TCI`, `TTRI`, `TQUANT`, `TPRINT`

### Manual Resource Binding - 6 operations
**Document**: [Manual Resource Binding](manual-binding-ops.md)

- **Assignment**: `TASSIGN`
- **Mode Configuration**: `TSETHF32MODE`, `TSETTF32MODE`, `TSETFMATRIX`
- **IMG2COL Configuration**: `TSET_IMG2COL_RPT`, `TSET_IMG2COL_PADDING`

---

## Scalar Arithmetic Operations (47 operations)

**Document**: [Scalar Arithmetic Operations](scalar-arith-ops.md)

Standard scalar operations from MLIR `arith` dialect (scalar only, no vector/tensor):

- **Integer Arithmetic**: `addi`, `subi`, `muli`, `divsi`, `divui`, `remsi`, `remui`, `ceildivsi`, `ceildivui`, `floordivsi`
- **Floating-Point Arithmetic**: `addf`, `subf`, `mulf`, `divf`, `remf`, `negf`
- **Bitwise**: `andi`, `ori`, `xori`
- **Shift**: `shli`, `shrsi`, `shrui`
- **Comparison**: `cmpi`, `cmpf`
- **Min/Max**: `minsi`, `minui`, `maxsi`, `maxui`, `minimumf`, `maximumf`, `minnumf`, `maxnumf`
- **Type Conversion**: `extsi`, `extui`, `trunci`, `extf`, `truncf`, `sitofp`, `uitofp`, `fptosi`, `fptoui`, `bitcast`, `index_cast`, `index_castui`
- **Special**: `select`, `constant`
- **Extended Arithmetic**: `addui_extended`, `mulsi_extended`, `mului_extended`

---

## Control Flow Operations (7 operations)

**Document**: [Control Flow Operations](control-flow-ops.md)

Structured control flow operations from MLIR `scf` dialect:

- **Loops**: `scf.for`, `scf.while`
- **Conditionals**: `scf.if`, `scf.index_switch`
- **Regions**: `scf.execute_region`
- **Terminators**: `scf.yield`, `scf.condition`

---

## Related Resources

- [**ISA Instruction Reference**](../isa/README.md) - Per-instruction canonical semantics
- [**PTO-AS Language Overview**](PTO-AS.md) - Assembly language syntax and grammar
- [**PTO-AS Conventions**](conventions.md) - Assembly and ISA documentation conventions
- [**BNF Grammar**](PTO-AS.bnf) - Formal grammar definition for PTO-AS
