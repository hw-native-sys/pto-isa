<!-- Generated from `docs/isa/syntax-and-operands/assembly-model.md` -->

# Assembly Spelling And Operands

PTO ISA includes a textual assembly spelling — PTO-AS — but the architecture contract stays in the PTO ISA manual itself. This page defines the syntax rules, including the BNF grammar for all three forms, operand modifier rules, and attribute syntax. Per-instruction syntax pages add instruction-specific variants.

## Three-Level Syntax System

PTO defines three levels of textual syntax, all preserving the same ISA contract:

| Level | Name | Form | Typical Use |
|-------|------|------|-------------|
| **Assembly Form (PTO-AS)** | Human-readable | `tadd %dst, %src0, %src1` | Documentation, pseudocode |
| **SSA Form (AS Level 1)** | MLIR SSA | `%dst = pto.tadd %src0, %src1` | IR, code generators |
| **DPS Form (AS Level 2)** | Functional DPS | `pto.tadd ins(...) outs(...)` | C++ intrinsic surface |

All three forms are **semantically equivalent** — they describe the same ISA operation. A backend or verifier must accept any form and produce identical behavior.

## BNF Grammar

### Assembly Form (PTO-AS)

```
assembly-program  ::= assembly-stmt*
assembly-stmt     ::= label? op-name operands? ":" type-ref ("#" attribute)*
label             ::= identifier ":"
op-name           ::= ("pto.")? identifier ("_" identifier)*
operands          ::= operand ("," operand)*
operand           ::= register | immediate | memory-operand | mask-operand
register          ::= "%" identifier
immediate         ::= integer | hex-integer | floating-point
hex-integer       ::= "0x" [0-9a-fA-F]+
floating-point    ::= [0-9]+ "." [0-9]+ ("e" [+-]? [0-9]+)?
memory-operand     ::= register "[" register ("," register)* "]"
mask-operand      ::= "%" identifier ":" "!" pto.mask
type-ref          ::= "!" pto "." type-key "<" type-params ">"
type-key          ::= "tile" | "tile_buf" | "vreg" | "ptr" | "partition_tensor_view" | "mask" | "event"
```

### SSA Form (AS Level 1)

```
ssa-program       ::= ssa-stmt*
ssa-stmt          ::= ssa-result "=" op-name operands ":" ssa-type -> ssa-type
ssa-result        ::= "%" identifier
op-name           ::= "pto." identifier ("_" identifier)*
operands          ::= ssa-operand ("," ssa-operand)*
ssa-operand       ::= ssa-result | immediate | memory-operand
ssa-type          ::= ssa-type-key "<" type-params ">"
ssa-type-key      ::= "tile" | "tile_buf" | "vreg" | "ptr" | "partition_tensor_view" | "mask"
```

### DPS Form (AS Level 2)

```
dps-program       ::= dps-stmt*
dps-stmt          ::= op-name "ins(" dps-ins ")" "outs(" dps-outs ")"
dps-ins           ::= dps-ins-item ("," dps-ins-item)*
dps-outs          ::= dps-out-item ("," dps-out-item)*
dps-ins-item      ::= ssa-result ":" ssa-type
dps-out-item      ::= ssa-result ":" ssa-type
```

## Operand Modifier Rules

### Tile Operands

A tile operand may carry optional modifiers in PTO-AS:

```
%tile                     -- bare tile register
%tile[%r, %c]            -- tile with GM base offset (row, col offsets)
%tile!loc=vec            -- tile with location intent annotation
```

In SSA form, location intent and valid-region information are encoded in the tile type:

```
!pto.tile<loc=vec, f32, 16, 16, RowMajor, NoneBox, None, Zero>
!pto.tile_buf<loc=vec, f32, 16, 16, RowMajor, NoneBox, None, Zero>
```

### GlobalTensor Operands

In PTO-AS, a `GlobalTensor` operand appears as a `memref` or `partition_tensor_view`:

```
%tensor                      -- bare GlobalTensor register
%tensor[%r, %c]             -- with 2D base offset
%tensor[%b, %h, %w, %r, %c] -- with 5D base offset (partition_tensor_view)
```

### Predicate Operands

A predicate operand is written as a mask register:

```
%mask : !pto.mask           -- predicate operand in SSA form
```

Vector instructions that take a mask write it as an explicit operand:

```
%result = pto.vadd %src0, %src1, %mask : ... -> ...
```

### Immediate Operands

Immediate operands are encoded directly in the instruction:

```
tadds %dst, %src, 0x3F800000   -- 32-bit float immediate (1.0f)
tshrs %dst, %src, 16            -- 16-bit shift amount
taddc %dst, %src0, %src1       -- carry-variant, no immediate
```

## Instruction Suffixes

PTO uses suffixes to distinguish operation variants:

| Suffix | Meaning | Example |
|--------|---------|---------|
| *(none)* | Standard binary op | `tadd` |
| `s` | Scalar variant: second operand is an immediate scalar | `tadds %dst, %src, 0x3F800000` |
| `c` | Carry variant: saturating arithmetic | `taddc`, `tsubc` |
| `sc` | Scalar + carry variant | `taddsc`, `tsubsc` |
| `_fp` | Floating-point special handling | `tstore_fp`, `tinsert_fp` |
| `_acc` | Accumulating variant | `tmatmul_acc` |
| `_bias` | Bias-addition variant | `tmatmul_bias` |
| `_mx` | MX format (int8 matmul) variant | `tgemv_mx` |

## Attribute Syntax

Attributes modify operation behavior. In PTO-AS, they appear after `#`:

```
tstore %tile, %tensor #atomic=add    -- atomic store
tcmps %dst, %src, 0   #cmp=gt        -- comparison mode
tmatmul %c, %a, %b    #phase=relu    -- matmul phase mode
```

In SSA form, attributes appear as `{key = value}`:

```
%result = pto.tcmp %src0, %src1 {cmp = "lt"} : ... -> ...
```

## Complete Examples

### Tile Compute: Elementwise Addition

**Assembly Form (PTO-AS)**:
```
tadd %dst, %src0, %src1 : !pto.tile<f32, 16, 16>
```

**SSA Form (AS Level 1)**:
```
%dst = pto.tadd %src0, %src1 : (!pto.tile<f32, 16, 16>, !pto.tile<f32, 16, 16>) -> !pto.tile<f32, 16, 16>
```

**DPS Form (AS Level 2)**:
```
pto.tadd ins(%src0, %src1 : !pto.tile_buf<f32, 16, 16>, !pto.tile_buf<f32, 16, 16>)
          outs(%dst : !pto.tile_buf<f32, 16, 16>)
```

**C++ Intrinsic**:
```cpp
TADD(TileDst, TileSrc0, TileSrc1);
```

### Tile Load: From GlobalTensor

**Assembly Form (PTO-AS)**:
```
tload %tile, %tensor[%r, %c] : (!pto.tile<f32,16,16>, !pto.memref<f32,5>) -> !pto.tile<f32,16,16>
```

**SSA Form (AS Level 1)**:
```
%tile = pto.tload %mem : !pto.partition_tensor_view<1x1x1x16x16xf32>
    -> !pto.tile_buf<loc=vec, f32, 16, 16, RowMajor, NoneBox, None, Zero>
```

### Vector Compute: Vector Addition with Mask

**SSA Form (AS Level 1)**:
```
%result = pto.vadd %src0, %src1, %mask : (!pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask) -> !pto.vreg<64xf32>
```

### Scalar Compare: Predicate Generation

**SSA Form (AS Level 1)**:
```
%pred = pto.pge_b32 %src0, %src1 : (!pto.vreg<64xi32>, !pto.vreg<64xi32>) -> !pto.mask
```

## What Textual Spelling Does Not Replace

Textual spelling does not replace:

- the PTO ISA machine model
- the PTO ISA memory model
- target-profile rules for CPU, A2/A3, and A5
- the architecture-level legality rules that are independent of textual spelling

## Contract Notes

- Textual assembly forms MUST preserve the same visible operation meaning as their documented intrinsic forms.
- Assembly syntax rules MUST stay in the PTO ISA syntax-and-operands pages, not in backend-private notes.
- Syntax variants that change semantics must be documented as explicit variants, not as undocumented assembler convenience.
- The three syntactic levels (Assembly / SSA / DPS) are semantically equivalent; a backend MUST NOT assign different behavior to different syntactic forms of the same operation.

## See Also

- [Operands and Attributes](./operands-and-attributes.md)
- [Type System](../state-and-types/type-system.md)
- [Parallel Tile Operation ISA Version 1.0](../introduction/what-is-pto-visa.md)
- [Instruction surfaces](../instruction-surfaces/README.md)
- [Common conventions](../conventions.md)
