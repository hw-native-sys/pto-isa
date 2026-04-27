# Source Of Truth

Use this order when rewriting or validating PTO ISA documentation:

1. `include/pto/common/pto_instr.hpp` — C++ intrinsic declarations; the public API contract
2. Current PTO ISA docs in this repo — authoritative prose descriptions
3. PTO-AS syntax page ([Assembly Spelling And Operands](../syntax-and-operands/assembly-model.md)) — syntax, assembly spelling, assembly-level forms
4. Target profile notes and implementation notes for backend-specific narrowing

If a prose source conflicts with the code-visible PTO instruction set, do not document unsupported behavior as architecture.

## Source Order

When the specification boundary is unclear, use the following order of authority:

1. **PTO ISA manual** and per-op ISA pages — architecture-visible semantics
2. **Code** (C++ headers, backend implementations) — legal instruction set
3. **PTO-AS syntax page** ([Assembly Spelling And Operands](../syntax-and-operands/assembly-model.md)) — syntax, assembly spelling, assembly-level forms
4. **Target profile notes** — backend-specific narrowing

## Two Compilation Flows

PTO programs flow through the toolchain in two ways. Both paths share the same PTO ISA semantics:

```
PTO program (.pto text)
        │
        ├──► ptoas ──► C++ ──► bisheng ──► binary  (Flow A)
        │
        └──► ptoas ──────────────────► binary           (Flow B)
```

The `ptoas` tool is the authoritative assembler. When documentation describes what "PTO" does, it refers to the semantics defined by PTO ISA, regardless of which flow is used to produce the final binary.

## What the Source Order Means for Authors

- If the manual says an operation is legal and the code rejects it, file a bug — the code should match the manual.
- If the manual is silent and the code accepts it, the code is authoritative — the manual should be updated.
- If the manual and the code disagree, the code is authoritative — the manual is wrong.
- If the manual is silent and the code rejects it, the code is authoritative — the behavior is backend-specific.

## PTOAS as the Authoritative Assembler

`ptoas` is the reference implementation of the PTO assembler. It defines:

- The PTO-AS grammar and syntax
- The parsing and validation rules
- The lowering semantics from PTO-AS to C++ or binary

When the PTO ISA manual specifies syntax forms (SSA, DPS), it refers to what `ptoas` accepts. See [Assembly Spelling And Operands](../syntax-and-operands/assembly-model.md) for the full grammar reference.
