# PTO AS Documentation Guide

The canonical PTO-AS syntax page is [Assembly Spelling And Operands](../isa/syntax-and-operands/assembly-model.md) in the PTO ISA manual. Use that page for manual navigation and for links from instruction pages.

PTO AS documentation mainly covers the following areas:

- PTO-AS syntax, grammar, and textual representation
- ISA-level tile operations and auxiliary AS constructs
- Scalar arithmetic and control-flow operations reused from MLIR
- Assembly-related conventions and supporting references

## Recommended Reading Path

If you are new to PTO-AS, we recommend reading in the following order:

1. [Assembly Spelling And Operands](../isa/syntax-and-operands/assembly-model.md): PTO-AS spelling, operands, and syntax
2. [PTO-AS Conventions](conventions.md): naming and documentation conventions
3. Operation category documents: read the category pages relevant to your task

## Documentation Categories

### 1. PTO-AS Syntax and Core Specification

- [Assembly Spelling And Operands](../isa/syntax-and-operands/assembly-model.md): canonical syntax page published in the PTO ISA manual
- [PTO-AS Conventions](conventions.md): assembly syntax conventions and related documentation rules

### 2. PTO Tile Operation Categories

- [Elementwise Operations](elementwise-ops.md): tile-tile elementwise operations
- [Tile-Scalar Operations](tile-scalar-ops.md): tile-scalar arithmetic, comparison, and activation operations
- [Axis Reduction and Expansion](axis-ops.md): row/column reductions and broadcast-like expansion operations
- [Memory Operations](memory-ops.md): GM and tile data movement operations
- [Matrix Multiplication](matrix-ops.md): GEMM and GEMV related operations
- [Data Movement and Layout](data-movement-ops.md): extraction, insertion, transpose, reshape, and padding operations
- [Complex Operations](complex-ops.md): sorting, gather/scatter, random, quantization, and utility operations
- [Manual Resource Binding](manual-binding-ops.md): assignment and hardware/resource configuration operations

### 3. Auxiliary AS and MLIR-Derived Operations

- [Auxiliary Functions](nonisa-ops.md): tensor views, tile allocation, indexing, and synchronization helpers
- [Scalar Arithmetic Operations](scalar-arith-ops.md): scalar-only arithmetic operations from MLIR `arith`
- [Control Flow Operations](control-flow-ops.md): structured control-flow operations from MLIR `scf`

### 4. Related References

- [ISA Instruction Reference](../isa/scalar/ops/micro-instruction/README.md): canonical per-instruction semantics
- [docs Entry Guide](../isa/scalar/ops/micro-instruction/README.md): top-level documentation navigation for PTO Tile Lib

## Directory Structure

Key entries are listed below:

```text
├── PTO-AS*                     # PTO-AS syntax and specification documents
├── conventions*                # Assembly conventions documents
├── elementwise-ops*            # Elementwise tile operation references
├── tile-scalar-ops*            # Tile-scalar operation references
├── axis-ops*                   # Axis reduction and expansion references
├── memory-ops*                 # Memory operation references
├── matrix-ops*                 # Matrix multiplication references
├── data-movement-ops*          # Data movement and layout references
├── complex-ops*                # Complex operation references
├── manual-binding-ops*         # Manual resource binding references
├── scalar-arith-ops*           # Scalar arithmetic references
├── control-flow-ops*           # Control-flow references
└── nonisa-ops*                 # Auxiliary AS construct references
```

## Related Entry Points

- [ISA Instruction Reference](../isa/scalar/ops/micro-instruction/README.md): browse canonical PTO instruction semantics
- [docs Entry Guide](../isa/scalar/ops/micro-instruction/README.md): return to the main docs navigation page
- [Machine Documentation](../machine/README.md): understand the abstract execution model
