# PTO Virtual Instruction Set Architecture Manual

## 0.1 Scope

This manual defines the architecture-level contract of the PTO Virtual Instruction Set Architecture (VISA).
It specifies what a conforming frontend, IR pipeline, backend, and runtime MUST preserve when executing PTO programs.

Per-instruction pages in `docs/isa/*.md` remain the canonical source for opcode-specific semantics.
This manual defines the system-level contract around those semantics.

## 0.2 Audience

This manual is intended for:

- compiler and IR engineers implementing PTO lowering pipelines
- backend engineers implementing target legalization and code generation
- kernel authors validating architecture-visible behavior
- simulator and conformance-test developers

## 0.3 Document conventions

This manual uses a PTX/Tile-IR-inspired structure while preserving PTO-specific architecture design.
Each chapter follows a normative pattern where applicable:

- scope
- syntax/form
- semantics
- constraints
- diagnostics
- compatibility

## 0.4 Conformance language

The key words `MUST`, `MUST NOT`, `SHOULD`, and `MAY` are normative.

- `MUST` / `MUST NOT`: mandatory architectural requirement.
- `SHOULD`: recommended requirement; deviations require explicit rationale.
- `MAY`: optional behavior explicitly allowed by the architecture.

## 0.5 Authority order

When documents differ, resolve in this order:

1. `docs/isa/*.md` for per-instruction semantics and constraints.
2. `include/pto/common/pto_instr.hpp` for public API surface and overload shape.
3. This manual for architecture layering, contracts, and conformance policy.

## 0.6 Reading order

1. [Overview](01-overview.md)
2. [Execution Model](02-machine-model.md)
3. [State and Types](03-state-and-types.md)
4. [Tiles and GlobalTensor](04-tiles-and-globaltensor.md)
5. [Synchronization](05-synchronization.md)
6. [PTO Assembly (PTO-AS)](06-assembly.md)
7. [Instruction Set (overview)](07-instructions.md)
8. [Programming Guide](08-programming.md)
9. [Virtual ISA and IR](09-virtual-isa-and-ir.md)
10. [Bytecode and Toolchain](10-bytecode-and-toolchain.md)
11. [Memory Ordering and Consistency](11-memory-ordering-and-consistency.md)
12. [Backend Profiles and Conformance](12-backend-profiles-and-conformance.md)
13. [Appendix A: Glossary](appendix-a-glossary.md)
14. [Appendix B: Instruction Contract Template](appendix-b-instruction-contract-template.md)
15. [Appendix C: Diagnostics Taxonomy](appendix-c-diagnostics-taxonomy.md)
16. [Appendix D: Instruction Family Matrix](appendix-d-instruction-family-matrix.md)
