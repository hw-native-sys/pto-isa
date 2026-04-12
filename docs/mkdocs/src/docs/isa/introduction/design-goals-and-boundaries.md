<!-- Generated from `docs/isa/introduction/design-goals-and-boundaries.md` -->

# Scope And Boundaries

This page defines the scope of the PTO ISA specification and the boundary between core ISA guarantees and neighboring layers.

## What PTO ISA Defines

PTO ISA defines the architecture-visible meaning of legal PTO programs. In this manual that includes:

- the semantics of `pto.t*`, `pto.v*`, `pto.*`, and other architecture-visible operations
- the programming model for tiles, GlobalTensor objects, events, and explicit synchronization
- the machine model and memory-ordering rules that make execution visible to programmers, simulators, and backends
- the legality surface that must remain stable across CPU simulation and supported Ascend NPU targets

If two supported targets both accept the same legal PTO program, the architecture-visible meaning of that program shall come from PTO ISA and shall not be redefined by target-specific interpretation.

## What Target Profiles May Narrow

PTO ISA is stable, but it is not unlimited. Target profiles may narrow the accepted or efficient subset for a particular implementation.

For example, a target profile may restrict:

- tile shapes or tile ranks
- data types and layout combinations
- specific vector micro-instruction forms
- synchronization variants or memory spaces
- instruction subsets tied to a hardware generation

Those restrictions narrow the accepted or efficient subset on that target. They do not change the meaning of PTO ISA itself.

## What PTO-AS Adds

PTO-AS is the textual syntax for PTO ISA. It adds exact spelling for:

- instruction names
- operand order
- attributes and modifiers
- textual conventions needed for parsing, assembly, and round-tripping

PTO-AS is therefore part of the expression of PTO ISA. It is not a second architecture with different semantics.

## What PTOBC Adds

PTOBC is the distribution and transport form for PTO programs. It exists so PTO code can be packaged, cached, shipped in middleware, and handed between tools without collapsing directly to one hardware generation.

PTOBC does not redefine the ISA. It carries PTO programs in serialized form.

## What PTO ISA Does Not Freeze

This manual does not define every compiler-internal stage or backend lowering step as part of the public contract. PTO ISA does not freeze:

- compiler-internal IR structure
- pass ordering
- backend-specific scheduling strategy
- hardware-private pipeline internals
- binary encodings of native hardware instructions

Those details belong to compilers, assemblers, runtime systems, and target-specific backend documentation.

## Source Of Truth Order

When the specification boundary is unclear, use the following order of authority:

1. the PTO ISA manual and per-op ISA pages
2. the legal instruction surface exposed by code and verification
3. PTO-AS and PTOBC documentation for syntax and distribution rules
4. backend-profile notes for target-specific narrowing

If a backend depends on behavior that is not covered by that authority chain, that behavior is a backend requirement and not yet a PTO ISA guarantee.
