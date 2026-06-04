# 1. Overview

## 1.1 Design goals

The PTO Virtual ISA is designed to provide:

- a stable architecture contract across evolving hardware generations
- tile-centric semantics with explicit valid-region behavior
- explicit boundaries between architecture-defined and implementation-defined behavior
- a practical bridge from intrinsics to IR and backend codegen

## 1.2 PTO architectural identity

PTO distinguishes itself from generic GPU ISAs by making the following architecture concepts first-class:

- **Tile as primary compute unit**: instruction semantics are defined over tile domains.
- **Valid-region-first semantics**: `Rv/Cv` define architectural compute coverage.
- **Location-intent model**: tile classes such as `Mat/Left/Right/Acc/Bias/Scale` participate in legality rules.
- **Dual programming model**: Auto and Manual modes are both architectural citizens.
- **Event-centric synchronization**: ordering is explicit through events and `TSYNC`.

## 1.3 Architecture boundary

The architecture defines:

- observable instruction results in valid regions
- required ordering and synchronization semantics
- legality boundaries exposed to users and toolchains

The architecture does **not** define:

- microarchitectural scheduling details
- exact on-chip layout implementation
- backend-specific optimization choices

Backend-specific details MUST be documented as implementation-defined constraints.

## 1.4 Source of truth

Authoritative PTO sources:

- per-op semantics: [PTO ISA Reference](/docs/isa/README.md)
- public API signatures: `include/pto/common/pto_instr.hpp`

This chaptered manual composes those sources into a complete Virtual ISA contract.

## 1.5 Instruction-family taxonomy

PTO instruction families are organized as:

- synchronization and resource binding
- elementwise and scalar/tile operations
- reduce/expand operations
- memory movement (`GM <-> Tile`) and indexed memory operations
- matrix and vector matrix operations
- layout/data-movement transforms
- irregular and complex operations

[Family-level contracts](06-instructions.md) are defined in `manual/06-instructions.md`.
[Per-op semantics](/docs/isa/README.md) remain in `docs/isa/*.md`.

## 1.6 Compatibility principles

- Additive evolution SHOULD be preferred over breaking changes.
- Breaking architectural behavior changes MUST include explicit versioning and migration notes.
- Implementation-defined behavior MUST remain explicitly tagged in all layers (manual, IR, backend docs).
