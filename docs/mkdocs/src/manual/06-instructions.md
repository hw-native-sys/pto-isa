# 6. Instruction families and contracts

## 6.1 Scope

This chapter defines family-level normative contracts.
Per-op normative details remain in `docs/isa/*.md`.

## 6.2 Family taxonomy

PTO instruction families:

1. synchronization and resource binding
2. elementwise tile-tile operations
3. tile-scalar and tile-immediate operations
4. axis reduce and expand operations
5. memory operations (`GM <-> Tile` and indexed variants)
6. matrix multiply and GEMV operations
7. data movement and layout transforms
8. irregular/complex operations

The source-synchronized inventory is maintained by `docs/isa/manifest.yaml`.

## 6.3 Common family contract

Every instruction family MUST define:

- operand/result classes and position rules
- semantic domain (valid-region handling)
- required constraints (dtype/layout/location/shape)
- synchronization/ordering implications
- diagnostics behavior for illegal use
- implementation-defined boundaries

## 6.4 Valid-region-first rule

Unless a specific instruction states otherwise:

- semantics are defined only on the operation's valid domain
- out-of-domain results are unspecified
- family contracts MUST state domain-composition rules for multi-input operations

## 6.5 Family-level summaries

### 6.5.1 Synchronization and resource binding

Includes `TSYNC`, `TASSIGN`, mode/config instructions.
These operations define ordering or state-configuration effects and MUST preserve architecture ordering semantics.

### 6.5.2 Elementwise and scalar variants

Includes arithmetic, bitwise, compare, select, unary math, and scalar-fused forms.
Operations MUST define per-element behavior and mode-specific constraints.

### 6.5.3 Reduce/expand families

Includes row/column reductions and broadcast-like expansions.
Operations MUST define axis semantics and domain compatibility.

### 6.5.4 Memory families

Includes load/store/prefetch and indexed gather/scatter forms.
Operations MUST define mapping between tile domains and memory domains.

### 6.5.5 Matrix families

Includes `TMATMUL*` and `TGEMV*` families.
Contracts MUST define accumulation domain, operand-role legality, and precision-mode interactions.

### 6.5.6 Movement/layout families

Includes extract/insert/reshape/transpose/fillpad-like transforms.
Contracts MUST define index mapping and domain preservation rules.

### 6.5.7 Complex/irregular families

Includes sort/quant/partial/gather variants and other special operations.
Contracts MUST explicitly identify implementation-defined portions.

## 6.6 Documentation contract for per-op pages

Each per-instruction page SHOULD follow Appendix B template sections:

- Syntax
- Operands
- Semantics
- Constraints
- Diagnostics
- Implementation-defined behavior
- Compatibility notes

## 6.7 Coverage and synchronization policy

Family and instruction indexes MUST stay synchronized with:

- `docs/isa/manifest.yaml`
- `include/pto/common/pto_instr.hpp`
- generated index/matrix tooling in `docs/tools/`
