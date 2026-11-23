# 11. Backend Profiles and Conformance

## 11.1 Scope

This chapter defines how backend capability subsets are described and how conformance levels are evaluated.

## 11.2 Backend profile model

A backend profile MUST document:

- supported instruction families and operation forms
- supported dtype/layout/location/shape tuples
- synchronization and memory-ordering limitations
- implementation-defined behavior surface
- diagnostics policy for unsupported features

Profiles MAY correspond to concrete targets (for example A2/A3/A5/CPU simulator).

## 11.3 Capability gating

Toolchains MUST gate backend-specific specialization by declared profile capability.
If requested behavior is outside profile support:

- compilation/legalization MUST fail deterministically, or
- an explicitly defined fallback path MUST be selected.

## 11.4 Conformance dimensions

Conformance is evaluated along these dimensions:

1. semantic conformance (instruction behavior)
2. legality conformance (contract validation)
3. ordering conformance (sync and memory visibility)
4. diagnostic conformance (deterministic actionable errors)

## 11.5 Conformance levels

Recommended levels:

- **Level 0 (parse/shape)**: structural toolchain correctness only.
- **Level 1 (family legality)**: documented family-level legality and diagnostics.
- **Level 2 (instruction semantic)**: per-op semantics validated on representative suites.
- **Level 3 (cross-layer stability)**: semantic, ordering, and diagnostics stability across AS/bytecode/backend transitions.

A backend SHOULD publish the highest validated level and known gaps.

## 11.6 Required test matrix

A profile conformance suite SHOULD include:

- legal/illegal tuple tests by instruction family
- synchronization and memory-ordering scenarios
- precision/mode interaction tests (including mixed precision paths)
- round-trip toolchain tests (text/AS/bytecode)
- deterministic diagnostics snapshots

## 11.7 Change management

When backend behavior changes:

- profile documents MUST be updated in the same change set
- conformance impact MUST be stated
- regressions against published levels MUST be treated as release blockers unless explicitly waived with rationale
