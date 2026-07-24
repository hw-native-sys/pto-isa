# 7. Programming model contracts

## 7.1 Scope

This chapter defines architecture-safe programming contracts for Auto and Manual modes.
It focuses on correctness and portability boundaries rather than backend-specific optimization tricks.

## 7.2 Auto and Manual contract split

### 7.2.1 Auto mode

- Toolchain SHOULD infer legal placement, ordering, and scheduling.
- Generated code MUST preserve Virtual ISA semantics.
- User-visible behavior MUST remain deterministic under equivalent source and options.

### 7.2.2 Manual mode

- Programmers MAY explicitly control placement and synchronization.
- User-authored dependencies and ordering points MUST be preserved.
- Illegal manual configurations MUST fail with actionable diagnostics.

## 7.3 Portability-safe programming rules

Programs intended for cross-backend portability SHOULD:

- stay within documented family-level legality domains
- avoid relying on implementation-defined side effects
- use explicit synchronization when dependence is not guaranteed by dataflow
- keep dtype/layout/location tuples within backend-intersection profiles

## 7.4 Performance-aware but portable patterns

Portable patterns include:

- explicit domain-safe tiling and valid-region management
- clear producer/consumer phase boundaries with events/`TSYNC`
- backend-gated specialization with capability checks
- deterministic fallback paths for unsupported tuples

## 7.5 Anti-patterns

The following are non-portable and SHOULD be avoided:

- reading out-of-valid-domain values as meaningful data
- depending on undocumented pipeline timing behavior
- assuming implicit ordering where no dependency is specified
- encoding backend-specific assumptions without explicit profile gating

## 7.6 Debug and validation workflow

Recommended workflow:

1. structural correctness checks (types, arity, attributes)
2. legal-domain checks (shape/layout/location tuple validity)
3. synchronization checks (dependency completeness)
4. backend conformance checks (profile-specific)
5. differential behavior checks across representative targets

## 7.7 Compatibility notes

When code relies on implementation-defined behavior:

- assumptions MUST be documented
- backend profile constraints MUST be declared
- fallback behavior SHOULD be provided where feasible
