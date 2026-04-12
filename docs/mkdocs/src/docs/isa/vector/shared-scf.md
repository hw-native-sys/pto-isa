<!-- Generated from `docs/isa/vector/shared-scf.md` -->

# Vector Families: Shared Structured Control Flow

Vector code in PTO is surrounded by structured control, not by hidden launch magic. This page explains the control shell that wraps `pto.v*` execution without claiming that `scf` is itself a vector mnemonic family.

## Summary

Shared `scf` operations provide the loop and branch structure around vector regions. They keep vector execution analyzable, explicit, and compatible with the rest of the PTO manual.

## Mechanism

Around vector regions, `scf` is used to:

- iterate over repeated vector work
- carry scalar state across iterations
- branch around target-specific vector paths
- model vector execution scopes using structured regions instead of opaque launch syntax

The canonical scalar-side explanation lives in [Scalar And Control Families: Shared Structured Control Flow](../scalar/shared-scf.md). This vector page keeps the relationship visible for readers following the `pto.v*` path.

## Inputs

- scalar loop bounds
- scalar predicates
- loop-carried SSA values
- yielded state from vector-adjacent branches or loops

## Expected Outputs

- explicit structured regions around vector work
- loop-carried scalar or stateful results
- analyzable control boundaries for vector lowering

## Constraints

- Vector-side control MUST keep carried values and branch results explicit through `scf.yield`.
- Structured control SHOULD remain in `scf` form unless a truly architecture-visible PTO synchronization mechanism is required.
- The manual MUST distinguish between vector payload effects and the shared control shell that surrounds them.

## Cases That Are Not Allowed

- treating structured control as backend-only hidden behavior
- collapsing vector loop state into vague prose instead of explicit carried SSA values
- documenting `scf` as though it were a `pto.v*` opcode family

## Related Ops And Family Links

- [Scalar And Control Families: Shared Structured Control Flow](../scalar/shared-scf.md)
- [Vector Families: Pipeline Sync](./pipeline-sync.md)
- [Vector Families: Shared Scalar Arithmetic](./shared-arith.md)
