# 10. Memory Ordering and Consistency

## 10.1 Scope

This chapter defines architecture-visible memory ordering and visibility guarantees for PTO Virtual ISA operations.

## 10.2 Memory objects and domains

Architecture-visible memory domains include:

- tile-local values
- global memory views accessed by memory operations
- synchronization state affecting visibility boundaries

Backend-private caches/buffers are implementation-defined, but MUST respect architecture-visible ordering outcomes.

## 10.3 Consistency baseline

The baseline model is dependency-ordered consistency:

- data dependencies and explicit synchronization define required visibility order
- independent operations MAY be reordered internally
- required synchronization points MUST establish visibility as specified

## 10.4 Ordering guarantees

A conforming implementation MUST ensure:

- producer writes become visible to dependent consumers after required synchronization/ordering points
- memory operations participating in explicit dependency chains preserve those chains
- semantics defined by `TSYNC` and event dependencies are reflected in memory visibility

## 10.5 Unspecified and implementation-defined behavior

The following are architecture-restricted:

- accesses or interpretations outside defined domains may be unspecified
- timing and cache policy details are implementation-defined
- backend-specific memory optimizations are allowed only when they preserve required visible behavior

## 10.6 Programming requirements

Programs SHOULD:

- use explicit synchronization at producer/consumer boundaries
- avoid assuming implicit global ordering without a defined dependency
- avoid relying on unspecified out-of-domain values

Manual mode programmers MUST ensure required ordering when tool-managed synchronization is not used.

## 10.7 Diagnostics and conformance tests

Backends SHOULD provide diagnostics for:

- missing ordering assumptions in illegal contexts
- unsupported memory-ordering forms
- profile-specific restrictions

Conformance tests SHOULD include ordered visibility scenarios across representative dependency patterns.
