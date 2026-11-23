# 2. Machine model

## 2.1 Scope

This chapter defines the abstract execution model that Virtual ISA programs target.
It specifies architecture-visible ordering and responsibility boundaries, not microarchitecture internals.

## 2.2 Execution agents

The abstract PTO machine has three conceptual agents:

- **Host machine**: prepares workloads, submits execution, and manages global resources.
- **Device machine**: schedules tile programs across execution resources.
- **Core machine**: executes tile/scalar instructions and synchronization primitives.

A conforming implementation MAY map these agents differently internally, but MUST preserve architecture-visible behavior.

## 2.3 Program granularity

PTO programs operate at tile granularity:

- A program is an ordered sequence of PTO operations over tile, scalar, memory, and event values.
- Execution units MAY process independent tile programs concurrently.
- Visible ordering MUST follow data dependencies and explicit synchronization semantics.

## 2.4 Dispatch and scheduling

Scheduling policy is implementation-defined, subject to architecture rules:

- Independent work MAY execute out of order.
- Dependence-ordered work MUST observe required happens-before relations.
- Backend/runtime MAY use SPMD, MPMD, or hybrid dispatch models.

## 2.5 Architecture-visible ordering domains

Ordering is defined across three domains:

1. **Program order domain**
- In a single dependent chain, later operations MUST observe earlier committed effects.

2. **Event/synchronization domain**
- Event operations and event synchronization MUST establish the architecture-defined ordering points.

3. **Memory visibility domain**
- `TLOAD`/`TSTORE` visibility rules apply according to memory-ordering constraints in chapter 11.

## 2.6 Auto vs Manual responsibilities

PTO supports two architecture-level responsibility modes:

- **Auto mode**
  - Compiler/runtime SHOULD insert legal synchronization and placement decisions.
  - User intent remains architecture-visible but operational details are tool-managed.

- **Manual mode**
  - Programmer is responsible for explicit placement, ordering, and pipeline-safe scheduling.
  - Toolchain MUST preserve explicitly authored synchronization semantics.

## 2.7 Implementation-defined surface

The following remain implementation-defined and MUST be documented per backend profile:

- scheduler heuristics
- pipeline occupancy and issue details
- internal buffering and transient placement
- backend-specific legality subsets

These details MUST NOT change architecture-defined instruction semantics.
