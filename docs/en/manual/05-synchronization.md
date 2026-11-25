# 5. Synchronization

## 5.1 Scope

This chapter defines architecture-visible synchronization and ordering behavior for PTO Virtual ISA programs.

## 5.2 Synchronization primitives

PTO synchronization includes:

- event-based dependency chaining
- event synchronization ordering points between producer/consumer domains
- backend-specific low-level primitives abstracted by architecture semantics

Programs MAY use implicit tool-managed synchronization in Auto mode, but explicit synchronization remains architecturally valid in all modes.

## 5.3 event synchronization contract

event synchronization establishes ordering between operation sets.
A conforming implementation MUST ensure that:

- operations ordered-before the synchronization point become visible to ordered-after consumers according to the memory model
- synchronization semantics are preserved through optimization and lowering
- unsupported synchronization forms are rejected with deterministic diagnostics

## 5.4 Hazard classes

Synchronization requirements commonly arise from:

- read-after-write (RAW) dependencies
- write-after-read (WAR) interactions when resources are reused
- write-after-write (WAW) ordering constraints
- cross-pipeline handoff hazards (memory/vector/matrix domains)

A backend MAY internally optimize hazard handling, but MUST preserve architecture-observable ordering.

## 5.5 Event and dependency model

The event model MUST provide a deterministic dependency relation suitable for:

- pipeline handoff between producer and consumer instruction groups
- safe reuse of tile and memory resources
- reproducible execution under equivalent program order and dependency specification

## 5.6 Auto vs Manual synchronization responsibilities

- In Auto mode, compiler/runtime SHOULD insert required synchronization for legal execution.
- In Manual mode, programmers MUST provide required synchronization when dependencies are not otherwise guaranteed.
- Toolchains MUST NOT remove required user-authored synchronization unless a provably equivalent ordering is preserved.

## 5.7 Diagnostics requirements

Synchronization diagnostics SHOULD include:

- missing or invalid dependency context
- conflicting ordering assumptions
- backend capability limitations for requested synchronization form
- deterministic error class and message text
