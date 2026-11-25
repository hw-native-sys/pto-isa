# 9. Bytecode and toolchain

## 9.1 Scope

This chapter defines the practical interchange and validation contract for PTO IR and bytecode forms.

## 9.2 Representation layers

PTO representation layers:

1. Virtual ISA semantics
2. PTO IR structured form
3. bytecode serialized interchange form

Layer transitions MUST preserve architecture-observable meaning.

## 9.3 Bytecode module contract (v1)

A conforming v1 module MUST preserve:

- operation/block/function ordering
- SSA def-use topology
- operand/result type information
- required attributes and mode metadata
- symbol and entrypoint identity

If lossless preservation is impossible, serialization MUST fail deterministically.

## 9.4 Validation pipeline

Recommended pipeline:

1. produce PTO IR from the frontend
2. run structural verifier
3. serialize IR to bytecode
4. deserialize bytecode to IR
5. re-run structural verifier
6. optionally run target legality verifier

CI SHOULD enforce steps 1-5.

## 9.5 Diagnostics contract

Diagnostics MUST be:

- location-aware for textual forms
- deterministic for equivalent inputs
- actionable with expected-vs-actual constraints

Minimum error classes:

- parse error
- structural verification error
- bytecode format/compatibility error
- target legality error

## 9.6 Compatibility policy

Evolution policy MUST define:

- schema version field
- backward compatibility window
- unknown-field and unknown-op handling policy

Default policy:

- unknown required fields: reject
- unknown optional fields: reject unless explicit compatibility mode permits
- unknown operations: reject with deterministic unsupported-op diagnostics

## 9.7 Round-trip guarantees

For supported features, `text -> IR -> bytecode -> IR -> text` SHOULD preserve:

- semantics
- verifier-relevant structure
- required metadata

Byte-for-byte textual formatting equivalence is not required.

## 9.8 Operational acceptance checklist

Each release SHOULD validate:

- parser positive and negative suites
- structural verifier conformance suites
- malformed bytecode robustness tests
- round-trip regression corpus
- diagnostic stability snapshots
