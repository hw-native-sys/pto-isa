# Appendix C. Diagnostics Taxonomy

## C.1 Scope

This appendix defines the diagnostic taxonomy and stability requirements for PTO Virtual ISA toolchains.

## C.2 Diagnostic quality contract

All diagnostics SHOULD satisfy:

- deterministic error class
- deterministic primary message shape
- actionable context (expected vs actual)
- source location when available

## C.3 Primary diagnostic classes

### C.3.1 Parse diagnostics (`PARSE_*`)

Use for textual PTO-AS errors:

- malformed token
- grammar violation
- invalid literal/attribute syntax

### C.3.2 Structural diagnostics (`STRUCT_*`)

Use for IR shape violations:

- wrong operand/result arity
- missing required attributes
- incompatible type classes

### C.3.3 Legality diagnostics (`LEGAL_*`)

Use for backend/profile legality failures:

- unsupported dtype/layout/location/shape tuple
- unsupported mode combination
- unsupported instruction variant in selected profile

### C.3.4 Ordering diagnostics (`ORDER_*`)

Use for synchronization/ordering failures:

- missing required dependency edge
- invalid synchronization form
- ordering contract violation

### C.3.5 Bytecode diagnostics (`BCODE_*`)

Use for interchange/serialization failures:

- unsupported bytecode version
- malformed section/record
- unknown required field/opcode

## C.4 Recommended message fields

Diagnostics SHOULD include:

- error class (stable identifier)
- operation name and operand position (if applicable)
- expected contract summary
- actual offending value/shape/type/mode
- location or source context

## C.5 Stability policy

- Error class identifiers MUST be stable across patch releases.
- Message wording SHOULD remain stable for CI snapshots.
- If message wording changes materially, release notes SHOULD document the change.

## C.6 Example format

```text
LEGAL_UNSUPPORTED_TUPLE: tmatmul operand src1 has unsupported tuple
  expected: layout in {fractal_a, fractal_b}, dtype in {fp16, bf16}
  actual: layout=row_major, dtype=int8
  context: backend_profile=A3, op_loc=line 42
```
