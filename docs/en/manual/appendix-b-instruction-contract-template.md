# Appendix B. Instruction Contract Template

## B.1 Purpose

This template defines the canonical section structure for PTO per-instruction documentation.
Use this template for new instruction pages and during refactors.

## B.2 Required section order

1. `# <INSTR>`
2. `## Scope`
3. `## Syntax`
4. `## Operands`
5. `## Semantics`
6. `## Constraints`
7. `## Diagnostics`
8. `## Implementation-defined behavior`
9. `## Compatibility`
10. `## Examples`

## B.3 Normative requirements per section

### Scope

- MUST identify instruction family and intent.
- MUST state whether this page defines architecture semantics or backend-specific supplement.

### Syntax

- MUST provide PTO-AS form and public API signature references.
- SHOULD keep one canonical syntax shape before optional variants.

### Operands

For each operand/result, MUST define:

- role (`dst`, `src0`, `src1`, ...)
- type class
- domain/shape expectations
- location/layout requirements (if any)

### Semantics

- MUST define the valid-domain iteration model.
- MUST define output meaning in-domain.
- MUST explicitly address domain-outside behavior (defined or unspecified).

### Constraints

- MUST list legality dimensions (`dtype`, `layout`, `location`, `shape`, mode attrs).
- MUST distinguish architecture requirements from backend profile restrictions.

### Diagnostics

- MUST define deterministic rejection conditions.
- SHOULD include expected-vs-actual examples for common failure classes.

### Implementation-defined behavior

- MUST enumerate every implementation-defined point.
- MUST state where backend-specific details are documented.

### Compatibility

- MUST state versioning/migration notes when behavior changed.
- SHOULD include additive-vs-breaking classification.

## B.4 Template body (copy/paste)

```markdown
# <INSTR>

## Scope

## Syntax

## Operands

## Semantics

## Constraints

## Diagnostics

## Implementation-defined behavior

## Compatibility

## Examples
```
