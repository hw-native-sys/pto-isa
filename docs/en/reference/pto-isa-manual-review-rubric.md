# PTO ISA Manual Review Rubric

Use this rubric when reviewing a rewritten manual chapter.

## Content Quality

- Does the chapter say what question it answers?
- Does it explain why PTO chose this design?
- Does it include at least one concrete example?
- Does it identify the portability boundary clearly?
- Does it separate architecture-defined, implementation-defined, unspecified, and illegal behavior?

## Normative Precision

- Is every `MUST`, `MUST NOT`, `SHOULD`, and `MAY` testable?
- Are diagnostics and legality rules stated in a way that a verifier or backend can implement?
- Are compatibility claims tied to an actual contract rather than a style preference?

## Readability

- Is the explanation carried by prose rather than bullet overload?
- Are abstract terms introduced only when needed?
- Does the chapter avoid sounding like a repeated template?
- Could a new PTO engineer understand the main model without leaving the chapter immediately?

## Bilingual Consistency

- Do the English and Chinese versions use the same examples?
- Do they preserve the same architecture boundaries?
- Are key terms translated consistently without forcing unnatural sentence structure?
