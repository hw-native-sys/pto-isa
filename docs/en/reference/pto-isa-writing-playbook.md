# PTO ISA Writing Playbook

This playbook is for authors who maintain the PTO architectural manual and related high-level documentation.

## What Good PTO Documentation Should Feel Like

Good PTO documentation should read like an experienced engineer explaining a system clearly, not like a glossary that was expanded into full sentences. The reader should learn:

- what the model is
- why PTO chose it
- where the hard boundaries are
- what is portable and what is not

The manual is allowed to sound authoritative. It should not sound anonymous.

## The Main Writing Rules

### Explain Before You Normalize

Do not open a chapter with "scope / audience / terminology" unless the reader truly cannot proceed without it. Start with the problem the chapter answers.

### Use Normative Language Sparingly

Use `MUST`, `MUST NOT`, `SHOULD`, and `MAY` only when the statement can be checked by a verifier, backend, test, or review. Do not use them for taste, philosophy, or roadmaps.

### Always Include A Worked Example

Every major manual chapter should include at least one example from this repository. Preferred sources:

- `docs/coding/tutorials/`
- `tests/cpu/st/testcase/`
- `demos/`

Examples do not need to be large. They need to make the abstract point concrete.

### Answer "Why Not The Alternative?"

Whenever a chapter introduces a PTO-specific concept such as valid regions, location intent, or explicit synchronization, answer the obvious alternative. For example:

- Why not a generic thread-centric model?
- Why not infer tile role after lowering?
- Why not treat backend behavior as implied architecture?

This is one of the simplest ways to make the manual sound authored instead of templated.

### Prefer Paragraphs For Explanation

Use bullets for enumerations, matrices, or checklists. Use paragraphs for explanation. A chapter made entirely of bullets usually means the author has listed facts without teaching the model.

## PTO-Specific Do And Don't

Do:

- state architecture-visible behavior in plain language before the formal rule
- separate architecture-defined, implementation-defined, unspecified, and illegal behavior
- connect claims to real repo artifacts when possible
- keep the English and Chinese versions aligned in example choice and structure

Don't:

- hide important semantics behind abstract phrases such as "architecture-level contract" when a direct explanation exists
- turn style preference into normative language
- repeat the same chapter skeleton mechanically
- use bilingual translation as an excuse to write unnatural prose

## A Recommended Chapter Shape

For high-level manual chapters, use this order:

1. what this chapter answers
2. why this topic matters in PTO
3. one concrete example
4. core concepts in prose
5. normative contract
6. portability or backend-variation notes
7. common mistakes

Not every chapter needs every label literally, but every chapter should cover that information.

## Bilingual Policy

English is the structural source for the manual, but Chinese is not a line-by-line translation target. The Chinese version should preserve meaning, examples, and boundaries while still reading naturally to a technical reader.

Rules:

- keep section order aligned across languages
- keep example choice aligned across languages
- keep key terminology stable across languages
- rewrite for natural prose instead of mirroring sentence shape mechanically

## Review Questions

Before merging a doc change, ask:

- Does this chapter teach the model, or only define terms?
- Is every `MUST/SHOULD/MAY` actually testable?
- Is there at least one concrete example?
- Have we named the backend-defined surface clearly?
- Could a new PTO engineer understand the main boundary without opening five extra pages?
