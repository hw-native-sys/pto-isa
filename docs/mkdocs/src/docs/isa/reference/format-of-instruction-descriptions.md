<!-- Generated from `docs/isa/reference/format-of-instruction-descriptions.md` -->

# Format Of Instruction Descriptions

This section defines how **per-instruction** and **family** pages in this manual are written. Readers should know what to expect from every opcode page, and authors should keep pages comparable across families.

PTO is **tile-first** and **valid-region-first**. Instruction text always means what happens in the declared valid region unless the page explicitly defines behavior outside it.

## Family Pages

A **family** page (for example sync and config, elementwise tile–tile, vector load/store) states:

- what the family is for, in one short opening section
- shared legality rules, operand roles, and interaction with valid regions
- pointers into the per-op pages

Family pages do not need to repeat every opcode; they set the contract for the group.

## Per-Op Pages

Each `pto.*` operation page should make the following easy to find. Section titles may vary if a different shape reads better, but the information should be present.

1. **Name and surface** — Mnemonic (`pto.tadd`, `pto.vlds`, …) and which instruction surface it belongs to (tile, vector, scalar/control).

2. **Summary** — One or two sentences: what the operation does on the meaningful domain.

3. **Mechanism** — Precise mathematical or dataflow description over the valid region (and any documented exceptions).

4. **Syntax** — Reference to PTO-AS spelling where relevant; optional **AS** and **IR** patterns when they help interchange and tooling (many pages use SSA and DPS-style examples).

5. **C++ intrinsic** — When the public C++ API is normative for authors, the `pto_instr.hpp` declaration is cited.

6. **Inputs and outputs** — Operands, including tile roles and immediate operands.

7. **Side effects** — Synchronization edges, configuration state, or “none beyond the destination tile” as appropriate.

8. **Constraints and illegal cases** — What verifiers and backends reject; target-profile narrowing may be called out here or under a dedicated subsection.

9. **Examples** — At least one concrete snippet or pseudocode where it clarifies use.

10. **Related links** — Family overview, neighbors in the nav, and cross-links to the programming or memory model when ordering matters.

## Normative Language

Use **MUST**, **SHOULD**, and **MAY** only for rules that a test, verifier, or review can check. Prefer plain language for explanation.

## See Also

- [Instruction surfaces](../instruction-surfaces/README.md)
- [Instruction families](../instruction-families/README.md)
- [Diagnostics and illegal cases](./diagnostics-and-illegal-cases.md)
- [Document structure](../introduction/document-structure.md)
