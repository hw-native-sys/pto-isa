# Format Of Instruction Descriptions

Per-instruction and instruction-set pages in this manual follow a common structure so opcode contracts remain easy to compare across instruction sets.

PTO is **tile-first** and **valid-region-first**. Instruction text always means what happens in the declared valid region unless the page explicitly defines behavior outside it.

## Instruction Set Pages

A **instruction set** page (for example sync and config, elementwise tile–tile, vector load/store) states:

- what the instruction set is for, in one short opening section
- shared legality rules, operand roles, and interaction with valid regions
- pointers into the per-op pages

Instruction set pages do not need to repeat every opcode; they set the contract for the group.

## Per-Op Pages

Each `pto.*` operation page should make the following easy to find. Section titles may vary if a different shape reads better, but the information should be present.

1. **Name and instruction set** — Mnemonic (`pto.tadd`, `pto.vlds`, …) and which instruction set it belongs to (tile, vector, scalar/control).

2. **Summary** — One or two sentences: what the operation does on the meaningful domain.

3. **Mechanism** — Precise mathematical or dataflow description over the valid region (and any documented exceptions).

4. **Syntax** — Reference to PTO-AS spelling where relevant; optional **AS** and **IR** patterns when they help interchange and tooling (many pages use SSA and DPS-style examples).

5. **C++ intrinsic** — When the public C++ API is normative, cite the corresponding declaration from `pto_instr.hpp`.

6. **Inputs and outputs** — Operands, including tile roles and immediate operands.

7. **Side effects** — Synchronization edges, configuration state, or “none beyond the destination tile” as appropriate.

8. **Constraints and illegal cases** — What verifiers and backends reject; target-profile narrowing may be called out here or under a dedicated subsection.

9. **Performance** — When timing data is documented, keep **A2A3** and **A5** in separate subsections. Do not merge their latency or throughput into one unlabeled table.

10. **Examples** — Include at least one concrete snippet or pseudocode when it makes an abstract rule materially clearer.

11. **Related links** — Instruction set overview, neighbors in the nav, and cross-links to the programming or memory model when ordering matters.

## Normative Language

Use **MUST**, **SHOULD**, and **MAY** only for rules that a test, verifier, or review can check. Prefer plain language for explanation.

## See Also

- [Instruction sets](../instruction-families/README.md)
- [Instruction sets](../instruction-families/README.md)
- [Diagnostics and illegal cases](./diagnostics-and-illegal-cases.md)
- [Document structure](../introduction/document-structure.md)
