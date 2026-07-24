# PTO ISA Manual

<div class="landing-shell">
  <section class="landing-hero">
    <p class="landing-kicker">Parallel Tile Operation</p>
    <h2>Read PTO as one coherent manual.</h2>
    <p class="landing-lede">
      PTO ISA is a multi-target virtual ISA for tile, vector, scalar, and communication operations. Start with the
      architecture model if you are new to PTO, or jump straight into the grouped instruction reference if you already
      know the instruction set you need.
    </p>
    <div class="landing-actions">
        <a class="landing-action-card" href="docs/isa/introduction/what-is-pto-visa.md">
        <span class="landing-action-eyebrow">Start learning</span>
        <strong>Introduction and reading order</strong>
        <span>Get the scope, document structure, and design boundaries before reading individual ops.</span>
      </a>
      <a class="landing-action-card" href="docs/isa/README.md">
        <span class="landing-action-eyebrow">Browse the reference</span>
        <strong>ISA index and instruction sets</strong>
        <span>Open the merged ISA tree, then drill into tile, vector, scalar, or communication instructions.</span>
      </a>
      <a class="landing-action-card" href="index_zh.md">
        <span class="landing-action-eyebrow">Switch language</span>
        <strong>Open the Chinese landing page</strong>
        <span>Use the Chinese reading track when a translated counterpart exists, or fall back to the Chinese hub.</span>
      </a>
    </div>
  </section>

  <section class="landing-grid">
    <article class="landing-panel">
      <h2>Learn PTO</h2>
      <ul>
        <li><a href="docs/isa/introduction/what-is-pto-visa.md">Introduction</a></li>
        <li><a href="docs/isa/introduction/document-structure.md">Document structure</a></li>
        <li><a href="docs/isa/introduction/goals-of-pto.md">Goals of PTO</a></li>
        <li><a href="docs/isa/introduction/design-goals-and-boundaries.md">Scope and boundaries</a></li>
        <li><a href="docs/isa/programming-model/tiles-and-valid-regions.md">Programming model</a></li>
        <li><a href="docs/isa/machine-model/execution-agents.md">Machine model</a></li>
        <li><a href="docs/isa/memory-model/consistency-baseline.md">Memory model</a></li>
      </ul>
    </article>

- [Manual entry page](docs/PTO-Virtual-ISA-Manual.md)
- [Manual preface and reading order](manual/index.md)
- [Instruction reference (one page per instruction)](docs/isa/README.md)
- [Programming model (Tiles/GlobalTensor/Events)](docs/coding/ProgrammingModel.md)
- [Abstract machine model](docs/machine/abstract-machine.md)
- [Virtual ISA / AS chapter](manual/08-virtual-isa-and-ir.md)
- [Bytecode / toolchain chapter](manual/09-bytecode-and-toolchain.md)
- [Memory ordering / consistency chapter](manual/10-memory-ordering-and-consistency.md)
- [Backend profiles / conformance chapter](manual/11-backend-profiles-and-conformance.md)

    <article class="landing-panel">
      <h2>Reference Notes</h2>
      <ul>
        <li><a href="docs/isa/syntax-and-operands/assembly-model.md">Assembly spelling and operands</a></li>
        <li><a href="docs/isa/conventions.md">Common conventions</a></li>
        <li><a href="docs/isa/state-and-types/type-system.md">Type system</a></li>
        <li><a href="docs/isa/state-and-types/location-intent-and-legality.md">Location intent and legality</a></li>
        <li><a href="docs/isa/reference/format-of-instruction-descriptions.md">Format of instruction descriptions</a></li>
        <li><a href="docs/isa/reference/README.md">Reference notes</a></li>
      </ul>
    </article>
  </section>
</div>
