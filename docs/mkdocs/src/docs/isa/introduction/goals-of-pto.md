<!-- Generated from `docs/isa/introduction/goals-of-pto.md` -->

# Goals Of PTO

This page lists what PTO is trying to achieve in the Ascend stack. It complements the narrative introduction in [What Is PTO VISA](./what-is-pto-visa.md) and the normative scope statement in [Scope And Boundaries](./design-goals-and-boundaries.md).

PTO is meant to solve a small set of important problems in the Ascend software stack.

- Keep the instruction set stable across multiple Ascend NPU generations. Hardware changes from generation to generation, but low-level software still needs one instruction language that does not have to be reinvented every time the machine changes.
- Preserve performance that is comparable to native NPU software. PTO is not meant to hide the machine behind a generic compute API. It keeps tile shape, data movement, synchronization, vector micro-instructions, and scalar control visible because those details often decide whether a kernel is merely correct or actually fast.
- Give C, C++, Python, and other frontends one machine-independent target. The same applies to tile-based systems and code generators such as TileLang and PyPTO. They should be able to target PTO instead of learning a separate low-level contract for each NPU generation.
- Provide a distribution form through PTOBC. Applications and middleware need a way to cache, package, and transport PTO programs without collapsing them immediately into one target-specific binary format.
- Give optimizing code generators and translators a common source-level ISA. PTO is the place where legalization, transformation, specialization, and verification can be shared before the final mapping to a particular hardware generation.
- Support hand-written libraries, performance kernels, and architecture tests. PTO is not only for compiler output. It also needs to be explicit and readable enough for people who write or inspect low-level code directly.
- Scale from a single NPU unit to many parallel units. Parallel execution, explicit synchronization, and machine-visible data movement are part of the model from the start, not features bolted on later.

## See Also

- [What Is PTO VISA](./what-is-pto-visa.md)
- [Scope And Boundaries](./design-goals-and-boundaries.md)
- [PTO ISA Version 1.0](./pto-isa-version-1-0.md)
