# PTO: Goals

PTO in the Ascend stack pursues the goals listed below. They complement the narrative introduction in [What Is PTO VISA](./what-is-pto-visa.md) and the normative scope statement in [Scope And Boundaries](./design-goals-and-boundaries.md).

## Core Objectives

### 1. Multi-Generation Stability

Keep the instruction set stable across multiple Ascend NPU generations. Hardware changes from generation to generation, but low-level software needs one instruction language that does not have to be reinvented every time the machine changes.

> **Example**: A `TMATMUL` kernel written against PTO ISA 1.0 should compile and run correctly on both A2/A3 and A5 backends without modification to the PTO-level code.

### 2. Performance Transparency

Preserve performance that is comparable to native NPU software. PTO is not meant to hide the machine behind a generic compute API. It keeps tile shape, data movement, synchronization, vector micro-instructions, and scalar control visible because those details often decide whether a kernel is merely correct or actually fast.

> **Example**: The tile-first model exposes valid regions, layout, and location intent so that the compiler or author can avoid unnecessary copies or padding.

### 3. Single Target for Multiple Frontends

Give C, C++, Python, and other frontends one machine-independent target. The same applies to tile-based systems and code generators such as TileLang and PyPTO. They should be able to target PTO instead of learning a separate low-level contract for each NPU generation.

### 4. Portable Distribution Format

Provide a distribution form through PTOBC. Applications and middleware need a way to cache, package, and transport PTO programs without collapsing them immediately into one target-specific binary format.

### 5. Common ISA for Code Generators

Give optimizing code generators and translators a common source-level ISA. PTO is the place where legalization, transformation, specialization, and verification can be shared before the final mapping to a particular hardware generation.

### 6. Human-Readable for Kernel Authors

Support hand-written libraries, performance kernels, and architecture tests. PTO is not only for compiler output. It also needs to be explicit and readable enough for people who write or inspect low-level code directly.

### 7. Scalable Parallelism

Scale from a single NPU unit to many parallel units. Parallel execution, explicit synchronization, and machine-visible data movement are part of the model from the start, not features bolted on later.

---

## Non-Goals

PTO is intentionally **not** trying to solve the following:

- **It is not a high-level DSL.** PTO is a low-level ISA, not a tensor expression language. Use TileLang, PyPTO, or a custom DSL for high-level authoring. PTO is the compilation target, not the source language.

- **It does not define hardware-specific micro-operations.** PTO exposes architecturally visible behavior. Whether a `TADD` lowers to 1 or 16 micro-ops on A5 is an implementation detail that does not belong in the ISA manual.

- **It does not replace the backend compiler.** PTO programs are lowered to target-specific binary through a backend (e.g., bisheng). The ISA defines the contract at the PTO level; the backend is responsible for optimization and instruction selection.

- **It does not guarantee bit-exact results across profiles.** CPU simulation, A2/A3, and A5 may produce slightly different floating-point results due to different microarchitectures. The ISA guarantees numerical *correctness* within each profile's defined semantics, not bit-for-bit identity across profiles.

---

## Design Trade-offs

When the goals conflict, PTO prioritizes in this order:

1. **Correctness** — illegal programs must be rejected; legal programs must produce correct results
2. **Portability** — the same PTO program must run correctly across all profiles (within profile-specific constraints)
3. **Performance** — PTO exposes enough detail for hand-tuned kernels, but does not mandate specific micro-optimizations

---

## See Also

- [What Is PTO VISA](./what-is-pto-visa.md)
- [Scope And Boundaries](./design-goals-and-boundaries.md)
- [Current PTO ISA Scope](./current-isa-scope.md)
