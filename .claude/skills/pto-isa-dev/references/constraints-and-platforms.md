# Constraints and Platforms

This reference explains where instruction legality comes from and how to reason about A2/A3 versus A5 when an agent is changing code, docs, or tests.

## How to Find Instruction Constraints

Use this order:

1. Public contract:
   - [include/pto/common/pto_instr.hpp](../../../../include/pto/common/pto_instr.hpp)
   - [docs/PTO-Virtual-ISA-Manual.md](../../../../docs/PTO-Virtual-ISA-Manual.md)
   - per-instruction docs under [docs/isa](../../../../docs/isa)
2. Shared tile and memory rules:
   - [include/pto/common/pto_tile.hpp](../../../../include/pto/common/pto_tile.hpp)
   - [include/pto/common/tassign_check.hpp](../../../../include/pto/common/tassign_check.hpp)
   - [include/pto/common/buffer_limits.hpp](../../../../include/pto/common/buffer_limits.hpp)
3. Backend-specific implementation:
   - [include/pto/cpu](../../../../include/pto/cpu)
   - [include/pto/npu/a2a3](../../../../include/pto/npu/a2a3)
   - [include/pto/npu/a5](../../../../include/pto/npu/a5)

When searching, this is the fastest pattern:

```bash
rg -n "TADD|static_assert|PTO_STATIC_ASSERT|Unsupported|only support" include/pto
```

## What Usually Constrains an Instruction

- tile location such as `Vec`, `Mat`, `Acc`, `Left`, `Right`
- layout such as row-major or fractal form
- dtype availability
- shape alignment and buffer limits
- backend-only resources such as `ScaleLeft` or `ScaleRight`
- synchronization or cross-core assumptions

Recent reviews show that the fragile cases are usually combinations, not single constraints. Before changing a backend helper, check the relevant combined axis in [review-derived-guardrails.md](review-derived-guardrails.md): split direction, runtime `subblock_dim`, injected CPU-SIM hooks, tile layout, entry offset, and target backend.

## A2/A3 vs A5

### Backend split

- A2/A3 share the `a2a3` backend and are selected by `-v a3`.
- A5 uses `-v a5` and the dedicated [include/pto/npu/a5](../../../../include/pto/npu/a5) implementation.

### Practical differences

- A5 exposes features that are not always available on A2/A3, including some scale-tile flows and communication paths.
- A2/A3 often have tighter shape or location assumptions carried by older hardware interfaces.
- CPU-SIM can model a portability envelope, but it is not a proof that every hardware backend accepts the same encoding.

### Good files to compare

- [include/pto/common/buffer_limits.hpp](../../../../include/pto/common/buffer_limits.hpp)
- [include/pto/common/arch_macro.hpp](../../../../include/pto/common/arch_macro.hpp)
- [include/pto/npu/README.md](../../../../include/pto/npu/README.md)
- [docs/reference/pto-cvid-cluster-id-mapping.md](../../../../docs/reference/pto-cvid-cluster-id-mapping.md)

## PTO-ISA and PTO-AS

PTO-ISA is the semantic contract and backend library. PTO-AS is the textual assembly layer and toolchain surface.

### Use PTO-AS docs when

- validating textual syntax
- checking SSA-like operand conventions
- debugging assembler or disassembler mismatches
- reasoning about explicit `tsync`, events, and directive syntax

Primary references:

- [docs/assembly/README.md](../../../../docs/assembly/README.md)
- [docs/isa/syntax-and-operands/assembly-model.md](../../../../docs/isa/syntax-and-operands/assembly-model.md)

### Typical linkage workflow

- define or confirm semantics in PTO-ISA docs and headers
- ensure PTO-AS can express the legal operand and type forms
- reproduce the behavior in CPU-SIM or a focused ST
- if lowering is external, compare emitted PTO-AS with the instruction contract before changing backend code

## Cross-Platform Authoring Rules

- Prefer the most restrictive common contract when writing portable examples.
- Guard backend-only capabilities in docs and tests.
- If a feature is A5-only or A2/A3-only, say so explicitly.
- Treat CPU-SIM as a portability gate, then verify the intended backend separately.
- For new public overloads, keep `include/pto/common/pto_instr.hpp`, CPU-SIM, A2/A3, A5, PTO-AS/docs, and tests either aligned or explicitly marked unsupported.
