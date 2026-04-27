---
name: pto-isa-dev
description: Work effectively in PTO-ISA: choose the right backend, run CPU/SIM/NPU flows, trace instruction constraints, understand A2/A3 vs A5 differences, align with PTO-AS, debug failures, and apply review-derived guardrails from recent PRs.
---

# PTO-ISA Development

Use this skill when working in `pto-isa` with Claude Code, Codex, or a similar repository-aware coding agent on instruction behavior, backend portability, tests, docs, or tooling integration.

## Quick Start

- Start from CPU-SIM unless the task is explicitly hardware-only.
- Use the smallest reproducer first: one testcase, one gtest filter, one backend.
- Treat public docs and headers as the ISA contract, then confirm backend-specific legality in the implementation.
- When assembly or bytecode is involved, cross-check PTO-AS docs before changing backend code.
- Before changing pipe sync, layout/stride math, backend template dispatch, ST target registration, or ISA docs, read the review-derived guardrails.

## Core Workflow

- Orient with:
  - [README.md](../../../README.md)
  - [docs/agent.md](../../../docs/agent.md)
  - [docs/getting-started.md](../../../docs/getting-started.md)
  - [include/README.md](../../../include/README.md)
- Pick the execution lane:
  - CPU correctness and portability: [references/build-and-run.md](references/build-and-run.md)
  - Backend legality and platform differences: [references/constraints-and-platforms.md](references/constraints-and-platforms.md)
  - PTO-AS linkage and failure analysis: [references/debugging.md](references/debugging.md)
  - Recent PR failure patterns and review gates: [references/review-derived-guardrails.md](references/review-derived-guardrails.md)
- Validate with the narrowest command that proves the change.
- Only scale up to full suites after the focused reproducer passes.

## Source of Truth

- Public ISA/API surface:
  - [include/pto/common/pto_instr.hpp](../../../include/pto/common/pto_instr.hpp)
  - [docs/PTO-Virtual-ISA-Manual.md](../../../docs/PTO-Virtual-ISA-Manual.md)
  - [docs/isa/README.md](../../../docs/isa/README.md)
- Backend implementations:
  - [include/pto/cpu](../../../include/pto/cpu)
  - [include/pto/npu/a2a3](../../../include/pto/npu/a2a3)
  - [include/pto/npu/a5](../../../include/pto/npu/a5)
- Toolchain and assembly contract:
  - [docs/assembly/README.md](../../../docs/assembly/README.md)
  - [docs/isa/syntax-and-operands/assembly-model.md](../../../docs/isa/syntax-and-operands/assembly-model.md)

## Working Rules

- Prefer CPU-SIM first for new behavior, regressions, and docs examples.
- Use `-v a3` for the shared A2/A3 backend and `-v a5` for A5-specific behavior.
- Assume backend constraints matter unless the virtual ISA explicitly says otherwise.
- Convert every review-discovered bug class into a focused regression test or docs validation check before broad refactoring.
- When documentation is touched, keep it hub-and-spoke: short overview here, deeper detail in reference docs.
- If an instruction works on one backend but not another, document whether it is:
  - virtual-ISA legal but backend-limited
  - backend-specific by design
  - an implementation gap

## Deliverables This Skill Supports

- Implement or debug a PTO instruction on CPU-SIM, A2/A3, or A5.
- Add or update tests in CPU-SIM, costmodel, or NPU ST.
- Trace why an instruction shape, dtype, or location is rejected.
- Compare PTO source, PTO-AS text, and backend lowering expectations.
- Update docs while keeping claims aligned with code.

## References

- [references/build-and-run.md](references/build-and-run.md)
- [references/constraints-and-platforms.md](references/constraints-and-platforms.md)
- [references/debugging.md](references/debugging.md)
- [references/review-derived-guardrails.md](references/review-derived-guardrails.md)
