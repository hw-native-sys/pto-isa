# Review-Derived Guardrails

Use this reference when a change touches an area that has recently produced PR review feedback. Each guardrail is anchored to concrete PR evidence and should turn into a focused test, validation command, or docs check.

## Cross-Core FIFO and Split-Lane Semantics

Evidence:
- PR #61: hook refactor review blocked on missing platform state propagation for dlopen'd kernel SOs and a pipe shared-state key that omitted `SlotSize`.
- PR #77 and PR #85: hook-backed V2C split push with runtime `subblock_dim=1` could record lane 0 but wait for lane 1, leaving the FIFO slot unpublished.
- PR #93, PR #95, and PR #100 continued to change `TPUSH`, `TPOP`, `TFREE`, cross-core compatibility, or fine-grained sync.

Guardrail:
- For `TPUSH`/`TPOP`/`TFREE`, check C2V and V2C, split and no-split, `subblock_dim=1` and `2`, hook and no-hook, plus the target backend (`cpu`, `a2a3`, `a5`).
- If a shared-state key or execution-context path changes, prove that all dimensions affecting storage layout and worker identity are represented.
- Add the smallest regression that observes publication state: `occupied`, `next_producer_slot`, `producers_done`, and `producers_allocated` where the API exposes those through CPU-SIM test hooks.

## Layout, Stride, and Offset Algebra

Evidence:
- PR #86 review flagged integer-overflow risk, an incorrect stride definition, and golden-data layout mismatch in ColMajor DN `TLoad`.
- PR #95 review flagged applying `entryOffset` to a row index instead of the memory base address, and an `isSplitN` branch that did not use subblock ID.
- PR #89 review flagged reversed stride parameter order in DMA documentation examples.

Guardrail:
- Write address math as named formulas before editing implementation code: base offset, row offset, column offset, split-lane offset, and entry offset.
- Use `size_t` or an explicitly wider type for shape, stride, byte, and slot-size products unless the implementation contract proves a narrower bound.
- Golden tests must include a case where RowMajor, ColMajor, flattened rows, and multi-column data would produce different output if the stride formula is wrong.
- Docs examples must keep source operands and source strides before destination operands and destination strides unless the documented ISA explicitly says otherwise.

## Backend Template Contracts

Evidence:
- PR #92 review asked for named constants and template validation after replacing runtime subblock detection with constexpr tile-type dispatch.
- PR #100 adds chunk-mode overloads that must stay coherent across public declarations and backend implementations.

Guardrail:
- Replace unexplained numeric constants with named `constexpr` values tied to tile type, pipe width, event count, or backend lane count.
- Add `PTO_STATIC_ASSERT`, `static_assert`, or a local concept check when a template assumes a tile category, layout, dtype, or backend-only resource.
- When adding overloads, update public declaration headers, CPU-SIM behavior, A2/A3, A5, and docs together or explicitly document the unsupported backends.

## ST Target and Build Closure

Evidence:
- PR #85 review caught a CMake target registration for `tpushpop_cv` without a matching source directory.
- PR #80 added broad CPU ST coverage and shared helpers, increasing the risk of stale runner registration.
- PR #95 was an A5 compile regression caused by a stale `TInsertMode` enum after an interface refactor.

Guardrail:
- For each new testcase, verify: directory exists, `CMakeLists.txt` exists, parent `CMakeLists.txt` registers it, the runner includes it if required, and the gtest filter names match the generated tests.
- For interface refactors, `rg` old enum/function/type names across `include/pto`, `tests`, `demos`, and docs before claiming closure.
- Prefer a targeted CPU-SIM command first; add `run_st.py -v a3` or `-v a5` when the touched path is backend-specific.

## Docs as Contract

Evidence:
- PR #78 aligned instruction docs with installed public intrinsics.
- PR #89 review found invalid parameter counts, missing `%tmp` operands, and a Pages-critical navigation target mismatch.
- PR #91 review found instruction-name typos and a broken Chinese README anchor.
- PR #98 review found taxonomy drift, copy-paste errors, and parameter-name consistency issues.

Guardrail:
- Treat `docs/isa` as the authored ISA source and `docs/mkdocs/src/docs` as generated or mirrored output unless the repository state says otherwise.
- Before docs PRs, run a strict MkDocs build, changed-file markdown link scan, and a `_zh` navigation closure check when Chinese pages are touched.
- Verify examples against `include/pto/common/pto_instr.hpp` and PTO-AS grammar; do not make friendlier names diverge from public headers.
- When moving an instruction between taxonomy groups, update family pages, surface pages, wrappers, MkDocs navigation, and both English/Chinese pages together.

## Performance-Sensitive Paths

Evidence:
- PR #100 benchmarked chunked `TPUSH` sync and showed that reducing per-chunk synchronization can be material.
- PR #80 review called out `TFMOD_IMPL` for needing layout-aware loops and parallelization.
- PR #86 introduced optimized flattened-row loading for ColMajor DN data.

Guardrail:
- For hot tile movement or arithmetic loops, review contiguous, strided, split, and fallback paths separately.
- Preserve layout-aware loop order; do not collapse row-major and col-major handling into a single generic loop without a performance reason and test coverage.
- Pair performance claims with a benchmark command, input shape, backend, and before/after numbers.
