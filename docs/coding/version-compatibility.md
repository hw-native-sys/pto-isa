# Version Compatibility

This document summarizes version and platform compatibility in a way that is consistent with the current PTO Tile Lib repository.

It intentionally avoids unsupported claims about release lifecycles, deprecated historical APIs, or framework integration matrices that are not defined in this repository.

## 1. Compatibility Scope

PTO Tile Lib targets the following execution environments:

- **Ascend A2**
- **Ascend A3**
- **Ascend A5**
- **CPU simulation** (`x86_64` / `AArch64`)

The repository provides:

- a common C++ intrinsic interface under `include/pto/`
- backend-specific implementations for CPU simulation and supported NPU targets
- per-instruction ISA documentation under `docs/isa/`

For the project-level platform statement, see `README.md`. For backend implementation coverage by instruction, see `include/README.md`.

## 2. What “compatibility” means in PTO Tile Lib

In PTO Tile Lib, compatibility should be understood along three dimensions:

### 2.1 API compatibility

The public programming surface is defined by the C++ intrinsic headers, primarily:

- `include/pto/common/pto_instr.hpp`
- related public headers under `include/pto/`

Instruction semantics and usage constraints are documented in:

- `docs/isa/*.md`
- `docs/isa/comm/*.md`

When checking whether code is compatible, these headers and ISA documents are the source of truth.

### 2.2 Backend compatibility

Not every instruction is implemented on every backend.

For example, a PTO instruction may be:

- available on CPU simulation
- available on A2/A3 but not yet on A5
- documented in the ISA but still marked `TODO` for some backends

Therefore, backend compatibility must be checked per instruction rather than assumed globally.

The authoritative backend coverage table is maintained in `include/README.md`.

### 2.3 Program portability

A kernel that is valid in PTO source form may still require backend-specific review for:

- instruction availability
- tile layout constraints
- valid-region constraints
- performance tuning parameters
- manual placement and synchronization strategy

In practice, PTO Tile Lib encourages validating logic on CPU simulation first, then validating behavior and performance on the target Ascend platform.

## 3. Supported platforms

The repository currently documents the following main targets:

| Target | Notes |
| --- | --- |
| **Ascend A2** | NPU backend target |
| **Ascend A3** | NPU backend target |
| **Ascend A5** | NPU backend target |
| **CPU simulation** | Functional development and debugging path |

For platform naming and project overview, see `README.md`.

For per-instruction support status across CPU / Costmodel / A2 / A3 / A5 / Kirin, see `include/README.md`.

## 4. Build and environment notes

The current repository documentation and build scripts indicate the following practical expectations:

- **C++20 or later is required**.
- **CPU simulation** is the recommended first step for functional validation.
- **NPU execution** requires an Ascend CANN environment.
- Some CPU-simulation scenarios have additional compiler requirements; for example, repository guidance notes that **bfloat16 support on CPU simulation requires GCC >= 14**.

Typical entry points used in this repository include:

```bash
# CPU simulation
python3 tests/run_cpu.py --clean --verbose

# NPU / simulator-side ST execution
python3 tests/script/run_st.py -r sim -v a3 -t tadd -g TADDTest.case_float_64x64_64x64
```

For exact commands, use the project documentation and test scripts in the repository as the authoritative reference.

## 5. API and instruction compatibility guidance

### 5.1 Intrinsics and ISA docs are the source of truth

To determine whether an API usage is valid, check:

- the intrinsic declaration in `include/pto/common/pto_instr.hpp`
- the corresponding instruction page under `docs/isa/`

For example:

- `TASSIGN` defines manual address binding
- `TLOAD` / `TSTORE` define GM-to-tile and tile-to-GM movement
- `TSYNC` and event-related APIs define explicit ordering

### 5.2 Event compatibility is backend-sensitive

PTO Tile Lib supports an explicit event model, but the exact behavior depends on the backend:

- on device builds, typed `Event<SrcOp, DstOp>` objects are used to model dependencies
- on CPU simulation, synchronization behavior is simplified and some event-related paths act as no-ops

For the detailed event model, see [Event Programming Model](Event.md).

### 5.3 Auto mode vs manual mode

Compatibility should also be considered in terms of programming mode:

- **PTO-Auto** emphasizes direct dataflow expression
- **PTO-Manual** exposes explicit tile placement and explicit ordering

For example, `TASSIGN(tile, addr)` may be a no-op in auto mode depending on build configuration, while it is part of the manual placement workflow in manual mode.

See:

- [PTO ISA Quickstart](tutorial.md)
- [Tile Programming Model](Tile.md)
- [TASSIGN Instruction](../isa/TASSIGN.md)

## 6. Recommended compatibility workflow

When developing or porting a PTO kernel, the recommended process is:

1. **Write against the public PTO intrinsic interface**.
2. **Check instruction legality and constraints** in `docs/isa/`.
3. **Validate functional correctness on CPU simulation first**.
4. **Check backend support status** in `include/README.md`.
5. **Run and tune on the target Ascend platform**.

This workflow is more reliable than assuming that all documented instructions are implemented uniformly on all targets.

## 7. What this document does not guarantee

This document does **not** define:

- a formal semantic-versioning policy for PTO Tile Lib releases
- LTS / support-lifecycle commitments
- historical migration guarantees across unpublished API generations
- framework compatibility matrices outside what is explicitly documented in this repository

If such policies are introduced in the future, they should be documented in official release notes or dedicated policy documents.
