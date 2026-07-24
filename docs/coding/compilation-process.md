# Compilation Process

This document describes the build and compilation flow for PTO Tile Lib from the perspective of source organization, public intrinsics, backend selection, and repository build entry points.

It focuses on the developer-visible workflow and does not expand undocumented internal compiler stages into normative interface descriptions.

## 1. Overview

PTO kernels are written in C++ using PTO intrinsics such as `TLOAD`, `TADD`, `TMATMUL`, `TSYNC`, and `TSTORE`.

The common public entry is:

```cpp
#include <pto/pto-inst.hpp>
```

The intrinsic layer is implemented primarily through headers under [PTO Public Headers](../../include/pto/README.md), especially `../../include/pto/common/pto_instr.hpp`.

## 2. Build and compilation characteristics

PTO Tile Lib uses a **C++ intrinsic interface**.

From the public API perspective, the library is primarily **header-based / template-based**.
The same PTO source can be built against different backends depending on build configuration.
CPU simulation is the recommended first validation path, while NPU execution depends on an Ascend CANN environment.
The codebase requires **C++20 or later**.

For project-level build guidance, see [Project Overview](../../README.md) and [Getting Started](../getting-started.md).

## 3. Build flow

At a high level, the build flow is:

```text
PTO C++ source
  -> C++ preprocessing / compilation
  -> PTO intrinsic headers select backend-specific implementations
  -> build system compiles test cases / kernels / demos
  -> binaries or test artifacts are produced
```

This description is intentionally written from the developer’s point of view to summarize the main relationship between source code and build artifacts.

This document does not define a complete proprietary compiler pipeline as a public contract, such as a fixed sequence of “frontend -> PTO intrinsic expansion -> middle-end IR -> backend lowering”. Such stages may exist in toolchains, but they are not presented here as normative interface definitions.

## 4. Public intrinsic layer and backend selection

The public intrinsic entry point is `../../include/pto/common/pto_instr.hpp`.

That header exposes APIs such as:

- `TASSIGN`
- `TSYNC`
- `TLOAD`
- `TSTORE`
- vector instructions such as `TADD`, `TMUL`, `TEXP`
- matrix instructions such as `TMATMUL`

The header also includes backend-specific implementation headers based on build conditions.

This means that, from a developer point of view, the compilation process is centered on:

1. writing C++ code against the PTO intrinsics
2. compiling it with the repository build configuration
3. letting the selected backend provide the concrete implementation path

## 5. Build tools used in this repository

The repository clearly depends on:

- **CMake**
- **Python** for scripts and tests
- a **C++20-capable compiler**

Typical commands used in this repository include:

```bash
# CPU simulation
python3 tests/run_cpu.py --clean --verbose

# Run a demo on CPU simulation
python3 tests/run_cpu.py --demo gemm --verbose

# Run ST on simulator backend
python3 tests/script/run_st.py -r sim -v a3 -t tadd -g TADDTest.case_float_64x64_64x64
```

If you are building in this repository, prefer the existing scripts and documented commands over inventing a standalone build flow.

## 6. CPU simulation path vs NPU path

### 6.1 CPU simulation

The CPU simulation path is intended for functional development and validation.

In this path:

- PTO intrinsics remain visible at the C++ source level
- backend behavior is modeled by the CPU simulation implementation
- some device-only synchronization details are simplified or become no-ops

Relevant documents:

- [CPU Simulation](cpu_sim.md)
- [Quickstart Tutorial](tutorial.md)
- [Events and Synchronization](Event.md)

### 6.2 NPU path

The NPU path targets Ascend hardware or simulator-side execution.

In this path:

- backend-specific NPU implementations are used
- device-side constraints matter more directly
- instruction availability must be checked against the backend support table

Relevant references:

- [Backend Implementation Status](../../include/README.md)
- [PTO ISA Reference](../isa/README.md)

## 7. Compilation-related checks

When a PTO kernel does not compile or run as expected, the most reliable checks are:

1. **Header-level API usage**
   - Is the intrinsic used according to `../../include/pto/common/pto_instr.hpp`?

2. **ISA constraints**
   - Does the instruction documentation under `docs/isa/` allow the tile type, layout, and operand combination?

3. **Tile and GlobalTensor definitions**
   - Are tile shapes, valid regions, and layouts legal?
   - Are `GlobalTensor` shape/stride declarations correct?

4. **Backend support**
   - Is the target instruction implemented on the selected backend according to [Backend Implementation Status](../../include/README.md)?

5. **Build environment**
   - Are the required compiler, Python environment, and CANN environment available?

## 8. Notes on build examples

Some commonly written build examples on the internet, such as generic `find_package(PTO REQUIRED)` snippets or imagined standalone `PTO::pto` link targets, are **not** established as the canonical integration model by this repository.

When documenting or extending PTO Tile Lib, use the repository build scripts, the top-level `CMakeLists.txt`, and existing test or demo build patterns as the primary reference.

## 9. Notes

The compilation flow of PTO Tile Lib can be summarized as follows:

- PTO code is written in C++ with public intrinsics.
- The build system selects the corresponding backend implementation according to configuration.
- CPU simulation is the preferred first validation path.
- Backend support and instruction legality are checked explicitly during development.

The documentation describes the public programming surface and usage model, while internal compiler stages remain implementation details unless stated otherwise in dedicated toolchain documents.
