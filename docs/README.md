<p align="center">
  <img src="../figures/pto_logo.svg" alt="PTO Tile Lib" width="200" />
</p>

# PTO ISA Guide

This directory documents the PTO ISA (Instruction Set Architecture) used by PTO Tile Lib. It explains instruction naming, common notation, and how to navigate the per-instruction reference pages.

## Naming and Notation

- **Tile**: Fundamental data type for small tensors (e.g., `MatTile`, `LeftTile`, `RightTile`, `BiasTile`, `AccumulationTile`, `VecTile`).
- **GlobalTensor**: A tensor stored in global memory (GM). `TLOAD`/`TSTORE` move data between GM and Tiles.
- **`%R`**: A scalar immediate register. Fields like `cmpMode` and `rmode` are instruction modifiers.
- **Shape and alignment**: Enforced by a combination of compile-time constraints and runtime assertions; invalid usage should fail fast.

## Where to Start

- [Virtual ISA manual entry](PTO-Virtual-ISA-Manual.md)
- [ISA overview](PTOISA.md)
- [Instruction index](isa/README.md)
- [PTO IR index](ir/README.md)
- [Common conventions](isa/conventions.md)
- [PTO assembly syntax reference (PTO-AS)](grammar/PTO-AS.md)
- [Virtual ISA / IR guide](../manual/09-virtual-isa-and-ir.md)
- [Bytecode / toolchain guide](../manual/10-bytecode-and-toolchain.md)
- [Memory ordering / consistency guide](../manual/11-memory-ordering-and-consistency.md)
- [Backend profiles / conformance guide](../manual/12-backend-profiles-and-conformance.md)
- [Programming model (Tiles/GlobalTensor/Events/Scalars)](coding/ProgrammingModel.md)
- [PTO ISA programming tutorial (C++ intrinsics)](coding/tutorial.md)
- [Abstract machine model (core/device/host)](machine/abstract-machine.md)
- [Getting started (recommended: run on CPU first)](getting-started.md)
- [Doc tooling (manifest/index/svg/consistency)](tools/)
- [Implementation and extension notes](coding/README.md)
- [Kernel examples (NPU-focused)](../kernels/README.md)

## Documentation Layout

- `docs/isa/`: Instruction reference (one file per instruction, plus category pages)
- `docs/grammar/`: PTO assembly grammar and specification (PTO-AS)
- `docs/coding/`: Developer notes for extending PTO Tile Lib
