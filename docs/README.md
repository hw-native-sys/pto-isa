<p align="center">
  <img src="figures/pto_logo.svg" alt="PTO Tile Lib" width="200" />
</p>

# PTO ISA Documentation Guide

This page is the main documentation entry for PTO Tile Lib. It helps readers locate documents by topic instead of navigating directories one by one.

The PTO documentation mainly covers the following areas:

- ISA fundamentals and an overall reading path
- Instruction indexes and per-instruction reference pages
- PTO assembly spelling as part of the ISA syntax-and-operands chapter
- Tile programming model, event synchronization, and performance tuning
- Getting started, test execution, and documentation build instructions

## Recommended Reading Path

If you are new to PTO Tile Lib, we recommend reading in the following order:

1. [Getting Started](getting-started.md): set up the environment and run the CPU simulator first
2. [ISA Overview](PTOISA.md): build an overall understanding of the PTO ISA
3. [PTO Instruction List](isa/scalar/ops/micro-instruction/README.md): browse the standard operations by category
4. [Tile Programming Model](coding/Tile.md): understand tile shape, tile mask, and data organization
5. [Events and Synchronization](coding/Event.md): understand set/wait flag usage and pipeline synchronization
6. [Performance Optimization](coding/opt.md): understand common bottlenecks and tuning directions

## Documentation Categories

### 1. ISA and Instruction Reference

- [Virtual ISA Manual Entry](PTO-Virtual-ISA-Manual.md): top-level entry for the PTO ISA manual
- [ISA Overview](PTOISA.md): background, goals, and overall structure of the PTO ISA
- [PTO Instruction List](isa/scalar/ops/micro-instruction/README.md): index of PTO standard operations organized by category
- [General Conventions](isa/conventions.md): common naming rules, constraints, and usage conventions

### 2. PTO Syntax and Representation

- [Assembly Spelling and Operands](isa/syntax-and-operands/assembly-model.md): canonical PTO-AS spelling and operand syntax inside the PTO ISA manual

### 3. Programming Model and Development Notes

- [Development Documentation Index](coding/README.md): entry for developer-facing PTO Tile Lib documentation
- [Tile Programming Model](coding/Tile.md): tile shape, tile mask, and data layout
- [Events and Synchronization](coding/Event.md): event recording, waiting, and synchronization behavior
- [Performance Optimization](coding/opt.md): performance analysis and tuning guidance
- [CostModel Backends (stub / fit)](costmodel-backends.md): backend split, formula-fit path, and run guide

### 4. Getting Started, Testing, and Documentation Build

- [Getting Started](getting-started.md): environment setup and CPU / NPU execution guide
- [Test Guide](../tests/README.md): test entry points, scripts, and common commands
- [Documentation Build Guide](mkdocs/README.md): how to build the docs locally with MkDocs

### 5. Other Related Documents

- [Machine Documentation](machine/README.md): abstract machine model and related notes

## Directory Structure

Key entries are listed below:

```text
├── isa/                        # PTO instruction reference and category indexes
├── coding/                     # Programming model, development, and optimization docs
├── auto_mode/                  # Auto Mode related documents
├── machine/                    # Abstract machine model documents
├── mkdocs/                     # Documentation site build config and scripts
├── figures/                    # Images and diagram assets used in docs
├── README*                     # Documentation entry pages
├── PTOISA*                     # ISA overview documents
└── getting-started*            # Getting started guides
```

## Related Entry Points

- [Root README](isa/scalar/ops/micro-instruction/README.md): project overview, quick start, and repository entry page
- [kernels Directory Guide](../kernels/README.md): kernel and operator implementation entry point
- [include Directory Guide](../include/README.md): headers and public interface overview
- [tests Directory Guide](../tests/README.md): testing and execution entry point
