# 4. Tiles and GlobalTensor

## 4.1 Scope

This chapter defines data-model contracts between tile operands and global memory operands.
It specifies architecture-visible movement and interpretation rules.

## 4.2 Tile data model

A tile is the primary architectural data object for compute instructions.
A tile contract includes:

- element type and shape class
- valid-region metadata (`Rv`, `Cv`)
- location-intent role where required by instruction legality
- layout/alignment properties required by backend legality

## 4.3 GlobalTensor data model

A GlobalTensor (or equivalent memory view) represents addressable global-memory data.
Its architecture-visible contract includes:

- element type compatibility with participating tile operations
- address and stride interpretation required by memory instructions
- visibility behavior under ordering constraints

## 4.4 GM <-> Tile movement contracts

`TLOAD` and `TSTORE` families define the primary GM <-> Tile bridge.
Conforming implementations MUST preserve:

- element mapping semantics in the defined valid domain
- required ordering guarantees under event/event synchronization and memory model rules
- documented behavior of quantization/scaling and mode attributes where present

## 4.5 Shape and domain compatibility

For movement and layout-transform operations:

- source and destination domains MUST satisfy instruction-specific compatibility constraints
- out-of-domain behavior MUST be either explicitly defined (for example pad/fill) or declared unspecified
- backend legality checks MUST reject unsupported shape/layout tuples deterministically

## 4.6 Layout-transform operations

Operations such as extract/insert/reshape/transpose are architecture-level transforms over tile domains.
They MUST define:

- index-space mapping
- valid-domain mapping
- behavior for partially covered domains
- implementation-defined constraints where hardware-specific behavior exists

## 4.7 Diagnostics requirements

Movement/layout diagnostics SHOULD report:

- offending operand and operation
- incompatible shape/layout/location dimensions
- relevant index/offset parameter context
- deterministic wording for reproducible CI behavior
