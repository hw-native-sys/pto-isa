# PTO Abstract Machine

This folder defines the abstract execution model used by the PTO ISA and PTO Tile Lib. It is intentionally written as a *programmer-facing* model: it describes what a correct program may assume, without requiring the reader to understand every micro-architectural detail of a specific device generation.

## Documents

- [PTO machine model (core/device/host)](abstract-machine.md)

## How this relates to other docs

- Data model and programming model:
  - [Tiles](../coding/Tile.md)
  - [Global memory tensors](../coding/GlobalTensor.md)
  - [Events and synchronization](../coding/Event.md)
  - [Scalar values and enums](../coding/Scalar.md)
- [Instruction reference](../isa/README.md)

