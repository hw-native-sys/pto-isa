# System Scheduling Instruction Set

System scheduling instructions expose PTO-visible runtime protocols for tile-buffer lifetime and producer-consumer flow. They do not cover tile payload transformations; those instructions live in the tile families that define their data semantics.

## Instruction Overview

| Operation | PTO Name | Role |
| --- | --- | --- |
| Resource release | `pto.tfree` | End a tile or buffer resource lifetime |
| Producer-consumer push | `pto.tpush` | Publish tile work or resources into a TPipe/TMPipe stream |
| Producer-consumer pop | `pto.tpop` | Acquire tile work or resources from a TPipe/TMPipe stream |

## Contract

System scheduling operations are PTO ISA instructions. Their visible effects are observed through resource lifetime, stream state, and producer-consumer ordering. A backend may lower them through scalar or runtime mechanisms, but the resulting PTO-visible state must match the operation contract.

## Shared Constraints

- Resource-lifetime operations must not leave a later instruction with a dangling tile or buffer handle.
- Push/pop operations must define the stream, matching endpoint, and blocking behavior.
- Producer-consumer ordering must not be weakened by lowering to scalar synchronization or runtime calls.

## See Also

- [System scheduling reference](../system/README.md)
- [Instruction set contracts](./README.md)
- [Format of instruction descriptions](../reference/format-of-instruction-descriptions.md)
