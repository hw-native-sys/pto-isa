# System Scheduling Instruction Set

System scheduling instructions describe PTO-visible runtime protocols that coordinate tile-buffer lifetime and producer-consumer flow. This set is intentionally narrow: tile aliasing, tile-scalar compute, layout packing, quantization, random generation, and histogram work belong to the tile instruction families that define their payload semantics.

## Operations

| Operation | Description | Category |
| --- | --- | --- |
| [pto.tfree](./ops/TFREE.md) | Release a tile or buffer resource | Resource lifetime |
| [pto.tpop](./ops/TPOP.md) | Pop from a TPipe/TMPipe producer-consumer stream | Scheduling protocol |
| [pto.tpush](./ops/TPUSH.md) | Push into a TPipe/TMPipe producer-consumer stream | Scheduling protocol |

## Contract

System scheduling instructions are PTO ISA instructions. Their effects are visible through buffer ownership, resource lifetime, and TPipe/TMPipe ordering. A backend must preserve the documented protocol even when it lowers an operation into scalar synchronization or runtime steps.

## Related Pages

- [Instruction Set Overview](../instruction-families/README.md)
- [Tile ISA Reference](../tile/README.md)
- [Communication ISA Reference](../comm/README.md)
