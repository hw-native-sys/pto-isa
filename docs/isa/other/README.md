# Other Instruction Set

Other and communication operations cover behavior that does not fit cleanly into the tile, vector, or scalar/control buckets.

## Communication And Runtime

Inter-NPU collective communication and synchronization.

| Instruction Set | Description |
|--------|-------------|
| [TBROADCAST](../comm/TBROADCAST.md) | Broadcast data from root NPU to all ranks |
| [TGET](../comm/TGET.md) | Get data from a remote NPU |
| [TGET_ASYNC](../comm/TGET_ASYNC.md) | Asynchronous variant of TGET |
| [TNOTIFY](../comm/TNOTIFY.md) | Notify other ranks of an event |
| [TPUT](../comm/TPUT.md) | Put data to a remote NPU |
| [TPUT_ASYNC](../comm/TPUT_ASYNC.md) | Asynchronous variant of TPUT |
| [TREDUCE](../comm/TREDUCE.md) | Collective reduction across all ranks |
| [TSCATTER](../comm/TSCATTER.md) | Scatter data from root NPU to all ranks |
| [TGATHER](../comm/TGATHER.md) | Gather data from all ranks to root NPU |
| [TTEST](../comm/TTEST.md) | Test if a notification has been received |
| [TWAIT](../comm/TWAIT.md) | Wait for a notification |

See [Communication and Runtime](./communication-and-runtime.md) for the instruction set contract.

## Non-ISA Supporting Operations

Convenience operations over tile sequences or memory management.

| Operation | Description | Category |
|-----------|-------------|----------|
| [TALIAS](../TALIAS.md) | Create an alias view of a tile without copying | Alias |
| [TAXPY](../TAXPY.md) | Fused multiply-add: `dst = src0 * scalar + src1` | Fused compute |
| [TCONCAT](../TCONCAT.md) | Concatenate two tiles along a dimension | Tile sequence |
| [pto.tdequant](./ops/non-isa-and-supporting-ops/tdequant.md) | Dequantize a tile from quantized format | Quantize |
| [TFREE](../TFREE.md) | Free a previously allocated tile or buffer | Memory |
| [THISTOGRAM](../THISTOGRAM.md) | Compute histogram of tile values | Statistics |
| [TPACK](../TPACK.md) | Pack multiple tiles into a single tile buffer | Tile sequence |
| [TPOP](../TPOP.md) | Population count of predicate mask | Predicate |
| [TPUSH](../TPUSH.md) | Push count of predicate mask | Predicate |
| [TRANDOM](../TRANDOM.md) | Fill tile with random values | Generation |

See [Non-ISA and Supporting Ops](./non-isa-and-supporting-ops.md) for the instruction set contract.

## See Also

- [Other instruction set](../instruction-surfaces/other-instructions.md) — High-level instruction set description
- [Instruction set contracts](../instruction-families/README.md) — Normative contracts for all instruction sets
- [Instruction set overview](../instruction-surfaces/README.md) — Map of all four instruction sets
