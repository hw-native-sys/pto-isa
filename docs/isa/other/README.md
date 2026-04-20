# Other Instruction Set

Other and communication operations cover behavior that does not fit cleanly into the tile, vector, or scalar/control buckets.

## Two Categories

### Communication and Runtime

Inter-NPU collective communication and synchronization primitives.

| Instruction | Description | Sync Type |
|-------------|-------------|-----------|
| [TBROADCAST](../comm/TBROADCAST.md) | Broadcast data from root NPU to all ranks | Sync |
| [TGET](../comm/TGET.md) | Get data from a remote NPU | Sync |
| [TGET_ASYNC](../comm/TGET_ASYNC.md) | Asynchronously get data from a remote NPU | Async |
| [TPUT](../comm/TPUT.md) | Put data to a remote NPU | Sync |
| [TPUT_ASYNC](../comm/TPUT_ASYNC.md) | Asynchronously put data to a remote NPU | Async |
| [TNOTIFY](../comm/TNOTIFY.md) | Notify other ranks of an event | Sync |
| [TWAIT](../comm/TWAIT.md) | Wait for a notification | Sync |
| [TTEST](../comm/TTEST.md) | Test if a notification has been received | Sync |
| [TGATHER](../comm/TGATHER.md) | Gather data from all ranks to root NPU | Sync |
| [TSCATTER](../comm/TSCATTER.md) | Scatter data from root NPU to all ranks | Sync |
| [TREDUCE](../comm/TREDUCE.md) | Collective reduction across all ranks | Sync |

[Communication and Runtime contract →](./communication-and-runtime.md)

### Non-ISA Supporting Operations

Convenience operations over tile sequences or memory management.

| Operation | Description | Category |
|-----------|-------------|----------|
| [TALIAS](../TALIAS.md) | Create an alias view of a tile without copying | Alias |
| [TAXPY](../TAXPY.md) | Fused multiply-add: `dst = src0 * scalar + src1` | Fused compute |
| [TCONCAT](../TCONCAT.md) | Concatenate two tiles along a dimension | Tile sequence |
| [TDEQUANT](../TDEQUANT.md) | Dequantize a tile from quantized format | Quantize |
| [TFREE](../TFREE.md) | Free a previously allocated tile or buffer | Memory |
| [THISTOGRAM](../THISTOGRAM.md) | Compute histogram of tile values | Statistics |
| [TPACK](../TPACK.md) | Pack multiple tiles into a single tile buffer | Tile sequence |
| [TPOP](../TPOP.md) | Population count of predicate mask | Predicate |
| [TPUSH](../TPUSH.md) | Push count of predicate mask | Predicate |
| [TRANDOM](../TRANDOM.md) | Fill tile with random values | Generation |
| [TQUANT](../TQUANT.md) | Quantize a tile to integer format | Quantize |

[Non-ISA and Supporting Ops contract →](./non-isa-and-supporting-ops.md)

## See Also

| Page | Content |
|------|---------|
| [Instruction overview](../instruction-surfaces/other-instructions.md) | High-level description of Other instruction set |
| [Instruction families](../instruction-families/README.md) | Normative contracts for all instruction sets |
| [Instruction set overview](../instruction-surfaces/README.md) | Map of all four instruction sets |
