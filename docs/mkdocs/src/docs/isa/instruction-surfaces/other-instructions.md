<!-- Generated from `docs/isa/instruction-surfaces/other-instructions.md` -->

# Other Instruction Surface

The "other" surface covers operations that do not fit cleanly into the tile, vector, or scalar/control buckets. This includes inter-NPU communication, collective operations, and supporting operations that extend the core ISA.

## Surface Overview

Communication and supporting operations carry their own side effects and ordering rules that differ from the standard tile/vector/scalar model. These operations are architecturally visible but serve a different role:

- **Communication operations** express inter-NPU data exchange and collective reduction across parallel groups.
- **Supporting operations** provide convenience semantics over tile sequences or memory allocation.

These operations are **NOT available on the CPU simulator**. They require A2/A3 or A5 profiles with inter-NPU interconnect hardware.

## Instruction Classes

| Class | Description | Availability |
|-------|-------------|------------|
| Communication and Runtime | Inter-NPU collective communication | A2/A3, A5 |
| Non-ISA Supporting Ops | Convenience operations over tile sequences | All profiles |

### Communication And Runtime

These operations span multiple NPUs in a parallel group. They require a `ParallelGroup` handle and involve network or interconnect traffic.

| Operation | Description |
|-----------|-------------|
| `tbroadcast` | Broadcast data from root NPU to all ranks in parallel group |
| `tget` | Get data from a remote NPU |
| `tget_async` | Asynchronous variant of `tget` |
| `tput` | Put data to a remote NPU |
| `tput_async` | Asynchronous variant of `tput` |
| `treduce` | Collective reduction across all ranks in parallel group |
| `tscatter` | Scatter data from root NPU to all ranks |
| `tgather` | Gather data from all ranks to root NPU |
| `tnotify` | Notify other ranks of an event |
| `ttest` | Test if a notification has been received |
| `twait` | Wait for a notification |

### Non-ISA Supporting Operations

These operations provide higher-level semantics over tile sequences or memory management. Some are convenience wrappers that expand to multiple core ISA operations.

| Operation | Description | Target Profile |
|-----------|-------------|------------|
| `talias` | Create an alias view of a tile without copying data | All |
| `taxpy` | Fused multiply-add: `dst = src0 * scalar + src1` | All |
| `tconcat` | Concatenate two tiles along a specified dimension | All |
| `tdequant` | Dequantize a tile from quantized format | All |
| `tfree` | Free a previously allocated tile or buffer | All |
| `thistogram` | Compute histogram of tile values | A5 |
| `tpack` | Pack multiple tiles into a single tile buffer | A5 |
| `tpop` | Population count of predicate mask | All |
| `tpush` | Push count of predicate mask | All |
| `trandom` | Fill tile with random values | A5 |
| `tquant` | Quantize a tile to quantized format | All |

## Inputs

Other instructions consume combinations of:

- Parallel group handles (`!pto.group<N>`)
- Tile operands or tile sequences
- Scalar parameters (reduction operator, axis, scale/zero-point for quant, etc.)
- Allocation handles

## Expected Outputs

Other instructions produce:

- Modified tiles or tile sequences
- Scalar results (e.g., population count from `tpop`)
- Allocation state changes (e.g., freed buffers from `tfree`)

## Side Effects

| Class | Side Effects |
|-------|-------------|
| Communication And Runtime | Network/interconnect traffic; ordering across NPUs |
| Non-ISA Supporting Ops | May copy, allocate, or free memory; `tquant`/`tdequant` modify numeric representation |

## Constraints

- **Communication operations** require all participating NPUs to call the operation with matching `ParallelGroup` handles.
- **Non-root ranks** for collective operations (broadcast, scatter) must ensure destination buffers are allocated and writable for the duration of the operation.
- **Tile shape compatibility** for `tconcat` requires compatible dimensions along the concatenation axis.
- **Quantization parameters** for `tquant`/`tdequant` must be valid scale/zero-point values.
- **CPU simulator** does not support communication operations; using them produces a runtime error.

## Cases That Are Not Allowed

- Calling collective operations with mismatched `ParallelGroup` handles across ranks.
- Using uninitialized or improperly sized destination buffers for communication operations.
- Calling `tfree` on a tile that is still in use.
- Relying on `taxpy` being expanded to separate `tmul`/`tadd` on backends that do not implement it natively.
- Using A5-only operations (`thistogram`, `tpack`, `trandom`) on CPU or A2/A3 profiles.

## Syntax

### Assembly Form (PTO-AS)

```asm
tbroadcast %group, %src : (!pto.group<8>, !pto.tile<f32, 16, 16>)
treduce %group, %src, %dst : (!pto.group<8>, !pto.tile<f32, 16, 16>, !pto.tile<f32, 16, 16>) {op = "sum"}
```

### SSA Form (AS Level 1)

```mlir
%result = pto.tbroadcast %group, %src
    : (!pto.group<8>, !pto.tile<f32, 16, 16>) -> !pto.tile<f32, 16, 16>
```

See [Assembly Spelling And Operands](../syntax-and-operands/assembly-model.md) for the full syntax specification.

## C++ Intrinsic

Communication and supporting operations are declared in `include/pto/comm/pto_comm_inst.hpp` and `include/pto/common/pto_instr.hpp`:

```cpp
#include <pto/comm/pto_comm_inst.hpp>
using namespace pto::comm;

// Broadcast across a parallel group
template <typename GroupType, typename GlobalData, typename TileData>
PTO_INST RecordEvent TBROADCAST(GroupType& group, GlobalData& src, TileData& stagingTile);

// Collective reduction
template <typename GroupType, typename GlobalData, typename TileData, ReduceOp Op>
PTO_INST RecordEvent TREDUCE(GroupType& group, GlobalData& src, TileData& stagingTile);
```

## See Also

- [Other ISA reference](./other/README.md) — Full communication and supporting ops reference
- [Other families](./instruction-families/other-families.md) — Family-level contracts
- [Instruction families](./instruction-families/README.md) — All family groups
- [Format of instruction descriptions](../reference/format-of-instruction-descriptions.md) — Per-op page standard
