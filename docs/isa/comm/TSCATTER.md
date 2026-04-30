# pto.tscatter

`pto.tscatter` is part of the [Communication](./README.md) instruction set.

## Summary

Scatter operation: the calling NPU (root) distributes data to all ranks in the parallel group by splitting the local source tensor along **DIM_3** (row dimension). This is the inverse of `pto.tgather`.

Only the root needs to execute `pto.tscatter`. Non-root ranks only need to ensure their destination buffers are allocated and writable for the duration of the operation. Calling `pto.tscatter` on non-root ranks is undefined behavior.

**Large Tile Support**: When the per-rank data exceeds the UB tile capacity in rows and/or columns, the transfer is automatically chunked via 2D sliding.

## Mechanism

The local source tensor has shape $(D_0, D_1, D_2, N \times H, W)$, where $N$ is the number of ranks and each rank receives $H$ rows. After the operation:

$$\mathrm{dst}^{(r)}_{d_0, d_1, d_2,\; i,\; j} = \mathrm{src}^{\mathrm{local}}_{d_0, d_1, d_2,\; r \cdot H + i,\; j} \quad \forall\, r \in [0, N),\; i \in [0, H),\; j \in [0, W)$$

## Syntax

PTO-AS form: see [Assembly Spelling And Operands](../syntax-and-operands/assembly-model.md).

Synchronous form:

```text
pto.tscatter %group, %src : (!pto.group<...>, !pto.memref<...>)
```

Lowering introduces UB staging tile(s) for the GM→UB→GM data path; the C++ intrinsic requires explicit `stagingTileData` (or `pingTile` / `pongTile`) operand(s).

## C++ Intrinsic

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
// Basic scatter (single staging tile)
template <typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent SCATTER(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                             TileData &stagingTileData, WaitEvents&... events);

// Ping-pong scatter (double buffering with two staging tiles)
template <typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent SCATTER(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                             TileData &pingTile, TileData &pongTile, WaitEvents&... events);
```

## Inputs

| Operand | Description |
|---------|-------------|
| `parallelGroup` | A `ParallelGroup<GPerRank>` enumerating each rank's destination buffer; root identified via `GetRootIdx()`. |
| `srcGlobalData` | Source GlobalTensor on the root NPU; concatenation of per-rank slices along DIM_3. |
| `stagingTileData` | UB staging tile used as the GM→UB→GM relay buffer (single-buffer form). |
| `pingTile`, `pongTile` | Two UB staging tiles for double-buffered (ping-pong) form, enabling MTE2/MTE3 overlap. |
| `events...` | Optional `RecordEvent` tokens to wait on before issuing. |

## Expected Outputs

| Result | Type | Description |
|--------|------|-------------|
| `RecordEvent` | token | Signals completion of the scatter to all participating ranks. |
| `parallelGroup.tensors[r]` | GlobalTensor (remote) | Each rank `r`'s destination receives `src[r*H : (r+1)*H, :]` of the source. |

## Side Effects

- Issues local MTE2 reads from `srcGlobalData` and remote MTE3 writes to each rank's destination buffer.
- Cross-core synchronisation flags are toggled as part of the rank-fan-out protocol.
- Non-root ranks must keep their destination buffers writable until the root signals completion; otherwise behavior is undefined.
- No implicit fence on unrelated tile traffic.

## Constraints

!!! warning "Constraints"
    - **Type constraints**:
        - `ParallelGroup::value_type::RawDType` must equal `GlobalSrcData::RawDType`.
        - `TileData::DType` must equal `GlobalSrcData::RawDType`.
    - **Memory constraints**:
        - `srcGlobalData` must point to local memory (current NPU) and be large enough to hold data for all ranks. Specifically, `srcGlobalData.GetShape(DIM_3)` must be $\geq N \times H$ where $H$ is each rank's `GetShape(DIM_3)`.
        - If `srcGlobalData.GetShape(DIM_3) > N × H`, only the first `N × H` rows are read; remaining rows are ignored.
        - `stagingTileData` (or `pingTile` / `pongTile`) must be pre-allocated in UB.
    - **ParallelGroup constraints**:
        - `parallelGroup.tensors[r]` must refer to rank `r`'s destination buffer (remote GM as seen by the root).
        - `parallelGroup.GetRootIdx()` identifies the calling NPU as the scatter root.
        - All destination tensors are assumed to have the same shape and strides; behavior is undefined if they differ.
    - **Chunked mode constraints** (when per-rank data exceeds a single UB tile):
        - If `TileData` has static `ValidRow`, `GetShape(DIM_3)` of each rank's destination must be divisible by `ValidRow`. Use a Tile with `DYNAMIC` ValidRow for partial row support.
        - If `TileData` has static `ValidCol`, `GetShape(DIM_4)` must be divisible by `ValidCol`. Use a Tile with `DYNAMIC` ValidCol for partial column support.

## Performance

### A2/A3 Cycle Count

`pto.tscatter` is bounded by the **inter-NPU MTE3** path: the root reads its local source buffer via MTE2 and writes to each rank's remote destination over the cross-core fabric.

**Cycle model**:

```
total ≈ startup + N × (per_rank_local_load + per_rank_remote_store) + sync_overhead
```

where `N` is the participating rank count.

The ping-pong form overlaps local MTE2 of chunk `k+1` with remote MTE3 of chunk `k`, hiding most of `per_rank_local_load` behind `per_rank_remote_store`.

### Layout and Shape Impact

| Form | When to use | Effect |
|------|------------|--------|
| Single staging tile | Small per-rank data (≤ one tile) | Simpler control flow; serialised MTE2/MTE3 |
| Ping-pong (double buffering) | Large per-rank data | MTE2/MTE3 overlap; near-2× throughput on long transfers |
| 2D sliding (per-tile chunking) | Per-rank data > UB tile | Automatic; chunk size set by `TileData::ValidRow/ValidCol` |

> Note: cycle numbers are first-order estimates; populate with measured values from `pto-isa/a2a3_benchmark.csv` and `pto-isa/a5_benchmark.csv`.

## Exceptions

!!! danger "Exceptions"
    - Calling `pto.tscatter` on a non-root NPU is undefined behavior.
    - Mismatched per-rank tensor shapes / strides yield undefined behavior; no runtime check is guaranteed.
    - Using a `srcGlobalData` shape with `GetShape(DIM_3) < N × H` is rejected by the verifier on static shapes; on dynamic shapes the call reads only what fits and remaining ranks receive partial data.
    - Type-mismatch between `ParallelGroup::value_type::RawDType` and `TileData::DType` / `GlobalSrcData::RawDType` is rejected at compile time via `static_assert`.
    - Programs must not rely on behavior outside the documented legal domain of this operation.

## Examples

### Basic Scatter (Single Staging Tile)

Root has `NRANKS * ROWS` rows of width `COLS`. Each rank receives `ROWS × COLS`, split along DIM_3.
The tile size (`TILE_ROWS × TILE_COLS`) can be smaller than the per-rank data — when it is, the implementation automatically chunks the transfer along both DIM_3 and DIM_4 via 2D sliding.

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int ROWS, int COLS, int TILE_ROWS, int TILE_COLS, int NRANKS>
void scatter(__gm__ T* local_data, __gm__ T* group_addrs[NRANKS], int my_rank) {
    using TileT = Tile<TileType::Vec, T, TILE_ROWS, TILE_COLS, BLayout::RowMajor, -1, -1>;
    using GPerRank = GlobalTensor<T, Shape<1,1,1,ROWS,COLS>,
                                  BaseShape2D<T, ROWS, COLS, Layout::ND>, Layout::ND>;
    using GSource = GlobalTensor<T, Shape<1,1,1,NRANKS*ROWS,COLS>,
                                  BaseShape2D<T, NRANKS*ROWS, COLS, Layout::ND>, Layout::ND>;

    GPerRank tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) {
        tensors[i] = GPerRank(group_addrs[i]);
    }

    comm::ParallelGroup<GPerRank> group(tensors, NRANKS, my_rank);
    GSource srcG(local_data);
    TileT stagingTile(TILE_ROWS, TILE_COLS);

    comm::SCATTER(group, srcG, stagingTile);
}
```

### Ping-Pong Scatter (Double Buffering)

Uses two UB tiles to overlap TLOAD of the next chunk (MTE2) with TSTORE of the current chunk (MTE3).

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int ROWS, int COLS, int TILE_ROWS, int TILE_COLS, int NRANKS>
void scatter_pingpong(__gm__ T* local_data, __gm__ T* group_addrs[NRANKS], int my_rank) {
    using TileT = Tile<TileType::Vec, T, TILE_ROWS, TILE_COLS, BLayout::RowMajor, -1, -1>;
    using GPerRank = GlobalTensor<T, Shape<1,1,1,ROWS,COLS>,
                                  BaseShape2D<T, ROWS, COLS, Layout::ND>, Layout::ND>;
    using GSource = GlobalTensor<T, Shape<1,1,1,NRANKS*ROWS,COLS>,
                                  BaseShape2D<T, NRANKS*ROWS, COLS, Layout::ND>, Layout::ND>;

    GPerRank tensors[NRANKS];
    for (int i = 0; i < NRANKS; ++i) {
        tensors[i] = GPerRank(group_addrs[i]);
    }

    comm::ParallelGroup<GPerRank> group(tensors, NRANKS, my_rank);
    GSource srcG(local_data);
    TileT pingTile(TILE_ROWS, TILE_COLS);
    TileT pongTile(TILE_ROWS, TILE_COLS);

    comm::SCATTER(group, srcG, pingTile, pongTile);
}
```

## See Also

- Instruction set overview: [Communication](./README.md)
- Inverse op: [pto.tgather](./TGATHER.md)
- Related collective ops: [pto.treduce](./TREDUCE.md), [pto.tbroadcast](./TBROADCAST.md)
- One-sided variants: [pto.tput](./TPUT.md), [pto.tget](./TGET.md)
