# Distributed FFN GridPipe Demo

## Goal

`distributed_ffn_grid` is an A2/A3 NPU demo that validates the GridPipe programming model for a distributed FFN. It mocks the LPU WSE Grid `TPUSH/TPOP`, cross-core `SPR_RDY/SPR_FREE`, and `WFE` semantics with HCCL shared windows, GM flags, explicit `dcci/dsb` fences, and spin waits.

The current implementation is an M4/D-6 style end-to-end demo, not the older launch-only skeleton:

- The logical grid is `gridRows x gridCols`, defaulting to `1x2`.
- Rows are data-parallel: each row owns a different token slice.
- Columns are model-parallel: each column owns one shard of the FFN intermediate dimension.
- Each rank computes its local FFN partial, then reduces fp32 partials along `EAST` within the row.
- The last-column rank of each row writes the final `[T, H]` fp32 `yOutput` and compares it against the matching slice of `golden.bin` with `1e-3` tolerance.

This demo is meant to validate the GridPipe semantics, the A2/A3 mock backend, the host/runtime glue, and the complete FFN data flow. It is not silicon validation evidence for LPU WSE.

## Files

| File | Purpose |
| --- | --- |
| `README.md` / `README_zh.md` | English / Chinese documentation. |
| `CMakeLists.txt` | Builds the host executable plus the cube/compute and vec/comm device kernel shared libraries. |
| `run.sh` | One-shot helper that sets up CANN/MPI, generates data, configures CMake, builds, and launches the demo with `mpirun`. |
| `ffn_config.hpp` | Compile-time configuration: grid shape, tile shape, GridPipe slot/window sizes, buffer sizes, chunk flag indices, and PReLU alpha. |
| `kernel_launch.hpp` | Host-side launch declarations and the parameter contract for the two device kernels. |
| `main.cpp` | Host driver: MPI/HCCL/ACL bootstrap, HCCL window allocation, device buffer allocation, data loading, dual-stream launch, synchronization, debug peeks, golden comparison, and cleanup. |
| `distributed_ffn_grid_compute_kernel.cpp` | Cube kernel: computes `X @ W_gate`, `X @ W_up`, waits for hidden, then computes `hidden @ W_down`; coordinates with the comm kernel through chunk flags. |
| `distributed_ffn_grid_comm_kernel.cpp` | Vec/comm kernel: waits for gate/up partials, computes `PReLU(gate) * up -> hidden`, waits for down partial, then performs row-local `EAST` reduction with GridPipe `TPOP/TPUSH`. |
| `ready_queue.hpp` | Defines `ChunkFlagMatrix` and device-side flag helpers for producer/consumer synchronization between the cube and comm streams within one rank. |
| `gridpipe_payload_inl.hpp` | A2/A3 GridPipe payload hooks: resolves peer windows through `HcclRemotePtr`, and moves payloads between tiles and GridPipe GM slots with `TSTORE/TLOAD`. |
| `scripts/gen_data.py` | Generates per-rank fp16 X/weight shards and an fp32 golden reference. |
| `build/` | Generated build directory created by `run.sh` / CMake. |
| `out/` | Generated data directory containing `pe_<rank>_*.bin` files and `golden.bin`. |

## Execution Flow

1. `run.sh` parses runtime arguments. Defaults are `gridRows=1`, `gridCols=2`, `T=16`, `H=64`, `Fi=64`.
2. Unless `--build-only` is set, `scripts/gen_data.py` generates:
   - `pe_<rank>_x.bin`: `[T, H]` fp16.
   - `pe_<rank>_w_gate.bin`: `[H, Fi]` fp16.
   - `pe_<rank>_w_up.bin`: `[H, Fi]` fp16.
   - `pe_<rank>_w_down.bin`: `[Fi, H]` fp16.
   - `golden.bin`: `[gridRows * T, H]` fp32.
3. CMake builds three targets:
   - `distributed_ffn_grid_compute_kernel`: `dav-c220-cube`.
   - `distributed_ffn_grid_comm_kernel`: `dav-c220-vec`.
   - `distributed_ffn_grid`: host executable.
4. `mpirun -n N_RANKS ./distributed_ffn_grid` starts one MPI rank per NPU device.
5. The host initializes ACL/HCCL. Rank 0 obtains and broadcasts `HcclRootInfo`; each rank initializes its HCCL context.
6. Each rank carves two regions from its HCCL window:
   - `reduce_pipe_window`: GridPipe ready/free flags and per-direction ring slots.
   - `chunk_flag_shmem`: `ChunkFlagMatrix` for cube/comm stream synchronization.
7. The host allocates and zeroes device buffers: `x_dev`, `w_*_dev`, `gate_partial_dev`, `up_partial_dev`, `hidden_dev`, `down_partial_dev`, and `y_output_dev`.
8. The host copies X and weight shards from `out/` to the rank-local device buffers.
9. The host launches two kernels concurrently on two streams:
   - comm stream: `DistributedFfnGridCommKernel`.
   - compute stream: `DistributedFfnGridComputeKernel`.
10. The compute kernel computes gate/up partials and publishes chunk flags 0/1.
11. The comm kernel waits for flags 0/1, runs `TLRELU + TMUL + TCVT`, stores `hidden_dev`, and publishes chunk flag 2.
12. The compute kernel waits for flag 2, computes down partial, and publishes chunk flag 3.
13. The comm kernel waits for flag 3, loads down partial, and performs row-local `EAST` reduction:
    - `col > 0`: `TPOP<EAST>` receives the west neighbor partial and `TADD`s it.
    - `col + 1 < gridCols`: `TPUSH<EAST>` forwards the accumulated result to the east neighbor.
    - `col == gridCols - 1`: `TSTORE`s the reduced result to `yOutput`.
14. The host synchronizes both streams. The last-column rank of each row copies `yOutput` back and compares it with the matching row slice from `golden.bin`.
15. Rank 0 prints PASS/FAILED, and all ranks clean up resources.

Data-flow summary:

```text
host files
  -> x_dev, w_gate, w_up, w_down

compute/cube:
  X @ W_gate -> gatePartial --flag0-->
  X @ W_up   -> upPartial   --flag1-->

comm/vec:
  wait flag0/1
  hidden = fp16(PReLU(gatePartial) * upPartial)
  hidden_dev --flag2-->

compute/cube:
  wait flag2
  hidden @ W_down -> downPartial --flag3-->

comm/vec:
  wait flag3
  downPartial --GridPipe EAST reduce across cols--> yOutput on final col

host:
  yOutput vs golden row slice
```

## Key Designs

### 1. Grid TPUSH/TPOP mock on A2/A3

The LPU WSE design expects Grid `TPUSH/TPOP` to lower to a combination of `SPR` reads/writes, `WFE` waits, and payload movement. A2/A3 has no matching cross-core mesh SPR/event line, so this demo mocks the behavior with GM slots and GM flags inside an HCCL window.

Mock expansion of `TPUSH<EAST>`:

```text
1. Check that the current coord can push EAST.
2. If the ring is full, wait on the local free flag.
3. Compute the direction-local slot offset from prodIndex.
4. Resolve that local slot address into the east neighbor window through HcclRemotePtr.
5. TSTORE the tile into the neighbor slot.
6. Run pipe_barrier + dsb so the payload becomes visible before the signal.
7. Resolve the neighbor ready flag through HcclRemotePtr.
8. MockMtsprReady writes the ready counter and wakes the peer TPOP mock WFE.
9. Increment prodIndex.
```

Mock expansion of `TPOP<EAST>`:

```text
1. Check that the current coord can pop from the EAST channel.
2. Wait for the local ready flag to reach consIndex + 1.
3. Compute the local slot offset from consIndex.
4. TLOAD the slot into the tile.
5. Resolve the west neighbor free flag through HcclRemotePtr.
6. MockMtsprFree writes the free counter and releases the peer ring credit.
7. Increment consIndex.
```

`TPOP<EAST>` means "receive from the eastbound channel", where the producer is the west neighbor. It does not mean "read from the east neighbor".

### 2. Ready/free flags and ring credit

Each direction has independent ready/free counters:

- `ready`: the producer has written a slot and the consumer can read it.
- `free`: the consumer has read a slot and the producer can reuse it.

The flags are monotonic counters rather than booleans, so they compose with `prodIndex/consIndex` and `SlotCount` to model ring-buffer credit.

### 3. Cache coherence and fences

A2/A3 AICORE GM caches are not automatically coherent across cores. The mock SPR/WFE path cannot use plain loads and stores:

- `MockMtsprReady/Free` uses `dcci -> store -> dcci -> dsb(DSB_DDR)`.
- `MockWfeReady/Free` invalidates the flag cache line with `dcci` during spin polling.
- `TPUSH` inserts `pipe_barrier(PIPE_ALL)` and `dsb(DSB_DDR)` between payload `TSTORE` and the ready flag write.

These fences are the cost of the A2/A3 mock backend. They should not be interpreted as the intended kernel-author burden for final LPU WSE silicon.

### 4. Compute/comm split

The target LPU WSE design can be expressed as one logical kernel. The A2/A3 demo splits it into:

- cube `.so`: TMATMUL-heavy compute.
- vec `.so`: activation, cast, and GridPipe communication.

The two halves synchronize through `ChunkFlagMatrix` in GM. The cube side cannot directly include the comm instruction header, so several flag writes and waits use raw volatile pointers plus `dcci/dsb`.

### 5. Coordinate source

The comm kernel launches `gridRows * gridCols` blocks but only `block 0` runs the body. Therefore it must not derive its logical grid coordinate from `get_block_idx()`. The current code uses the host-provided `myRankId`:

```text
row = myRankId / gridCols
col = myRankId % gridCols
```

This ensures `TPUSH<EAST>` routes to the real peer rank.

### 6. SOURCE direction

The GridPipe layout still reserves the `SOURCE` direction, and semantically `SOURCE` is valid only for `TPOP`. The current M4 demo does not read X through `TPOP<SOURCE>`; the host copies X directly into `x_dev`, and the compute kernel loads from `x_dev`.

### 7. fp32 EAST reduction

The EAST reduce slot carries fp32 `[T, H]`, so `FFN_SLOT_BYTES = T * H * 4`. This avoids fp16 accumulation overflow and keeps both `yOutput` and `golden.bin` in fp32 for direct tolerance-based comparison.

## How to Run

### Build only

```bash
bash run.sh --build-only -v Ascend910B1
```

### Run on simulator

This is usually launched through `task-submit`:

```bash
task-submit --device auto --run "cd $(pwd)/kernels/manual/a2a3/distributed_ffn_grid && bash run.sh -r sim -v Ascend910B1 -n 2"
```

### Run on NPU

```bash
task-submit --device auto --run "cd $(pwd)/kernels/manual/a2a3/distributed_ffn_grid && bash run.sh -r npu -v Ascend910B1 -n 2"
```

### 2x2 example

```bash
task-submit --device auto --run "cd $(pwd)/kernels/manual/a2a3/distributed_ffn_grid && bash run.sh -r npu -v Ascend910B1 --grid-rows 2 --grid-cols 2"
```

`N_RANKS` defaults to `gridRows * gridCols`. If `-n/--n-ranks` is provided explicitly, it must match `gridRows * gridCols`, otherwise the host driver reports an MPI world-size mismatch.

### Common arguments

```text
-r, --run-mode      sim or npu, default npu
-v, --soc-version   default Ascend910B1
-n, --n-ranks       MPI rank count, default gridRows * gridCols
--grid-rows         grid row count, default 1
--grid-cols         grid column count, default 2
--token-tile        token tile T per row, default 16
--model-tile        hidden dim H, default 64
--ffn-tile          intermediate dim Fi per column, default 64
--build-only        build only; skip data generation and execution
```

### Expected result

On success, rank 0 prints:

```text
[SUCCESS] Distributed FFN GridPipe (M4) PASS.
```

The run fails with a non-zero exit status if the golden file is missing, the MPI rank count is wrong, stream synchronization fails, or ResultCmp fails on a last-column rank.
