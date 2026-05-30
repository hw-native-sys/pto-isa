# Single-device Multi-block FFN GridPipe Demo

## Goal

`distributed_ffn_grid_reducesum` validates a single-device logical FFN grid on A2/A3. The host runs one process on the selected device and launches `gridRows * gridCols` blocks. Each block owns one logical cell:

- Rows are data-parallel token slices.
- Columns are model-parallel FFN intermediate shards.
- The mixed Cube/Vec kernel computes gate/up, activation, down projection, and row-local `EAST` reduce in one launch.
- The last column in each row writes the final fp32 `[T, H]` output tile, which the host compares with `golden.bin` using `1e-3` tolerance.

The `EAST` reduce uses the A2/A3 GridPipe mock backend: local SRAM windows backed by GM in the mock, fake `HcclDeviceContext` window pointers, ready/free counters, `dcci/dsb` fences, and spin waits. This validates the programming model and same-device mock path; it is not multi-card communication validation.

An AllGather variant is also provided in `run_allgather.sh` / `distributed_ffn_grid_allgather`. It gathers hidden shards across columns before down projection, so the post-down ReduceSum is removed and each column writes its own output-H shard.

## Files

| File | Purpose |
| --- | --- |
| `README.md` / `README_zh.md` | English / Chinese documentation. |
| `CMakeLists.txt` | Builds the host executables and mixed Cube/Vec device kernel shared libraries. |
| `run_reducesum.sh` | Sets up CANN, generates ReduceSum data, configures CMake, builds, and runs the ReduceSum demo. |
| `run_allgather.sh` | Sets up CANN, generates AllGather data, configures CMake, builds, and runs the AllGather demo. |
| `ffn_config.hpp` | Compile-time grid shape, tile shape, GridPipe window sizes, buffer sizes, and PReLU alpha. |
| `kernel_launch.hpp` | Host-side mixed kernel launch declaration. |
| `main_reducesum.cpp` | ReduceSum host driver: ACL setup, fake HCCL context/local GridPipe windows, device buffers, data loading, kernel launch, golden comparison, and cleanup. |
| `distributed_ffn_grid_reducesum_compute_kernel.cpp` | ReduceSum mixed Cube/Vec kernel. Cube computes GEMMs; Vec computes activation/cast and GridPipe `EAST` reduce. |
| `main_allgather.cpp` | AllGather host driver. |
| `distributed_ffn_grid_allgather_compute_kernel.cpp` | AllGather mixed Cube/Vec kernel. Vec gathers hidden shards; Cube writes output-H shards. |
| `gridpipe_payload_inl.hpp` | Local GridPipe payload hooks and fake-window remote pointer adapter. |
| `../../../../include/pto/common/grid_counter_intrinsic.hpp` | CCE-intrinsic-style neighbor counter API used by GridPipe ready/free waits and notifications. |
| `../../../../include/pto/common/grid_sram_intrinsic.hpp` | CCE-intrinsic-style neighbor SRAM address and payload transfer API used by GridPipe payload movement. |
| `scripts/gen_data.py` | Generates per-cell fp16 X/weight shards and an fp32 golden reference. |
| `build/` | Ignored generated build directory. |
| `out/` | Ignored generated data directory. |

## Execution Flow

1. `run_reducesum.sh` parses arguments. Defaults are `gridRows=2`, `gridCols=2`, `T=16`, `H=64`, `Fi=64`, and `n-ranks=1`.
2. Unless `--build-only` is set, `scripts/gen_data.py` generates per-cell input and weight files plus `golden.bin`.
3. CMake builds two targets:
   - `distributed_ffn_grid_reducesum_mixed_kernel`: `dav-c220` mixed Cube/Vec.
   - `distributed_ffn_grid_reducesum`: host executable.
4. The host initializes ACL on the selected device.
5. The host allocates contiguous device buffers for `gridRows * gridCols` cells.
6. The host allocates one local GridPipe SRAM window per cell, backed by GM in the mock, and builds a fake `HcclDeviceContext`:

```text
windowsIn[cell] = reduce_pipe_windows_dev + cell * FFN_GRID_WINDOW_BYTES
rankNum = gridRows * gridCols
winSize = FFN_GRID_WINDOW_BYTES
```

7. The host loads X and weights into row-major per-cell buffers.
8. The host obtains the FFTS base address with `rtGetC2cCtrlAddr()` and launches `DistributedFfnGridMixedKernel` once with `gridRows * gridCols` blocks.
9. Inside each block, Cube and Vec branches exchange intermediate tiles through A2/A3 `TPipe` FIFOs:

```text
Cube:
  X[row] @ W_gate[col] -> gatePartial[row,col] --TPipe C2V-->
  X[row] @ W_up[col]   -> upPartial[row,col]   --TPipe C2V-->

Vec:
  hidden[row,col] = fp16(PReLU(gatePartial) * upPartial)
  hidden[row,col] --TPipe V2C-->

Cube:
  hidden[row,col] @ W_down[col] -> downPartial[row,col] --TPipe C2V-->

Vec:
  downPartial --GridPipe EAST reduce across cols--> yOutput[row] on final col
```

10. The host synchronizes the stream, checks GridPipe fault flags, copies `yOutput` back, and compares it with `golden.bin`.

## Key Designs

### Mixed Cube/Vec launch

The device kernel is compiled for `dav-c220`. Cube and Vec code paths are guarded by `__DAV_CUBE__` and `__DAV_VEC__`, so both sides live in one kernel source and synchronize through regular A2/A3 `TPipe` ready/free handshakes.

### Single-device logical grid

`get_block_idx()` is the row-major cell id:

```text
cell = get_block_idx()
row  = cell / gridCols
col  = cell % gridCols
```

All cells run on one device. `gridRows` controls data-parallel token tiles, and `gridCols` controls model-parallel FFN shards.

### Local GridPipe mock

The host allocates `gridRows * gridCols` local SRAM windows, backed by GM in the mock. `TPUSH<EAST>` resolves the east neighbor's SRAM slot with `get_neighbor_sram_addr`, writes the payload, then publishes the ready counter; `TPOP<EAST>` waits on the local ready counter, loads the local SRAM slot, and returns free credit to the west neighbor.

The mock uses GM flag polling and cache maintenance to emulate the intended LPU WSE `SPR` / `WFE` behavior on A2/A3.

### Neighbor counter intrinsic API

GridPipe ready/free synchronization goes through two CCE-intrinsic-style APIs in `include/pto/common/grid_counter_intrinsic.hpp`. The canonical call form keeps hardware semantic operands first and the mock backend operand last:

- `mtspr_neighbor_counter(kind, dir, value, operand)` publishes a monotonic `Ready` or `Free` counter to the neighbor-visible counter for `dir`.
- `wfe_neighbor_counter(kind, dir, threshold, operand, maxSpins)` waits until the local mirror of that counter reaches `threshold`.

GridPipe payload address resolution goes through `include/pto/common/grid_sram_intrinsic.hpp`:

- `get_neighbor_sram_addr(dst, src, dir, peerRank, operand)` resolves a local slot offset to the same offset in a neighbor SRAM slot address register.
- `copy_sram_to_neighbor_sram(dst, src, bytes, config)` writes local SRAM payloads to a neighbor SRAM slot.
- `copy_neighbor_sram_to_sram(dst, src, bytes, config)` is the read-side interface counterpart. Native lowering is intentionally only an interface placeholder for now; the A2/A3 mock uses it for TPOP-side validation against the GM-backed slot.

The current A2/A3 mock implements the SRAM write/read path with MTE copies through the GM-backed fake window. Once native hardware provides the corresponding write builtin, GridPipe call sites do not need to change.

`TPUSH<EAST>` waits on `Free` with `wfe_neighbor_counter`, writes the payload slot, then publishes `Ready` with `mtspr_neighbor_counter`. `TPOP<EAST>` waits on `Ready`, reads the payload slot, then publishes `Free` back upstream.

On current A2/A3 boards, `NeighborCounterOperand::addr` points to a GM-backed counter in the local/fake peer GridPipe window, and `NeighborSramOperand::runtimeCtx` points to the fake HCCL context. When real hardware supports neighbor SPR/WFE counters and neighbor SRAM address registers, this demo should be adapted by compiling GridPipe with `PTO_GRID_COUNTER_NATIVE_INTRINSIC` and `PTO_GRID_SRAM_NATIVE_INTRINSIC` and providing the compiler builtins. In that mode the mock operands are ignored, and the host/device setup should move from fake GM windows to hardware-provided per-neighbor counter/event registers and SRAM slot bases while keeping the GridPipe `TPUSH/TPOP` call sites unchanged.

### fp32 EAST reduction

The reduce slot carries fp32 `[T, H]`, so `FFN_SLOT_BYTES = T * H * 4`. This keeps `downPartial`, `yOutput`, and `golden.bin` in fp32 for direct tolerance-based comparison.

### AllGather variant

`run_allgather.sh` uses `scripts/gen_data.py --split-mode allgather` to generate data. In this mode, `W_down` is split as `[F, Hc]`, GridPipe carries fp16 `hidden [T, Fi]`, and each column writes one `Y[:, Hc]` output shard. The host still compares the stitched full output with `golden.bin`. AllGather requires `--model-tile` to be divisible by `--grid-cols` so `Hc = H / gridCols` is an integer tile width.

## How to Run

### Build only

```bash
bash run_reducesum.sh --build-only -v Ascend910B1 --grid-rows 2 --grid-cols 2
bash run_allgather.sh --build-only -v Ascend910B1 --grid-rows 2 --grid-cols 2
```

### Run on NPU

```bash
bash run_reducesum.sh -r npu -v Ascend910B1 --device-id 0 --grid-rows 3 --grid-cols 3
bash run_allgather.sh -r npu -v Ascend910B1 --device-id 0 --grid-rows 3 --grid-cols 3 --model-tile 96
```

### Common arguments

```text
-r, --run-mode      sim or npu, default npu
-v, --soc-version   default Ascend910B1
-n, --n-ranks       fixed to 1
-d, --device-id     selected ACL device id; defaults to FFN_GRID_DEVICE_ID, ASCEND_DEVICE_ID, DEVICE_ID, then 0
--grid-rows         logical grid row count, default 2
--grid-cols         logical grid column count, default 2
--token-tile        token tile T per cell, default 16
--model-tile        hidden dim H, default 64; AllGather requires H % gridCols == 0
--ffn-tile          intermediate dim Fi per column, default 64
--build-only        build only; skip data generation and execution
```

## Expected Result

On success, the ReduceSum executable prints:

```text
[SUCCESS] Single-device multi-block FFN GridPipe PASS.
```

On success, the AllGather executable prints:

```text
[SUCCESS] Single-device multi-block FFN GridPipe AllGather PASS.
```
