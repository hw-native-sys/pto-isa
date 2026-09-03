# CPU_SIM
CPU_SIM is a CPU-based backend designed for execution on CPU-only systems.
It has some limitations and differences comparing to NPU backends at this moment:
- PTO instructions execute synchronously in each CPU worker. Supported synchronization and communication operations,
  including the TileData `TPUSH`/`TPOP`/`TFREE` FIFO flow, are simulated with CPU synchronization primitives.
- Specific memory model to mimic NPU memory (see below)
- Multithreading support is not complete. Memory access in Tile objects is not synchronized across threads, so tiles
  should not be shared across threads except through supported communication operations. Tile lazy allocation is also
  not synchronized.

## Enabling CPU_SIM
You may enable CPU backend (CPU_SIM) by setting `__CPU_SIM` compiler definition. In this case, programs can be built using standard CPU-targeted compiler (gcc or clang).

Note, for compatibility with NPU-based programs, some Ascend-specific functions are implemented for CPU platform in **include/pto/common/cpu_stub.hpp**. Including this file into existing program already using NPU-backend will make it compilable for CPU with only minor changes. You may not include this file, but in this you'll have to remove all functions like aclInit, aclrtSetDevice from your code or replace them with corresponding CPU-based code if needed.

The CPU stubs for `aclInit`, `aclrtSetDevice`, and `aclrtCreateStream` initialize the CPU_SIM runtime, retain the
selected non-negative device ID, and create a lightweight host stream handle. Runtime environment variables are read
by `aclInit` or by the first API that lazily initializes the runtime, whichever occurs first.

## CPU_SIM memory model
Generally, all tiles memory in CPU_SIM is allocated in system memory (contrary to NPU backend where memory is divided into host and device memory, and device itself has several different memory locations). But to make CPU_SIM memory model closer to NPU, it simulates separate memory locations corresponding to NPU architecture.

CPU_SIM memory model allocates following memory locations for each thread: UB, L1, L0A, L0B, L0C. Each of this locations is basically pre-allocated array of the size corresponding to simulating NPU architecture. TASSIGN operation uses one of these arrays to assign some memory chunk from it to the tile. I.e., if TASSIGN is called for the tile with Loc==Mat and offset 10, it will assign memory starting from the L1[10] to that tile.

Currently A2A3 and A5 architectures supported, specific architecture can be chosen using pto::NPUMemoryModel::Initialize function, that should be called once for each thread (can be omitted, in this case default A2A3 architecture will be used). For more information please refer to **include/pto/cpu/NPUMemoryModel.hpp**

The simulated memory capacities can be overridden with the following environment variables. Values are specified in
bytes and must be positive integers:

- `PTO_CPU_SIM_UB_BYTES`
- `PTO_CPU_SIM_L1_BYTES`
- `PTO_CPU_SIM_L0A_BYTES`
- `PTO_CPU_SIM_L0B_BYTES`
- `PTO_CPU_SIM_L0C_BYTES`

By default, CPU_SIM provides at least 512 KiB of UB scratch space. Set the variables before initializing the memory
model.

When `__PTO_AUTO__` is defined, regular `Tile` objects support lazy fallback storage in CPU_SIM. If a tile has not
been bound by `TASSIGN`, its first `data()` access allocates private host storage. Without `__PTO_AUTO__`, regular
tiles must be bound explicitly before access.

Fallback storage is allocated from host memory regardless of the tile location and does not overlap the simulated UB,
L1, L0A, L0B, or L0C buffers. Use `TASSIGN` when the simulated memory location, offset, aliasing, or communication
behavior matters. Tile abstractions that do not provide lazy fallback storage must still be explicitly bound.

To avoid concurrent first access in `__PTO_AUTO__` mode, the CPU_SIM implementation of `TMATMUL` materializes the
backing storage for the destination, optional accumulator, and both matrix input Tiles on the caller thread before
launching parallel workers. The `TMATMUL_MX` path also materializes both scale Tiles.

### To summarize:
For regular `Tile` objects, use one of these strategies:

- **Direct memory assignment:** Every tile should have corresponding TASSIGN operation call to assign memory directly. Proper offset should be calculated manually and provided to TASSIGN operation.
- **Lazy fallback storage:** Define `__PTO_AUTO__`, then leave the tile unbound and let its first `data()` access
  allocate private host storage.

Direct assignment models the configured NPU memory regions. Lazy fallback storage is intended for private CPU-side
correctness testing and does not model a specific on-chip address.

## Supported behavior and backend differences

- CPU_SIM `TADD` and `TABS` accept independently typed operand Tiles when their element types and runtime
  valid shapes match. This includes mixing static and dynamic `ValidRow`/`ValidCol` template arguments. Each operand
  is indexed using its own Tile layout and physical shape; a runtime valid-shape mismatch triggers an assertion.
- CPU_SIM implements both `TCI(dst, start)` and `TCI(dst, start, tmp)`. The three-argument form accepts but does not
  access `tmp`, and otherwise has the same ascending or descending sequence semantics as the two-argument form.
  Keep the target-specific scratch allocation required by the NPU backend when writing portable kernels.
- CPU_SIM implements all four `TCVT` overloads, with or without an explicit scratch Tile and `SaturationMode`.
  Scratch-Tile forms accept but do not access `tmp` and match the corresponding no-scratch conversion. The default
  CPU_SIM saturation mode is `SaturationMode::OFF`; portable kernels must retain any NPU scratch allocation.
- `TSTORE` from a `TileType::Vec` tile with `SLayout::NoneBox` picks its GM traversal from the tile's own layout
  in one case, the way the hardware DMA does: the single-row or single-column pairing that `TSTORE` allows on top
  of matching ND/DN/NZ layouts. A ColMajor `[N, 1]` tile stored through an ND `GlobalTensor`, or a RowMajor
  `[1, N]` tile stored through a DN one, lands as a contiguous vector instead of one element per the other axis'
  stride. Every other combination, including a ColMajor tile wider than one column, still takes its mapping from
  the `GlobalTensor` layout. `TLOAD` needs no such case: `TileType::Vec` loads only accept matching layouts.
- TileData `TPUSH`/`TPOP`/`TFREE` use a host-side `TPipe` FIFO model. The model waits for free slots and ready data,
  keeps C2V and V2C traffic separate for `Direction::DIR_BOTH`, and coordinates split lanes through the simulated
  block/subblock context. TileData payloads stay in host-owned shared slot storage even if generated code supplies a
  non-null NPU GM workspace; CPU_SIM does not access that workspace for this flow. `TFREE` participates in the CPU
  FIFO release protocol; it is not the A2A3 TileData no-op. For `TileSplitAxis::TILE_NO_SPLIT`, a TileData `TPUSH`
  lays the slot payload out with the shape of the window it actually transfers, that is the pushed tile's valid
  shape, so pushing a narrow view of a wider tile stays row-aligned with the tile the consumer pops it into; the
  split axes keep using the producer tile's declared shape. The public GlobalData
  `TALLOC`/`TPUSH`/`TPOP`/`TFREE` flow is not currently available in CPU_SIM.
- The Tile-vs-Tile `TCMPS` overload compares `src0[i,j]` with `src1[i,j]` in CPU_SIM. This matches A5 and differs
  from the A2/A3 scalar-broadcast behavior. The scalar overload has the usual scalar comparison semantics.
- CPU arg-reduce implementations (`TCOLARGMIN`, `TCOLARGMAX`, `TROWARGMIN`, and `TROWARGMAX`) accept integral,
  `half`, `bfloat16_t`, and `float` source elements. Index outputs must use `int32_t` or `uint32_t`; the temporary tile
  argument is unused by CPU_SIM.
- CPU_SIM supports the AIV paths of `TGATHER` and `TSCATTER`, but does not provide a functional `CollEngine::CCU`
  path. The CPU headers include deferred compile-time rejection for unsupported CCU calls; use the A5 NPU backend.
- `SYNCALL`, including its workspace-bearing Soft forms, is currently a compatibility no-op in CPU_SIM. It must not
  be used as a CPU worker barrier. Use a synchronization or communication operation with an implemented CPU path when
  workers need to exchange data.
- Simulator integrations can provide subblock-ID and shared-`TPipe`-state callbacks through
  `pto::cpu_sim::register_hooks`. If callbacks are not registered directly, CPU_SIM also resolves the
  `pto_sim_get_subblock_id` and `pto_sim_get_pipe_shared_state` symbols from the host process.

## Multicore execution

`pto::cpu_sim::LaunchKernelMultiCore` launches one CPU worker for each active simulated core and initializes the
worker's block and subblock execution context. The default configured core count is 4 and can be changed with
`PTO_CPU_SIM_NUM_CORES`. Set this environment variable before CPU_SIM runtime initialization.

The launch options can specify the requested core count, total work items, work quantum, subblocks per block, and an
explicit block count. `get_block_idx()`, `get_subblockid()`, and `get_subblockdim()` reflect the worker's launch
context, and `get_coreid()` aliases `get_block_idx()`. `get_block_num()` returns the configured
`PTO_CPU_SIM_NUM_CORES` value; it can be greater than the active block count when a launch is explicitly limited or
reduced to fit its work items.

## Instruction tracing

Instruction tracing is disabled at build time by default. For CPU STs, enable it with:

```bash
python3 tests/run_cpu.py --trace-mode
```

This sets the `PTO_CPU_SIM_TRACE_MODE` CMake option. A trace-enabled build records instruction opcodes, block indexes,
sequence IDs, tile operands, and scalar operands. `LaunchKernelMultiCore` writes the combined JSON Lines trace to:

```text
cpu_sim_traces/<kernel_name>/launch_<id>/trace.jsonl
```

The following environment variables control tracing at runtime:

- `PTO_CPU_SIM_TRACE_ENABLE`: set to `0` or `false` to disable trace collection for a trace-enabled build.
- `PTO_CPU_SIM_TRACE_DIR`: override the default `cpu_sim_traces` output directory.

Set these variables before CPU_SIM runtime initialization. The trace APIs in `include/pto/cpu/trace.hpp` can also be
used to reset, inspect, copy, or serialize the current thread's instruction records.
