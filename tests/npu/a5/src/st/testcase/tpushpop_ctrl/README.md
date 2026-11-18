# Scalar control FIFOs

`TPipe` control FIFOs exchange `uint32_t` values through SSBUF and `PIPE_S`. They do
not transfer accumulator tiles through FIXP. The A5 NPU backend supports AIV0,
AIV1, or both. The shared A6 backend permits AIV0 only. Support for the PTO CPU
simulator (`__CPU_SIM`) and costmodel (`__COSTMODEL`) is not available for this
control protocol. CANN's A5 instruction simulator runs the native NPU
implementation without either macro.
The examples and NPU tests use manual NPU mode (`__PTO_AUTO__` is not defined).
Compilation and execution in PTO automatic mode have not been validated.
Ctrl payloads are scalar values; the caller explicitly supplies the FIFO's
SSBUF base. The pipe manages ring-slot progression and enabled ready/free
dependencies automatically in the tested manual mode.

On the native NPU backend, `Tile<TileType::Ctrl, uint32_t, 1, 1>` holds its
value directly in `data()`; it does not contain a UB pointer. Initialize NPU control payloads with
`value.data() = scalar`; the NPU `TASSIGN` path does not support Ctrl tiles.
V2C also accepts a `uint32_t` Vec tile and
reads its first element. Complete any producing vector operation with a
V-to-scalar dependency before that scalar UB read.

## Configuration

Append a `CtrlPipeConfig` to the `TPipe` template:

```cpp
using Config = CtrlPipeConfig<
    CtrlMergeMode::OR, VecCubeRatio::V2C1_VECS,
    true,   // EnableForward: ready notification and matching consumer wait
    true>;  // EnableBackward: release notification and matching reuse wait

using Request = TPipe<0, Direction::DIR_V2C_CTRL, 4, 2, 2, false, false, Config>;
Request request(nullptr, 0, 0x100); // Third argument is the V2C SSBUF base.
```

The policy's AIV selection controls participation, independently of the
`TileSplitAxis` argument. Calls on an inactive AIV perform no transfer or
notification. The explicit `subBlockId` overload does not redirect a control
message to a different physical AIV.

| Policy | Meaning for V2C |
| --- | --- |
| `CtrlMergeMode::NONE` | Preserve the raw 32-bit value; requires one active AIV. |
| `CtrlMergeMode::OR` | Return 1 when any active input is nonzero, otherwise 0. |
| `CtrlMergeMode::AND` | Return 1 when every active input is nonzero, otherwise 0. |

With one AIV selected, both OR and AND return `value != 0`. For example,
an input of `7` produces `1` with either merge mode and `7` with NONE.
The unselected AIV is excluded from payload reads and ready/release synchronization.

Each active V2C producer has separate storage in every ring slot. AIC waits
for the selected producers and reduces values from the same message. C2V
always broadcasts the raw 32-bit value; it does not apply the merge mode.

The default control policy is OR over both AIVs, or AIV0 when `IsNoSplit`
is true. Explicitly select AIV0 on A6.

## Return path and storage

Use `Direction::DIR_C2V_CTRL` for AIC-to-AIV delivery:

```cpp
using Broadcast = CtrlPipeConfig<CtrlMergeMode::NONE, VecCubeRatio::V2C1_VECS>;
using Reply = TPipe<2, Direction::DIR_C2V_CTRL, 4, 2, 2, false, false, Broadcast>;
Reply reply(nullptr, 0x120, 0); // Second argument is the C2V SSBUF base.

using Ctrl = Tile<TileType::Ctrl, uint32_t, 1, 1>;
Ctrl value;
#ifdef PTO_COMPILE_VEC
value.data() = predicate; // Caller-provided scalar predicate on each AIV.
TPUSH<Request, Ctrl, TileSplitAxis::TILE_NO_SPLIT>(request, value);
TPOP<Reply, Ctrl, TileSplitAxis::TILE_NO_SPLIT>(reply, value);
TFREE<Reply, TileSplitAxis::TILE_NO_SPLIT>(reply);
#endif
#ifdef PTO_COMPILE_CUBE
TPOP<Request, Ctrl, TileSplitAxis::TILE_NO_SPLIT>(request, value);
TPUSH<Reply, Ctrl, TileSplitAxis::TILE_NO_SPLIT>(reply, value);
TFREE<Request, TileSplitAxis::TILE_NO_SPLIT>(request);
#endif
```

Including `pto/pto-inst.hpp` supplies `PTO_COMPILE_VEC` and `PTO_COMPILE_CUBE`
through `arch_macro.hpp`, based on the compiler's `__DAV_VEC__` and
`__DAV_CUBE__` target macros.

Keep the pipe instances alive across the message loop. Both AIVs receive the
same merged result. Distinct pipes need distinct enabled events and nonoverlapping
SSBUF storage. Bases and entry offsets must be 4-byte aligned.

Control storage is compact: 4 bytes per slot for single-AIV V2C or C2V, and
8 bytes per slot for dual-AIV V2C. `Pipe::CTRL_SLOT_BYTES` and
`Pipe::FIFO_BYTES` expose the required allocation. The `TPipe` slot-size
argument does not select a padded control stride; it must be at least 4.

## Dependency and lifetime contract

| Switch | Enabled | Disabled |
| --- | --- | --- |
| Forward | Store payload, signal `FlagID`; consumer waits before reading. | No ready set/wait; caller supplies publication and read ordering. |
| Backward | `TFREE` signals `FlagID + 1`; producer waits before slot reuse. | No release/reuse set/wait; caller protects storage until all readers finish. |

Each switch applies to both endpoints and final draining. Disabling either
dependency retains payload transfer, reduction, and slot progression. Local
visibility dependencies, including V-to-scalar ordering, remain necessary.

Control FIFOs return one credit per message, including depth two; they do not
use data-FIFO free-credit batching. `TPOP` advances the consumer cursor, and
`TFREE` releases one popped message. Every active consumer must release each
message exactly once when backward synchronization is enabled. In broadcast
mode, the producer waits for all selected AIV readers before reuse. Producer
destruction drains outstanding releases; it must run only after matching
consumers have been scheduled. No startup release tokens are inserted.

`TPipe` advances message cursors automatically. Skipped work
must skip corresponding calls on every endpoint; attention-tile indices are
not necessarily message indices. Use `setTileId` to reposition matching endpoints
only when no popped messages await `TFREE`; changing the slot cursor does not
reset the producer's outstanding release credits. Runtime status overrides require matching
external synchronization. Prefer the compile-time policy when disabling an
entire edge, so setup and drain behavior remain consistent.

A request/reply wrapper may replace an automatic edge with its own ordering.
It must wait once per notification, retain the reply until every reader has
consumed it, and explicitly handle empty jobs and the final drain. Sharing an
event between two independent generic waits does not implement that coupling.

## Tests

From the repository root, with the CANN environment and A5 instruction simulator
configured, run the mixed-core ST:

```bash
python3 tests/script/run_st.py -r sim -v a5 -t tpushpop_ctrl
```

This builds the kernel with `bisheng -xcce --cce-aicore-arch=dav-c310` and
links `runtime_camodel` for simulator execution. It does not define
`__CPU_SIM` or `__COSTMODEL`.

The ST launches one logical mixed block. Its twelve cases cover the `TPipe` control API,
all Boolean input combinations, raw and normalized single-AIV messages,
inactive-producer calls, scalar and UB-backed V2C inputs, C2V broadcasts,
all four dependency configurations, delayed producers, ring wraparound, and
multiple jobs. Cases with a disabled dependency use separate synchronized
guard pipes to provide the ordering that the disabled edge would otherwise
supply. These are protocol tests, not performance benchmarks. A6 runtime
behavior is not covered by the A5 ST.
