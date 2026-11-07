# include/pto/comm/

PTO communication instruction set for inter-NPU data transfer, signal synchronization, and collective operations.

## Recommended Include

- `pto_comm_inst.hpp`: Unified public API header. Upper-layer code should include this file only; it pulls in all necessary types and dispatches to the correct backend (NPU native or CPU simulation).

## Layout

```
comm/
├── pto_comm_inst.hpp            # Public API: TPUT, TGET, TNOTIFY, TWAIT, TTEST,
│                                #   TGATHER, TSCATTER, TBROADCAST, TREDUCE,
│                                #   TPUT_ASYNC, TGET_ASYNC
├── pto_comm_instr_impl.hpp      # Backend dispatcher — includes NPU or CPU impl
│                                #   based on __CCE_AICORE__ / __CPU_SIM / PTO_NPU_ARCH_A5
├── comm_types.hpp               # Shared types: ParallelGroup, Signal, Signal2D,
│                                #   NotifyOp, WaitCmp, ReduceOp, DmaEngine, AsyncEvent
│
├── a2a3/                        # A2/A3 (Ascend 910B/910C) architecture implementations
│   ├── TPut.hpp                 # TPUT_IMPL  — remote write (local GM → UB → remote GM)
│   ├── TGet.hpp                 # TGET_IMPL  — remote read  (remote GM → UB → local GM)
│   ├── TNotify.hpp              # TNOTIFY_IMPL — send flag notification
│   ├── TWait.hpp                # TWAIT_IMPL — blocking wait on signal(s)
│   ├── TTest.hpp                # TTEST_IMPL — non-blocking signal test
│   ├── TGather.hpp              # TGATHER_IMPL  — root collects from all ranks
│   ├── TScatter.hpp             # TSCATTER_IMPL — root distributes to all ranks
│   ├── TBroadCast.hpp           # TBROADCAST_IMPL — root broadcasts to all ranks
│   ├── TReduce.hpp              # TREDUCE_IMPL — root gathers and reduces (Sum/Max/Min)
│   └── async/
│       ├── TPutAsync.hpp        # TPUT_ASYNC_IMPL (SDMA only)
│       └── TGetAsync.hpp        # TGET_ASYNC_IMPL (SDMA only)
│
├── a5/                          # A5 (Ascend 950) architecture implementations
│   ├── T*.hpp                   # Sync instructions (include a2a3/ counterparts)
│   └── async/
│       ├── TPutAsync.hpp        # TPUT_ASYNC_IMPL (SDMA with MTE fallback + URMA + RDMA)
│       └── TGetAsync.hpp        # TGET_ASYNC_IMPL (SDMA + URMA + RDMA)
│
└── async_common/                # Common async implementations (shared by a2a3/a5)
    ├── async_types.hpp          # SDMA/URMA/RDMA session and context types
    ├── async_event_impl.hpp     # AsyncEvent::Wait/Test, BuildAsyncSession
    ├── TPutAsyncCommonDetail.hpp # Common TPUT_ASYNC detail helpers + SDMA impl
    └── TGetAsyncCommonDetail.hpp # Common TGET_ASYNC detail helpers + SDMA impl
```

## Architecture

```
 User code
    │
    ▼
 pto_comm_inst.hpp              ← public API (template wrappers + event handling)
    │                              includes async_common/async_event_impl.hpp
    ▼
 pto_comm_instr_impl.hpp        ← compile-time dispatch
    │
    ├── PTO_NPU_ARCH_A5    →  a5/T*.hpp / a5/async/T*Async.hpp
    ├── __CCE_AICORE__     →  a2a3/T*.hpp / a2a3/async/T*Async.hpp
    └── __CPU_SIM          →  pto/cpu/comm/T*.hpp  (CPU simulation stubs)
```

## Instruction Categories

| Category | Instructions | Description |
|---|---|---|
| Point-to-Point (sync) | `TPUT`, `TGET` | Remote write / read through UB staging tile. Supports single-buffer and ping-pong double-buffering modes. |
| Point-to-Point (async) | `TPUT_ASYNC`, `TGET_ASYNC` | GM-to-GM DMA via SDMA, URMA, or RDMA engine. Returns `AsyncEvent` for later Wait/Test. |
| Signal Synchronization | `TNOTIFY`, `TWAIT`, `TTEST` | Flag-based cross-NPU synchronization. Signals are `int32_t` scalars or 2D grids. |
| Collective | `TGATHER`, `TSCATTER`, `TBROADCAST`, `TREDUCE` | Multi-rank operations via `ParallelGroup`. Root-initiated; support chunked 2D sliding and ping-pong. |

## Key Types (comm_types.hpp)

- **`ParallelGroup<GlobalData>`** — Lightweight view wrapping an array of `GlobalTensor` objects, one per rank.
- **`Signal`** — Scalar `GlobalTensor<int32_t, Shape<1,1,1,1,1>>` for single-flag synchronization.
- **`Signal2D<Rows, Cols>`** — 2D signal grid with compile-time shape; supports dense and strided sub-region views.
- **`AsyncEvent`** — Handle returned by async instructions; call `.Wait(session)` or `.Test(session)` to synchronize.
- **`AsyncSession`** — Engine-agnostic session built via `BuildAsyncSession<engine>()`.

## Related

- ISA semantics and examples: `docs/isa/`
- CPU simulation stubs: `pto/cpu/comm/`
- NPU async backends: `pto/comm/async/`
