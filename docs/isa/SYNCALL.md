# SYNCALL

## Instruction Diagram

> No `SYNCALL.svg` is provided (unlike most vector ops). `SYNCALL` is a **cross-core control-plane** primitive rather than a per-element Tile transform; semantically it means "all selected participants rendezvous at this point before any may proceed."

The following conceptual diagram distinguishes the hardware (FFTS) and software (GM polling) paths:

```mermaid
flowchart TB
  subgraph hard [Hard Mode / FFTS]
    H1[Each participant reaches call site] --> H2[ffts_cross_core_sync etc.]
    H2 --> H3[wait_flag_dev etc.]
    H3 --> H4[Barrier complete]
  end
  subgraph soft [Soft Mode / GM Polling]
    S1[Write local GM slot counter] --> S2[Poll all slots until threshold]
    S2 --> S3[Barrier complete]
  end
```

## Summary

`SYNCALL` is a cross-core synchronization barrier supporting A2/A3 and A5 NPU backends. The template parameter `SyncCoreType` selects the core-type mode:

- **AIV-only** (default): `SYNCALL()` synchronizes all AIV cores.
- **AIC-only**: `SYNCALL<SyncCoreType::AICOnly>()` synchronizes all AIC cores (A2/A3 supports both hardware and software modes; A5 supports hardware mode only).
- **MIX (AIC+AIV)**: `SYNCALL<SyncCoreType::Mix>()` synchronizes mixed AIC and AIV cores.

`SyncAllMode` (specified explicitly in workspace-bearing overloads) selects **hardware mode (FFTS)** or **software mode (GM polling)**. The workspace-free overload corresponds to the hardware path.

## Mathematical Semantics

Not applicable as an elementwise arithmetic operation. `SYNCALL` expresses a **barrier arrival** relation:

- At a given dynamic program point, every core in the participant set defined by the current `SyncCoreType` must execute past the `SYNCALL` call before any participant may proceed beyond that point.
- Hardware mode: cross-core visibility is guaranteed by FFTS flags and device-side `wait_flag_dev` primitives.
- Software mode: each participant owns a monotonically-increasing counter slot in GM; `dcci`/`dsb` coherency primitives and polling determine "all participants have reached the current generation."

This semantic does **not** provide additional guarantees on GM or other buffer contents after the barrier; cross-core data visibility must be maintained by the caller — see "Cross-Core GM Communication Notes".

## C++ Built-in Interface

Declared in `include/pto/common/pto_instr.hpp`. Software-mode interfaces use type-safe `GlobalTensor` and `Tile` parameters (constrained via SFINAE):

```cpp
// Hardware mode (all CoreType variants)
template <SyncCoreType CoreType = SyncCoreType::AIVOnly>
PTO_INST void SYNCALL();

// Software mode — AIV-only (GlobalTensor + Vec Tile)
template <SyncAllMode Mode, SyncCoreType CoreType = SyncCoreType::AIVOnly,
          typename GlobalData, typename TileData,
          std::enable_if_t<is_global_data_v<GlobalData> &&
                           is_tile_data_v<TileData> && TileData::Loc == TileType::Vec, int> = 0>
PTO_INST void SYNCALL(GlobalData &gmWorkspace, TileData &ubWorkspace, int32_t usedCores = 0);

// Software mode — AIC-only (GlobalTensor + Mat Tile)
template <SyncAllMode Mode, SyncCoreType CoreType = SyncCoreType::AICOnly,
          typename GlobalData, typename TileData,
          std::enable_if_t<is_global_data_v<GlobalData> &&
                           is_tile_data_v<TileData> && TileData::Loc == TileType::Mat, int> = 0>
PTO_INST void SYNCALL(GlobalData &gmWorkspace, TileData &l1Workspace, int32_t usedCores = 0);

// Software mode — MIX (GlobalTensor + Vec Tile + Mat Tile)
template <SyncAllMode Mode, SyncCoreType CoreType = SyncCoreType::Mix,
          typename GlobalData, typename UbTileData, typename L1TileData,
          std::enable_if_t<is_global_data_v<GlobalData> &&
                           is_tile_data_v<UbTileData> && UbTileData::Loc == TileType::Vec &&
                           is_tile_data_v<L1TileData> && L1TileData::Loc == TileType::Mat, int> = 0>
PTO_INST void SYNCALL(GlobalData &gmWorkspace, UbTileData &ubWorkspace, L1TileData &l1Workspace,
                       int32_t usedCores = 0);
```

## Parameters

- `gmWorkspace`: `GlobalTensor<int32_t, pto::Shape<>, pto::Stride<>>` (when `using namespace pto` coexists with Ascend C headers, qualify with `pto::` to avoid name collision with the compiler-intrinsic `Stride` enum). GM workspace for software mode; must be zero-initialized before the call. Each participating core occupies 8 `int32_t` values (cache-line-isolated sync counter).
- `ubWorkspace`: `Tile<TileType::Vec, int32_t, 1, SYNCALL_SOFT_SLOT_INT32>` (template parameter is fixed at `SYNCALL_SOFT_SLOT_INT32 = 8`, one cache-line slot per core). UB scratch for AIV-only and MIX software mode; the runtime backing memory capacity must be at least `usedCores * 8 * sizeof(int32_t)` (the implementation accesses via raw pointer and does not validate the template capacity; examples declare it as compile-time max participant count × `SYNCALL_SOFT_SLOT_INT32` to guarantee sufficient backing memory).
- `l1Workspace`: `Tile<TileType::Mat, int32_t, 1, SYNCALL_SOFT_SLOT_INT32>`. L1 (cbuf) scratch for AIC-only and MIX software mode; used by `create_cbuf_matrix` to fill a sync value then DMA-transfer to GM.
- `usedCores`: Number of cores participating in the software barrier. When 0, automatically inferred — AIV-only / AIC-only use `get_block_num()`, MIX uses `SYNCALL_GET_MIX_PARTICIPANT_COUNT()` (i.e. `AIC blocks × (1 + AIV ratio)`).

## Kernel Meta Macros

The following scenarios require **hand-written** `.ascend.meta` in the ELF for correct runtime scheduling: **hard AIV-only**, **soft AIC-only**, and **register-ELF MIX** (e.g. 1:1 hard). For `dav-c220` auto-split builds, Bisheng generates meta automatically — see the note at the end of this section. Macros are defined in `include/pto/common/kernel_meta.hpp`:

> `kernelName` must **exactly match** the `__global__` entry symbol (stored in section `.ascend.meta.<kernelName>`).

```cpp
// AIV-side kernel (marked as MIX_AIV_MAIN, ratio fixed 0:1)
PTO_SYNCALL_AIV_KERNEL_META(kernelName);

// AIC-side kernel (marked as MIX_AIC_MAIN, specify AIC:AIV ratio)
PTO_SYNCALL_MIX_AIC_KERNEL_META(kernelName, aicRatio, aivRatio);
```

Usage example (1:2 mixed mode):

```cpp
PTO_SYNCALL_MIX_AIC_KERNEL_META(MyKernel_mix_aic, 1, 2);  // AIC kernel ELF
PTO_SYNCALL_AIV_KERNEL_META(MyKernel_mix_aiv);             // AIV kernel ELF
```

Soft AIC-only (single kernel, chevron launch):

```cpp
PTO_SYNCALL_AIC_KERNEL_META(MyKernel);
extern "C" __global__ AICORE void MyKernel(...) { SYNCALL<SyncAllMode::Soft, SyncCoreType::AICOnly>(...); }
```

register-ELF generic pairing (AIC side sets the ratio + AIV side). Note: the MIX 1:2 case in the current `syncall` ST now uses `dav-c220` auto-split and needs no hand-written meta; the example below only demonstrates the register-ELF macro pairing:

```cpp
PTO_SYNCALL_MIX_AIC_KERNEL_META(MyKernel_mix_aic, 1, 2);
PTO_SYNCALL_AIV_KERNEL_META(MyKernel_mix_aiv);
```

register-ELF MIX 1:1 hard (**both** AIC and AIV sides use `PTO_SYNCALL_MIX_AIC_KERNEL_META(..., 1, 1)` — do **not** use `PTO_SYNCALL_AIV_KERNEL_META` on the AIV side):

```cpp
PTO_SYNCALL_MIX_AIC_KERNEL_META(MyKernel_mix_aic, 1, 1);
PTO_SYNCALL_MIX_AIC_KERNEL_META(MyKernel_mix_aiv, 1, 1);
```

**Common scenarios that do not require hand-written meta** (full mapping in the "Compilation and Launch Guide" quick-reference table below):

- AIV-only soft (`dav-c220-vec`)
- MIX 1:2 hard/soft, hard AIC-only on A2/A3 (`dav-c220` auto-split)
- MIX 1:1 soft (dual-stream chevron)

> **dav-c220 auto-split**: When compiled with `--cce-aicore-arch=dav-c220`, Bisheng automatically generates AIC/AIV sub-kernels and corresponding `.ascend.meta` at a fixed physical ratio of **1:2** (two AIV subblocks per AIC block). In this case you do **not** need to hand-write `PTO_SYNCALL_MIX_AIC_KERNEL_META`, and you **cannot** override the ratio to 1:1 via meta alone (see "MIX 1:1" below).

## Compilation and Launch Guide (A2/A3)

This section uses the ST suite [`tests/npu/a2a3/src/st/testcase/syncall/`](../../tests/npu/a2a3/src/st/testcase/syncall/) as the reference for which **compile arch**, **kernel meta**, and **host launch** pattern to use for each `SyncCoreType` / mode / AIC:AIV ratio. The host decides the launch grid at runtime via [`syncall_core_config.hpp`](../../tests/npu/a2a3/src/st/testcase/syncall/syncall_core_config.hpp) (910B1: 24 AIC + 48 AIV; 910B4: 20 AIC + 40 AIV), so the same kernel binaries work across chips.

### Scenario Quick Reference

| Scenario | Sync Mode | Participants | Compile `--cce-aicore-arch` | Kernel Meta | Host Launch | Reference Source |
|----------|-----------|--------------|----------------------------|-------------|-------------|------------------|
| AIV-only | Hard | `aiv` | `dav-c220-vec` | `PTO_SYNCALL_AIV_KERNEL_META` | chevron `<<<aiv>>>` | `syncall_kernel.cpp` |
| AIV-only | Soft | `aiv` | `dav-c220-vec` | none | chevron `<<<aiv>>>` | `syncall_soft_kernel.cpp` |
| AIC-only | Hard | `aic` | **`dav-c220`** (MIX auto-split, empty AIV stub) | auto-generated by Bisheng | chevron `<<<aic>>>` | `syncall_aic_hard_kernel.cpp` |
| AIC-only | Soft | `aic` | `dav-c220-cube` | `PTO_SYNCALL_AIC_KERNEL_META` | chevron `<<<aic>>>` | `syncall_aic_kernel.cpp` |
| MIX 1:2 | Hard / Soft | `aic×3` | **`dav-c220`** | auto-generated by Bisheng | chevron `<<<aic>>>` (hard + soft share one `.so`) | `syncall_mix_1_2_kernel.cpp` |
| MIX 1:1 | Soft | `aic×2` | cube + vec `.o` each | none | **dual-stream** chevron: AIC `<<<aic>>>` + AIV `<<<aic>>>` | `syncall_mix_1_1_soft_kernel.cpp` |
| MIX 1:1 | Hard | `aic×2` | cube + vec `.o` each | **`PTO_SYNCALL_MIX_AIC_KERNEL_META(..., 1, 1)`** | **register ELF** + `rtKernelLaunchWithHandleV2` | `syncall_mix_1_1_kernel.cpp` |

Hard and soft kernels **must not share the same `.so`** in AIV-only / AIC-only cases (soft corrupts hard FFTS config and hangs). MIX 1:2 hard and soft may live in the same dav-c220 auto-split source and `.so`.

### Path Details

#### 1. Chevron single-arch (AIV-only / AIC-only soft)

- **Compile**: one source file + matching arch (`dav-c220-vec` or `dav-c220-cube`) → separate `.so`.
- **Launch**: `kernel<<<blockDim, nullptr, stream>>>(..., totalBlocks)`; `blockDim` and `totalBlocks` come from the host at runtime (ST: `syncall_cfg::GetCoreConfig()`).
- **Hard AIV-only**: declare `PTO_SYNCALL_AIV_KERNEL_META`.

#### 2. Chevron MIX auto-split (MIX 1:2, hard AIC-only)

- **Compile**: `--cce-aicore-arch=dav-c220`; CMake helper `pto_syncall_chevron_kernel(<target> <source>)`.
- **Launch**: single chevron `<<<aic>>>`; runtime brings up all MIX participants at physical 1:2.
- **Kernel args**: `aicBlocks` and `totalParticipants` passed as scalars from the host (read on both AIC and AIV sides) for 910B1/910B4 portability.
- **Hard AIC-only exception**: pure `dav-c220-cube` cannot establish the FFTS context for AIC-only hard sync. Use `dav-c220` MIX compile: AIC runs `SYNCALL<AICOnly>()`, AIV is an empty stub; `totalBlocks` passed from the host.

#### 3. Dual-arch dual-stream (MIX 1:1 soft)

- **Why**: on the ccec/bisheng path `GetTaskRation()` is always **2**; `dav-c220` auto-split is physically fixed at **1:2** and cannot produce true 1:1.
- **Compile**: same source compiled once as `dav-c220-cube` (`-DSYNCALL_MIX_BUILD_AIC`) and once as `dav-c220-vec` (`-DSYNCALL_MIX_BUILD_AIV`), linked into one `.so`; CMake helper `pto_syncall_mix11_soft_kernel`.
- **Launch**: AIC and AIV chevron `<<<aic>>>` on separate `aclrtStream`s; `aicBlocks` / `totalParticipants` from the host at runtime.

#### 4. Register ELF (MIX 1:1 hard)

- **Why**: hard MIX sync needs a single MIX FFTS context; chevron auto-split cannot produce true 1:1 on ccec.
- **Compile**: cube / vec objects with `PTO_SYNCALL_MIX_AIC_KERNEL_META(name, 1, 1)`, register-only objects with `-DSYNCALL_MIX_REGISTER_BUILD`, merged by `make_mix_register_elf.py`; CMake helper `pto_syncall_mix_kernel`.
- **Launch**: `rtRegisterAllKernel` + `rtKernelLaunchWithHandleV2(handle, tilingKey, aicBlocks, ...)`; device derives participant count via `get_block_num()` (register path passes only `ffts/out/flags`).

## Mode Support Matrix

### A2/A3

| Core Type | Hardware Mode | Software Mode |
|-----------|--------------|--------------|
| AIV-only | Supported | Supported |
| AIC-only | Supported | Supported |
| MIX | Supported | Supported |

### A5

| Core Type | Hardware Mode | Software Mode |
|-----------|--------------|--------------|
| AIV-only | Supported | Supported |
| AIC-only | Supported | Not supported |
| MIX | Not supported | Supported |

## Constraints

- Software-mode GM write paths per platform:
  - A2/A3 (AIC-only and the AIC side of MIX): AIC writes GM slots via `copy_cbuf_to_gm` (L1→GM DMA); the AIV side of MIX writes via UB workspace.
  - A5 MIX: A5 AIC (`dav-c310-cube`) lacks `copy_cbuf_to_gm`; instead it delegates UB→GM writes to AIV subblock 0 of the same block via `intra_block` signaling.
- A5 platform limitations (cf. "Mode Support Matrix"):
  - AIC-only software unavailable: A5 AIC lacks an independent GM DMA write path (no `copy_cbuf_to_gm`), so GM-polling sync is infeasible.
  - Hardware MIX unavailable: `rtGetC2cCtrlAddr` returns `RT_ERROR_FEATURE_NOT_SUPPORT` (207000) on A5 (`CHIP_DAVID`), preventing FFTS base address retrieval.
  - AIC-only hardware: implemented via `ffts_cross_core_sync` + `wait_flag_dev`, without needing `set_ffts_base_addr`.
- Software mode requires all participating cores to enter the same barrier group in the same order (based on monotonic generation counting; mismatched entry count/order causes misalignment or deadlock).
- `SYNCALL` does not participate in PTO's automatic Event dependency scheduling: it neither accepts `WaitEvents` nor returns a `RecordEvent` that later instructions can wait on. Consequently it does not automatically wait for preceding data instructions (e.g. `TSTORE`) to complete; the ordering and visibility between `SYNCALL` and surrounding data instructions must be ensured by the caller (see "Cross-Core GM Communication Notes").
- In the auto build path (`__PTO_AUTO__`), `SYNCALL` is a no-op and emits no cross-core hardware synchronization (consistent with `TSYNC` etc.); actual synchronization happens only in manual kernels.

## Cross-Core GM Communication Notes

`SYNCALL` only provides barrier **arrival** semantics (for both hard and soft), and does **not** guarantee cross-core cache visibility of business data around the barrier. When an operator writes GM per core before the barrier and reads other cores' GM after the barrier (e.g. cross-core histogram / prefix-sum), the caller must satisfy the two points below, otherwise stale reads or lost writes will occur.

### 1. Cache coherency: explicit `dcci` / `dsb` is required

- **Writer**: after `copy_ubuf_to_gm` / `copy_cbuf_to_gm`, issue `dcci(addr, SINGLE_CACHE_LINE)` + `dsb(DSB_DDR)` to flush data out to DDR.
- **Reader**: before reading, issue `dcci(addr, SINGLE_CACHE_LINE)` (invalidate) + `dsb` to ensure the latest DDR value is read instead of a stale local cache.
- `set_flag` / `wait_flag` alone (intra-core pipeline sync) is **not** sufficient for cross-core visibility.
- This is independent of the barrier mode: the **hardware FFTS barrier also does not flush cache**; it only guarantees the "all arrived" control-plane ordering.
- `SYNCALL` internally applies full `dcci` + `dsb(DDR)` to its own sync slots, but does **not** flush the caller's business data.

### 2. Per-core slot must own a full cache line: avoid false-sharing lost writes

- `dcci` / DMA operate at **32B cache-line** granularity; if adjacent cores' slots share one cache line, cross-core flushes overwrite / lose each other's writes.
- Each core's slot should be 32B-aligned and **own one full cache line** (for `int32`, that means stride = 8, not 4).
- `SYNCALL`'s own sync slots follow this design: `SYNCALL_SOFT_SLOT_INT32 = 8` (see `include/pto/common/type.hpp`); the caller's business workspace should follow the same isolation principle.

## Examples

### Auto

In the **auto** build path, `SYNCALL` is consistent with existing sync strategies and **does not directly emit** cross-core hardware synchronization; it serves as a placeholder or for host/compiler-coordinated graph-level semantics. Typical operator development uses explicit `SYNCALL` in **manual** kernels.

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

// In auto mode SYNCALL is a no-op (same as TSYNC etc. under auto)
void example_auto_noop() {
  SYNCALL();  // Does not trigger FFTS
}
```

### Manual — Hardware Mode

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

// AIV-only: FFTS barrier across all AIV cores (requires correct kernel meta / ELF)
void example_hard_aiv() {
  SYNCALL();
}

// AIC-only: only available when compiled for AIC (__DAV_CUBE__); verified on A5 hard mode
void example_hard_aic() {
  SYNCALL<SyncCoreType::AICOnly>();
}

// MIX: paired AIC and AIV ELFs, see "Kernel Meta Macros" section above
void example_hard_mix() {
  SYNCALL<SyncCoreType::Mix>();
}
```

### Manual — Software Mode

Software mode requires a **zero-initialized** GM workspace and a correctly-sized UB/L1 Tile. `Mode` must be `SyncAllMode::Soft` (`Hard` ignores the workspace and behaves like the workspace-free `SYNCALL_IMPL`).

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_soft_aiv(__gm__ int32_t *gmPtr) {
  GlobalTensor<int32_t, pto::Shape<>, pto::Stride<>> gmWs(gmPtr);
  Tile<TileType::Vec, int32_t, 1, SYNCALL_SOFT_SLOT_INT32> ub;
  SYNCALL<SyncAllMode::Soft, SyncCoreType::AIVOnly>(gmWs, ub, 0);
}
```

MIX software mode requires both UB and L1 (Mat) Tiles; on A5 the AIC side delegates GM writes via a proxy path — see the "Constraints" section.
