<!-- Generated from `docs/isa/scalar/ops/pipeline-sync/mem-bar.md` -->

# pto.mem_bar

Standalone reference page for `pto.mem_bar`. This page belongs to the [Pipeline Sync](../../pipeline-sync.md) family in the PTO ISA manual.

## Summary

Intra-vector-pipe memory fence within `__VEC_SCOPE__`. Required when UB addresses alias between vector load/store operations.

## Mechanism

`pto.mem_bar` is a `pto.*` control/configuration operation. It changes ordering, buffer, event, or DMA-visible state that later payload work depends on. The portable guarantee is the dependency/configuration effect, while concrete pipe/event spaces remain target-profile details.

## Syntax

```mlir
pto.mem_bar "BARRIER_TYPE"    // BARRIER_TYPE ∈ { "VV_ALL", "VST_VLD", "VLD_VST" }
```

## Inputs

The inputs are the architecture-visible control operands shown in the syntax: pipe ids, event ids, buffer ids, loop/stride values, pointers, or configuration words used to drive later execution.

## Expected Outputs

This form is primarily defined by the side effect it has on control state, predicate state, or memory. It does not publish a new payload SSA result beyond any explicit state outputs shown in the syntax.

## Side Effects

This operation updates control, synchronization, or DMA configuration state. Depending on the form, it may stall a stage, establish a producer-consumer edge, reserve or release a buffer token, or configure later copy behavior.

## Constraints

This operation inherits the legality and operand-shape rules of its family overview. Any target-specific narrowing of element types, distributions, pipe/event spaces, or configuration tuples must be stated by the selected target profile.

## Exceptions

- It is illegal to use unsupported pipe ids, event ids, buffer ids, or configuration tuples for the selected target profile.
- Waiting on state that was never established by a matching producer or prior configuration is an illegal PTO program.

## Target-Profile Restrictions

- CPU simulation preserves the visible dependency/configuration contract, but it may not expose every low-level hazard that motivates the form on hardware targets.
- A2/A3 and A5 profiles may use different concrete pipe, DMA, predicate, or event spaces. Portable code must rely on the documented PTO contract plus the selected target profile.

## Examples

```c
mem_bar(barrier_type);
```

```mlir
pto.vsts %v0, %ub[%c0] : !pto.vreg<64xf32>, !pto.ptr<f32, ub>
pto.mem_bar "VST_VLD"
%v1 = pto.vlds %ub[%c0] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
```

```mlir
// ─── Stage 1: MTE2 loads data from GM into UB ───
pto.copy_gm_to_ubuf %gm_ptr, %ub_ptr, ...

// MTE2 signals: "UB data is ready for Vector pipe"
pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

// ─── Stage 2: Vector pipe consumes UB data ───
// Vector waits until MTE2's signal arrives
pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

scf.for %dummy = %c0 to %c1 step %c1 {
  %v   = pto.vlds %ub_ptr[%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %mask = pto.pset_b32 "PAT_ALL" : !pto.mask
  %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
  pto.vsts %abs, %ub_out[%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask
} {llvm.loop.aivector_scope}

// Vector signals: "UB output is ready for MTE3"
pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]

// ─── Stage 3: MTE3 stores result from UB back to GM ───
// MTE3 waits until Vector's signal arrives
pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]

pto.copy_ubuf_to_gm %ub_out, %gm_out, ...
```

```mlir
// ─── Stage 1: MTE2 loads data into UB ───
// MTE2 acquires ub_ptr — blocks if Vector hasn't released it from a prior iteration
pto.get_buf "PIPE_MTE2", %bufid_ub_ptr, %mode : i64, i64
pto.copy_gm_to_ubuf %gm_ptr, %ub_ptr, ...
// MTE2 done writing ub_ptr — release it so Vector can consume
pto.rls_buf "PIPE_MTE2", %bufid_ub_ptr, %mode : i64, i64

// ─── Stage 2: Vector computation ───
// Vector acquires ub_ptr (input) — blocks until MTE2 releases it (RAW: MTE2 write → V read)
pto.get_buf "PIPE_V", %bufid_ub_ptr, %mode : i64, i64
// Vector acquires ub_out (output) — blocks until MTE3 releases it from a prior iteration (WAR: MTE3 read → V write)
pto.get_buf "PIPE_V", %bufid_ub_out, %mode : i64, i64

scf.for %dummy = %c0 to %c1 step %c1 {
  %mask = pto.pset_b32 "PAT_ALL" : !pto.mask
  %v   = pto.vlds %ub_ptr[%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
  pto.vsts %abs, %ub_out[%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask
} {llvm.loop.aivector_scope}

// Vector done reading ub_ptr — release so MTE2 can reuse it in next iteration
pto.rls_buf "PIPE_V", %bufid_ub_ptr, %mode : i64, i64
// Vector done writing ub_out — release so MTE3 can consume
pto.rls_buf "PIPE_V", %bufid_ub_out, %mode : i64, i64

// ─── Stage 3: MTE3 stores result to GM ───
// MTE3 acquires ub_out — blocks until Vector releases it (RAW: V write → MTE3 read)
pto.get_buf "PIPE_MTE3", %bufid_ub_out, %mode : i64, i64
pto.copy_ubuf_to_gm %ub_out, %gm_out, ...
// MTE3 done reading ub_out — release so Vector can reuse it in next iteration
pto.rls_buf "PIPE_MTE3", %bufid_ub_out, %mode : i64, i64
```

```mlir
// ═══ Pre-loop: prime ALL reverse-dependency signals ═══
// Both input and output buffers start unused. We must pre-send
// reverse-dep signals so the first iteration's wait_flags don't deadlock.
pto.set_flag["PIPE_V",    "PIPE_MTE2", "EVT_IN_REV_0"]   // ◀ PRIME: buf_in[0] "free"
pto.set_flag["PIPE_V",    "PIPE_MTE2", "EVT_IN_REV_1"]   // ◀ PRIME: buf_in[1] "free"
pto.set_flag["PIPE_MTE3", "PIPE_V",    "EVT_OUT_REV_0"]  // ◀ PRIME: buf_out[0] "free"
pto.set_flag["PIPE_MTE3", "PIPE_V",    "EVT_OUT_REV_1"]  // ◀ PRIME: buf_out[1] "free"

scf.for %i = %c0 to %N step %c1 {
  // ── All 3 stages in same iteration, indexed by i%2 ──
  // %pp = i % 2  (ping/pong selector for buffer & event IDs)

  // ── MTE2: load tile[i] into buf_in[i%2] ──
  // WAR: wait until Vector has released buf_in[i%2] from iteration i-2
  pto.wait_flag["PIPE_V", "PIPE_MTE2", "EVT_IN_REV_{pp}"]
  pto.copy_gm_to_ubuf %gm_ptr[%i], %ub_in[%pp], ...
  // RAW: signal Vector that buf_in[i%2] data is ready
  pto.set_flag["PIPE_MTE2", "PIPE_V", "EVT_IN_FWD_{pp}"]

  // ── Vector: compute buf_in[i%2] → buf_out[i%2] ──
  // RAW: wait for MTE2 to finish loading buf_in[i%2]
  pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVT_IN_FWD_{pp}"]
  // WAR: wait for MTE3 to finish reading buf_out[i%2] from iteration i-2
  pto.wait_flag["PIPE_MTE3", "PIPE_V", "EVT_OUT_REV_{pp}"]
  scf.for %dummy = %c0 to %c1 step %c1 {
    %v   = pto.vlds %ub_in[%pp][%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
    %mask = pto.pset_b32 "PAT_ALL" : !pto.mask
    %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
    pto.vsts %abs, %ub_out[%pp][%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask
  } {llvm.loop.aivector_scope}
  // WAR: tell MTE2 "done reading buf_in[i%2]"
  pto.set_flag["PIPE_V", "PIPE_MTE2", "EVT_IN_REV_{pp}"]
  // RAW: tell MTE3 "buf_out[i%2] result ready"
  pto.set_flag["PIPE_V", "PIPE_MTE3", "EVT_OUT_FWD_{pp}"]

  // ── MTE3: store result from buf_out[i%2] to GM ──
  // RAW: wait for Vector to finish writing buf_out[i%2]
  pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVT_OUT_FWD_{pp}"]
  pto.copy_ubuf_to_gm %ub_out[%pp], %gm_out[%i], ...
  // WAR: tell Vector "done reading buf_out[i%2]"
  pto.set_flag["PIPE_MTE3", "PIPE_V", "EVT_OUT_REV_{pp}"]
}

// ═══ Post-loop: drain — match every pre-loop prime with a wait ═══
// Each priming set_flag must be paired. The last loop iteration's
// set_flags are consumed by wait_flags that will never fire inside the
// loop (there is no iteration i+2). Drain them here.
pto.wait_flag["PIPE_V",    "PIPE_MTE2", "EVT_IN_REV_{(N-1)%2}"]  // ◀ DRAIN
pto.wait_flag["PIPE_V",    "PIPE_MTE2", "EVT_IN_REV_{(N-2)%2}"]  // ◀ DRAIN
pto.wait_flag["PIPE_MTE3", "PIPE_V",    "EVT_OUT_REV_{(N-1)%2}"] // ◀ DRAIN
pto.wait_flag["PIPE_MTE3", "PIPE_V",    "EVT_OUT_REV_{(N-2)%2}"] // ◀ DRAIN
```

```mlir
scf.for %i = %c0 to %N step %c1 {
  // %pp = i % 2  (ping/pong selector)

  // ── MTE2: load tile[i] into buf[i%2] ──
  // Acquires buf[i%2] — on first iteration, buffer is free so proceeds immediately.
  // On later iterations, blocks until Vector releases buf[i%2] (WAR: automatic).
  pto.get_buf %bufid_buf[%pp], "PIPE_MTE2"
  pto.copy_gm_to_ubuf %gm_ptr[%i], %ub_buf[%pp], ...
  pto.rls_buf %bufid_buf[%pp], "PIPE_MTE2"

  // ── Vector: compute on buf[i%2] ──
  // Acquires buf[i%2] — blocks until MTE2 releases it (RAW: automatic)
  pto.get_buf %bufid_buf[%pp], "PIPE_V"
  pto.get_buf %bufid_out[%pp], "PIPE_V"
  scf.for %dummy = %c0 to %c1 step %c1 {
    %v   = pto.vlds %ub_buf[%pp][%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
    %mask = pto.pset_b32 "PAT_ALL" : !pto.mask
    %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
    pto.vsts %abs, %ub_out[%pp][%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask
  } {llvm.loop.aivector_scope}
  // Release buf[i%2] — MTE2 can reuse in iteration i+2 (WAR resolved)
  pto.rls_buf %bufid_buf[%pp], "PIPE_V"
  pto.rls_buf %bufid_out[%pp], "PIPE_V"

  // ── MTE3: store result ──
  // Acquires out[i%2] — blocks until Vector releases it (RAW: automatic)
  pto.get_buf %bufid_out[%pp], "PIPE_MTE3"
  pto.copy_ubuf_to_gm %ub_out[%pp], %gm_out[%i], ...
  pto.rls_buf %bufid_out[%pp], "PIPE_MTE3"
}
// No post-loop drain needed — last rls_buf completes the pipeline.
```

```
Core Cluster (1:2 ratio)
┌─────────────────────────────────────────────┐
│  ┌──────────────┐    ┌──────────────┐       │
│  │  AIC (Cube)  │    │  AIV0 (Vec)  │       │
│  │  ┌────────┐  │    │  ┌────────┐  │       │
│  │  │   SU   │──┼────┼──│   SU   │  │       │
│  │  └────────┘  │    │  └────────┘  │       │
│  │  CUBE pipe   │    │  MTE2/V/MTE3 │       │
│  │  L0C buffer  │    │  UB (256KB)  │       │
│  └──────────────┘    └──────────────┘       │
│                      ┌──────────────┐       │
│                      │  AIV1 (Vec)  │       │
│                      │  ┌────────┐  │       │
│                      │  │   SU   │  │       │
│                      │  └────────┘  │       │
│                      │  MTE2/V/MTE3 │       │
│                      │  UB (256KB)  │       │
│                      └──────────────┘       │
└─────────────────────────────────────────────┘
```

```c
// mode2 broadcast/reduce semantics for 1:2 cluster
set_cross_core(pipe, semaphore_id);   // pipe: VEC/MTE2/CUBE/FIX
wait_flag_dev(semaphore_id);          // SU-level blocking
```

```
C→V Broadcast (one set reaches both):
    AIC ──set_cross_core──┬──> AIV0 sema++
                          └──> AIV1 sema++

V→C Reduce (one wait for both):
    AIV0 ──set_cross_core──┐
                           ├──> AIC wait_flag_dev (blocks until BOTH)
    AIV1 ──set_cross_core──┘
```

## Detailed Notes

```c
mem_bar(barrier_type);
```

**Barrier types:**

| Type | Semantics |
|------|-----------|
| `VV_ALL` | All prior vector ops complete before subsequent |
| `VST_VLD` | All prior vector stores visible before subsequent loads |
| `VLD_VST` | All prior vector loads complete before subsequent stores |

**Example:** Ensure stores are visible before loads to same UB region:
```mlir
pto.vsts %v0, %ub[%c0] : !pto.vreg<64xf32>, !pto.ptr<f32, ub>
pto.mem_bar "VST_VLD"
%v1 = pto.vlds %ub[%c0] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
```

## Intra-Core Sync Patterns & Examples

### Example 1: `set_flag` / `wait_flag` (Explicit Events)

Each cross-pipeline data dependency requires an explicit signal/wait pair. The programmer must manually insert `set_flag` after the producer and `wait_flag` before the consumer.

```mlir
// ─── Stage 1: MTE2 loads data from GM into UB ───
pto.copy_gm_to_ubuf %gm_ptr, %ub_ptr, ...

// MTE2 signals: "UB data is ready for Vector pipe"
pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

// ─── Stage 2: Vector pipe consumes UB data ───
// Vector waits until MTE2's signal arrives
pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

scf.for %dummy = %c0 to %c1 step %c1 {
  %v   = pto.vlds %ub_ptr[%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %mask = pto.pset_b32 "PAT_ALL" : !pto.mask
  %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
  pto.vsts %abs, %ub_out[%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask
} {llvm.loop.aivector_scope}

// Vector signals: "UB output is ready for MTE3"
pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]

// ─── Stage 3: MTE3 stores result from UB back to GM ───
// MTE3 waits until Vector's signal arrives
pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]

pto.copy_ubuf_to_gm %ub_out, %gm_out, ...
```

**Key property:** Every cross-pipeline edge is an explicit `(set_flag, wait_flag)` pair. Simple for straight-line code, but gets verbose in loops (see Example 3).

### Example 2: `get_buf` / `rls_buf` (Resource-Based)

Instead of naming events, each pipeline declares when it **acquires** (`get_buf`) and **releases** (`rls_buf`) a shared UB buffer. Cross-pipeline RAW/WAR dependencies are resolved implicitly by program order — if MTE2 releases `buf_A` and Vector later acquires `buf_A`, the hardware ensures the acquire cannot proceed until the release completes.

```mlir
// ─── Stage 1: MTE2 loads data into UB ───
// MTE2 acquires ub_ptr — blocks if Vector hasn't released it from a prior iteration
pto.get_buf "PIPE_MTE2", %bufid_ub_ptr, %mode : i64, i64
pto.copy_gm_to_ubuf %gm_ptr, %ub_ptr, ...
// MTE2 done writing ub_ptr — release it so Vector can consume
pto.rls_buf "PIPE_MTE2", %bufid_ub_ptr, %mode : i64, i64

// ─── Stage 2: Vector computation ───
// Vector acquires ub_ptr (input) — blocks until MTE2 releases it (RAW: MTE2 write → V read)
pto.get_buf "PIPE_V", %bufid_ub_ptr, %mode : i64, i64
// Vector acquires ub_out (output) — blocks until MTE3 releases it from a prior iteration (WAR: MTE3 read → V write)
pto.get_buf "PIPE_V", %bufid_ub_out, %mode : i64, i64

scf.for %dummy = %c0 to %c1 step %c1 {
  %mask = pto.pset_b32 "PAT_ALL" : !pto.mask
  %v   = pto.vlds %ub_ptr[%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
  pto.vsts %abs, %ub_out[%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask
} {llvm.loop.aivector_scope}

// Vector done reading ub_ptr — release so MTE2 can reuse it in next iteration
pto.rls_buf "PIPE_V", %bufid_ub_ptr, %mode : i64, i64
// Vector done writing ub_out — release so MTE3 can consume
pto.rls_buf "PIPE_V", %bufid_ub_out, %mode : i64, i64

// ─── Stage 3: MTE3 stores result to GM ───
// MTE3 acquires ub_out — blocks until Vector releases it (RAW: V write → MTE3 read)
pto.get_buf "PIPE_MTE3", %bufid_ub_out, %mode : i64, i64
pto.copy_ubuf_to_gm %ub_out, %gm_out, ...
// MTE3 done reading ub_out — release so Vector can reuse it in next iteration
pto.rls_buf "PIPE_MTE3", %bufid_ub_out, %mode : i64, i64
```

**Key property:** No event IDs needed. Dependencies are implicit from program order of `get_buf`/`rls_buf` on the same buffer ID. This becomes much more convenient in multi-iteration loops (see Example 3).

### Example 3: Ping/Pong Double-Buffering Loop

Double-buffering overlaps DMA and compute by using two UB buffers alternately. All three stages (MTE2, Vector, MTE3) appear in the **same iteration** — the hardware pipelines them across iterations because different iterations operate on different buffers (`buf[i%2]`).

#### Event ID scheme (`set_flag` / `wait_flag`)

With 2 ping/pong buffers and 2 pipeline pairs (MTE2↔V, V↔MTE3), `set_flag`/`wait_flag` needs **8 event IDs** = 2 pipe-pairs × 2 buffers × (forward + reverse):

**MTE2 ↔ Vector (input buffers):**

| Event ID | Direction | Purpose |
|----------|-----------|---------|
| `EVT_IN_FWD_0` | MTE2 → V | RAW: buf_in[0] data ready |
| `EVT_IN_FWD_1` | MTE2 → V | RAW: buf_in[1] data ready |
| `EVT_IN_REV_0` | V → MTE2 | WAR: Vector done reading buf_in[0] |
| `EVT_IN_REV_1` | V → MTE2 | WAR: Vector done reading buf_in[1] |

**Vector ↔ MTE3 (output buffers):**

| Event ID | Direction | Purpose |
|----------|-----------|---------|
| `EVT_OUT_FWD_0` | V → MTE3 | RAW: buf_out[0] result ready |
| `EVT_OUT_FWD_1` | V → MTE3 | RAW: buf_out[1] result ready |
| `EVT_OUT_REV_0` | MTE3 → V | WAR: MTE3 done reading buf_out[0] |
| `EVT_OUT_REV_1` | MTE3 → V | WAR: MTE3 done reading buf_out[1] |

#### 3a. `set_flag` / `wait_flag` version

```mlir
// ═══ Pre-loop: prime ALL reverse-dependency signals ═══
// Both input and output buffers start unused. We must pre-send
// reverse-dep signals so the first iteration's wait_flags don't deadlock.
pto.set_flag["PIPE_V",    "PIPE_MTE2", "EVT_IN_REV_0"]   // ◀ PRIME: buf_in[0] "free"
pto.set_flag["PIPE_V",    "PIPE_MTE2", "EVT_IN_REV_1"]   // ◀ PRIME: buf_in[1] "free"
pto.set_flag["PIPE_MTE3", "PIPE_V",    "EVT_OUT_REV_0"]  // ◀ PRIME: buf_out[0] "free"
pto.set_flag["PIPE_MTE3", "PIPE_V",    "EVT_OUT_REV_1"]  // ◀ PRIME: buf_out[1] "free"

scf.for %i = %c0 to %N step %c1 {
  // ── All 3 stages in same iteration, indexed by i%2 ──
  // %pp = i % 2  (ping/pong selector for buffer & event IDs)

  // ── MTE2: load tile[i] into buf_in[i%2] ──
  // WAR: wait until Vector has released buf_in[i%2] from iteration i-2
  pto.wait_flag["PIPE_V", "PIPE_MTE2", "EVT_IN_REV_{pp}"]
  pto.copy_gm_to_ubuf %gm_ptr[%i], %ub_in[%pp], ...
  // RAW: signal Vector that buf_in[i%2] data is ready
  pto.set_flag["PIPE_MTE2", "PIPE_V", "EVT_IN_FWD_{pp}"]

  // ── Vector: compute buf_in[i%2] → buf_out[i%2] ──
  // RAW: wait for MTE2 to finish loading buf_in[i%2]
  pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVT_IN_FWD_{pp}"]
  // WAR: wait for MTE3 to finish reading buf_out[i%2] from iteration i-2
  pto.wait_flag["PIPE_MTE3", "PIPE_V", "EVT_OUT_REV_{pp}"]
  scf.for %dummy = %c0 to %c1 step %c1 {
    %v   = pto.vlds %ub_in[%pp][%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
    %mask = pto.pset_b32 "PAT_ALL" : !pto.mask
    %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
    pto.vsts %abs, %ub_out[%pp][%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask
  } {llvm.loop.aivector_scope}
  // WAR: tell MTE2 "done reading buf_in[i%2]"
  pto.set_flag["PIPE_V", "PIPE_MTE2", "EVT_IN_REV_{pp}"]
  // RAW: tell MTE3 "buf_out[i%2] result ready"
  pto.set_flag["PIPE_V", "PIPE_MTE3", "EVT_OUT_FWD_{pp}"]

  // ── MTE3: store result from buf_out[i%2] to GM ──
  // RAW: wait for Vector to finish writing buf_out[i%2]
  pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVT_OUT_FWD_{pp}"]
  pto.copy_ubuf_to_gm %ub_out[%pp], %gm_out[%i], ...
  // WAR: tell Vector "done reading buf_out[i%2]"
  pto.set_flag["PIPE_MTE3", "PIPE_V", "EVT_OUT_REV_{pp}"]
}

// ═══ Post-loop: drain — match every pre-loop prime with a wait ═══
// Each priming set_flag must be paired. The last loop iteration's
// set_flags are consumed by wait_flags that will never fire inside the
// loop (there is no iteration i+2). Drain them here.
pto.wait_flag["PIPE_V",    "PIPE_MTE2", "EVT_IN_REV_{(N-1)%2}"]  // ◀ DRAIN
pto.wait_flag["PIPE_V",    "PIPE_MTE2", "EVT_IN_REV_{(N-2)%2}"]  // ◀ DRAIN
pto.wait_flag["PIPE_MTE3", "PIPE_V",    "EVT_OUT_REV_{(N-1)%2}"] // ◀ DRAIN
pto.wait_flag["PIPE_MTE3", "PIPE_V",    "EVT_OUT_REV_{(N-2)%2}"] // ◀ DRAIN
```

**What `set_flag`/`wait_flag` requires outside the loop:**

#### 3b. `get_buf` / `rls_buf` version

Same ping/pong double-buffering, but **no pre-loop priming or post-loop draining needed.** Buffer acquire/release semantics handle everything.

```mlir
scf.for %i = %c0 to %N step %c1 {
  // %pp = i % 2  (ping/pong selector)

  // ── MTE2: load tile[i] into buf[i%2] ──
  // Acquires buf[i%2] — on first iteration, buffer is free so proceeds immediately.
  // On later iterations, blocks until Vector releases buf[i%2] (WAR: automatic).
  pto.get_buf %bufid_buf[%pp], "PIPE_MTE2"
  pto.copy_gm_to_ubuf %gm_ptr[%i], %ub_buf[%pp], ...
  pto.rls_buf %bufid_buf[%pp], "PIPE_MTE2"

  // ── Vector: compute on buf[i%2] ──
  // Acquires buf[i%2] — blocks until MTE2 releases it (RAW: automatic)
  pto.get_buf %bufid_buf[%pp], "PIPE_V"
  pto.get_buf %bufid_out[%pp], "PIPE_V"
  scf.for %dummy = %c0 to %c1 step %c1 {
    %v   = pto.vlds %ub_buf[%pp][%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
    %mask = pto.pset_b32 "PAT_ALL" : !pto.mask
    %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
    pto.vsts %abs, %ub_out[%pp][%lane], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask
  } {llvm.loop.aivector_scope}
  // Release buf[i%2] — MTE2 can reuse in iteration i+2 (WAR resolved)
  pto.rls_buf %bufid_buf[%pp], "PIPE_V"
  pto.rls_buf %bufid_out[%pp], "PIPE_V"

  // ── MTE3: store result ──
  // Acquires out[i%2] — blocks until Vector releases it (RAW: automatic)
  pto.get_buf %bufid_out[%pp], "PIPE_MTE3"
  pto.copy_ubuf_to_gm %ub_out[%pp], %gm_out[%i], ...
  pto.rls_buf %bufid_out[%pp], "PIPE_MTE3"
}
// No post-loop drain needed — last rls_buf completes the pipeline.
```

**No priming, no draining, no event IDs.** The acquire/release protocol on buffer IDs indexed by `i%2` implicitly resolves all cross-pipeline dependencies:
- **RAW** (MTE2→V): Vector's `get_buf` blocks until MTE2's `rls_buf` on `buf[i%2]`
- **WAR** (V→MTE2): MTE2's `get_buf` in iteration `i+2` blocks until Vector's `rls_buf` in iteration `i` (same buffer)

## Comparison Summary

| Aspect | `set_flag` / `wait_flag` | `get_buf` / `rls_buf` |
|--------|--------------------------|------------------------|
| Dependency model | Explicit event signals | Implicit via buffer acquire/release |
| IDs per pipe-pair | **8** = 2 buffers × 2 dirs × 2 (fwd+rev) | 1 fwd + 1 rev per buffer (shared global pool) |
| Total HW IDs | 8 per pipe-pair, grows with buffers | **32 global** across all pipes |
| Reverse (WAR) deps | Extra `set_flag`/`wait_flag` pair per buffer | Handled automatically |
| Pre-loop setup | `set_flag` to prime each reverse dep | None |
| Post-loop teardown | `wait_flag` to drain all primed signals | None |
| Straight-line code | Simple, clear | Slightly more verbose (bracket each stage) |
| Ping/pong loops | 8 event IDs + 4 prime + 4 drain | Same pattern, no overhead |
| Best used for | Simple pipelines, fine-grained control | Double/multi-buffering, complex loops |

## Inter-Core Sync

> **Note:** Inter-core sync is only needed for **mixed Cube+Vector tasks** where Cube produces data that Vector consumes (or vice versa). **Vec-only tasks can ignore this section entirely.**

These ops coordinate execution across the Cube block and Vector subblocks within a cluster. Each core cluster consists of **1 Cube block : 2 Vector subblocks**, each with its own **SU (Sequencer Unit)** running independent instruction streams.

```
Core Cluster (1:2 ratio)
┌─────────────────────────────────────────────┐
│  ┌──────────────┐    ┌──────────────┐       │
│  │  AIC (Cube)  │    │  AIV0 (Vec)  │       │
│  │  ┌────────┐  │    │  ┌────────┐  │       │
│  │  │   SU   │──┼────┼──│   SU   │  │       │
│  │  └────────┘  │    │  └────────┘  │       │
│  │  CUBE pipe   │    │  MTE2/V/MTE3 │       │
│  │  L0C buffer  │    │  UB (256KB)  │       │
│  └──────────────┘    └──────────────┘       │
│                      ┌──────────────┐       │
│                      │  AIV1 (Vec)  │       │
│                      │  ┌────────┐  │       │
│                      │  │   SU   │  │       │
│                      │  └────────┘  │       │
│                      │  MTE2/V/MTE3 │       │
│                      │  UB (256KB)  │       │
│                      └──────────────┘       │
└─────────────────────────────────────────────┘
```

### Platform Comparison

| Aspect | A2A3 (Ascend 910) | A5 (A5) |
|--------|-------------------|-----------------|
| **Signal op** | `set_cross_core` (mode2) | `set_intra_block` |
| **Wait op** | `wait_flag_dev` | `wait_intra_core` |
| **Wait behavior** | SU-level blocking (entire core stalls) | Per-pipeline (only named pipe stalls) |
| **Semaphore pool** | 16 IDs per cluster, 4-bit counter | 16 IDs, but 32-ID address space (see below) |
| **C→V** | **Broadcast**: one `set` reaches both AIV0+AIV1 | **1:1**: separate `set` per subblock required |
| **V→C** | **Reduce**: Cube waits for both subblocks in one `wait` | **1:1**: Cube needs separate `wait` per subblock |

### A2A3: `set_cross_core` / `wait_flag_dev`

```c
// mode2 broadcast/reduce semantics for 1:2 cluster
set_cross_core(pipe, semaphore_id);   // pipe: VEC/MTE2/CUBE/FIX
wait_flag_dev(semaphore_id);          // SU-level blocking
```

```
C→V Broadcast (one set reaches both):
    AIC ──set_cross_core──┬──> AIV0 sema++
                          └──> AIV1 sema++

V→C Reduce (one wait for both):
    AIV0 ──set_cross_core──┐
                           ├──> AIC wait_flag_dev (blocks until BOTH)
    AIV1 ──set_cross_core──┘
```

## Related Ops / Family Links

- Family overview: [Pipeline Sync](../../pipeline-sync.md)
- Previous op in family: [pto.rls_buf](./rls-buf.md)
- Next op in family: [pto.set_cross_core](./set-cross-core.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
