# GPU Backend Plan for PTO Tile Lib

## Selected Decisions (2026-03-31)

- Git identity: use the already configured global Git identity
- Backend substrate: **CUDA C++ + inline PTX for hot kernels**
- Architecture scope: **keep all scaffolded SM dirs, implement `sm121` first**
- First implementation slice: **mixed** (`TLOAD`, `TSTORE`, `TADD`, `TMATMUL`)
- Event semantics in v1: **defer advanced semantics until compute kernels work**

## Current Repo Analysis

### 1. Existing backend split

The public entry point is `include/pto/pto-inst.hpp`, which pulls in:

- shared tile/type/instruction declarations from `include/pto/common/`
- CPU simulator support through `__CPU_SIM`
- NPU implementations through `include/pto/common/arch_macro.hpp` and `include/pto/common/pto_instr_impl.hpp`

Today the repo has three real backend families:

- **CPU simulator**: header-only reference/simulation path
- **NPU A2/A3**: `include/pto/npu/a2a3/`
- **NPU A5**: `include/pto/npu/a5/`

There is **no NVIDIA GPU backend yet**:

- no `include/pto/gpu/`
- no GPU dispatch macro in `arch_macro.hpp`
- no GPU include block in `pto_instr_impl.hpp`
- no `tests/gpu/`
- no `kernels/manual/gpu/`

### 2. What the existing backends tell us

#### CPU backend

The CPU path is the functional/spec reference layer:

- broad instruction surface area
- flexible tile/layout checks
- useful for correctness and fallback semantics
- `TSYNC` is effectively a no-op in CPU simulation

This is the best source of **semantic truth** for a future GPU backend.

#### NPU A2/A3 backend

The A2/A3 path is the first real hardware mapping layer:

- instruction-specific files such as `TAdd.hpp`, `TLoad.hpp`, `TMatmul.hpp`
- explicit pipeline/event handling
- memory-move kernels mapped to hardware copy instructions
- vector and cube paths separated by tile/layout constraints

This is the best source of **how PTO instructions get lowered to hardware micro-ops**.

#### NPU A5 backend

The A5 path extends the same architecture with:

- richer data type support
- more advanced matmul variants (FP8 / MX / FP4-related handling)
- evolved sync/event behavior
- more specialized utility helpers and hardware-specific tuning

This is the best source of **how backend specialization grows with hardware generations**.

### 3. Architectural implications for a GPU backend

A CUDA/NVIDIA backend should not start as one monolithic implementation. It should follow the repo's existing pattern:

- shared PTO API remains in `include/pto/common/`
- GPU-specific lowering lives under `include/pto/gpu/`
- arch-specific overrides live in per-SM subdirectories
- common CUDA helpers, tile iterators, layout adapters, and launch helpers live in `include/pto/gpu/common/`

### 4. Backend bring-up priorities

To make a GPU backend useful early, the instruction bring-up order should be:

1. **Dispatch and toolchain plumbing**
2. **Memory movement**: `TLOAD`, `TSTORE`, `TPREFETCH`
3. **Elementwise vector ops**: `TADD`, `TSUB`, `TMUL`, `TMAX`, `TMIN`, `TEXPANDS`, comparisons/selects
4. **Reductions / reshapes / transforms**
5. **Matrix kernels**: `TMATMUL`, `TMATMUL_ACC`, `TMATMUL_BIAS`, `TMATMUL_MX`
6. **Synchronization / event semantics**
7. **Collective / comm semantics**, if GPU multi-device becomes in scope

---

## Proposed Directory Layout

```text
include/pto/gpu/
  README.md
  common/
    README.md
  sm70/
    README.md
    microkernels/
      README.md
  sm75/
    README.md
    microkernels/
      README.md
  sm80/
    README.md
    microkernels/
      README.md
  sm86/
    README.md
    microkernels/
      README.md
  sm87/
    README.md
    microkernels/
      README.md
  sm89/
    README.md
    microkernels/
      README.md
  sm90/
    README.md
    microkernels/
      README.md
  sm90a/
    README.md
    microkernels/
      README.md
  sm100/
    README.md
    microkernels/
      README.md
  sm100a/
    README.md
    microkernels/
      README.md
  sm103/
    README.md
    microkernels/
      README.md
  sm110/
    README.md
    microkernels/
      README.md
  sm120/
    README.md
    microkernels/
      README.md
  sm121/
    README.md
    microkernels/
      README.md

tests/gpu/
  README.md

kernels/manual/gpu/
  README.md
```

### Why these SM targets

These folders cover the modern NVIDIA targets that matter for PTO-style tile programming and current deployment:

- `sm70`: Volta
- `sm75`: Turing
- `sm80`: Ampere datacenter
- `sm86`: Ampere client/workstation
- `sm87`: Orin
- `sm89`: Ada
- `sm90`: Hopper
- `sm90a`: Hopper architecture-conditional specialization
- `sm100`: Blackwell
- `sm100a`: Blackwell architecture-conditional specialization
- `sm103`: GB300 / B300 generation
- `sm110`: Thor / Jetson T-series generation
- `sm120`: Blackwell workstation / GeForce generation
- `sm121`: GB10 (DGX Spark)

This machine is **GB10 / compute capability 12.1**, so `sm121` should be the first optimization target.

---

## Phase Plan

### Phase 0 — Scope freeze and backend contract

**Goal:** Decide what “GPU backend” means in this repo before writing kernels.

#### Checklist

- [ ] Decide whether the first GPU backend is:
  - [ ] correctness-first
  - [ ] performance-first
  - [ ] hybrid (correctness-first, optimize hot instructions first)
- [ ] Decide the implementation substrate:
  - [ ] CUDA C++
  - [ ] CUDA C++ + inline PTX
  - [ ] CUDA C++ + CUTLASS/CuTe
  - [ ] CUDA C++ + generated microkernels
- [ ] Decide the minimum supported toolkit/driver baseline
- [ ] Decide the minimum supported GPU baseline (`sm80+` only vs broad support)
- [ ] Decide whether GPU backend must preserve exact PTO event semantics or allow a GPU-specific semantic subset first
- [ ] Decide whether multi-GPU comm is in phase 1 or deferred

### Phase 1 — Dispatch, macros, and build plumbing

**Goal:** Make GPU a first-class backend selection path.

#### Checklist

- [ ] Add GPU arch detection / selection macros
- [ ] Define a backend macro family such as:
  - [ ] `PTO_GPU_BACKEND`
  - [ ] `PTO_GPU_SM80`
  - [ ] `PTO_GPU_SM90`
  - [ ] `PTO_GPU_SM121`
- [ ] Add GPU include blocks to `include/pto/common/pto_instr_impl.hpp`
- [ ] Decide whether `include/pto/pto-inst.hpp` should expose GPU path under:
  - [ ] `__CUDACC__`
  - [ ] a project-defined macro
  - [ ] both
- [ ] Introduce initial CMake path for GPU tests/builds
- [ ] Add one minimal compile-only smoke target for GPU backend headers

### Phase 2 — Memory model and tile mapping

**Goal:** Define how PTO tiles live on NVIDIA GPUs.

#### Checklist

- [ ] Map PTO tile memory spaces onto CUDA memory spaces:
  - [ ] global memory
  - [ ] shared memory
  - [ ] registers
  - [ ] tensor memory / async staging abstractions where needed
- [ ] Define row-major / col-major / NZ-like tile layout adapters on GPU
- [ ] Decide whether PTO fractal layouts are represented as:
  - [ ] compile-time layout descriptors
  - [ ] iterator objects
  - [ ] shared-memory swizzles
- [ ] Implement common helper layer in `include/pto/gpu/common/`
- [ ] Document tile-to-warp / tile-to-warpgroup ownership rules

### Phase 3 — Memory instructions first

**Goal:** Make data movement work before math.

#### Checklist

- [ ] Implement `TLOAD`
- [ ] Implement `TSTORE`
- [ ] Implement `TPREFETCH`
- [ ] Validate padding, stride, and layout conversions
- [ ] Add tests for ND / DN / NZ-like cases
- [ ] Add baseline performance checks for bandwidth-sensitive paths

### Phase 4 — Core elementwise backend

**Goal:** Bring up the wide surface-area instructions with reusable microkernel templates.

#### Checklist

- [ ] Implement binary elementwise template family
- [ ] Implement unary elementwise template family
- [ ] Prioritize:
  - [ ] `TADD`
  - [ ] `TSUB`
  - [ ] `TMUL`
  - [ ] `TDIV`
  - [ ] `TMIN`
  - [ ] `TMAX`
  - [ ] `TCMP` / `TCMPS`
  - [ ] `TSEL` / `TSELS`
  - [ ] `TEXP` / `TLOG` / `TRSQRT` / `TSQRT`
- [ ] Split generic implementations from per-SM overrides
- [ ] Add correctness tests against CPU simulator outputs

### Phase 5 — Reduction, reshape, and transform ops

**Goal:** Cover the shape-changing and reduction-heavy instructions.

#### Checklist

- [ ] `TRow*` family
- [ ] `TCol*` family
- [ ] `TRESHAPE`
- [ ] `TTRANS`
- [ ] `TEXTRACT`
- [ ] `TFILLPAD`
- [ ] `TGATHER` / `TSCATTER` variants where practical
- [ ] Decide warp-level vs block-level reduction strategies per SM

### Phase 6 — Matrix and tensor-core path

**Goal:** Make PTO meaningful for high-performance GPU kernels.

#### Checklist

- [ ] Implement a generic `TMATMUL` path first
- [ ] Add accumulator forms:
  - [ ] `TMATMUL_ACC`
  - [ ] `TMATMUL_BIAS`
  - [ ] `TGEMV`
- [ ] Decide whether `TMATMUL_MX` is phase-1 GPU scope or deferred
- [ ] Create per-SM tensor-core microkernel registry
- [ ] Define tile shape families by SM
- [ ] Add tuning metadata per instruction / dtype / tile shape / SM
- [ ] Prioritize `sm121`, `sm120`, `sm90`, `sm89`, `sm80`

### Phase 7 — Sync/event semantics on GPU

**Goal:** Reconcile PTO pipeline/event model with CUDA execution semantics.

#### Checklist

- [ ] Define what `TSYNC` means on GPU
- [ ] Decide whether `Event<SrcOp, DstOp>` is:
  - [ ] fully modeled
  - [ ] partially modeled
  - [ ] translated into CUDA barriers / named barriers / cooperative groups
- [ ] Document unsupported or approximated event cases
- [ ] Add tests for ordering-sensitive instruction sequences

### Phase 8 — Tests, perf harness, and examples

**Goal:** Make the backend maintainable.

#### Checklist

- [ ] Add `tests/gpu/` structure
- [ ] Add CPU-vs-GPU oracle tests
- [ ] Add instruction-level microbenchmarks
- [ ] Add kernel-level examples under `kernels/manual/gpu/`
- [ ] Add CI matrix for at least compile coverage, even if hardware CI is unavailable
- [ ] Add documentation for supported instructions per SM

---

## Recommended first implementation slice

If the goal is fast progress with good signal, the first real slice should be:

1. `sm121` + `sm120` + `sm90` shared bring-up path
2. `TLOAD`
3. `TSTORE`
4. `TADD`
5. `TMUL`
6. `TMAX`
7. `TMATMUL`
8. basic correctness tests vs CPU simulator

That gets the backend from “directory scaffold” to “real execution path” with a minimal but meaningful kernel set.

---

## Open design questions for the next step

1. **Backend substrate**
   - Option A: plain CUDA C++ first
   - Option B: CUDA + inline PTX for hot kernels
   - Option C: CUDA + CUTLASS/CuTe for matmul, custom kernels for the rest

2. **Initial architecture scope**
   - Option A: `sm121` only
   - Option B: `sm121` + `sm120` + `sm90`
   - Option C: all scaffolded SMs, but only `sm121` gets real kernels first

3. **Bring-up priority**
   - Option A: memory ops first
   - Option B: matmul first
   - Option C: elementwise first
   - Option D: mixed (`TLOAD/TSTORE/TADD/TMATMUL`)

4. **How strict should PTO event semantics be in v1?**
   - Option A: exact where possible, fail-fast otherwise
   - Option B: approximate with documented caveats
   - Option C: defer advanced event semantics until after compute kernels land
