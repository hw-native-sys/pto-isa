<!-- Generated from `docs/isa/scalar/predicate-load-store.md` -->

# Predicate Load Store

Predicate load/store family moves predicate-register state (`!pto.mask`) between UB-visible storage and the architectural predicate surface. Predicates are the lane-masking mechanism that `pto.v*` vector operations consume.

## Mechanism

Predicate state lives on the scalar/control surface. `pld*/pst*` operations transfer predicate bits to or from UB memory locations, enabling predicates to persist across kernel boundaries or to be shared with scalar address calculations.

### Data Flow

```
Predicate Register File ──(plds/pld/pldi)──► UB location (64-bit aligned)
UB location ──(psts/pst/psti/pstu)──► Predicate Register File
```

### Predicate Width

| Element Type | Vector Width N | Predicate Width |
|-------------|:-------------:|:--------------:|
| f32 | 64 | 64 bits |
| f16 / bf16 | 128 | 128 bits (2 × 64-bit transfers) |
| i8 / u8 | 256 | 256 bits (4 × 64-bit transfers) |

A single predicate load/store operation covers the full predicate width for the element type in use. Partial predicate loads are **not supported**.

### Alignment Requirements

| Operation | Alignment Requirement | Consequence of Violation |
|-----------|----------------------|--------------------------|
| `plds` / `psts` | 64-bit (8 bytes) at UB address | Illegal if address not 8-byte aligned |
| `pld` / `pst` (areg offset) | 64-bit; offset must be register-aligned | Illegal if address or offset violates alignment |
| `pldi` / `psti` (immediate offset) | 64-bit; offset must be compile-time constant | Illegal if immediate violates alignment |
| `pstu` (stream form) | None; tracks alignment state internally | Alignment state is implementation-defined on first use |

## Distribution Modes

Distribution modes (`dist` attribute) control how predicate bits are packed into UB storage. All load/store forms accept a `dist` attribute:

| Mode | Description | Load Behavior | Store Behavior |
|------|-------------|---------------|----------------|
| `NORM` | Normal packing | Read 64-bit predicate word directly | Write 64-bit predicate word directly |
| `PK` | Packed (store only) | Not applicable | Pack two 32-bit predicate segments into one 64-bit word |
| `US` | Unsigned streaming | UB bits as-is | UB bits as-is |
| `DS` | Signed streaming | UB bits as-is, sign-extend | UB bits as-is |

## Shared Constraints

All predicate load/store operations MUST satisfy:

1. **UB address space**: The pointer operand MUST have type `!pto.ptr<T, ub>`. Predicates cannot be transferred directly to/from GM.
2. **Alignment**: The effective UB address (base + offset) MUST be 64-bit aligned. The stream form (`pstu`) relaxes this but imposes its own ordering requirements.
3. **Predicate width match**: The transfer covers the full predicate width for the active element type. Partial transfers are not permitted.
4. **Event ordering**: When used in a producer-consumer chain with DMA, the program MUST use `set_flag`/`wait_flag` to order the predicate transfer before or after the dependent operation.
5. **Single active predicate**: At any point in program order, at most one predicate register is architecturally active. Concurrent predicate transfers that would overwrite an in-flight predicate are **illegal**.

## Stream Form (`pstu`)

`pto.pstu` is the high-throughput stream variant of predicate store. It differs from `psts` in the following ways:

| Aspect | `psts` | `pstu` |
|--------|--------|--------|
| Alignment | 64-bit required | None required |
| Write atomicity | Single predicate word is atomic | Writes may be batched; individual 64-bit words are **not** guaranteed atomic |
| Alignment state | Not updated | Updates `%align_out` with new alignment base |
| Use case | Exact predicate save/restore | Streaming predicate writes with internal buffering |

Programs that require exact predicate state restoration (e.g., saving and restoring a mask for later reuse) MUST use `psts`. Programs that stream predicates as part of a larger pipeline SHOULD use `pstu`.

## Predicate Lifecycle

A typical predicate load/store lifecycle:

```
// Kernel entry: load saved predicate
%mask = pto.plds %ub_saved : !pto.ptr<i64, ub> -> !pto.mask

// Use predicate for vector computation
%result = pto.vsel %v_true, %v_false, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// At kernel exit: save predicate for next kernel
pto.psts %mask, %ub_saved : !pto.mask, !pto.ptr<i64, ub>
```

## Target-Profile Restrictions

| Feature | CPU Simulator | A2/A3 | A5 |
|---------|:------------:|:------:|:--:|
| `plds` / `psts` | Simulated | Supported | Supported |
| `pld` / `pst` (areg) | Simulated | Supported | Supported |
| `pldi` / `psti` (immediate) | Simulated | Supported | Supported |
| `pstu` stream form | Not supported | Supported | Supported |
| `PK` distribution mode | Not supported | Supported | Supported |
| Alignment relaxation (`pstu`) | Not applicable | Supported | Supported |

## Per-Op Pages

- [pto.plds](./ops/predicate-load-store/plds.md) — Contiguous predicate load
- [pto.pld](./ops/predicate-load-store/pld.md) — Predicate load with areg offset
- [pto.pldi](./ops/predicate-load-store/pldi.md) — Predicate load with immediate offset
- [pto.psts](./ops/predicate-load-store/psts.md) — Contiguous predicate store
- [pto.pst](./ops/predicate-load-store/pst.md) — Predicate store with areg offset
- [pto.psti](./ops/predicate-load-store/psti.md) — Predicate store with immediate offset
- [pto.pstu](./ops/predicate-load-store/pstu.md) — Predicate unaligned stream store

## Related Material

- [Control and configuration](./control-and-configuration.md)
- [Vector Families: Predicate And Materialization](../vector/predicate-and-materialization.md)
- [Predicate Generation And Algebra](./predicate-generation-and-algebra.md)
