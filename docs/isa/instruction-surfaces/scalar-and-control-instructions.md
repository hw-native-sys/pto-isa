# Scalar And Control Instruction Set

`pto.*` (scalar/control) is the configuration, synchronization, and scalar-orchestration instruction set of PTO ISA. It sets up the execution shell around tile and vector payload regions.

## Instruction Overview

Scalar/control instructions do not produce tile or vector payloads. Instead, they produce:

- Control effects (pipeline barriers, control flow)
- Event tokens for explicit producer-consumer ordering
- Predicate masks for conditional execution
- DMA configuration state for memory transfer
- Scalar register updates

**Scalar operands** are single-value registers. Scalar instructions are the glue that orchestrates tile and vector work.

## Instruction Classes

| Class | Description | Examples |
|-------|-------------|----------|
| Control and Configuration | NOP, barrier, yield, and mode/config setup | `nop`, `barrier`, `yield`, `tsethf32mode`, `tsetfmatrix` |
| Pipeline Sync | Event and barrier synchronization between pipelines | `set_flag`, `wait_flag`, `pipe_barrier` |
| DMA Copy | GM↔vector-tile-buffer transfer configuration and initiation | `copy_gm_to_ubuf`, `copy_ubuf_to_gm`, `set_loop_size_outtoub` |
| Predicate Load/Store | Mask-based scalar memory access | `pld`, `plds`, `pdi`, `pst`, `psts`, `psti`, `pstu` |
| Predicate Generation | Predicate construction and algebra | `pset_b8`, `pge_b8`, `plt_b8`, `pand`, `por`, `pxor`, `pnot`, `pintlv_b16` |

## Inputs

Scalar/control instructions consume combinations of:

- Scalar registers (`!pto.scalar<T>` or built-in C++ types)
- Pipe identifiers: `PIPE_MTE1`, `PIPE_MTE2`, `PIPE_MTE3`, `PIPE_V`, `PIPE_M`
- Event identifiers: `EVENT_ID0`–`EVENT_ID15` (profile-specific range)
- Buffer identifiers: UB buffer slots
- Memory addresses: `!pto.ptr<T, ub>` or `!pto.ptr<T, gm>`
- DMA loop sizes and stride values

## Expected Outputs

Scalar/control instructions produce:

- Control state changes (pipeline barriers, control flow)
- Event tokens for explicit synchronization (`RecordEvent`)
- Predicate masks (`!pto.mask`)
- Configured DMA state ready for transfer
- UB buffer handles

## Side Effects

Scalar/control instructions may have significant architectural side effects:

| Class | Side Effects |
|-------|-------------|
| Pipeline Sync | Establishes producer-consumer ordering; may stall pipeline stages |
| DMA Copy | Initiates memory transfer between GM and UB; may stall DMA engine |
| Predicate Load/Store | Reads from or writes to scalar memory locations in UB |

## Event Model

Scalar/control sync operations use an event-based model. Events are identified by a triple `(src_pipe, dst_pipe, event_id)`:

| Field | Values | Meaning |
|-------|--------|---------|
| `src_pipe` | `PIPE_MTE1`, `PIPE_MTE2`, `PIPE_MTE3`, `PIPE_V`, `PIPE_M` | Source pipeline that produces the event |
| `dst_pipe` | `PIPE_MTE1`, `PIPE_MTE2`, `PIPE_MTE3`, `PIPE_V`, `PIPE_M` | Destination pipeline that consumes the event |
| `event_id` | 0–15 (profile-specific) | Event slot identifier |

```
Producer pipeline                          Consumer pipeline
  │                                          │
  │  issue DMA or compute                    │
  │  ▼                                       │
  │  set_flag(src_pipe, dst_pipe, EVENT_ID)  │
  │  (produces the event)                    │
  │                                          │
  │                            wait_flag(src_pipe, dst_pipe, EVENT_ID)
  │                            (consumes the event)
  │                                          │
  │  data/result available                   │
  ▼                                          ▼
```

## Pipe Spaces by Target Profile

| Pipe | CPU Sim | A2/A3 | A5 |
|------|:-------:|:------:|:--:|
| `PIPE_MTE1` | Simulated | Supported | Supported |
| `PIPE_MTE2` | Simulated | Supported | Supported |
| `PIPE_MTE3` | Simulated | Supported | Supported |
| `PIPE_V` | Emulated | Emulated | Native |
| `PIPE_M` | Simulated | Supported | Supported |

## Constraints

- **Pipe/event spaces** differ between A2/A3 and A5 profiles; portable code must use the documented PTO contract plus the selected profile.
- **Event ordering** requires matching `set_flag`/`wait_flag` pairs; waiting on an unestablished event is illegal.
- **DMA parameters** must be configured before initiating transfer; incorrect loop sizes or strides produce undefined results.
- **Predicate width** must match the expected mask width for the target profile.
- **Pipe identifiers** not supported by the target profile produce verification errors.

## Cases That Are Not Allowed

- Using pipe or event identifiers not supported by the target profile.
- Waiting on an event that was never established by a matching producer.
- Configuring DMA with inconsistent loop sizes and strides.
- Mixing predicate widths that do not match the target operation.
- Issuing a vector load before `copy_gm_to_ubuf` completes without an intervening `wait_flag`.
- Issuing `copy_ubuf_to_gm` before vector store completes without an intervening `wait_flag`.

## Syntax

### Assembly Form (PTO-AS)

```asm
set_flag PIPE_MTE2, PIPE_V, EVENT_ID0
wait_flag PIPE_MTE2, PIPE_V, EVENT_ID0
pipe_barrier PIPE_V
```

### SSA Form (AS Level 1)

```mlir
pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
pto.pipe_barrier["PIPE_V"]
```

See [Assembly Spelling And Operands](../syntax-and-operands/assembly-model.md) for the full syntax specification.

## C++ Intrinsic

Scalar/control instructions are available as C++ intrinsics declared in `include/pto/common/pto_instr.hpp`:

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

// Set synchronization flag
PTO_INST void set_flag(pipe_t src_pipe, pipe_t dst_pipe, event_t event_id);

// Wait on synchronization flag
PTO_INST void wait_flag(pipe_t src_pipe, pipe_t dst_pipe, event_t event_id);

// Pipeline barrier
PTO_INST void pipe_barrier(pipe_t pipe);

// DMA copy: GM → vector tile buffer / hardware UB
PTO_INST void copy_gm_to_ubuf(ub_ptr dst, gm_ptr src, uint64_t sid,
                              uint64_t n_burst, uint64_t len_burst,
                              uint64_t dst_stride, uint64_t src_stride);

// DMA copy: vector tile buffer / hardware UB → GM
PTO_INST void copy_ubuf_to_gm(gm_ptr dst, ub_ptr src, uint64_t sid,
                              uint64_t n_burst, uint64_t len_burst,
                              uint64_t reserved, uint64_t dst_stride, uint64_t src_stride);
```

## See Also

- [Scalar ISA reference](../scalar/README.md) — Full scalar instruction set reference
- [Scalar instruction sets](../instruction-families/scalar-and-control-families.md) — Instruction-set-level contracts
- [Instruction sets](../instruction-families/README.md) — All instruction sets
- [Format of instruction descriptions](../reference/format-of-instruction-descriptions.md) — Per-op page standard
- [Ordering and Synchronization](../machine-model/ordering-and-synchronization.md) — PTO memory and execution ordering model
