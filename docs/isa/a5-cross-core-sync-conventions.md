# A5 Cross-Core Synchronization Conventions (Mix Architecture)

> **Scope:** A5 (Ascend 950PR / 950DT, `dav-c310`) mix-architecture kernels where the AIC (cube core) and AIV (vector core) run as separate cores in the same block and exchange data over GM or UB. This page documents the producer/consumer pipe-pairing rules that are otherwise only inferable from `include/pto/npu/a5/custom/TSync_Custom.hpp` and the mix testcases.

## Why this page exists

On A5 the AIC (cube core) and AIV (vector core) are **separate cores** that run concurrently, and every cross-core data handoff needs an explicit `set_intra_block` / `wait_intra_block` flag pair. The A5 block pairs one AIC with multiple AIVs, so the same handoff must also cover the `id` / `id + 16` shadow flag. The **pipe ids on the set side and the wait side are not symmetric** — they depend on which core produces, which core consumes, and whether the data crosses cores through GM or through UB. Getting the pairing wrong produces silent symptom/cause separation: all-zero output (consumer ran before producer), wrong values (consumer ran mid-overwrite), or a hang (consumer waits on a flag that never arrives). The rules below let you write the correct flag pair without reading the implementation headers.

## Pipe assignment by data path

| Direction | Path | Producer data op (set pipe) | Consumer data op (wait pipe) | Evidence |
|-----------|------|------------------------------|------------------------------|----------|
| Cube → Vec | GM | `TSTORE` C→GM (`PIPE_FIX`) | `TLOAD` GM→UB (`PIPE_MTE2`) | `TSync_Custom.hpp` `is_c2v_gm` branch |
| Cube → Vec | UB, vector op | `TMOV` L0C→UB (`PIPE_FIX`) | vector op on UB (`PIPE_V`) | `TSync_Custom.hpp` `is_c2v_ub` branch |
| Cube → Vec | UB, GM store | `TMOV` L0C→UB (`PIPE_FIX`) | `TSTORE` UB→GM (`PIPE_MTE3`) | `tests/npu/a5/src/st/testcase/tmov_acc2vec/tmov_acc2vec_kernel.cpp:342-346` |
| Cube → Vec | L1→UB copy | `tf_copy_cbuf_to_ubuf` (`PIPE_MTE1`) | `TSTORE` UB→GM (`PIPE_MTE3`) | `tests/npu/a5/src/st/testcase/tload_mix/tload_mix_kernel.cpp:58-59`, `:99-102` |
| Vec → Cube | GM | `TSTORE` V→GM (`PIPE_MTE3`) | `TLOAD` GM→L1 (`PIPE_MTE2`) | `TSync_Custom.hpp` `is_v2c_gm` branch |
| Vec → Cube | UB | `TINSERT` UB→L1 (`PIPE_MTE3`) | cube op on L1 (`PIPE_MTE1`) | `TSync_Custom.hpp` `is_v2c_ub` branch |
| Cube → Vec (ack / back-pressure) | flag-only | Vec sets on the pipe it consumed on (`PIPE_MTE2` / `PIPE_V` / `PIPE_MTE3`) | Cube waits on `PIPE_FIX` (both ids) | `TSync_Custom.hpp` `free` / `allocate`, `tests/npu/a5/src/st/testcase/tmov_acc2vec/tmov_acc2vec_kernel.cpp:782-786` |
| Vec → Cube (ack / back-pressure) | flag-only | Cube sets on the pipe it consumed on (`PIPE_MTE1`) | Vec waits on `PIPE_MTE3` | `TSync_Custom.hpp` `free` / `allocate` |

### Reading the table

- The **producer's set pipe** is the pipe of the producer's own data instruction: `TSTORE`/`TMOV` from cube run on `PIPE_FIX`, `TSTORE`/`TINSERT` from vec run on `PIPE_MTE3`, a cube-side L1→UB copy runs on `PIPE_MTE1`.
- The **consumer's wait pipe** is the pipe of the consumer instruction that **reads the data**, not the pipe that loaded it — except when the consumer itself loads (`TLOAD` → `PIPE_MTE2`). A vector op reads UB on `PIPE_V`; `TSTORE` UB→GM runs on `PIPE_MTE3`; a cube op reading L1 runs on `PIPE_MTE1`; a cube op writing L0C runs on `PIPE_FIX`. Copying the wait pipe from a neighbouring testcase without checking the consumer's own instruction is what produced the run3 all-zero failure.
- **Back-pressure (ack) flags** flow in the opposite direction and use the pipe the consumer consumed on as the set pipe; the producer waits on its own pipe (`PIPE_FIX` for cube producers, `PIPE_MTE3` for vec producers).

## Shadow event id (`id` / `id + 16`)

When Cube signals Vec it must set **two** flags: `flag_id` and `flag_id + VEC_CORE_ID_OFFSET` (`VEC_CORE_ID_OFFSET = 16`), both on the producer's set pipe (`PIPE_FIX` for `TSTORE`/`TMOV` producers, `PIPE_MTE1` for the L1→UB copy in `tload_mix`). The A5 block pairs one AIC with multiple AIVs; the offset selects which AIV receives the flag. A single `set_intra_block(..., flag_id)` without the shadow id reaches only the first AIV and the others hang.

```cpp
// Cube -> Vec with a TSTORE/TMOV producer: set BOTH the primary and the shadow id on PIPE_FIX
set_intra_block(PIPE_FIX, flag_id);
set_intra_block(PIPE_FIX, flag_id + VEC_CORE_ID_OFFSET);
```

Vec→Cube uses the reverse convention: Vec sets **only** `flag_id` on `PIPE_MTE3` (the hardware maps subblock 1's flag to `flag_id + 16`), and the AIC waits on **both** `flag_id` and `flag_id + VEC_CORE_ID_OFFSET`.

### Event id reuse pitfalls

An event id is a per-pipe resource, not a global one. Reusing the same `flag_id` on the same pipe before the previous `wait_flag` has matched causes a lost wakeup. The safe pattern is:

- Use a fresh `flag_id` for each logically distinct handoff, or
- Reuse an id only after the matching `wait_flag` has executed, and never reuse an id across two producers that can run concurrently.

The id space is small (per pipe); for long kernel loops, cycle ids within the loop and ensure the cycle length exceeds the pipeline depth so no in-flight flag is overwritten.

## Reference implementation

The canonical flag-pair implementation is in `include/pto/npu/a5/custom/TSync_Custom.hpp` (see the `set_intra_block` / `wait_intra_block` wrappers and the `TSTORE_C2GM` / `TMOV_C2UB` / `TSTORE_V2GM` / `TINSERT_V2L1` producer enum). The mix testcases listed in the table above are the proven end-to-end references; when adding a new mix handoff, mirror the flag pair from the closest testcase in that table.

## See also

- [SYNCALL.md](SYNCALL.md) — cross-core *barrier* (all participants rendezvous), as opposed to the per-handoff flags documented here.
- [conventions.md](conventions.md) — shared ISA conventions, including the Events and synchronization section.
- [a5-on-chip-storage-reuse.md](a5-on-chip-storage-reuse.md) — on-chip buffer reuse and the barrier insertion rules that complement the cross-core flags here.
