# A5 On-Chip Storage Reuse and Barrier Conventions

> **Scope:** A5 (Ascend 950PR / 950DT) kernels that reuse L0A / L0B / L0C(ACC) / UB buffers across loop iterations or across pipeline stages. This page documents the producer/consumer pipe ownership of each buffer and where the `set_flag` / `wait_flag` (or `pipe_barrier`) must be inserted when a buffer is reused, so that the producer of the next iteration does not overwrite the buffer while the consumer of the previous iteration is still reading it.

## Why this page exists

PTO's pipeline is asynchronous: `TLOAD` (GM→L1/UB, MTE2), `TMOV` (L1→L0, MTE1), `TMATMUL` (M), `TMOV`/`TSTORE` (FIX), and vector ops (V) run on independent pipes. When a buffer (e.g. L0B) is reused for the next tile while the previous `TMATMUL` is still reading it, the refill overwrites the operand mid-multiply and the result is silently wrong — no hardware fault, just wrong values. The barrier that prevents this is a one-line `set_flag`/`wait_flag` pair, but its **insertion position and pipe pairing** are not obvious from the instruction signatures. The rule below lets you derive the barrier from the buffer being reused and the pipes that produce/consume it.

## Pipe ownership by buffer

| Buffer | Producer pipe | Consumer pipe | Typical instructions |
|--------|--------------|--------------|---------------------|
| L1 (Mat tile) | `PIPE_MTE2` (GM→L1 via `TLOAD`) | `PIPE_MTE1` (L1→L0A/L0B via `TMOV`) | `TLOAD` → `TMOV` |
| L0A | `PIPE_MTE1` (L1→L0A via `TMOV`) | `PIPE_M` (`TMATMUL` reads L0A) | `TMOV` → `TMATMUL` |
| L0B | `PIPE_MTE1` (L1→L0B via `TMOV`) | `PIPE_M` (`TMATMUL` reads L0B) | `TMOV` → `TMATMUL` |
| L0C / ACC | `PIPE_M` (`TMATMUL` writes L0C) | `PIPE_FIX` (`TMOV`/`TSTORE` drains L0C) | `TMATMUL` → `TMOV`/`TSTORE` |
| UB (Vec tile) | `PIPE_MTE2` (GM→UB via `TLOAD`), `PIPE_V` (vector op writes UB), `PIPE_FIX` (L0C→UB via `TMOV` acc2vec) | `PIPE_V` (vector op reads UB), `PIPE_MTE3` (`TSTORE` UB→GM) | `TLOAD`/`TMOV` → vector op → `TSTORE` |

## Barrier insertion rule for loop reuse

When a buffer is reused for iteration `i+1` while iteration `i`'s consumer is still running, insert a barrier from the **consumer pipe of iteration i** to the **producer pipe of iteration i+1**:

```
set_flag(<consumer_pipe_i>, <producer_pipe_i+1>, EVENT_IDn);
wait_flag(<consumer_pipe_i>, <producer_pipe_i+1>, EVENT_IDn);
```

The `set_flag` runs on the consumer pipe (it signals "I'm done reading"), the `wait_flag` runs on the producer pipe (it blocks "don't overwrite until the reader is done"). The two pipe ids are the ones from the table above for the buffer being reused.

### Worked example — L1 mat tile and L0 operand reuse across `TMATMUL` iterations

This is the canonical pattern in `RunTMATMUL_SPLIT_K` (`tests/npu/a5/src/st/testcase/tmatmul/tmatmul_kernel.cpp:286-287`):

```cpp
// iteration i: TLOAD loaded the L1 mat tiles (PIPE_MTE2), TMOV copied them to
// L0A/L0B (PIPE_MTE1), TMATMUL consumed L0A/L0B (PIPE_M).
// Wait for TMATMUL to drain before iteration i+1's TLOAD refills L1. The
// MTE2->MTE1 barrier at the top of the next iteration then orders the TMOV
// that overwrites L0A/L0B, so this single pair protects both L1 and L0.
set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
// now safe to TLOAD the next mat tile into L1
```

### Anti-pattern — missing M→MTE2 barrier (run5 from issue #4)

If iteration `i+1`'s refill of L0B is issued **without** the `PIPE_M`→`PIPE_MTE2` barrier above, and `TMATMUL` of iteration `i` has a long drain tail, the refill chain (`TLOAD` into L1, then `TMOV` into L0B) overwrites L0B while the previous multiply is still reading it — in run5, the first-iteration `TMOV` of the `w1` block overwrote L0B@`0x0` while Cube1's `TMATMUL` was reading the `w_gate` tile from the same address. The symptom is a result that looks like `x @ W1[:, :8]` (the first 8 columns of the next tile bleed into the current multiply) — a plausible-looking but wrong value, not a fault. The fix is the one-line barrier pair above.

### Anti-pattern — V-pipe op ordered before MTE2→V barrier (run13 from issue #4)

When a `TCOLEXPANDADD` (runs on `PIPE_V`) consumes a UB tile that was just loaded by `TLOAD` (`PIPE_MTE2`), the `set_flag(PIPE_MTE2, PIPE_V)` / `wait_flag(PIPE_MTE2, PIPE_V)` must execute **before** the `TCOLEXPANDADD`. If the V op is issued before the MTE2→V barrier, it reads the UB tile before the load has landed and produces a broadcast over stale/zero data (out diff 0.39~2.71, all 256/256 cases bad). The rule: **the barrier between a load pipe and its consumer pipe is issued at the consumer's entry, not after the load.**

## UB multi-tile layout — offset spacing and default base

When multiple tiles share the UB space via `TASSIGN`, two rules are **not** enforced by the current `TASSIGN` checks (`TASSIGN_zh.md` SA-0351~0354 only check single-tile capacity, alignment, and bounds):

1. **Inter-tile overlap is not checked.** Two `TASSIGN` calls with offsets that overlap produce silent write-through — the later write corrupts the earlier tile. The safe spacing is the tile's physical size (rows × `RowStride` × element size), rounded up to the 32-byte alignment. For a 1 KB tile, leave at least `0x400` (1024 B) between base offsets, **not** `0x200` — the `0x200` gap leaves the tail 512 B overlapping the next tile's head.
2. **A tile that is never `TASSIGN`-ed falls back to a default base address (`0x10`), which is not 32-byte aligned.** The first vector op on such a tile hangs silently on A5. Always `TASSIGN` every UB tile to an explicitly aligned base before first use, even tiles that "obviously" start at offset 0.

The convention is proven in `tests/npu/a5/src/st/testcase/tmatmul/tmatmul_kernel.cpp` (see the `TASSIGN` block at the top of the kernel, which spaces the mat tiles `0x0` / `0x10000` apart); the check set that does **not** cover these two rules is listed in [TASSIGN.md](TASSIGN.md) (SA-0351~0354, default base address).

## See also

- [a5-cross-core-sync-conventions.md](a5-cross-core-sync-conventions.md) — cross-core (AIC↔AIV) flag pairing; this page covers the **intra-core** pipe barriers.
- [conventions.md](conventions.md) — shared ISA conventions, including Events and synchronization.
- [TASSIGN.md](TASSIGN.md) — the `TASSIGN` instruction reference and its current single-tile check set.
