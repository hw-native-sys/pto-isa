# UB Budget and Tile Sizing (S1_TILE, row_slice shrink, empirical S1 >= 16384 recommendation)

The Flash Attention four-stage pipeline lives or dies on the vec UB budget. This file is the standalone reference for **how the working tile is sized, why the row_slice shrink is needed, and why the default suite recommends `S1_TILE=512` for `S1 >= 16384`**. The companion pipeline reference covers the timing knobs (`QK_PRELOAD`, `EXP_RING`); this file is purely about memory.

## 1. Why UB Sizing Is the Real Constraint

The four-stage pipeline keeps three fp32 working tiles co-resident in vec UB during steady state:

- the QK working tile (fp32 from cube, `[Vec_S0, S1_TILE]`),
- the P / softmax working tile (fp32 -> fp16, `[Vec_S0, S1_TILE]`),
- the PV / O working tile (fp32, `[VecGuRows, HEAD]`),

plus the `exp_max_ring` slots (`[EXP_RING]` x `[TILE_FACTOR]`) and the running `O` and `running_sum` per row_slice. The 192 KiB UB budget on A3 fits all of them only when the per-iteration working tile stays at **32 KiB**. The whole reason `row_slice` exists is to keep the per-iteration tile at 32 KiB even at `S1_TILE = 256`.

## 2. Row-Slice Shrink Formula

Vec walks each logical S1 tile in `TILE_FACTOR` slices along the M direction:

```text
HEAD        = 128
VEC_CORES   = 2
CUBE_S1     = 128
S1_TILE     = 256 or 512
TILE_FACTOR = S1_TILE // CUBE_S1                # 2 at 256, 4 at 512
Vec_S0      = S0 // VEC_CORES // TILE_FACTOR    # 32 at 256, 16 at 512
VecGuRows   = S0 // VEC_CORES                   # 64 (full subblock, GU/PV)
```

Per-iteration working-tile size (fp32, `[Vec_S0, S1_TILE]`):

```text
work_tile_bytes = Vec_S0 * S1_TILE * 4
                = 32 * 256 * 4 = 32 KiB         at S1_TILE = 256
                = 16 * 512 * 4 = 32 KiB         at S1_TILE = 512
```

The size is **constant in 32 KiB by construction** -- the row_slice count grows with `S1_TILE` so the per-slice tile shrinks. Without the row_slice loop, the working tile at `S1_TILE = 256` would be `64 * 256 * 4 = 64 KiB`, which does not co-exist with the other two working tiles in 192 KiB UB. This is the original reason the row_slice loop exists in the manual kernel and the reason the DSL kernel inherits it.

GU and PV do **not** row-split (they operate on the full subblock `VecGuRows = S0 / VEC_CORES = 64`). Only the QK -> softmax -> P chain walks per row_slice. The vec side therefore reads a P slot as `TILE_FACTOR` row_slices of `[Vec_S0, S1_TILE]`, but feeds it into PV consumption as one full `[VecGuRows, HEAD]` view.

## 3. The S1 >= 16384 Tile Recommendation

`run.py` picks `S1_TILE` per shape, not per kernel build. The default suite currently uses:

```python
def _default_case_s1_tile(seq_len):
    return 512 if seq_len >= 16384 else 256
```

This is a **scheduling-overhead heuristic**, not an architectural limit and not a proof of global optimality:

- `S1_TILE = 256` is the baseline empirical value and is the default builder value.
- For the current large-S1 default shapes (`case5..case8`, `S1 >= 16384`), `S1_TILE = 512` has measured better because it halves `num_tiles_s1` and reduces per-tile FIFO + sync overhead. The working tile stays 32 KiB because `TILE_FACTOR` doubles to 4 and `Vec_S0` halves to 16.
- For other shape mixes, `S1 >= 16384 -> 512` is a recommended starting point, not a mandatory rule. If a sweep shows 256 or another supported tile size is faster while satisfying the UB and FIFO constraints, document the shape-specific result.

This is the **flash-attn analogue of the GEMM 32 MiB safety-ratio cliff** only in the sense that it is an empirical threshold where one knob may jump because the cost model crosses a boundary. Treat it as a heuristic default, not a hard rule.

Porters with a different shape mix can sweep within the tile sizes implemented by the current library (`{256, 512}`). Supporting any other tile size is an implementation change: update the builder allow-list, re-derive section 2, and adapt the pipeline/barrier assumptions before treating the result as supported.

## 4. Why a Higher S1_TILE Is Not Free

Two costs scale with `S1_TILE`:

- **GM FIFO slot size.** `SLOT_SIZE_QK = S0 * S1_TILE * 4` and `SLOT_SIZE_P = S0 * S1_TILE * 2`. With `SLOT_NUM = 8` on A3 the GM bytes per AIC block grow linearly. This is GM, not UB, so it is generally fine -- but it costs DMA bandwidth proportionally.
- **`exp_max` ring footprint.** The ring is `EXP_RING * TILE_FACTOR` rescale-factor tiles, and `TILE_FACTOR = S1_TILE / CUBE_S1`. Going to `S1_TILE = 512` doubles `TILE_FACTOR` (2 -> 4), which doubles the per-ring-slot rescale storage on the vec side. The row_slice shrink rebalances the per-iteration working tile, but the ring itself grows.

The cumulative effect is why the default recommendation only switches to 512 at `S1 >= 16384`. At smaller S1 the per-tile FIFO overhead has not yet dominated in the current benchmark suite, so paying the extra ring footprint has not measured better.

## 5. Constraints the Porter Must Enforce

Validate on the host before building the kernel:

- `S1_TILE in {256, 512}` -- the tile-size allow-list implemented by the current library. Extending it requires code changes and shape-specific validation, not just a documentation change.
- `S1_TILE % CUBE_S1 == 0` -- `TILE_FACTOR` integer.
- `S0 % (VEC_CORES * TILE_FACTOR) == 0` -- `Vec_S0` integer (`32` at 256, `16` at 512 for `S0 = 128`, `VEC_CORES = 2`).
- `S1 % S1_TILE == 0` and `S1 >= S1_TILE * QK_PRELOAD` -- see the pipeline reference for why.
- `EXP_RING == QK_PRELOAD` -- the ring is sized by the pipeline knobs, but it lives in vec UB. Re-check section 1 after any change.

The host validates the first three today via `ValueError` in `fa_builder.py`; the last two are checked in `run.py::_num_tiles`.

## 6. The "FA UB Cliff": Symptom Catalogue

If the kernel builds but ptoas reports local-memory allocation failure or the vec stalls profile out, the working tile is too big:

- Symptom: `local memory allocation failed` from ptoas on a build that previously worked. Cause: someone raised `EXP_RING` without re-checking UB, or added a new vec working tile. Fix: revert the ring change, or split the new tile by row_slice.
- Symptom: `case5..case8` speedup drops from ~1.0x to ~0.7x with no other change. Possible cause: `S1_TILE` stayed at 256 (e.g. `FA_S1_TILE` env var pinned for a manual sweep) even though 512 is the current recommended default for those shapes. Fix: rerun with the default 512 recommendation and compare before changing the heuristic.
- Symptom: `case1..case4` speedup drops while `case5..case8` stays put. Possible cause: `S1_TILE` switched to 512 too aggressively below the empirical 16384 boundary. Fix: restore the default 256 baseline for small S1 unless a shape-specific sweep proves otherwise.

## 7. Anti-Patterns

- Removing the row_slice loop "because it adds a loop level". The loop is what keeps the per-iteration tile at 32 KiB. Without it the kernel does not fit in UB.
- Setting `EXP_RING > QK_PRELOAD` to "have headroom". It does not buy timing slack -- the timing invariant is set by `QK_PRELOAD` alone. It only burns UB.
- Sweeping `S1_TILE` above 512 without re-deriving section 2. The working-tile budget rebalance only works because `TILE_FACTOR` cleanly divides `S0 / VEC_CORES`. At `S1_TILE = 1024`, `Vec_S0` would be 8 and the row_slice count 8 -- the row_slice loop overhead starts to dominate, and PV / GU do not split, so the asymmetry grows. If you go there, treat it as a new schedule, not a tuning step.

## In-Tree Reference

The numbers in this file come from `kernels/python/flash_atten/kernels/fa_builder.py` constants (`HEAD`, `S0`, `VEC_CORES`, `CUBE_S1`, `S1_TILE`, `TILE_FACTOR`, `Vec_S0`, `VecGuRows`, `SLOT_SIZE_*`, `SLOT_NUM`) and the `_default_case_s1_tile` heuristic in `kernels/python/flash_atten/run.py`. Open those files only when you need to see the budget wired end-to-end; the formulas, cutover, and validation list in this file should be enough to size a new shape on its own.
