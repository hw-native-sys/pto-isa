# PIPE_V Barrier Removal Patterns (Generated C++ Post-Patch)

`ptoas` emits conservative `pipe_barrier(PIPE_V);` statements in `build_artifacts/fa.cpp` after every vec-side op that could theoretically race a later vec op. Most of them are redundant because some other event (MTE wait, direct tile-dependency edge) already provides the spacing. Removing them is worth between a few percent and double-digit percent on vec-bound shapes.

This file documents the **op-pattern** removal strategy used by `scripts/patch_vec_barriers.py` (the line-number strategy is legacy -- see section 5). Apply only the patterns listed here, and only with the safety checks listed alongside.

## 1. The Three Supported Patterns

| Pattern name (and aliases) | Default | Removes the barrier in | Safety mechanism |
| --- | --- | --- | --- |
| `gu` (alias: `trowexpandmul-tadd`) | **on** | `TROWEXPANDMUL -> pipe_barrier(PIPE_V) -> wait_flag(PIPE_MTE2, PIPE_V) -> TADD` (the `compute_gu` rescale-and-add chain). | The `wait_flag(PIPE_MTE2, PIPE_V)` between the two ops already pins the order. The PIPE_V barrier is pure redundancy. |
| `softmax-exp-sum` (alias: `texp-trowsum`) | off | `TEXP -> pipe_barrier(PIPE_V) -> TROWSUM` (the `compute_p` exp-then-rowsum sequence). | Removed only if `patch_vec_barriers.py` confirms the two ops share a **direct tile dependency** (the rowsum reads the exp's output tile). If the patch tool reports `softmax-exp-sum:direct-tile-dependency`, the barrier is kept. |
| `softmax-sum-add` (alias: `trowsum-tadd`) | off | `TROWSUM -> pipe_barrier(PIPE_V) -> TADD` (the streaming-sum running-max update). | Same direct-tile-dependency check as `softmax-exp-sum`; emits `softmax-sum-add:direct-tile-dependency` when kept. |

`gu` is on by default everywhere (`compile.sh` default `FA_REMOVE_VEC_BARRIER_PATTERNS=gu`, `run.py` propagates it). The other two are off by default because the direct-tile-dependency check is sufficient but not necessary -- ptoas may emit them in contexts the patch tool's tile-dependency walker does not cover, in which case removing them silently can race.

To enable extras explicitly:

```bash
python3 run.py --remove-vec-barrier-patterns gu,softmax-sum-add
FA_REMOVE_VEC_BARRIER_PATTERNS=gu,softmax-sum-add python3 run.py
```

To disable everything (for an A/B run vs the unpatched build):

```bash
python3 run.py --remove-vec-barrier-patterns none
```

## 2. How the Patch Tool Works

`scripts/patch_vec_barriers.py` runs **between** ptoas (which emits `build_artifacts/fa.cpp`) and bisheng (which builds `build_artifacts/fa.so`). It is line-level text patching, but the scanner is op-aware:

1. Parse the C++ line-by-line.
2. For each pattern, identify the operation triple (e.g. `TROWEXPANDMUL` line, candidate `pipe_barrier(PIPE_V);` line, `TADD` line) and the spacing op between them (`wait_flag(PIPE_MTE2, PIPE_V)` for `gu`).
3. For the two `softmax-*` patterns, additionally walk the tile-dependency graph between the producer op and the consumer op. If the producer's output tile is the consumer's input tile, the barrier is **kept** (the direct dependency already serializes the two ops on the PIPE_V scheduler, so removing the barrier here would be wrong as well as redundant).
4. Emit `build_artifacts/fa_patched.cpp` with only the matched lines removed; bisheng then compiles this file.

The "kept due to direct-tile-dependency" decisions are tallied in the patch tool's summary so you can see how many barriers each pattern actually removed.

## 3. Reading the Patch Output

The patch step prints something like:

```text
patch_vec_barriers: applied patterns={gu}
                    removed=N1 (gu=N1)
                    skipped due to direct-tile-dependency: softmax-exp-sum=0 softmax-sum-add=0
```

What the numbers mean:

- `removed=N1 (gu=N1)` -- N1 PIPE_V barriers were dropped because they matched the `gu` triple and had the MTE wait spacing. This is the stable, expected count.
- A nonzero `skipped due to direct-tile-dependency` count when you asked for `softmax-*` patterns means those patterns matched but were kept. That is the patch tool refusing to introduce a race -- treat the count as informational, not a warning.
- If `removed=0` for `gu` on a build that previously had `removed > 0`, the ptoas emit changed shape (e.g. inserted a different op between `TROWEXPANDMUL` and `TADD`). Investigate the C++ diff, do not change the pattern names.

## 4. When to Add a New Pattern (and What to Reject)

Adding a new pattern is a non-trivial change -- both the pattern matcher and the dependency walker need entries. A new pattern is justifiable only when:

- The barrier is **provably redundant** -- another barrier, event, or direct tile dependency already serializes the two ops on the PIPE_V scheduler.
- The pattern matches a **specific op triple**, not a "this barrier looks unused" heuristic. A pattern that says "remove every PIPE_V barrier after TADD" is too broad and will race in some emit context the author has not seen.
- The removal is measurable on at least one of `case1..case8` and is neutral on the others.

Things to reject:

- Patterns whose safety argument is "we tested it on case4 and it passed". Pattern safety needs an event-ordering argument, not just a passing benchmark.
- Patterns that depend on a specific ptoas version. ptoas emit churns; pattern matchers need to fail closed (no match -> no removal) when the surrounding C++ shape changes.
- Patterns that overlap. If two patterns can match the same barrier, the patch summary becomes ambiguous. Make pattern matchers mutually exclusive in their op triple.

## 5. Legacy: Line-Number Removal (`FA_REMOVE_VEC_BARRIERS`)

There is a parallel removal path that takes generated-C++ line numbers directly:

```bash
FA_REMOVE_VEC_BARRIERS=123,145,167 bash compile.sh
```

This existed before the op-pattern path and is kept only for one-off experiments where a porter wants to A/B a single suspected barrier. It is **not suitable for default use**:

- Every ptoas emit churn (new op, new pass, new layout) renumbers the file. The line list is stale on the next build.
- There is no safety check. The line is removed if it is a `pipe_barrier(PIPE_V);`, regardless of what surrounds it.

If you find yourself reaching for it, the answer is almost always to add a new op-pattern (see section 4) -- not to maintain a line list.

## 6. Operational Rules

- Keep `gu` on for any benchmark you report. Disabling it makes the numbers incomparable to the documented case1..case8 watermark.
- Treat the `softmax-*` patterns as opt-in experiments. If they help on a specific shape, document the shape and the pattern set together; do not flip them on by default.
- Re-run the case sweep after any ptoas / bisheng version bump. If the patch tool's `removed=` count changes materially, audit the matched lines before trusting the build.
- Never combine line-number and op-pattern removal in the same build. `compile.sh` accepts both flags but the patch tool then applies them in sequence and the diagnostics get confusing.

## In-Tree Reference

- `kernels/python/flash_atten/scripts/patch_vec_barriers.py` -- the matcher and dependency walker. The pattern-to-alias table (`PATTERN_ALIASES`) is the authoritative list of supported names.
- `kernels/python/flash_atten/compile.sh` -- where the patch tool is invoked (between ptoas and bisheng) and where `FA_REMOVE_VEC_BARRIER_PATTERNS` / `FA_REMOVE_VEC_BARRIERS` are read.
- `kernels/python/flash_atten/run.py` -- `--remove-vec-barrier-patterns` / `--remove-vec-barriers` CLI plumbing through to `compile.sh`.
