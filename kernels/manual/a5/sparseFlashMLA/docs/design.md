# FlashMLA BF16 numerical and pipeline design

This reference describes the kernel's numerical behavior and pipeline ownership.
See the [build/run guide](../README.md) and [instruction interfaces](instructions.md)
for usage and tile contracts.

## Conditional rescaling

Q, decoded K/V and P are BF16; QK, PV and row/output state accumulate in
FP32. The held row basis `b` is separate from the true running maximum `m`.
Scores include the attention scale. Each AIV changes all of its 32 row bases
when any row exceeds the strict threshold `ln(256) = 5.545177444`:

```text
m_new = max(m_old, max(score))
local_trigger = not first_tile and max_rows(m_new - b_old) > ln(256)
b_new = m_new if first_tile or local_trigger else b_old
alpha = exp(b_old - b_new)                         # zero for unseen rows
p = exp(score - b_new)                            # invalid tokens are zero
l_new = alpha * l_old + sum(p)                    # before BF16 P conversion
if either AIV triggers:
    saved_O = (saved_O + pending_L0C) * alpha     # before current PV
    pending_L0C = 0
pending_L0C += BF16(p) @ BF16(V)
# Final tile always exports pending_L0C, then normalizes saved_O + pending_L0C.
```

The threshold is the factor-256 slack described in
[FlashAttention-4, section 3.1.4](https://arxiv.org/html/2603.05451v1#S3.SS1.SSS4):
8 in log2 units becomes `ln(256)` for natural `exp`. The other AIV can retain
its basis and use alpha one when only one half triggers. The first tile
initializes row state and probabilities directly; decision exchange and
segment rescaling begin with subsequent tiles.

Probabilities can exceed one while the basis is held. Numerator, denominator,
LSE and sink weighting use the same held basis directly. BF16 P rounding and
FP32 accumulation order can differ from full rescaling. Add and multiply are
separate FP32 operations. Finite Q/cache data and intermediate scores are
required. The API supports BF16 Q, the 288-byte FP4 cache record, H64/H128
and D512.

## Pipeline and memory ownership

The scheduler assigns successive logical jobs to a physical AIC and its two AIVs.
Three KV banks allow gather/dequantization two stream positions ahead of
softmax/PV. Paired gathers use fixed 32-token slices, and QK/PV unit flags
are always enabled at both N64 and N128. AIC consumes the oldest PV before
submitting the newest QK:
`QK0 QK1 PV0 QK2 PV1 ...`. In the steady state it submits PV(t-2), exports
score(t-1) within that PV stage, then submits QK(t).

Two L0C QK slots feed one physical UB score tile. Two row-state slots hold
state for overlapping jobs. The single 64-KiB L1 Q allocation has independently
ready and released 32-KiB halves. Each half is released after its last actual Q
extraction. The next job's query loads are queued after current QK submission
and before the following PV work, waiting on their respective releases.

| Storage | Producer | Last consumer / release |
| --- | --- | --- |
| Two raw FP4 slots | Paired MTE2 gather | Dequant VF |
| Two padded BF16 NZ slots | Dequant VF | MTE3 insert into L1 |
| Three L1 KV slots | AIV MTE3, regional ready credit | Each K64 region's last PV MTE1 read |
| Two halves of one L1 Q allocation | AIC MTE2, per-half ready | Each half's last QK MTE1 read of the job |
| UB P | Softmax VF | MTE3 insert |
| Two L1 P slots | AIV MTE3, P ready | Last P MTE1 read |
| UB QK score slot | AIC FIX | Final softmax score read |
| Two incoming PV fragments | AIC FIX | Running-output update or final VF |
| Packed O in running-O storage | Two final VFs | Both O stores and LSE complete |
| Three 256-token metadata banks | MTE2 and index VF | Final softmax mask read of the batch |

At N128, L1 use is **480 KiB**: Q 64 KiB, KV 384 KiB and P 32 KiB.
Application UB use is **250.875 KiB per AIV**, including the metadata banks
and 384 bytes of anchor/decision state. Static assertions check physical
capacity. The launch is `<<<launchCount, nullptr, stream>>>`; its second
argument supplies the launch descriptor; UB addresses follow the static allocation map.

## KV regions and operands

Both AIVs contribute 32 tokens to each K64 region. At N128, AIV0
handles global slices 0/2 and AIV1 handles 1/3. Raw refill follows the local
ordinal's mapped slice; MTE3 inserts at the corresponding global token row.
One unused row in each NZ panel distributes stores across UB banks.

The first decoded slice follows `Gather0 -> DQ0 -> insert0 -> Gather1`.
Submission of insert0 precedes the second local gather when present; the
gather can overlap insert0 completion. Raw/NZ credits protect later slices.
Each AIV acquires regional L1 reuse before its first insert and publishes readiness after its final local insert.
AIC consumes both endpoints' readiness credits, then completes all four
128-feature QK depth slices for that K64 region before acquiring the next.
Regional reuse returns after the final V-to-L0B extraction, while Cube may
consume the L0B copy.

L0A has two 32-KiB slots. Independent operand credits use four 16-KiB L0B slots,
QK depth128 and PV K64/output128 MMADs. QK keeps four 16-KiB Q slices in
L0A. At N128, PV's 8-KiB P slices overwrite only the beginning of each left slot,
so Q at offsets 16/48 KiB survives into the next QK of the same job. PV
loads each P region once, uses it across four 128-column views, and releases
L1 P after first-view P extractions. At N64, P stays in left slot0
while V rotates through all four right slots.

Each job's first QK uses Q-first loading and refreshes every Q slice.
Later N128 tiles use K-first and restore only Q bytes overwritten by PV.
Later N64 tiles also use K-first but restore both Q prefixes, even though
PV overwrites only left slot0.
The guarded upper-feature preload loads features 64–127 of each L0A half
before that half's reuse wait, then loads protected features 0–63 after
the wait. It is eligible only at split-operand N128 job starts after a
preceding PV acquisition. The ordinary load path handles cold starts,
singleton transitions with no preceding PV, and N64.

Final QK unit flags protect the K64 score views: two 16-KiB views at N128
or one at N64. They share one full-tile lease; score export and softmax
consume the complete tile.

## Softmax, PV retention and output

The index VF prepares 128 indices per loop iteration, including both halves
of its interleave result. Pair descriptors encode validity, offsets, strides
and reversal. Branchless paired MTE2 gathers duplicate the valid token for
a singleton and select zero bursts for a fully masked pair. Dequantization
restores token order and masks invalid entries.

The row softmax VF produces row state and a uint32 rescale predicate; the
probability VF produces P and the denominator. When needed, the ordinary
nonfinal running-output update runs between these VFs and returns its
incoming-PV credit before P packing finishes. It uses the other alpha parity.
Score and metadata credits stay held through the probability VF's reads.

A V-to-S event makes the row decision available before P packing completes.
Native V2C control `TPUSH` publishes the Boolean to a two-slot SSBUF mailbox;
AIC `TPOP` ORs the two AIV values and C2V `TPUSH` broadcasts the result.
Forward synchronization uses scalar event 7 in both directions. Each AIV
consumes the reply before its next request, protecting slot reuse with
backward synchronization disabled. AIC queues initial PV loads
before reading the decision. The schedule gathers the next raw
slices before publication; deferred metadata preparation publishes first.
Prefill acquires raw/NZ credits separately from L1 insertion ownership.

PV accumulates in two L0C fragments across skipped exports. Local M-to-FIX
event0 captures each nonfinal PV's completion before the next QK submission;
the FIX wait is queued even when the next export is skipped. On a trigger,
old-PV FIXP exports precede the next score copy and overlap independent QK
arithmetic. Per-view credits prevent overwriting those accumulators. A
last-tile trigger causes both an old-segment export and a final export.

Each 256-column PV fragment has two 128-column views at a common 64-row NZ
pitch. Conditional dual-destination FIXP copies return each view's credit
after transferring both AIVs, then publish full-fragment PV readiness.
Final exports span 256 columns and return both view credits together.
PV unit flags span the logical job: its first MMAD acquires, last MMAD
publishes and final FIXP releases. Intermediate segment flushes use explicit
subview credits within the same unit-flag phases.

Nonempty jobs initialize running O in their first consumed PV update, using
register zero instead of loading the previous contents. A per-job state bit
selects this specialization for both fragments, including the final update
when all conditional exports are skipped. The state resets when the output
stage acquires storage from the preceding job; softmax lookahead does not
reset it. Empty jobs retain the explicit zero fill.

Intermediate update adds the segment to saved O, then rescales it. The final
VF adds the final segment, applies the normalization multiplier and casts
directly to BF16. Each final VF packs 256 columns into running-O storage;
its 16-KiB DMA uses V-to-MTE3 ID4 or ID5. Packing writes behind reads and the
first half is disjoint from the second VF's input/output. Stores use a
2,048-byte UB row pitch and 1,024-byte GM row pitch. Backward output-reuse
ID4 returns only after both halves and LSE complete. Empty jobs store the
whole zero output.

For nonempty jobs, the final-fragment V-to-MTE3 ID5 handoff also orders the
preceding normalization and packed-LSE writes before their stores. Empty
jobs use a local V-to-MTE3 ID0 fence to order their initialization before
storing zero O and positive-infinity LSE.

LSE uses 32 contiguous FP32 values per AIV in its reserved UB storage.
Q1 uses one 128-byte GM copy; longer queries use a four-byte source pitch
and `queryLength * sizeof(float)` destination pitch. O and LSE share the
whole-output completion credit.

## Metadata and flag map

Each 3.25-KiB metadata bank holds 256 indices reused as validity flags,
128 64-bit pair offsets, 128 controls and a valid count per tile. A batch
covers two N128 or four N64 tiles and stops at cache/job boundaries.
Startup prepares the current batch, queues the initial gather/dequant/insert
and prefetches metadata for the next two stream positions. Same-batch parts
reuse the batch's index DMA/VF results; empty jobs use no metadata. Initial
loop iterations consume the preissued refill.

Metadata slots rotate 0/1/2 per batch; empty jobs consume none. Partial-batch
reuse is credit protected. Metadata selection is independent of the
modulo-3 KV and modulo-2 score/P phases; the shared stream phase wraps at six.
The first tile consumes batch readiness, and the last tile returns reuse
after its final softmax mask read. Initialization/drain cover every bank,
region and operand credit once per physical core; readiness starts empty.

| Local pipe pair / resource | Event IDs |
| --- | --- |
| V ↔ MTE2 metadata | 1–3 |
| V ↔ MTE2 raw gathers | 4–5 |
| V ↔ MTE2 job row state | 6–7 |
| V → S metadata descriptors | 1–3 |
| V → S rescale predicate | 4 |
| V ↔ MTE3 P source | 1 |
| V ↔ MTE3 decoded KV sources | 2–3 |
| V → MTE3 final output halves ready | 4–5 |
| MTE3 → V whole-output reuse | 4 |
| M → MTE1 left reuse | 2–3 |
| MTE1 → M left readiness for PV | 2–3 |
| MTE1 ↔ M right operands | 4–7 |
| MTE1 → MTE2 Q reuse / MTE2 → MTE1 Q ready | 6–7 |
| M → FIX QK slot relay / FIX → M QK slot reuse | 2–3 |
| FIX → M PV subview reuse | 4–7 |
| M → FIX final PV fragment relay | 4–5 |
| M → FIX PV completion fence | 0 |

QK right-ready follows both Q and K extraction on MTE1 and orders both
operands. Query readiness is consumed at each half's first Q extraction;
query reuse follows its last extraction. IDs on different pipe pairs are
independent; forward output readiness and backward output reuse are distinct.
ID0 is reserved for local fence handoffs. Metadata descriptors and rescale
predicates share V-to-S, so their ID ranges must remain disjoint even though
the same metadata IDs also serve V-to-MTE2 ownership.

| Intra-block resource | IDs / direction |
| --- | --- |
| KV ready | V → C: region0 0–2, region1 3–5 |
| KV release | C → V: region0 3–5, region1 0–2 |
| Score ready / release | C → V / V → C: 6, with one score lease |
| Decision / acknowledgement | V → C / C → V: 7 |
| P ready | V → C: 8–9 |
| PV ready | C → V: 10–11 |
| PV release | V → C: 12–13 |
| P release | C → V: 14–15 |

Both AIV endpoints participate. The table lists AIV0's base IDs; AIV1 uses
the same directional map at ID + 16. N64 uses only KV region0. Reusing a
number in opposite directions identifies separate tokens: each readiness or
reuse credit belongs to its direction, endpoint and physical resource.

## Profiling stamps

[flash_mla_profile.h](../flash_mla_profile.h) defines job, query-load,
dequant, QK, softmax, PV, GU and output stages. Logical ordinals are encoded
modulo 50; trace order disambiguates wraps. Stamps add same-pipe barriers and
must be disabled for primary timing comparisons.
Stamps require an NPU build; simulator builds reject them at compile time
because they can deadlock the simulator.

- Dequant uses global slice IDs: 0/2 for AIV0 and 1/3 for AIV1 at N128.
- QK MTE1 part is `region * 4 + depth_slice`; whole-tile Cube/FIX endpoints
  use part7 at N128 or part3 at N64.
- `Q_LOAD` stamps follow both query-half transfers. Prequeue can place the
  stamp before its job-start marker; the encoded job ID identifies ownership.
- PV MTE1 part is `(column / PV_MMAD_N) * (N / 64) + token_region`, parts
  0–7 at N128 or 0–3 at N64. Cube/FIXP use parts0/1 for 256-column
  fragments.
- Output MTE3 parts0/1 identify nonempty O halves and part2 identifies LSE.
  Empty jobs use part1 for the whole zero-output store and part2 for LSE.
