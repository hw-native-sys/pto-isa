# Tile instruction interfaces

`sparseFlashMLA` uses tile-based instruction interfaces for A5 BF16 attention
with an FP4 cache. The AIC/AIV scheduler binds tile views and manages pipeline
ownership; custom instructions implement vector arithmetic and transfers.

The instruction implementations are local to this example and target A5.
The Python oracle supplies the CPU numerical reference.

## Vector instructions

Each vector instruction has its own header under `custom/`, separating
arithmetic from buffer binding and pipeline scheduling. Shared metadata layout
constants and row-state tile types live in [common.hpp](../custom/common.hpp).
Public entries use `PTO_INST`, return
`RecordEvent` and accept optional wait events. Their implementation functions
use `PTO_INTERNAL`. All memory operands are tiles; scalar arguments describe
counts, capacity, initialization state or the attention scale.

| Entry | Tile contract |
| --- | --- |
| [`TGatherMetaData`](../custom/TGatherMetaData.hpp) | Replaces up to 256 loaded indices with validity masks, paired byte offsets, DMA controls and per-attention-tile valid counts in one metadata tile. |
| [`TDequantKey`](../custom/TDequantKey.hpp) | Reads packed FP4 records and pair controls, uses the BF16 scale scratch tile and writes one padded NZ key/value slice. |
| [`TSoftMaxRow<AllValid, FirstTile>`](../custom/TSoftMaxRow.hpp) | Reads score and validity tiles; updates row maxima, held bases, alpha and scratch; produces the scalar rescale decision in a UB tile. |
| [`TSoftMaxProb<AllValid, FirstTile>`](../custom/TSoftMaxProb.hpp) | Reads scores and row state, updates FP32 row sums and produces the padded NZ BF16 probability tile. |
| [`TGlobalUpdate<STEADY>`](../custom/TGlobalUpdate.hpp) | Adds a 256-column incoming PV fragment to the resident FP32 output and applies row alpha. With `initialize=true`, uses register zero instead of reading the resident output. |
| [`TGlobalUpdate<FINAL>`](../custom/TGlobalUpdate.hpp) | Adds the final PV fragment, applies normalization and writes BF16 output into the packed output view. With `initialize=true`, uses register zero for the previous output. |
| [`TOutputNormalize`](../custom/TOutputNormalize.hpp) | Produces packed FP32 LSE and row multipliers from maxima, sums and the attention sink. |
| [`custom::TEXPANDS<ScalarBits>`](../custom/TExpandS.hpp) | Fills the 32-lane FP32 row-state tile with a compile-time scalar encoded as IEEE-754 bits. |

Empty selections use `custom::TEXPANDS<0x7f800000u>(lse)` to fill their LSE
tile with positive infinity. This overload embeds the scalar directly in the
VF. Other scalar fills use standard `pto::TEXPANDS`.

The templates select first-tile and all-valid specializations at compile time.
Row and probability softmax use separate launches so the decision can be
published early and a global update can run between them.

The `initialize` argument is true for both fragments of a job's first consumed
PV update. When every conditional export is skipped, this is the final update.
The VF selects a specialized row loop once at entry. The addition to positive
zero is retained before scaling, preserving the arithmetic of an explicitly
cleared output.

Tile storage dimensions include the physical pitches. The KV and probability
tiles include NZ padding. A running-output fragment has 256 valid columns
inside a 512-FP32-column row; its packed BF16 view uses that same byte pitch.
Final output may alias the running output only in fragment order: the first
256-column fragment is read and packed before the second.

## Transfer and control instructions

[TPipe.hpp](../custom/TPipe.hpp) provides explicit-slot `C2V`
and `V2C` data policies for the standard PTO `TPUSH`/`TPOP` dispatch.
Rescale control uses native `TPipe`, a 1×1 `TileType::Ctrl` tile with a
`uint32_t` payload, and the request/reply aliases in
[flash_mla_common.h](../flash_mla_common.h).

The transfer entries use PTO instruction names in the `custom` namespace:

| Entry | Header and contract |
| --- | --- |
| `custom::MGATHER` | [MGather.hpp](../custom/MGather.hpp): gathers cache records using paired offset/control tiles. |
| `custom::TSTORE<PackedRows>` | [TStore.hpp](../custom/TStore.hpp): stores an explicitly pitched tile or a packed row buffer with one DMA command. |

The descriptor format and packed-row views are part of these local overloads'
contracts. Standard `TLOAD`, `TPUSH` and `TPOP` are called directly.

The scheduler selects physical slots and controls acquisition, publication and
release for each resource:

- QK and PV use independently allocated L0C/UB slots.
- A conditional PV flush returns each 128-column L0C view after both AIV
  destination copies, while GU receives a complete 256-column fragment.
- KV readiness and release operate on K64 regions in three L1 banks.
- The rescale mailbox is published after the local V-to-S handoff, consumed by
  AIC after its scalar wait, and acknowledged before the next publication.
- The final unit-flag export ends a whole-job accumulation lifetime.
  Intermediate conditional exports share that lifetime.

The explicit-slot policies give acquire, ready, release and drain ownership to
the scheduler, including every required barrier. Their transfer implementations
use the selected source/destination pitches and distinguish ordinary exports
from `STPhase::Final` exports.

These policies instantiate the shared PTO `TPUSH_IMPL`/`TPOP_IMPL` algorithm.
`TPOP` uses `TPipe`'s local tile binding primitives and performs each explicitly
selected ready wait. Consumer calls use `TPOP(pipe, tile, 0)` because the bound
local address selects the destination view. The literal zero supplies the API's
sub-block argument. Split V2C waits acquire both producer endpoints. Producer
calls use the standard `TPUSH(pipe, tile)` entry.

The policies specify destination pitches and insertion offsets, explicit
data buffer addresses, and final unit-flag accumulator copies. The scheduler owns
their lifetime and explicit drain. Standard `TLOAD` loads query parts, sinks
and the dynamic tail of the index batch.

`C2V<DestinationTile, STPhase::Final>` performs two single-destination
FP32 NZ-to-ND copies, with source row offsets 0 and 32. Each copy clears its
own complete 512-byte L0C units. The ordinary conditional-flush path uses
`TEXTRACT` with dual M splitting and a 256-column destination pitch for each
128-column source view. It uses four independent FIX-to-M leases on IDs 4–7.
A lease returns after both AIV destination copies for its view, and GU
readiness is published after the complete 256-column fragment is available.

`V2C<DestinationTile>` inserts a valid slice into the destination's physical
layout and selected row offset. This maps 33-row padded NZ source panels
while constructing the complete N-row KV or 64-row probability L1
view. The destination tile defines the L1 row pitch independently of the
padded source dimensions. Readiness and release operate at K64-region or
probability-slot scope.

Each AIV waits on local V-to-S event 4 before passing the softmax VF's UB
decision tile directly to `TPUSH`. The native V2C producer reads its first
32-bit word. The two requests occupy offsets 0 and 4 within each 32-byte
SSBUF mailbox. On AIC, `TPOP` uses `CtrlMergeMode::OR` to combine the two
predicates. `RescaleReply` broadcasts the result from offset 16. The native
pipes use scalar event 7 in their respective directions to signal readiness.

Both pipes disable backward synchronization. Each AIV consumes its previous
reply before publishing the next request, and AIC receives both requests
before overwriting the reply. This ordering protects slot reuse without
`TFREE` or destructor credits; event 8 remains available for `P_READY`.
Each helper constructs a depth-one pipe at the scheduler's selected mailbox,
so first-tile skips and cross-job state do not advance an independent cursor.

For paired gather, four VF-produced descriptors are read before issuing four
DMA commands. A single valid partner is duplicated; dequantization restores
reversed pairs to their logical order. Fully invalid pairs issue a zero-burst
descriptor. `custom::MGATHER` consumes the prepared descriptors directly and outputs
one contiguous 32-record byte tile. Each offset/control input holds 16 `uint64_t`
descriptors; these views select the current metadata part and slice before the call. The
cache operand is a byte `GlobalTensor`; descriptor offsets refer to validated
records in its underlying allocation. Each descriptor represents two
288-byte records, with four descriptor reads followed by four DMA commands.

`custom::TSTORE(destinationGlobal, sourceTile)` emits the single 2D output DMA
through PTO's `TStoreInstr` implementation.
The BF16 source tile has 1024 physical columns and 256 or 512 valid columns,
using the 2048-byte FP32-running-output row pitch. The GM row pitch is
512 BF16 elements (1024 bytes). `custom::TSTORE<true>` views the packed 32-float LSE
tile as either one 128-byte burst or 32 consecutive 4-byte source bursts;
its GM row pitch is `queryLength * sizeof(float)`. It transfers the packed LSE
buffer directly. Both `custom::TSTORE` and `custom::MGATHER` use `PTO_INST RecordEvent`
wrappers with optional wait events and `PTO_INTERNAL` implementations. Optional
events request handoffs; the default call issues the transfer directly.

## Scheduling ownership

The AIC/AIV files bind tile views, invoke instructions and implement the explicit
event protocol. Vector implementation details and raw transfer intrinsics live
in the custom instruction headers. See [design.md](design.md) for the complete
memory map, event IDs and conditional-rescale recurrence.
