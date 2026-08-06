/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Grid TPUSH/TPOP CCE facade layer (A2/A3 backend) -- V8 IPC_SCB scoreboard route.
//
// Design_spec: Grid_TPUSH_TPOP_ISA...V8.md, section 3.3 (machine operands) and
// section 3.4 (layering / naming).  V8 freezes a strict TWO-name lowering with NO
// intermediate PTO wrapper: each Grid handshake op is a *CCE facade name* (the
// header declaration name) that forwards to a *CCE builtin name* (__builtin_cce_*),
// which the compiler lowers to one machine instruction.  The Grid TPUSH/TPOP
// sequence (GridTPush.hpp / GridTPop.hpp / GridTBroadcast.hpp) calls these facades
// DIRECTLY -- there is deliberately no sync_neighbor_scb / wait_local_spr /
// mov_local_spr / ScbOperand vocabulary between the PTO instruction and the CCE name.
//
// clang-format off
//   V8 machine instr | CCE facade (this file)      | CCE builtin (native)                    | facade
//   -----------------+------------------------------+------------------------------------------+---------------------------
//   COPY_UBUF_TO_NBR | copy_ubuf_to_neighbor_ubuf   | __builtin_cce_copy_ubuf_to_neighbor_ubuf | copy_ubuf_to_neighbor_ubuf
//   SYNC_HSCB/ST_HSCB| __sync_hscb                  | __builtin_cce___sync_hscb                | sync_hscb
//   WAIT_SPR         | __wait_ipc_scb               | __builtin_cce___wait_ipc_scb             | wait_ipc_scb
// clang-format on
//
// V8 revision vs V7: WAIT_SPR alone reads the local IPC_SCB and blocks -- read+block
// is ONE instruction (entry reads the unsigned count and compares: >= threshold
// proceeds, < threshold suspends the current pipe until the peer's SYNC_HSCB store
// raises it).  The V7 "先 get_ipc_scb (MOV_SPR2X) 非阻塞 peek、不足才 WAIT_SPR 阻塞"
// two-step is GONE: get_ipc_scb / MOV_SPR2X no longer appears in the handshake path,
// so the machine-instruction count collapses from "新增 1 + 复用 3" to "新增 1 + 复用 2".
// (MOV_SPR2X remains a hardware fact -- the ScalarUnit *can* read an IPC_SCB via
// MOV_SPR2X -- but the handshake never uses it; see V8 §3.2.0.)
//
// Why the facade names here drop the leading "__" (sync_hscb / wait_ipc_scb):
// cce_aicore_intrinsics.h *already* declares __sync_hscb and __wait_ast_scb as real
// builtin aliases (auto-included by the CCE frontend), so a same-named redefinition
// for the GM mock would collide.  Each facade below is therefore a thin, 1:1
// dispatcher that carries the CCE name and, under PTO_GRID_CCE_NATIVE, forwards to
// the real __builtin_cce_* (via the header alias); otherwise it emulates the
// builtin's SEMANTICS in GM.
//
// GM-mock rationale (default on A3): A3 has no cross-AICORE fabric, and the compiler
// exposes neither __builtin_cce_copy_ubuf_to_neighbor_ubuf nor a blocking
// __builtin_cce___wait_ipc_scb on IPC_SCB (__sync_hscb exists, but its machine op
// cannot address a geometric neighbor's IPC_SCB / L1).  So the DEFAULT build models
// each direction IPC_SCB slot as a volatile GM word and the neighbor L1 as a GM
// window: SYNC_HSCB -> cross-core GM store + cache maintenance; WAIT_SPR -> GM
// spin-poll (read+block); COPY_UBUF_TO_NBR -> UB->GM window copy.  Define
// PTO_GRID_CCE_NATIVE on silicon that provides the builtins to route each facade to
// the real __builtin_cce_*; call sites do not change.

#ifndef PTO_A2A3_GRID_CCE_INTRINSIC_HPP
#define PTO_A2A3_GRID_CCE_INTRINSIC_HPP

#include <cstdint>

#include <pto/common/arch_macro.hpp>
#include <pto/common/constants.hpp> // REPEAT_BYTE (reduce A3-mock Vec-combine repeat chunking)
#include <pto/common/type.hpp>      // AICORE + address-space qualifiers
#include <pto/comm/comm_types.hpp>  // pto::comm::ReduceOp (Sum/Max/Min) -- reduce combine

namespace pto {

// Forward declaration of the GridPipe group type.  It is defined in
// grid_intrinsic.hpp, which includes THIS header (so a real include here would
// be circular).  The group intrinsic below only needs the NAME -- GridGroup as a
// non-type template parameter -- so a forward declaration suffices.  (Its
// underlying-type definition is complete by the time any translation unit that
// instantiates these templates is compiled.)
enum class GridGroup : uint8_t;

// ---------------------------------------------------------------------------
// ScbKind: the G2 SYNC_HSCB `kind` machine operand (V8 §3.3 G2).  READY stores the
// producer's prod_idx into the downstream consumer's ready_scb_<dir>; FREE stores the
// consumer's cons_idx into the upstream producer's free_scb_<dir>.  The mock resolves
// the specific ready/free target into the `peerScb` pointer already (via the runtime
// RemoteScbPtr helper), so sync_hscb need not carry kind/dir/dist redundantly; this
// enum is kept for documentation and for the native lowering's operand encoding.
// ---------------------------------------------------------------------------
enum class ScbKind : uint8_t {
    READY = 0, // SYNC_HSCB(READY): prod_idx -> downstream ready_scb_<dir>
    FREE = 1,  // SYNC_HSCB(FREE):  cons_idx -> upstream   free_scb_<dir>
};

// ---------------------------------------------------------------------------
// (1) COPY_UBUF_TO_NBR  ->  copy_ubuf_to_neighbor_ubuf  ->  __builtin_cce_copy_ubuf_to_neighbor_ubuf
//
// Cross-core payload write: local UB -> the target core's L1/SRAM slot (V8 §3.3 G1,
// HW-DEP-0, the ONLY new machine instruction).  Not self-syncing; data-ready is
// announced by the following sync_hscb(READY) after the publish fence (V8 R5).
//
// §3.3 G1 operands (dir, dist, nbr_off, local_off, bytes) map to this facade as:
// `dstNeighborSlot` = the resolved neighbor L1 slot (native: the encoded neighbor L1
// address resolved from (dir, dist, nbr_off); mock: the GM window standing in for
// it); `src` = the local UB source tile (local_off folded into the UB pointer);
// `bytes` = payload size.
// ---------------------------------------------------------------------------
AICORE inline void copy_ubuf_to_neighbor_ubuf(__gm__ void* dstNeighborSlot, __ubuf__ void* src, uint32_t bytes)
{
#if defined(PTO_GRID_CCE_NATIVE)
    __builtin_cce_copy_ubuf_to_neighbor_ubuf(dstNeighborSlot, src, bytes, /*config=*/0);
#elif defined(__CPU_SIM)
    // CPU_SIM: __gm__/__ubuf__ collapse to ordinary host pointers and the CCE DMA
    // intrinsic (copy_ubuf_to_gm_align_b8) is not declared in this build, so a plain
    // byte copy stands in for the neighbor L1 write.  A loop (not memcpy) keeps the
    // CPU-sim source lint happy.
    auto* dstBytes = reinterpret_cast<uint8_t*>(dstNeighborSlot);
    auto* srcBytes = reinterpret_cast<uint8_t*>(src);
    for (uint32_t i = 0; i < bytes; ++i) {
        dstBytes[i] = srcBytes[i];
    }
#else
    // A3 mock: chunked UB -> GM-window copy stands in for the neighbor L1 write.
    constexpr uint32_t kChunkBytes = 256;
    auto* dstBytes = reinterpret_cast<__gm__ uint8_t*>(dstNeighborSlot);
    auto* srcBytes = reinterpret_cast<__ubuf__ uint8_t*>(src);
    uint32_t offset = 0;
    while (offset < bytes) {
        uint32_t chunk = (bytes - offset > kChunkBytes) ? kChunkBytes : (bytes - offset);
        copy_ubuf_to_gm_align_b8(dstBytes + offset, srcBytes + offset, 0, 1, chunk, 0, 0, 0, 0);
        offset += chunk;
    }
#endif
}

// ---------------------------------------------------------------------------
// (2) SYNC_HSCB / ST_HSCB  ->  __sync_hscb  ->  __builtin_cce___sync_hscb
//
// Store this core's new absolute count into the direction scoreboard of the peer
// (READY -> downstream neighbor's ready_scb = prod_idx; FREE -> upstream neighbor's
// free_scb = cons_idx).  Single external writer per scoreboard (SPSC), so the
// overwrite store of a monotone absolute count is safe (V8 §2.1).
//
// §3.3 G2 operands (kind, dir, dist, abs_count): `peerScb` is the RESOLVED peer
// scoreboard (native: the encoded peer IPC_SCB address resolved from (kind, dir,
// dist), HW-DEP-1; mock: the peer GM word via the runtime RemoteScbPtr helper) and
// `absCount` is the absolute count to store.  kind/dir/dist are folded into peerScb
// by the caller's address resolver, so the facade operates on the resolved target.
//
// Memory ordering: release -- earlier payload writes must be visible first (V8 R5 /
// publish).  V8 prefers SYNC_HSCB from an async pipe so it naturally orders after the
// payload DMA; both SYNC_HSCB and ST_HSCB are HSCB stores and V8 lets ST_HSCB reuse
// __sync_hscb.
// ---------------------------------------------------------------------------
AICORE inline void sync_hscb(__gm__ uint32_t* peerScb, uint32_t absCount)
{
#if defined(PTO_GRID_CCE_NATIVE)
    __sync_hscb(peerScb, absCount); // -> __builtin_cce___sync_hscb; exact operand encoding per ISA manual
#else
    if (peerScb != nullptr) {
        // A3 mock: cross-core GM store + cache maintenance.  AICORE caches are not
        // coherent between cores, so the pre/post dcci + dsb(DSB_DDR) make the store
        // observable by the peer's wait_ipc_scb spin (matches the canonical TNotify
        // Set pattern).  volatile prevents the compiler caching the write.
        volatile __gm__ uint32_t* ptr = reinterpret_cast<volatile __gm__ uint32_t*>(peerScb);
        __asm__ __volatile__("" ::: "memory");
        dcci(reinterpret_cast<__gm__ void*>(const_cast<__gm__ uint32_t*>(ptr)), cache_line_t::SINGLE_CACHE_LINE);
        __asm__ __volatile__("" ::: "memory");
        *ptr = absCount;
        __asm__ __volatile__("" ::: "memory");
        dcci(reinterpret_cast<__gm__ void*>(const_cast<__gm__ uint32_t*>(ptr)), cache_line_t::SINGLE_CACHE_LINE);
        __asm__ __volatile__("" ::: "memory");
        dsb(DSB_DDR);
    }
#endif
}

// ---------------------------------------------------------------------------
// (2b) SYNC_HSCB_ADD -- the INCREMENT form of the HSCB store.
//
// Same store, but the peer's IPC_SCB is bumped by `delta` instead of being
// overwritten with an absolute count.  This is what makes a MANY-WRITER
// scoreboard expressible at all: K consumers can each add 1 to one producer's
// free_scb and the producer simply waits for the total, without any of them
// knowing what the others contributed.  With an overwrite-only store that fan-in
// needs the writers to agree on a shared sequence -- which means encoding the
// participant set into the value, and that breaks the moment the participant set
// is not the whole group (e.g. a single-source broadcast).
//
// HW-DEP: an add-form HSCB is a hardware ask.  The increment MUST be atomic at
// the scoreboard: unlike the overwrite store, its writers really do overlap --
// the K receivers of one broadcast all credit the SAME producer free_scb at the
// same instant, and a read-modify-write would drop credits (observed: a free
// count of 9 where 10 adds were issued).  So the mock does NOT emulate it with a
// scalar RMW; it issues the smallest atomic-accumulate DMA the A2/A3 backend has
// (set_atomic_s32 + set_atomic_add + a 4-byte UB->GM burst), which the L2
// accumulates atomically.
//
// MOCK-ONLY, two further points the native lowering does not have:
//   * UB scratch.  The DMA needs a source, so the mock reserves ONE word at the
//     top of UB (kMockHscbAddScratchUb).  Hardware SYNC_HSCB-add takes its addend
//     as an operand and needs no buffer.
//   * Sub-block collapse.  One GridPipe cell == one block, but a mix-mode block
//     runs the vector program on BOTH of its AIV sub-blocks, which execute the
//     collective redundantly.  The absolute sync_hscb store is idempotent under
//     that duplication; an increment is not (it doubles every count).  The cell
//     therefore rings from sub-block 0 only.  On silicon a cell is one core and
//     the guard is a no-op.
// ---------------------------------------------------------------------------
#if !defined(PTO_GRID_CCE_NATIVE) && !defined(__CPU_SIM)
// Top 32 B of the 192 KiB A2/A3 UB, reserved for the addend of the emulated
// atomic increment (32 B aligned: a UB->GM burst needs an aligned source).  No
// PTO tile allocator reaches here (the grid demos top out far below), and it is
// written only inside sync_hscb_add.
constexpr uint64_t kMockHscbAddScratchUb = 0x2FF80;
#endif

AICORE inline void sync_hscb_add(__gm__ uint32_t* peerScb, uint32_t delta)
{
#if defined(PTO_GRID_CCE_NATIVE)
    __sync_hscb_add(peerScb, delta); // -> __builtin_cce___sync_hscb_add; encoding per ISA manual
#elif defined(__CPU_SIM)
    // CPU_SIM: one thread, no DMA intrinsics -- a plain read-modify-write IS atomic.
    if (peerScb != nullptr) {
        volatile uint32_t* ptr = reinterpret_cast<volatile uint32_t*>(peerScb);
        *ptr = *ptr + delta;
    }
#else
    if (peerScb != nullptr && get_subblockid() == 0) {
        __ubuf__ uint32_t* addend = reinterpret_cast<__ubuf__ uint32_t*>(kMockHscbAddScratchUb);
        *addend = delta;
#ifndef __PTO_AUTO__
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
#endif
        // Invalidate first: this core may hold a stale copy of the peer's line,
        // and a dirty write-back after the atomic would undo it.
        dcci(reinterpret_cast<__gm__ void*>(peerScb), cache_line_t::SINGLE_CACHE_LINE);
        set_atomic_s32();
        set_atomic_add();
        copy_ubuf_to_gm_align_b32(peerScb, addend, 0, 1, sizeof(uint32_t), 0, 0, 0, 0);
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
#endif
        set_atomic_none();
        dcci(reinterpret_cast<__gm__ void*>(peerScb), cache_line_t::SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
    }
#endif
}

// ===========================================================================
// (3) WAIT_SPR  ->  __wait_ipc_scb  ->  __builtin_cce___wait_ipc_scb
//
// V8: WAIT_SPR reads the local IPC_SCB and blocks in ONE instruction (entry reads
// the unsigned count and compares: >= threshold proceeds, < threshold suspends the
// current pipe until the peer's SYNC_HSCB store raises it).  There is deliberately
// NO get_ipc_scb / MOV_SPR2X peek step -- read+block is a single instruction.
// ===========================================================================

namespace grid_cce_detail {
// Shared GM spin-poll for the mock: return true once *localScb >= threshold.
// `maxSpins == 0` means block-forever (matches hardware WAIT_SPR); `maxSpins > 0`
// bounds the poll so a handshake deadlock fails the test instead of hanging (mock
// diagnostic only -- hardware WAIT_SPR has no spin bound).  A dcci each iteration
// invalidates the local cache line, since AICORE caches are not coherent between
// cores; without it the AICORE may cache a stale value and never see the peer store.
AICORE inline bool poll_ipc_scb_ge(__gm__ uint32_t* localScb, uint32_t threshold, uint32_t maxSpins)
{
    if (localScb == nullptr) {
        return true;
    }
    volatile __gm__ uint32_t* p = reinterpret_cast<volatile __gm__ uint32_t*>(localScb);
    uint32_t spin = 0;
    constexpr uint32_t kFenceInterval = 64;
    while (true) {
        __asm__ __volatile__("" ::: "memory");
        dcci(reinterpret_cast<__gm__ void*>(const_cast<__gm__ uint32_t*>(p)), cache_line_t::SINGLE_CACHE_LINE);
        __asm__ __volatile__("" ::: "memory");
        if (*p >= threshold) {
            return true;
        }
        if (maxSpins != 0 && spin >= maxSpins) {
            return false;
        }
        if ((++spin % kFenceInterval) == 0) {
            pipe_barrier(PIPE_ALL);
        }
    }
}
} // namespace grid_cce_detail

// Mock-only peek at a local scoreboard, used by the group collectives to TRAP an
// out-of-order publish (kFaultGroupOutOfOrder) after their WAIT_SPR returns.  It
// is a diagnostic, not part of the handshake: WAIT_SPR already did the compare.
// Native has MOV_SPR2X for this; the mock reads the GM word it stands in for.
AICORE inline uint32_t peek_ipc_scb(__gm__ uint32_t* localScb)
{
#if defined(PTO_GRID_CCE_NATIVE)
    (void)localScb;
    return 0; // native: skip the check rather than spend a MOV_SPR2X on it
#else
    if (localScb == nullptr) {
        return 0;
    }
    volatile __gm__ uint32_t* p = reinterpret_cast<volatile __gm__ uint32_t*>(localScb);
    dcci(reinterpret_cast<__gm__ void*>(const_cast<__gm__ uint32_t*>(p)), cache_line_t::SINGLE_CACHE_LINE);
    __asm__ __volatile__("" ::: "memory");
    return *p;
#endif
}

// V8 WAIT_SPR: block until the local IPC_SCB reaches `threshold`.  void / blocking,
// mirroring the real __wait_ast_scb -- this is the documented CCE intrinsic for G3.
//
// §3.3 G3 operands (local_scb_id, threshold): `slot` selects the native IPC_SCB slot
// (0..15) -- ready_scb_<dir> -> slot dirIdx, free_scb_<dir> -> slot
// kGridDirectionCount+dirIdx; `localScb` is the GM word the mock reads instead.  Native
// ignores localScb; the mock ignores slot.  Memory ordering: acquire.
AICORE inline void wait_ipc_scb(__gm__ uint32_t* localScb, uint32_t threshold, uint32_t slot)
{
#if defined(PTO_GRID_CCE_NATIVE)
    (void)localScb;
    __builtin_cce___wait_ipc_scb(slot, threshold); // blocking WAIT_SPR; encoding per ISA manual
#else
    (void)slot;
    (void)grid_cce_detail::poll_ipc_scb_ge(localScb, threshold, /*maxSpins=*/0); // block-forever, like HW
#endif
}

// Mock-simulation wrapper around wait_ipc_scb: identical semantics but with a
// spin-timeout (`maxSpins`) so the simulation can flag a handshake deadlock instead
// of hanging forever.  Native has no spin bound -- it delegates to wait_ipc_scb
// (blocks until satisfied) and returns true.  Returns false ONLY on mock spin-timeout.
// GridPipe's handshake sequences (GridTPush / GridTPop / GridTBroadcast) call THIS
// wrapper so a bug surfaces as a fault sentinel rather than a dead test; the
// documented hardware interface remains the void wait_ipc_scb above.
AICORE inline bool wait_ipc_scb_sim(__gm__ uint32_t* localScb, uint32_t threshold, uint32_t slot, uint32_t maxSpins)
{
#if defined(PTO_GRID_CCE_NATIVE)
    (void)maxSpins;
    wait_ipc_scb(localScb, threshold, slot);
    return true;
#else
    (void)slot;
    return grid_cce_detail::poll_ipc_scb_ge(localScb, threshold, maxSpins);
#endif
}

// ===========================================================================
// GridCollOp: the MOV_UBUF_GROUP `op` machine operand -- the NoC collective
// communication MODE (design: 2026-07-24-bcast-reduce-合并mov_ubuf_group方案.md
// §2.1).  From the issuing core's perspective a group broadcast (1->N identity
// copy, push) and a group reduce (N->1 element-wise combine, pull) are the SAME
// action -- move a UB tile to/from the resolved group arena -- differing only in
// the NoC mode; that mode is this runtime operand, NOT a different instruction.
// COPY = replicate-fan-out (UB -> arena, the former bcast_ubuf_to_group);
// SUM/MAX/MIN = combine-fan-in (arena -> UB, the former reduce_group_to_ubuf).
// The datapath direction is IMPLIED by op (COPY => out, combine => in).
// Values are deliberately comm::ReduceOp{Sum=0,Max=1,Min=2} + 1 so the Tier-2
// reduce caller maps with static_cast<GridCollOp>(uint32_t(commOp) + 1).
// ===========================================================================
enum class GridCollOp : uint8_t {
    COPY = 0, // broadcast: 1 source -> N members, identity replicate (push/out); eltype ignored
    SUM = 1,  // reduce: N members -> 1 sink, element-wise sum (pull/in)
    MAX = 2,
    MIN = 3,
};

// ===========================================================================
// (4) MOV_UBUF_GROUP  ->  mov_ubuf_group  ->  __builtin_cce_mov_ubuf_group
//
// Unified template-free group collective transfer.  The issuing core moves
// `bytes` between its local UB tile and the resolved per-member group arena,
// with the NoC collective mode selected by the RUNTIME `op` operand:
//   * op == COPY          : broadcast -- this core is the SOURCE; its UB tile is
//     replicated once into every member's slot (1->N fan-out, push).  Byte-level
//     pure copy (does NOT read element values), so eltype is IGNORED -- mirrors
//     copy_ubuf_to_neighbor_ubuf above.
//   * op == SUM/MAX/MIN   : reduce -- this core is the SINK; it reads every
//     member's contribution slot and folds them element-wise by op into UB
//     (N->1 fan-in, pull).  Element-wise combine MUST know the element width, so
//     `eltype` (1/2/4 bytes) selects the combine granularity.
// This collapses the former bcast_ubuf_to_group + reduce_group_to_ubuf<T,Op> pair
// (two machine instructions, two template facades) into ONE machine instruction
// and ONE template-free facade: the NoC mode + dtype are runtime operand fields,
// matching how cce_aicore_intrinsics.h encodes direction/op as operand enums
// (MemoryDirection_t / atomic_op_t).  NOT self-syncing: data-ready is still
// announced by the caller's sync_hscb(READY) after the publish fence.
//
// `groupSlotBase` is the RESOLVED per-member arena base (member 0's slot); member
// m's slot = groupSlotBase + m*memberStride (memberStride==0 means packed, stride
// == bytes).  Writable for COPY (the source writes each slot); read-only for
// SUM/MAX/MIN (the sink only reads contributions -- the caller const-casts).  The
// Tier-2 caller resolves member spacing from the group topology; ROW and COL
// members occupy consecutive grid ranks, so they are always a uniform-stride
// arena and this is the only payload path the group collectives take.
//
// `combineScratch` is the A3-mock combine scratch (one member's worth of UB).
// REQUIRED on the A3 mock for op != COPY (no on-transit combine: the in-core Vec
// vadd/vmax/vmin needs each member k>=1 in UB); IGNORED for COPY and on native
// (hardware collective) / __CPU_SIM (host loop reads members directly).
// ===========================================================================
AICORE inline void mov_ubuf_group(
    __ubuf__ void* ubTile, __gm__ void* groupSlotBase, uint32_t bytes, uint32_t memberCount, uint32_t memberStride,
    pto::GridCollOp op, uint32_t eltype, __ubuf__ void* combineScratch = nullptr, uint64_t groupDesc = 0)
{
    const uint32_t stride = (memberStride == 0) ? bytes : memberStride;
#if defined(PTO_GRID_CCE_NATIVE)
    (void)memberCount; // native: member count is encoded into the group descriptor.
    (void)combineScratch;
    __builtin_cce_mov_ubuf_group(ubTile, groupSlotBase, bytes, stride, static_cast<uint32_t>(op), eltype, groupDesc);
#elif defined(__CPU_SIM)
    // CPU_SIM: __gm__/__ubuf__ collapse to ordinary host pointers and the CCE DMA
    // intrinsic is not declared, so a typed host loop (no memcpy, lint-clean) stands
    // in for the collective.  op selects direction+combine at runtime.
    (void)groupDesc;
    (void)combineScratch;
    auto* ub = reinterpret_cast<uint8_t*>(ubTile);
    if (op == pto::GridCollOp::COPY) {
        // broadcast: replicate ub -> every member's slot.
        for (uint32_t k = 0; k < memberCount; ++k) {
            auto* d = reinterpret_cast<uint8_t*>(groupSlotBase) + static_cast<uint64_t>(k) * stride;
            for (uint32_t i = 0; i < bytes; ++i) {
                d[i] = ub[i];
            }
        }
    } else {
        // reduce: member 0 seeds ub, k>=1 folds element-wise by op (eltype picks width).
        const uint32_t n = bytes / eltype;
        for (uint32_t k = 0; k < memberCount; ++k) {
            const uint8_t* in = reinterpret_cast<const uint8_t*>(groupSlotBase) + static_cast<uint64_t>(k) * stride;
            if (k == 0) {
                for (uint32_t i = 0; i < bytes; ++i) {
                    ub[i] = in[i];
                }
                continue;
            }
            if (eltype == 4) {
                auto* o = reinterpret_cast<float*>(ub);
                const auto* iv = reinterpret_cast<const float*>(in);
                if (op == pto::GridCollOp::SUM) {
                    for (uint32_t i = 0; i < n; ++i) {
                        o[i] = o[i] + iv[i];
                    }
                } else if (op == pto::GridCollOp::MAX) {
                    for (uint32_t i = 0; i < n; ++i) {
                        o[i] = o[i] > iv[i] ? o[i] : iv[i];
                    }
                } else {
                    for (uint32_t i = 0; i < n; ++i) {
                        o[i] = o[i] < iv[i] ? o[i] : iv[i];
                    }
                }
            } else { // eltype == 2 (half)
                auto* o = reinterpret_cast<half*>(ub);
                const auto* iv = reinterpret_cast<const half*>(in);
                if (op == pto::GridCollOp::SUM) {
                    for (uint32_t i = 0; i < n; ++i) {
                        o[i] = o[i] + iv[i];
                    }
                } else if (op == pto::GridCollOp::MAX) {
                    for (uint32_t i = 0; i < n; ++i) {
                        o[i] = o[i] > iv[i] ? o[i] : iv[i];
                    }
                } else {
                    for (uint32_t i = 0; i < n; ++i) {
                        o[i] = o[i] < iv[i] ? o[i] : iv[i];
                    }
                }
            }
        }
    }
#else
    // A3 mock: chunked DMA + (for combine ops) an in-core Vec combine.  op selects
    // direction+combine at runtime; eltype dispatches the Vec op width.
    (void)groupDesc;
    constexpr uint32_t kChunkBytes = 256;
    if (op == pto::GridCollOp::COPY) {
        // broadcast: chunked UB -> GM-window copy of ubTile into every member's slot
        // (mirrors copy_ubuf_to_neighbor_ubuf's 256B-chunked A3-mock pump).
        auto* dstBase = reinterpret_cast<__gm__ uint8_t*>(groupSlotBase);
        auto* srcBytes = reinterpret_cast<__ubuf__ uint8_t*>(ubTile);
        for (uint32_t k = 0; k < memberCount; ++k) {
            __gm__ uint8_t* d = dstBase + static_cast<uint64_t>(k) * stride;
            for (uint32_t off = 0; off < bytes; off += kChunkBytes) {
                uint32_t chunk = (bytes - off > kChunkBytes) ? kChunkBytes : (bytes - off);
                copy_ubuf_to_gm_align_b8(d + off, srcBytes + off, 0, 1, chunk, 0, 0, 0, 0);
            }
        }
    } else {
        // reduce: per-member GM->UB pull (member 0 -> accumulator ubTile, k>=1 ->
        // scratch) + an in-core Vec combine (vadd/vmax/vmin), mirroring the existing
        // ReduceTiles (TADD/TMAX/TMIN) path.  One scratch buffer is reused per member.
        auto* accBytes = reinterpret_cast<__ubuf__ uint8_t*>(ubTile);
        auto* scrBytes = reinterpret_cast<__ubuf__ uint8_t*>(combineScratch);
        auto* baseBytes = reinterpret_cast<__gm__ const uint8_t*>(groupSlotBase);
        // member 0: contribution -> accumulator.
        for (uint32_t off = 0; off < bytes; off += kChunkBytes) {
            uint32_t chunk = (bytes - off > kChunkBytes) ? kChunkBytes : (bytes - off);
            copy_gm_to_ubuf_align_b8(accBytes + off, baseBytes + off, 0, 1, chunk, 0, 0, 0, 0);
        }
        const uint32_t elemsPerRepeat = static_cast<uint32_t>(REPEAT_BYTE) / eltype; // 64 (float) / 128 (half)
        const uint32_t totalRepeats = bytes / static_cast<uint32_t>(REPEAT_BYTE);
        for (uint32_t k = 1; k < memberCount; ++k) {
            __gm__ const uint8_t* mk = baseBytes + static_cast<uint64_t>(k) * stride;
            for (uint32_t off = 0; off < bytes; off += kChunkBytes) {
                uint32_t chunk = (bytes - off > kChunkBytes) ? kChunkBytes : (bytes - off);
                copy_gm_to_ubuf_align_b8(scrBytes + off, mk + off, 0, 1, chunk, 0, 0, 0, 0);
            }
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_ALL); // MTE2 (GM->UB) -> V (combine)
#endif
            // in-core element-wise combine acc OP= scr, chunked into <=255 repeats;
            // eltype dispatches the Vec op (float vadd/vmax/vmin vs half variants).
            for (uint32_t r = 0; r < totalRepeats;) {
                uint32_t chunk = totalRepeats - r;
                if (chunk > 255u) {
                    chunk = 255u;
                }
                if (eltype == 4) {
                    __ubuf__ float* a = reinterpret_cast<__ubuf__ float*>(accBytes) + r * elemsPerRepeat;
                    __ubuf__ float* s = reinterpret_cast<__ubuf__ float*>(scrBytes) + r * elemsPerRepeat;
                    if (op == pto::GridCollOp::SUM) {
                        vadd(a, a, s, static_cast<uint8_t>(chunk), 1, 1, 1, 8, 8, 8);
                    } else if (op == pto::GridCollOp::MAX) {
                        vmax(a, a, s, static_cast<uint8_t>(chunk), 1, 1, 1, 8, 8, 8);
                    } else {
                        vmin(a, a, s, static_cast<uint8_t>(chunk), 1, 1, 1, 8, 8, 8);
                    }
                } else { // eltype == 2 (half)
                    __ubuf__ half* a = reinterpret_cast<__ubuf__ half*>(accBytes) + r * elemsPerRepeat;
                    __ubuf__ half* s = reinterpret_cast<__ubuf__ half*>(scrBytes) + r * elemsPerRepeat;
                    if (op == pto::GridCollOp::SUM) {
                        vadd(a, a, s, static_cast<uint8_t>(chunk), 1, 1, 1, 8, 8, 8);
                    } else if (op == pto::GridCollOp::MAX) {
                        vmax(a, a, s, static_cast<uint8_t>(chunk), 1, 1, 1, 8, 8, 8);
                    } else {
                        vmin(a, a, s, static_cast<uint8_t>(chunk), 1, 1, 1, 8, 8, 8);
                    }
                }
                r += chunk;
            }
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_ALL); // V (combine) -> next MTE2 (GM->UB into scratch)
#endif
        }
    }
#endif
}

} // namespace pto

#endif // PTO_A2A3_GRID_CCE_INTRINSIC_HPP
