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
//   COPY_L1_TO_NBR   | copy_l1_to_neighbor_l1       | __builtin_cce_copy_ubuf_to_neighbor_ubuf | copy_l1_to_neighbor_l1
//   SYNC_HSCB/ST_HSCB| __sync_hscb                  | __builtin_cce___sync_hscb                | sync_hscb
//   ATOM_ADD_HSCB    | __atom_add_hscb              | __builtin_cce___atom_add_hscb            | atom_add_hscb
//   WAIT_SPR         | __wait_ipc_scb               | __builtin_cce___wait_ipc_scb             | wait_ipc_scb
//   MOV_SPR2X        | __mov_ipc_scb_to_l1          | __builtin_cce___mov_ipc_scb_to_l1        | mov_ipc_scb_to_l1
//   MOVX2SPR         | __mov_x_to_ipc_scb           | __builtin_cce___mov_x_to_ipc_scb         | mov_x_to_ipc_scb
//   MOVX2GPR         | __mov_x_to_gpr               | __builtin_cce___mov_x_to_gpr             | mov_x_to_gpr
// clang-format on
//
// V8 revision vs V7: WAIT_SPR alone reads the local IPC_SCB and blocks -- read+block
// is ONE instruction (entry reads the unsigned count and compares: >= threshold
// proceeds, < threshold suspends the current pipe until the peer's SYNC_HSCB store
// raises it).  The V7 "先 get_ipc_scb (MOV_SPR2X) 非阻塞 peek、不足才 WAIT_SPR 阻塞"
// two-step is GONE from every steady-state TPUSH/TPOP wait: get_ipc_scb /
// MOV_SPR2X is not a pre-check before WAIT_SPR, so that hot path still collapses
// from "新增 1 + 复用 3" to "新增 1 + 复用 2".  MOV_SPR2X remains available on the
// infrequent time-division bind path, where the consumer must snapshot ready_scb
// and close_scb to choose a channel and relay an absolute baseline.
//
// The MOV-class facades serve GridPipe's dynamic close/relay bind.  Producer and
// consumer channel indices are independent: the response supplies the consumer's
// ready baseline and receive-channel index, while the consumer writes its cons_idx
// directly into the producer channel's free_scb and commits an explicit completion
// word last.
//
//   MOV_SPR2X (mov_ipc_scb_to_l1) -- SPR -> memory.  Snapshots ready/close during
//     dynamic consumer-channel selection.  A scoreboard cannot be read into a GPR
//     directly.
//   MOVX2SPR (mov_x_to_ipc_scb)   -- memory -> SPR.  Retained as a machine facade,
//     but the current GridPipe protocol does not use it: free_scb has an external
//     consumer writer even when its local producer channel is rebound.
//   MOVX2GPR (mov_x_to_gpr)       -- memory -> GPR.  Polls the request/completion
//     commits and installs the returned ready baseline into prod_idx.  V8 spells the
//     L1-source case MOV_L12X; the value must reach a register before TPUSH can
//     derive a ring slot or free threshold.
//
// Note the asymmetry -- one lands in memory, one in the scoreboard file, one in a
// GPR.  They are different machine instructions and must not share a facade.
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
// spin-poll (read+block); COPY_L1_TO_NBR -> local producer-window read followed
// by a remote receive-window write.  Define
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

// ---------------------------------------------------------------------------
// GridBlockRect: the MOV_UBUF_GROUP `group` machine operand -- WHICH CORES take
// part in a group collective, named the way this mesh names cores everywhere
// else: by BLOCK ID.  A group is the INCLUSIVE sub-rectangle of the mesh whose
// opposite corners are the block ids `topLeft` and `botRight`; its members are
// every cell inside it, ranked row-major (which is ascending block id).
// `meshCols` is the mesh width -- what turns a block id back into a coordinate
// (blockId = row*meshCols + col).  Silicon knows the mesh it is wired into; the
// mock has to be told, so the width travels in the descriptor.
//
// This REPLACES the (memberCount + rank-strided arena) pair the instruction used
// to take.  Two corners describe the old ROW / COL groups exactly (a one-row /
// one-column rectangle) and, unlike a rank stride, they also describe a
// MULTI-ROW rectangle: the jump in block id at a row boundary is the
// instruction's own arithmetic now, not a geometry the caller has to fold into a
// single stride that cannot express it.
//
// It is defined HERE, in the machine layer, because the member set is a machine
// operand; grid_intrinsic.hpp (which includes this header) adds the Tier-2
// helpers that build one out of mesh topology (GridBlockRectOfGroup et al).
// ---------------------------------------------------------------------------
struct GridBlockRect {
    uint32_t topLeft = 0;  // block id of the rectangle's top-left cell
    uint32_t botRight = 0; // block id of the rectangle's bottom-right cell (INCLUSIVE)
    uint32_t meshCols = 0; // mesh width; 0 => empty group, and the instruction is a no-op
};

AICORE constexpr uint32_t GridBlockRectRowSpan(const GridBlockRect& g)
{
    return (g.meshCols == 0 || g.botRight < g.topLeft) ? 0u : (g.botRight / g.meshCols - g.topLeft / g.meshCols + 1u);
}

AICORE constexpr uint32_t GridBlockRectColSpan(const GridBlockRect& g)
{
    if (g.meshCols == 0 || g.botRight < g.topLeft) {
        return 0u;
    }
    const uint32_t c0 = g.topLeft % g.meshCols;
    const uint32_t c1 = g.botRight % g.meshCols;
    return (c1 < c0) ? 0u : (c1 - c0 + 1u); // corners crossed in the column axis => empty
}

AICORE constexpr uint32_t GridBlockRectSize(const GridBlockRect& g)
{
    return GridBlockRectRowSpan(g) * GridBlockRectColSpan(g);
}

// Block id of the member whose rank-in-group is `rank`, row-major inside the
// rectangle.  Rank order is ascending block id, which is what keeps a row/column
// fan-in folding in the same order the directional relay accumulates in (so the
// two lowerings of a reduce stay bit-identical).
AICORE constexpr uint32_t GridBlockRectMember(const GridBlockRect& g, uint32_t rank)
{
    const uint32_t cols = GridBlockRectColSpan(g);
    return (cols == 0) ? g.topLeft : (g.topLeft + (rank / cols) * g.meshCols + (rank % cols));
}

// Is `blockId` one of the group's members?  A block id between the two corners
// is NOT enough -- a multi-row rectangle skips the cells outside its columns.
AICORE constexpr bool GridBlockRectContains(const GridBlockRect& g, uint32_t blockId)
{
    return GridBlockRectSize(g) != 0 && blockId >= g.topLeft && blockId <= g.botRight &&
           (blockId % g.meshCols) >= (g.topLeft % g.meshCols) && (blockId % g.meshCols) <= (g.botRight % g.meshCols);
}

// Pack the member set + the caller's own block id into the 64-bit `group_desc`
// machine operand the native builtin takes: 16 bits each of
// [topLeft | botRight | meshCols | root], where root is the issuing core -- the
// SOURCE of a COPY, the SINK of a combine.
AICORE constexpr uint64_t GridPackGroupDesc(const GridBlockRect& g, uint32_t selfBlockId)
{
    return (static_cast<uint64_t>(g.topLeft & 0xFFFFu)) | (static_cast<uint64_t>(g.botRight & 0xFFFFu) << 16) |
           (static_cast<uint64_t>(g.meshCols & 0xFFFFu) << 32) | (static_cast<uint64_t>(selfBlockId & 0xFFFFu) << 48);
}

// ---------------------------------------------------------------------------
// ScbKind: the G2 SYNC_HSCB `kind` machine operand (V8 §3.3 G2).  READY stores the
// producer's prod_idx into the consumer's ready_scb for that channel; FREE stores the
// consumer's cons_idx into the producer's independently negotiated free_scb channel;
// CLOSE stores the final prod_idx into the consumer's close_scb.  The mock
// resolves the specific ready/free target into the `peerScb` pointer already (via the
// runtime RemoteScbPtr helper), so sync_hscb need not carry the kind redundantly; this
// enum is kept for documentation and for the native lowering's operand encoding.
// ---------------------------------------------------------------------------
enum class ScbKind : uint8_t {
    READY = 0, // SYNC_HSCB(READY): prod_idx -> the consumer's ready_scb[chan]
    FREE = 1,  // SYNC_HSCB(FREE):  cons_idx -> the producer's free_scb[chan]
    CLOSE = 2, // SYNC_HSCB(CLOSE): final prod_idx -> the consumer's close_scb[chan]
};

// ---------------------------------------------------------------------------
// (1) COPY_L1_TO_NBR  ->  copy_l1_to_neighbor_l1
//                      ->  __builtin_cce_copy_ubuf_to_neighbor_ubuf (legacy compiler spelling)
//
// Cross-core payload write: a dedicated local producer L1 slot -> the target
// core's receive-side L1/SRAM slot (V8 §3.3 G1, HW-DEP-0, the ONLY new
// machine instruction).  Real WSE has one unified L1 SRAM; there is no separate
// vector UB address space that may be used as the source mapping.  Not
// self-syncing; data-ready is
// announced by the following sync_hscb(READY) after the publish fence (V8 R5).
//
// §3.3 G1 operands (dir, dist, nbr_off, local_off, bytes) map to this facade as:
// `dstNeighborSlot` = the resolved neighbor L1 slot (native: the encoded neighbor L1
// address resolved from (dir, dist, nbr_off); mock: the GM window standing in for
// it); `srcProducerSlot` = the isolated local L1 producer slot (mock: a disjoint
// range in this core's GM window); `transferScratch` is only the A3 mock's UB DMA
// pump and is not an architectural source address; `bytes` = payload size.
// ---------------------------------------------------------------------------
AICORE inline void copy_l1_to_neighbor_l1(
    __gm__ void* dstNeighborSlot, __gm__ const void* srcProducerSlot, __ubuf__ void* transferScratch, uint32_t bytes)
{
#if defined(PTO_GRID_CCE_NATIVE)
    (void)transferScratch;
    // The currently exposed builtin retains the historical "ubuf" spelling.
    // On WSE that qualifier names the same physical unified L1 SRAM; the pointer
    // value is the producer staging address, never the caller's tile address.
    auto* srcUnifiedL1 = reinterpret_cast<__ubuf__ void*>(reinterpret_cast<uint64_t>(srcProducerSlot));
    __builtin_cce_copy_ubuf_to_neighbor_ubuf(dstNeighborSlot, srcUnifiedL1, bytes, /*config=*/0);
#elif defined(__CPU_SIM)
    // CPU_SIM: address-space qualifiers collapse to host pointers.  Read from the
    // explicit producer range so the model catches source/receive-ring aliasing.
    (void)transferScratch;
    auto* dstBytes = reinterpret_cast<uint8_t*>(dstNeighborSlot);
    const auto* srcBytes = reinterpret_cast<const uint8_t*>(srcProducerSlot);
    for (uint32_t i = 0; i < bytes; ++i) {
        dstBytes[i] = srcBytes[i];
    }
#else
    // A3 mock: both L1 ranges are represented by GM.  Pump local producer GM ->
    // scratch UB -> peer GM in chunks.  The scratch pointer is the original tile
    // storage; each load restores the same staged bytes before the outbound DMA,
    // so it remains unchanged when this synchronous facade returns.
    constexpr uint32_t kChunkBytes = 256;
    auto* dstBytes = reinterpret_cast<__gm__ uint8_t*>(dstNeighborSlot);
    const auto* srcBytes = reinterpret_cast<__gm__ const uint8_t*>(srcProducerSlot);
    auto* scratchBytes = reinterpret_cast<__ubuf__ uint8_t*>(transferScratch);
    uint32_t offset = 0;
    while (offset < bytes) {
        uint32_t chunk = (bytes - offset > kChunkBytes) ? kChunkBytes : (bytes - offset);
        copy_gm_to_ubuf_align_b8(scratchBytes + offset, srcBytes + offset, 0, 1, chunk, 0, 0, 0, 0);
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);
        copy_ubuf_to_gm_align_b8(dstBytes + offset, scratchBytes + offset, 0, 1, chunk, 0, 0, 0, 0);
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);
        offset += chunk;
    }
#endif
}

// ---------------------------------------------------------------------------
// (2) SYNC_HSCB / ST_HSCB  ->  __sync_hscb  ->  __builtin_cce___sync_hscb
//
// Store this core's new absolute count into a resolved peer word (READY ->
// downstream ready_scb = prod_idx; FREE -> upstream free_scb = cons_idx; CLOSE ->
// downstream close_scb = final prod_idx).  The bind control path also uses the
// same resolved-store mechanism for its request/response L1 words.  Each live word
// has one external writer in the time-division protocol, so overwrite stores are
// safe (V8 §2.1).
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
#elif defined(__CPU_SIM)
    if (peerScb != nullptr) {
        __atomic_store_n(reinterpret_cast<uint32_t*>(peerScb), absCount, __ATOMIC_RELEASE);
    }
#else
    if (peerScb != nullptr) {
        // A3 mock: cross-core GM store + cache maintenance.  AICORE caches are not
        // coherent between cores, so the pre/post dcci + dsb(DSB_DDR) make the store
        // observable by the peer's wait_ipc_scb spin (matches the canonical TNotify
        // Set pattern).  volatile prevents the compiler caching the write.
        volatile __gm__ uint32_t* ptr = reinterpret_cast<volatile __gm__ uint32_t*>(peerScb);
        __asm__ __volatile__("" ::: "memory");
        dcci(reinterpret_cast<__gm__ void*>(const_cast<__gm__ uint32_t*>(ptr)), SINGLE_CACHE_LINE);
        __asm__ __volatile__("" ::: "memory");
        *ptr = absCount;
        __asm__ __volatile__("" ::: "memory");
        dcci(reinterpret_cast<__gm__ void*>(const_cast<__gm__ uint32_t*>(ptr)), SINGLE_CACHE_LINE);
        __asm__ __volatile__("" ::: "memory");
        dsb(DSB_DDR);
    }
#endif
}

// ---------------------------------------------------------------------------
// (2b) ATOM_ADD_HSCB -- atomically add `delta` to a resolved peer IPC_SCB.
//
// TPUSH and the channelised collectives have exactly one active forward writer
// per READY/CLOSE scoreboard, so those dependencies publish absolute counts with
// sync_hscb.  A TBROADCAST ownership handoff, however, collects reverse FREE
// credit from several receivers into the next source's one scoreboard.  That
// reverse fan-in edge would lose credits with overwrite stores and is the reason
// this atomic-add form remains in the protocol.  Group TREDUCE uses the same
// reverse primitive for consistency even though its sink is a single writer.
//
// HW-DEP: CANN exposes __atom_add_hscb, but WSE silicon/compiler support for
// targeting a peer's WAIT_SPR-visible IPC_SCB (including wakeup and release
// ordering) must be confirmed against the final ISA.  The A2/A3 mock below uses
// the smallest atomic-accumulate DMA available, so it validates the protocol but
// cannot prove that native routing property.
//
// The A3 DMA needs one aligned UB word as its addend.  A mix-mode block executes
// the vector program on two AIV subblocks, while one GridPipe cell is logical
// block-wide; only subblock 0 may issue a non-idempotent increment.  Grid kernels
// enforce the same active-subblock contract at entry, and the guard here prevents
// an accidental doubled count if a caller forgets it.
// ---------------------------------------------------------------------------
#if !defined(PTO_GRID_CCE_NATIVE) && !defined(__CPU_SIM)
inline constexpr uint64_t kMockHscbAddScratchUb = 0x2FF80;
#endif

AICORE inline void atom_add_hscb(__gm__ uint32_t* peerScb, uint32_t delta)
{
#if defined(PTO_GRID_CCE_NATIVE)
    (void)__atom_add_hscb(peerScb, delta); // -> __builtin_cce___atom_add_hscb; exact target encoding is HW-DEP
#elif defined(__CPU_SIM)
    if (peerScb != nullptr) {
        __atomic_fetch_add(reinterpret_cast<uint32_t*>(peerScb), delta, __ATOMIC_RELEASE);
    }
#else
    if (peerScb != nullptr && get_subblockid() == 0) {
        __ubuf__ uint32_t* addend = reinterpret_cast<__ubuf__ uint32_t*>(kMockHscbAddScratchUb);
        *addend = delta;
#ifndef __PTO_AUTO__
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID7);
#endif
        // Drop any stale local copy before the L2 atomic, otherwise a later
        // line-granular write-back from this core could undo another writer's add.
        dcci(reinterpret_cast<__gm__ void*>(peerScb), SINGLE_CACHE_LINE);
        set_atomic_s32();
        set_atomic_add();
        copy_ubuf_to_gm_align_b32(peerScb, addend, 0, 1, sizeof(uint32_t), 0, 0, 0, 0);
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
#endif
        set_atomic_none();
        dcci(reinterpret_cast<__gm__ void*>(peerScb), SINGLE_CACHE_LINE);
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
// Shared GM-mock scalar read of a word this core owns.  The leading dcci
// invalidates the local line first: AICORE caches are not coherent between cores,
// so without it the read can return a stale copy.
AICORE inline uint32_t read_local_word(__gm__ uint32_t* addr)
{
    if (addr == nullptr) {
        return 0;
    }
#if defined(__CPU_SIM)
    return __atomic_load_n(reinterpret_cast<uint32_t*>(addr), __ATOMIC_ACQUIRE);
#else
    volatile __gm__ uint32_t* ptr = reinterpret_cast<volatile __gm__ uint32_t*>(addr);
    __asm__ __volatile__("" ::: "memory");
    dcci(reinterpret_cast<__gm__ void*>(const_cast<__gm__ uint32_t*>(ptr)), SINGLE_CACHE_LINE);
    __asm__ __volatile__("" ::: "memory");
    return *ptr;
#endif
}

// Shared GM-mock scalar write of a word this core owns.  The trailing dcci writes
// the line back so a later read (this core's mov_x_to_gpr, or the host's D2H dump)
// observes it; AICORE caches are not coherent between cores.
AICORE inline void write_local_word(__gm__ uint32_t* addr, uint32_t value)
{
    if (addr == nullptr) {
        return;
    }
#if defined(__CPU_SIM)
    __atomic_store_n(reinterpret_cast<uint32_t*>(addr), value, __ATOMIC_RELEASE);
#else
    volatile __gm__ uint32_t* ptr = reinterpret_cast<volatile __gm__ uint32_t*>(addr);
    __asm__ __volatile__("" ::: "memory");
    *ptr = value;
    __asm__ __volatile__("" ::: "memory");
    dcci(reinterpret_cast<__gm__ void*>(const_cast<__gm__ uint32_t*>(ptr)), SINGLE_CACHE_LINE);
    __asm__ __volatile__("" ::: "memory");
#endif
}

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
#if !defined(__CPU_SIM)
    volatile __gm__ uint32_t* p = reinterpret_cast<volatile __gm__ uint32_t*>(localScb);
#endif
    uint32_t spin = 0;
    constexpr uint32_t kFenceInterval = 64;
    while (true) {
#if defined(__CPU_SIM)
        if (__atomic_load_n(reinterpret_cast<uint32_t*>(localScb), __ATOMIC_ACQUIRE) >= threshold) {
            return true;
        }
#else
        __asm__ __volatile__("" ::: "memory");
        dcci(reinterpret_cast<__gm__ void*>(const_cast<__gm__ uint32_t*>(p)), SINGLE_CACHE_LINE);
        __asm__ __volatile__("" ::: "memory");
        if (*p >= threshold) {
            return true;
        }
#endif
        if (maxSpins != 0 && spin >= maxSpins) {
            return false;
        }
        if ((++spin % kFenceInterval) == 0) {
            pipe_barrier(PIPE_ALL);
        }
    }
}
} // namespace grid_cce_detail

// V8 WAIT_SPR: block until the local IPC_SCB reaches `threshold`.  void / blocking,
// mirroring the real __wait_ast_scb -- this is the documented CCE intrinsic for G3.
//
// §3.3 G3 operands (local_scb_id, threshold): `slot` selects the native IPC_SCB slot
// (0..15) -- ready_scb of channel c -> slot c, free_scb of channel c -> slot
// kGridChanCount+c, close_scb of channel c -> slot 2*kGridChanCount+c;
// `localScb` is the GM word the mock reads instead.  Native ignores
// localScb; the mock ignores slot.  Memory ordering: acquire.
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
// (4) MOV_SPR2X  ->  __mov_ipc_scb_to_l1  ->  __builtin_cce___mov_ipc_scb_to_l1
//
// Copy a LOCAL IPC_SCB into a LOCAL memory word.  It is not a steady-state
// wait pre-check -- wait_ipc_scb compares inside the instruction.  GridPipe uses
// it on control paths to save free credit and to snapshot ready/close counts while
// selecting and rebasing a time-division channel.  There is no SPR -> GPR variant;
// the value must land in memory first.
//
// `srcSlot` selects the native IPC_SCB slot (0..15); `srcScb` is the GM word the
// mock reads instead (native ignores it, the mock ignores the slot).  `dst` is the
// local word to deposit into.  Null operands are no-ops, matching sync_hscb.
// ===========================================================================
AICORE inline void mov_ipc_scb_to_l1(__gm__ uint32_t* dst, __gm__ uint32_t* srcScb, uint32_t srcSlot)
{
#if defined(PTO_GRID_CCE_NATIVE)
    (void)srcScb;
    __builtin_cce___mov_ipc_scb_to_l1(dst, srcSlot); // MOV_SPR2X; encoding per ISA manual
#else
    (void)srcSlot;
    grid_cce_detail::write_local_word(dst, grid_cce_detail::read_local_word(srcScb));
#endif
}

// ===========================================================================
// (5) MOVX2SPR  ->  __mov_x_to_ipc_scb  ->  __builtin_cce___mov_x_to_ipc_scb
//
// Install a scalar into a LOCAL IPC_SCB.  This is NOT used by the current GridPipe
// handshake: a scoreboard has exactly one writer and that writer is the PEER
// (选型文档 §1.1 约束①: a core may not write its own IPC_SCB).  During rebinding the
// new consumer transfers its baseline directly with SYNC_HSCB to the selected local
// producer channel.  Calling MOVX2SPR on such a live GridPipe SCB could race that
// external writer and lose credit; the facade remains only as a direct machine
// primitive for callers that can independently prove exclusive ownership.
//
// `slot` selects the native IPC_SCB slot (0..15); `localScb` is the GM word the mock
// writes instead (native ignores it, the mock ignores the slot).  Null is a no-op,
// matching sync_hscb / wait_ipc_scb.
// ===========================================================================
AICORE inline void mov_x_to_ipc_scb(__gm__ uint32_t* localScb, uint32_t slot, uint32_t value)
{
#if defined(PTO_GRID_CCE_NATIVE)
    (void)localScb;
    __builtin_cce___mov_x_to_ipc_scb(slot, value); // MOVX2SPR; encoding per ISA manual
#else
    (void)slot;
    grid_cce_detail::write_local_word(localScb, value);
#endif
}

// ===========================================================================
// (6) MOVX2GPR  ->  __mov_x_to_gpr  ->  __builtin_cce___mov_x_to_gpr
//
// READ a LOCAL word into a scalar GPR -- the only way a value in memory becomes
// something the scalar unit can branch on or run a counter with.  V8 spells the
// same move MOV_L12X when the source is specifically L1; the instruction is named
// by where the value LANDS, so one facade covers both commit-word polling and the
// bind-response ready-baseline install into prod_idx.
//
// Unlike mov_x_to_ipc_scb above this addresses memory rather than naming a slot,
// which is exactly why the two cannot share a facade.  It carries no
// synchronisation of its own: the caller must already own the word it reads.
// ===========================================================================
AICORE inline uint32_t mov_x_to_gpr(__gm__ uint32_t* localWord)
{
#if defined(PTO_GRID_CCE_NATIVE)
    return __builtin_cce___mov_x_to_gpr(localWord); // MOVX2GPR; encoding per ISA manual
#else
    return grid_cce_detail::read_local_word(localWord);
#endif
}

// ===========================================================================
// GridCollOp: the MOV_UBUF_GROUP `op` machine operand -- the NoC collective
// communication MODE (design: 2026-07-24-bcast-reduce-合并mov_ubuf_group方案.md
// §2.1).  From the issuing core's perspective a group broadcast (1->N identity
// copy, push) and a group reduce (N->1 element-wise combine, pull) are the SAME
// action -- move local data to/from the resolved group arena -- differing only in
// the NoC mode; that mode is this runtime operand, NOT a different instruction.
// COPY = replicate-fan-out (dedicated producer L1 -> arena; GridPipe uses
// copy_l1_to_group so the source cannot alias a receive ring);
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

namespace grid_cce_detail {
// Byte distance from THIS core's copy of a symmetric group slot to member
// `blockId`'s copy of the same slot.  Signed: a member's block id may sit either
// side of the caller's, and the top-left member of a group usually sits below it.
// See the MOV_UBUF_GROUP note below for why a uniform per-block-id stride is what
// the mock uses to model symmetric addressing.
AICORE constexpr int64_t member_slot_delta(uint32_t blockId, uint32_t selfBlockId, uint32_t blockStride)
{
    return (static_cast<int64_t>(blockId) - static_cast<int64_t>(selfBlockId)) * static_cast<int64_t>(blockStride);
}
} // namespace grid_cce_detail

// Broadcast COPY counterpart of mov_ubuf_group for the unified-L1 address
// model.  The architectural source is `srcProducerSlot`, a dedicated L1 range;
// `transferScratch` exists only because the A3 GM mock needs UB as a DMA pump.
// Native still lowers to one group instruction (whose current builtin retains
// the historical `ubuf` spelling), while the mock expands it per member.
AICORE inline void copy_l1_to_group(
    __gm__ const void* srcProducerSlot, __gm__ void* groupSlot, __ubuf__ void* transferScratch, uint32_t bytes,
    uint32_t blockStride, const pto::GridBlockRect& group, uint32_t selfBlockId, uint64_t groupDesc = 0)
{
    const uint32_t stride = (blockStride == 0) ? bytes : blockStride;
#if defined(PTO_GRID_CCE_NATIVE)
    (void)transferScratch;
    const uint64_t desc = (groupDesc != 0) ? groupDesc : pto::GridPackGroupDesc(group, selfBlockId);
    auto* srcUnifiedL1 = reinterpret_cast<__ubuf__ void*>(reinterpret_cast<uint64_t>(srcProducerSlot));
    __builtin_cce_mov_ubuf_group(
        srcUnifiedL1, groupSlot, bytes, stride, static_cast<uint32_t>(pto::GridCollOp::COPY), /*eltype=*/1, desc);
#else
    (void)groupDesc;
    const uint32_t memberCount = pto::GridBlockRectSize(group);
    for (uint32_t k = 0; k < memberCount; ++k) {
        auto* dst = reinterpret_cast<__gm__ uint8_t*>(groupSlot) +
                    grid_cce_detail::member_slot_delta(pto::GridBlockRectMember(group, k), selfBlockId, stride);
        copy_l1_to_neighbor_l1(dst, srcProducerSlot, transferScratch, bytes);
    }
#endif
}

// ===========================================================================
// (7) MOV_UBUF_GROUP  ->  mov_ubuf_group  ->  __builtin_cce_mov_ubuf_group
//
// Unified template-free group collective transfer.  The issuing core moves
// `bytes` between its local UB tile and every group member's copy of one
// SYMMETRIC group slot, with the NoC collective mode selected by the RUNTIME
// `op` operand:
//   * op == COPY          : broadcast -- this core is the SOURCE; its UB tile is
//     replicated once into every member's copy of the slot (1->N fan-out, push).
//     Byte-level pure copy (does NOT read element values), so eltype is IGNORED
//     -- retained for non-GridPipe compatibility.  GridPipe broadcast uses
//     copy_l1_to_group above so its architectural source is dedicated L1.
//   * op == SUM/MAX/MIN   : reduce -- this core is the SINK; every member's copy
//     of the slot is read and folded element-wise by op into UB (N->1 fan-in,
//     pull).  Element-wise combine MUST know the element width, so `eltype`
//     (1/2/4 bytes) selects the combine granularity.
// This collapses the former bcast_ubuf_to_group + reduce_group_to_ubuf<T,Op> pair
// (two machine instructions, two template facades) into ONE machine instruction
// and ONE template-free facade: the NoC mode + dtype are runtime operand fields,
// matching how cce_aicore_intrinsics.h encodes direction/op as operand enums
// (MemoryDirection_t / atomic_op_t).  NOT self-syncing: data-ready is still
// announced by the caller's sync_hscb(READY) after the publish fence.
//
// WHO the collective ends at is now an OPERAND, not an implication of who ran
// the code: `selfBlockId` is the issuing core's own block id and names the
// collective's ROOT -- the SOURCE of a COPY, the SINK of a combine.  `group`
// (GridBlockRect) names the member set as the two corner block ids of a mesh
// sub-rectangle.  Together they read exactly as the collective is specified:
// every core in the rectangle contributes the data at its own copy of
// `groupSlot`, and the reduction of all of it lands in the UB of the core whose
// block id is `selfBlockId`.  The root is normally a member of the rectangle
// (GridBlockRectContains), but nothing here requires it -- the member set comes
// from the corners alone, so a core outside the rectangle can gather it.
//
// `groupSlot` is THIS core's copy of the symmetric slot; member b's copy sits at
// `groupSlot + (b - selfBlockId)*blockStride` (blockStride == 0 means packed by
// block id, stride == bytes).  The stride is how the MOCK models "the same
// address in every core's space" -- the per-cell HCCL windows and the demo's
// contribution arenas are both laid out uniformly by block id; silicon resolves a
// symmetric address in the NoC and ignores the operand.  Writable for COPY (the
// source writes each member's copy); read-only for SUM/MAX/MIN (the sink only
// reads contributions -- the caller const-casts).  Because members are walked by
// BLOCK ID, a multi-row rectangle costs nothing extra here: the row-boundary jump
// is this arithmetic, not a stride the caller has to fake.
//
// `combineScratch` is the A3-mock combine scratch (one member's worth of UB).
// REQUIRED on the A3 mock for op != COPY (no on-transit combine: the in-core Vec
// vadd/vmax/vmin needs each member k>=1 in UB); IGNORED for COPY and on native
// (hardware collective) / __CPU_SIM (host loop reads members directly).
// ===========================================================================
AICORE inline void mov_ubuf_group(
    __ubuf__ void* ubTile, __gm__ void* groupSlot, uint32_t bytes, uint32_t blockStride, pto::GridCollOp op,
    uint32_t eltype, const pto::GridBlockRect& group, uint32_t selfBlockId, __ubuf__ void* combineScratch = nullptr,
    uint64_t groupDesc = 0)
{
    const uint32_t stride = (blockStride == 0) ? bytes : blockStride;
    const uint32_t memberCount = pto::GridBlockRectSize(group);
#if defined(PTO_GRID_CCE_NATIVE)
    (void)memberCount; // native: the member set travels in the group descriptor.
    (void)combineScratch;
    // The descriptor carries the member rectangle AND the root; `stride` still rides
    // along as the machine's arena-spacing operand, but silicon resolves a symmetric
    // address in the NoC and does not need it.
    const uint64_t desc = (groupDesc != 0) ? groupDesc : pto::GridPackGroupDesc(group, selfBlockId);
    __builtin_cce_mov_ubuf_group(ubTile, groupSlot, bytes, stride, static_cast<uint32_t>(op), eltype, desc);
#elif defined(__CPU_SIM)
    // CPU_SIM: __gm__/__ubuf__ collapse to ordinary host pointers and the CCE DMA
    // intrinsic is not declared, so a typed host loop (no memcpy, lint-clean) stands
    // in for the collective.  op selects direction+combine at runtime.
    (void)groupDesc;
    (void)combineScratch;
    auto* ub = reinterpret_cast<uint8_t*>(ubTile);
    if (op == pto::GridCollOp::COPY) {
        // broadcast: replicate ub -> every member's copy of the slot.
        for (uint32_t k = 0; k < memberCount; ++k) {
            auto* d = reinterpret_cast<uint8_t*>(groupSlot) +
                      grid_cce_detail::member_slot_delta(pto::GridBlockRectMember(group, k), selfBlockId, stride);
            for (uint32_t i = 0; i < bytes; ++i) {
                d[i] = ub[i];
            }
        }
    } else {
        // reduce: member 0 seeds ub, k>=1 folds element-wise by op (eltype picks width).
        const uint32_t n = bytes / eltype;
        for (uint32_t k = 0; k < memberCount; ++k) {
            const uint8_t* in =
                reinterpret_cast<const uint8_t*>(groupSlot) +
                grid_cce_detail::member_slot_delta(pto::GridBlockRectMember(group, k), selfBlockId, stride);
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
        // Legacy non-GridPipe broadcast: chunked UB -> GM-window copy.  GridPipe's
        // corrected unified-L1 path exits through copy_l1_to_group above instead.
        auto* selfSlot = reinterpret_cast<__gm__ uint8_t*>(groupSlot);
        auto* srcBytes = reinterpret_cast<__ubuf__ uint8_t*>(ubTile);
        for (uint32_t k = 0; k < memberCount; ++k) {
            __gm__ uint8_t* d =
                selfSlot + grid_cce_detail::member_slot_delta(pto::GridBlockRectMember(group, k), selfBlockId, stride);
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
        auto* selfSlot = reinterpret_cast<__gm__ const uint8_t*>(groupSlot);
        // member 0 (the rectangle's top-left cell): contribution -> accumulator.
        __gm__ const uint8_t* m0 =
            selfSlot + grid_cce_detail::member_slot_delta(pto::GridBlockRectMember(group, 0), selfBlockId, stride);
        for (uint32_t off = 0; off < bytes; off += kChunkBytes) {
            uint32_t chunk = (bytes - off > kChunkBytes) ? kChunkBytes : (bytes - off);
            copy_gm_to_ubuf_align_b8(accBytes + off, m0 + off, 0, 1, chunk, 0, 0, 0, 0);
        }
        const uint32_t elemsPerRepeat = static_cast<uint32_t>(REPEAT_BYTE) / eltype; // 64 (float) / 128 (half)
        const uint32_t totalRepeats = bytes / static_cast<uint32_t>(REPEAT_BYTE);
        for (uint32_t k = 1; k < memberCount; ++k) {
            __gm__ const uint8_t* mk =
                selfSlot + grid_cce_detail::member_slot_delta(pto::GridBlockRectMember(group, k), selfBlockId, stride);
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
