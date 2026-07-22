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
#include <pto/common/type.hpp> // AICORE + address-space qualifiers

namespace pto {

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
AICORE inline void copy_ubuf_to_neighbor_ubuf(__gm__ void *dstNeighborSlot, __ubuf__ void *src, uint32_t bytes)
{
#if defined(PTO_GRID_CCE_NATIVE)
    __builtin_cce_copy_ubuf_to_neighbor_ubuf(dstNeighborSlot, src, bytes, /*config=*/0);
#elif defined(__CPU_SIM)
    // CPU_SIM: __gm__/__ubuf__ collapse to ordinary host pointers and the CCE DMA
    // intrinsic (copy_ubuf_to_gm_align_b8) is not declared in this build, so a plain
    // byte copy stands in for the neighbor L1 write.  A loop (not memcpy) keeps the
    // CPU-sim source lint happy.
    auto *dstBytes = reinterpret_cast<uint8_t *>(dstNeighborSlot);
    auto *srcBytes = reinterpret_cast<uint8_t *>(src);
    for (uint32_t i = 0; i < bytes; ++i) {
        dstBytes[i] = srcBytes[i];
    }
#else
    // A3 mock: chunked UB -> GM-window copy stands in for the neighbor L1 write.
    constexpr uint32_t kChunkBytes = 256;
    auto *dstBytes = reinterpret_cast<__gm__ uint8_t *>(dstNeighborSlot);
    auto *srcBytes = reinterpret_cast<__ubuf__ uint8_t *>(src);
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
AICORE inline void sync_hscb(__gm__ uint32_t *peerScb, uint32_t absCount)
{
#if defined(PTO_GRID_CCE_NATIVE)
    __sync_hscb(peerScb, absCount); // -> __builtin_cce___sync_hscb; exact operand encoding per ISA manual
#else
    if (peerScb != nullptr) {
        // A3 mock: cross-core GM store + cache maintenance.  AICORE caches are not
        // coherent between cores, so the pre/post dcci + dsb(DSB_DDR) make the store
        // observable by the peer's wait_ipc_scb spin (matches the canonical TNotify
        // Set pattern).  volatile prevents the compiler caching the write.
        volatile __gm__ uint32_t *ptr = reinterpret_cast<volatile __gm__ uint32_t *>(peerScb);
        __asm__ __volatile__("" ::: "memory");
        dcci(reinterpret_cast<__gm__ void *>(const_cast<__gm__ uint32_t *>(ptr)), SINGLE_CACHE_LINE);
        __asm__ __volatile__("" ::: "memory");
        *ptr = absCount;
        __asm__ __volatile__("" ::: "memory");
        dcci(reinterpret_cast<__gm__ void *>(const_cast<__gm__ uint32_t *>(ptr)), SINGLE_CACHE_LINE);
        __asm__ __volatile__("" ::: "memory");
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
AICORE inline bool poll_ipc_scb_ge(__gm__ uint32_t *localScb, uint32_t threshold, uint32_t maxSpins)
{
    if (localScb == nullptr) {
        return true;
    }
    volatile __gm__ uint32_t *p = reinterpret_cast<volatile __gm__ uint32_t *>(localScb);
    uint32_t spin = 0;
    constexpr uint32_t kFenceInterval = 64;
    while (true) {
        __asm__ __volatile__("" ::: "memory");
        dcci(reinterpret_cast<__gm__ void *>(const_cast<__gm__ uint32_t *>(p)), SINGLE_CACHE_LINE);
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

// V8 WAIT_SPR: block until the local IPC_SCB reaches `threshold`.  void / blocking,
// mirroring the real __wait_ast_scb -- this is the documented CCE intrinsic for G3.
//
// §3.3 G3 operands (local_scb_id, threshold): `slot` selects the native IPC_SCB slot
// (0..15) -- ready_scb_<dir> -> slot dirIdx, free_scb_<dir> -> slot
// kGridDirectionCount+dirIdx; `localScb` is the GM word the mock reads instead.  Native
// ignores localScb; the mock ignores slot.  Memory ordering: acquire.
AICORE inline void wait_ipc_scb(__gm__ uint32_t *localScb, uint32_t threshold, uint32_t slot)
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
AICORE inline bool wait_ipc_scb_sim(__gm__ uint32_t *localScb, uint32_t threshold, uint32_t slot, uint32_t maxSpins)
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

} // namespace pto

#endif // PTO_A2A3_GRID_CCE_INTRINSIC_HPP
