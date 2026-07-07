/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Grid TPUSH/TPOP CCE facade layer (A2/A3 backend) -- V7 IPC_SCB scoreboard route.
//
// Design_spec: Grid_TPUSH_TPOP_ISA...V7.md, section 3.4 (layering/naming) and
// section 6 point 4.  V7 freezes a strict TWO-name lowering with NO intermediate
// PTO wrapper: each Grid handshake op is a *CCE facade name* (the header
// declaration name) that forwards to a *CCE builtin name* (__builtin_cce_*),
// which the compiler lowers to one machine instruction.  The Grid TPUSH/TPOP
// sequence (GridTPush.hpp / GridTPop.hpp) calls these four facades DIRECTLY --
// there is deliberately no sync_neighbor_scb / wait_local_spr / mov_local_spr /
// ScbOperand vocabulary between the PTO instruction and the CCE name.
//
// clang-format off
//   V7 machine instr | CCE facade            | CCE builtin (native)                     | facade in this file
//   -----------------+-----------------------+------------------------------------------+--------------------
//   COPY_UBUF_TO_NBR | copy_ubuf_to_neighbor_ubuf | __builtin_cce_copy_ubuf_to_neighbor_ubuf | copy_ubuf_to_neighbor_ubuf
//   SYNC_HSCB/ST_HSCB| __sync_hscb           | __builtin_cce___sync_hscb                | sync_hscb
//   MOV_SPR2X        | get_ipc_scb_{0..15}   | __builtin_cce_get_ipc_scb_{0..15}        | get_ipc_scb
//   WAIT_SPR         | __wait_ipc_scb        | __builtin_cce___wait_ipc_scb             | wait_ipc_scb
// clang-format on
//
// Why the facade names here drop the leading "__" (sync_hscb / get_ipc_scb /
// wait_ipc_scb): cce_aicore_intrinsics.h *already* declares __sync_hscb,
// get_ipc_scb_{0..15} and __wait_ast_scb as real builtin aliases (auto-included
// by the CCE frontend), so a same-named redefinition for the GM mock would
// collide.  Each facade below is therefore a thin, 1:1 dispatcher that carries
// the CCE name and, under PTO_GRID_CCE_NATIVE, forwards to the real
// __builtin_cce_* (via the header alias); otherwise it emulates the builtin's
// SEMANTICS in GM.
//
// GM-mock rationale (default on A3): A3 has no cross-AICORE fabric, and the
// compiler exposes neither __builtin_cce_copy_ubuf_to_neighbor_ubuf nor
// __builtin_cce___wait_ipc_scb (get_ipc_scb_* / __sync_hscb exist, but their
// machine ops cannot address a geometric neighbor's IPC_SCB / L1).  So the
// DEFAULT build models each direction IPC_SCB slot as a volatile GM word and the
// neighbor L1 as a GM window: SYNC_HSCB -> cross-core GM store + cache
// maintenance; MOV_SPR2X -> GM read; WAIT_SPR -> GM spin-poll; COPY_UBUF_TO_NBR
// -> UB->GM window copy.  Define PTO_GRID_CCE_NATIVE on silicon that provides the
// builtins to route each facade to the real __builtin_cce_*; call sites do not
// change.

#ifndef PTO_A2A3_GRID_CCE_INTRINSIC_HPP
#define PTO_A2A3_GRID_CCE_INTRINSIC_HPP

#include <cstdint>

#include <pto/common/arch_macro.hpp>
#include <pto/common/type.hpp> // AICORE + address-space qualifiers

// The direction scoreboards map to IPC_SCB slots as ready_scb_<dir> -> slot
// dirIdx and free_scb_<dir> -> slot kGridDirectionCount+dirIdx (V7 section 3.2.1,
// generalized to include SOURCE).  The backend computes these slot indices inline
// and passes them to get_ipc_scb / wait_ipc_scb (native dispatch); the GM mock
// addresses the scoreboard by its GM pointer and ignores the slot.

namespace pto {

#if defined(PTO_GRID_CCE_NATIVE)
// V7 MOV_SPR2X == get_ipc_scb_* : read one of the 16 local IPC_SCB slots.  The
// per-slot builtins are distinct, so a runtime slot index needs this dispatch.
// get_ipc_scb_* returns int64_t; the low 16 bits latch the unsigned count.
AICORE inline int64_t ReadIpcScbSlot(uint32_t slot)
{
    switch (slot) {
        case 0:
            return get_ipc_scb_0();
        case 1:
            return get_ipc_scb_1();
        case 2:
            return get_ipc_scb_2();
        case 3:
            return get_ipc_scb_3();
        case 4:
            return get_ipc_scb_4();
        case 5:
            return get_ipc_scb_5();
        case 6:
            return get_ipc_scb_6();
        case 7:
            return get_ipc_scb_7();
        case 8:
            return get_ipc_scb_8();
        case 9:
            return get_ipc_scb_9();
        case 10:
            return get_ipc_scb_10();
        case 11:
            return get_ipc_scb_11();
        case 12:
            return get_ipc_scb_12();
        case 13:
            return get_ipc_scb_13();
        case 14:
            return get_ipc_scb_14();
        default:
            return get_ipc_scb_15();
    }
}
#endif

// ---------------------------------------------------------------------------
// (1) COPY_UBUF_TO_NBR  ->  copy_ubuf_to_neighbor_ubuf  ->  __builtin_cce_copy_ubuf_to_neighbor_ubuf
//
// Cross-core payload write: local UB -> the target core's L1/SRAM slot.  Not
// self-syncing; data-ready is announced by the following sync_hscb(READY) after
// the publish fence (V7 R5).  `dst` is the resolved neighbor slot (native: the
// encoded neighbor L1 address; mock: the GM window standing in for it).
// ---------------------------------------------------------------------------
AICORE inline void copy_ubuf_to_neighbor_ubuf(__gm__ void *dst, __ubuf__ void *src, uint32_t bytes)
{
#if defined(PTO_GRID_CCE_NATIVE)
    __builtin_cce_copy_ubuf_to_neighbor_ubuf(dst, src, bytes, /*config=*/0);
#elif defined(__CPU_SIM)
    // CPU_SIM: __gm__/__ubuf__ collapse to ordinary host pointers and the CCE DMA
    // intrinsic (copy_ubuf_to_gm_align_b8) is not declared in this build, so a plain
    // byte copy stands in for the neighbor L1 write.  A loop (not memcpy) keeps the
    // CPU-sim source lint happy.
    auto *dstBytes = reinterpret_cast<uint8_t *>(dst);
    auto *srcBytes = reinterpret_cast<uint8_t *>(src);
    for (uint32_t i = 0; i < bytes; ++i) {
        dstBytes[i] = srcBytes[i];
    }
#else
    // A3 mock: chunked UB -> GM-window copy stands in for the neighbor L1 write.
    constexpr uint32_t kChunkBytes = 256;
    auto *dstBytes = reinterpret_cast<__gm__ uint8_t *>(dst);
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
// (READY -> downstream neighbor's ready_scb_<dir> = prod_idx; FREE -> upstream
// neighbor's free_scb_<dir> = cons_idx).  Single external writer per scoreboard
// (SPSC), so the overwrite store of a monotone absolute count is safe (V7 2.1).
// Memory ordering: release -- earlier payload writes must be visible first
// (V7 R5 / publish).  V7 prefers SYNC_HSCB from an async pipe so it naturally
// orders after the payload DMA; both SYNC_HSCB and ST_HSCB are HSCB stores and
// V7 point 3 lets ST_HSCB reuse __sync_hscb.  `peerScb` is the resolved peer
// scoreboard (native: encoded peer IPC_SCB address, HW-DEP-1; mock: peer GM word).
// ---------------------------------------------------------------------------
AICORE inline void sync_hscb(__gm__ uint32_t *peerScb, uint32_t absCount)
{
#if defined(PTO_GRID_CCE_NATIVE)
    __sync_hscb(peerScb, absCount); // -> __builtin_cce___sync_hscb; exact operand encoding per ISA manual
#else
    if (peerScb != nullptr) {
        // A3 mock: cross-core GM store + cache maintenance.  AICORE caches are not
        // coherent between cores, so the pre/post dcci + dsb(DSB_DDR) make the
        // store observable by the peer's wait_ipc_scb spin (matches the canonical
        // TNotify Set pattern).  volatile prevents the compiler caching the write.
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

// ---------------------------------------------------------------------------
// (3) MOV_SPR2X  ->  get_ipc_scb_{0..15}  ->  __builtin_cce_get_ipc_scb_{0..15}
//
// Non-blocking peek of the LOCAL direction scoreboard; returns the current
// absolute count for the fast-path check before wait_ipc_scb (V7 3.5.2 C1).
// `slot` selects the native IPC_SCB slot (0..15); `localScb` is the GM word the
// mock reads.  Native ignores localScb; the mock ignores slot.
// ---------------------------------------------------------------------------
AICORE inline uint32_t get_ipc_scb(__gm__ uint32_t *localScb, uint32_t slot)
{
#if defined(PTO_GRID_CCE_NATIVE)
    (void)localScb;
    return static_cast<uint32_t>(ReadIpcScbSlot(slot) & 0xFFFFu);
#else
    (void)slot;
    if (localScb == nullptr) {
        return 0;
    }
    return *reinterpret_cast<volatile __gm__ uint32_t *>(localScb);
#endif
}

// ---------------------------------------------------------------------------
// (4) WAIT_SPR  ->  __wait_ipc_scb  ->  __builtin_cce___wait_ipc_scb
//
// Block until the LOCAL direction scoreboard reaches `threshold` (ready:
// cons_idx+1; free: prod_idx-SlotCount+1); woken by the peer's HSCB store
// (V7 3.1.4).  Memory ordering: acquire.  Returns true when the threshold is
// reached, false on mock spin-timeout (maxSpins != 0).
//
// V7 renames the wait facade to __wait_ipc_scb (section 3.4.1 note (1); the
// current stand-in in cce_aicore_intrinsics.h is __wait_ast_scb on the sister
// AST_SCB).  On A3 there is no blocking IPC_SCB wait, so the mock degrades to a
// GM spin-poll with a dcci each iteration to invalidate the local cache line
// (without it the AICORE may cache a stale value and never see the peer store).
// ---------------------------------------------------------------------------
AICORE inline bool wait_ipc_scb(__gm__ uint32_t *localScb, uint32_t threshold, uint32_t slot, uint32_t maxSpins = 0)
{
#if defined(PTO_GRID_CCE_NATIVE)
    (void)localScb;
    (void)maxSpins;
    __builtin_cce___wait_ipc_scb(slot, threshold); // blocking WAIT_SPR; encoding per ISA manual
    return true;
#else
    (void)slot;
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
#endif
}

} // namespace pto

#endif // PTO_A2A3_GRID_CCE_INTRINSIC_HPP
