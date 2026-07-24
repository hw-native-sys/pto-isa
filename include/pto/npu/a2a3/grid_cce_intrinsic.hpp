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

// Forward declarations of the GridPipe group/topology types.  They are defined
// in grid_intrinsic.hpp, which includes THIS header (so a real include here
// would be circular).  The broadcast/reduce group intrinsics below only need
// their NAMES -- GridGroup as a non-type template parameter and GridRect as an
// unused by-reference group descriptor -- so forward declarations suffice.
// (Their underlying-type / field definitions are complete by the time any
// translation unit that instantiates these templates is compiled.)
enum class GridGroup : uint8_t;
struct GridRect;

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

// ===========================================================================
// (4) BCAST_UBUF_TO_GROUP  ->  bcast_ubuf_to_group  ->  __builtin_cce_bcast_ubuf_to_group
//
// Single-source group broadcast: the source core (caller == root) copies its
// local UB tile once into every member's resolved receive slot in the group.
// This is the hardware-accelerated single-instruction form of the Tier-2
// TBROADCAST<Group> handshake fan-out (design: CCE-Intrinsic接口规范-广播与归约
// 语义扩展设计 §6.1 / §7.3).  It is a byte-level PURE copy (it does NOT read
// element values), so the facade is dtype-agnostic (void* + bytes), mirroring
// copy_ubuf_to_neighbor_ubuf above.  NOT self-syncing: data-ready is still
// announced by the caller's sync_hscb(READY) after the publish fence (§7.3).
//
// `groupSlotBase` is the RESOLVED per-member receive-slot arena base (member 0's
// slot); member m's slot = groupSlotBase + m*memberStride (memberStride==0 means
// packed, stride == bytes).  The Tier-2 caller resolves member spacing from the
// group topology (ROW/COL are uniformly spaced; a SUBRECT whose members are not
// uniformly spaced -- a multi-row rectangle -- is handled by the caller, which
// falls back to a per-member copy for that case).
// ===========================================================================
template <pto::GridGroup Group>
AICORE inline void bcast_ubuf_to_group(__gm__ void *groupSlotBase, __ubuf__ void *src, uint32_t bytes,
                                       uint32_t memberCount, const pto::GridRect &rect, uint32_t memberStride)
{
    (void)rect; // ROW/COL ignore; SUBRECT range already folded into base/stride by the caller.
    const uint32_t stride = (memberStride == 0) ? bytes : memberStride;
#if defined(PTO_GRID_CCE_NATIVE)
    (void)memberCount; // native: member count is encoded into the group descriptor.
    __builtin_cce_bcast_ubuf_to_group(src, groupSlotBase, bytes, stride, /*group_desc=*/0);
#elif defined(__CPU_SIM)
    // CPU_SIM: __gm__/__ubuf__ collapse to ordinary host pointers and the CCE DMA
    // intrinsic is not declared, so a byte loop (no memcpy, lint-clean) stands in
    // for the multicast.  Each member receives the identical payload.
    auto *s = reinterpret_cast<const uint8_t *>(src);
    for (uint32_t k = 0; k < memberCount; ++k) {
        auto *d = reinterpret_cast<uint8_t *>(groupSlotBase) + static_cast<uint64_t>(k) * stride;
        for (uint32_t i = 0; i < bytes; ++i) {
            d[i] = s[i];
        }
    }
#else
    // A3 mock: chunked UB -> GM-window copy of `src` into every member's slot
    // (mirrors copy_ubuf_to_neighbor_ubuf's 256B-chunked A3-mock pump).
    constexpr uint32_t kChunkBytes = 256;
    auto *dstBase = reinterpret_cast<__gm__ uint8_t *>(groupSlotBase);
    auto *srcBytes = reinterpret_cast<__ubuf__ uint8_t *>(src);
    for (uint32_t k = 0; k < memberCount; ++k) {
        __gm__ uint8_t *d = dstBase + static_cast<uint64_t>(k) * stride;
        for (uint32_t off = 0; off < bytes; off += kChunkBytes) {
            uint32_t chunk = (bytes - off > kChunkBytes) ? kChunkBytes : (bytes - off);
            copy_ubuf_to_gm_align_b8(d + off, srcBytes + off, 0, 1, chunk, 0, 0, 0, 0);
        }
    }
#endif
}

// ===========================================================================
// (5) REDUCE_GROUP_TO_UBUF  ->  reduce_group_to_ubuf<T>  ->
//   __builtin_cce_reduce_group_to_ubuf_{b16,b32}
//
// Group fan-in reduce to the target core: the target core (caller == sink)
// reads every member's resolved contribution slot in the group, folds them
// element-wise with Op (Sum/Max/Min), and writes the result to local UB.  This
// is the hardware-accelerated single-instruction form of the Tier-2 N->1 reduce
// (the TREDUCE directional relay / TADD loop) -- design: §6.2 / §7.3.  It is a
// genuine N->1 fan-in, a different collective SHAPE from the directional relay
// (§7.1); the Tier-2 caller chooses it for whole-row / whole-column reduces.
//
// Element-wise combine MUST know the element width, so the facade is templated
// on T and dispatches the native builtin by sizeof(T): _b16 for half /
// bfloat16_t (2B), _b32 for float (4B), mirroring the pto_copy_*<T> sizeof
// dispatch.  NOT self-syncing (same contract as bcast / copy_ubuf_to_neighbor).
//
// `groupSlotBase` is the RESOLVED per-member contribution-slot arena base
// (const: reduce only reads contributions); member m's slot = groupSlotBase +
// m*memberStride.  The combine folds in ASCENDING member order (member 0 seeds
// dst, member k>=1 folds in), so an SPMD row/col fan-in reproduces the relay's
// left-to-right accumulation BIT-FOR-BIT (IEEE-754 add is commutative, so the
// relay's `acc := acc + recv` per hop and the reduce's `out += member_k` give
// identical results when members are visited in the same order).
//
// `combineScratch` is the A3-mock combine scratch (one member's worth of UB).
// It is REQUIRED on the A3 mock (no on-transit combine: the in-core Vec op
// vadd/vmax/vmin needs each member k>=1 in UB) and IGNORED on native (hardware
// reduce tree) and __CPU_SIM (host loop reads members directly).
// ===========================================================================
template <pto::GridGroup Group, pto::comm::ReduceOp Op, typename T>
AICORE inline void reduce_group_to_ubuf(__ubuf__ T *dst, __gm__ const T *groupSlotBase, uint32_t bytes,
                                        uint32_t memberCount, const pto::GridRect &rect, uint32_t memberStride,
                                        __ubuf__ T *combineScratch = nullptr)
{
    static_assert(sizeof(T) == 2 || sizeof(T) == 4,
                  "reduce_group_to_ubuf: T must be 2B (half/bfloat16_t) or 4B (float)");
    (void)rect;
    const uint32_t stride = (memberStride == 0) ? bytes : memberStride;
#if defined(PTO_GRID_CCE_NATIVE)
    (void)memberCount;
    (void)combineScratch;
    if constexpr (sizeof(T) == 2) {
        __builtin_cce_reduce_group_to_ubuf_b16(dst, groupSlotBase, bytes, stride,
                                               /*op=*/static_cast<uint32_t>(Op), /*group_desc=*/0);
    } else {
        __builtin_cce_reduce_group_to_ubuf_b32(dst, groupSlotBase, bytes, stride,
                                               /*op=*/static_cast<uint32_t>(Op), /*group_desc=*/0);
    }
#elif defined(__CPU_SIM)
    (void)combineScratch;
    // host typed element-wise combine (no memcpy): member 0 seeds dst, k>=1 folds.
    auto *out = reinterpret_cast<T *>(dst);
    const uint32_t n = bytes / sizeof(T);
    for (uint32_t k = 0; k < memberCount; ++k) {
        const T *in = reinterpret_cast<const T *>(reinterpret_cast<const uint8_t *>(groupSlotBase)
                                                      + static_cast<uint64_t>(k) * stride);
        if (k == 0) {
            for (uint32_t i = 0; i < n; ++i) {
                out[i] = in[i];
            }
            continue;
        }
        if constexpr (Op == pto::comm::ReduceOp::Sum) {
            for (uint32_t i = 0; i < n; ++i) {
                out[i] = out[i] + in[i];
            }
        } else if constexpr (Op == pto::comm::ReduceOp::Max) {
            for (uint32_t i = 0; i < n; ++i) {
                out[i] = out[i] > in[i] ? out[i] : in[i];
            }
        } else {
            for (uint32_t i = 0; i < n; ++i) {
                out[i] = out[i] < in[i] ? out[i] : in[i];
            }
        }
    }
#else
    // A3 mock: per-member GM->UB pull (member 0 -> accumulator dst, k>=1 ->
    // scratch) + an in-core Vec combine (vadd/vmax/vmin), mirroring the existing
    // ReduceTiles (TADD/TMAX/TMIN) path.  One scratch buffer is reused per member.
    auto *acc = reinterpret_cast<__ubuf__ T *>(dst);
    auto *scr = reinterpret_cast<__ubuf__ T *>(combineScratch);
    auto *baseBytes = reinterpret_cast<__gm__ const uint8_t *>(groupSlotBase);
    auto *accBytes = reinterpret_cast<__ubuf__ uint8_t *>(acc);
    auto *scrBytes = reinterpret_cast<__ubuf__ uint8_t *>(scr);
    constexpr uint32_t kChunkBytes = 256;
    // member 0: contribution -> accumulator.
    for (uint32_t off = 0; off < bytes; off += kChunkBytes) {
        uint32_t chunk = (bytes - off > kChunkBytes) ? kChunkBytes : (bytes - off);
        copy_gm_to_ubuf_align_b8(accBytes + off, baseBytes + off, 0, 1, chunk, 0, 0, 0, 0);
    }
    const uint32_t elemsPerRepeat = static_cast<uint32_t>(REPEAT_BYTE) / sizeof(T); // 64 (float) / 128 (half)
    const uint32_t totalRepeats = bytes / static_cast<uint32_t>(REPEAT_BYTE);
    for (uint32_t k = 1; k < memberCount; ++k) {
        __gm__ const uint8_t *mk = baseBytes + static_cast<uint64_t>(k) * stride;
        for (uint32_t off = 0; off < bytes; off += kChunkBytes) {
            uint32_t chunk = (bytes - off > kChunkBytes) ? kChunkBytes : (bytes - off);
            copy_gm_to_ubuf_align_b8(scrBytes + off, mk + off, 0, 1, chunk, 0, 0, 0, 0);
        }
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL); // MTE2 (GM->UB) -> V (combine)
#endif
        // in-core element-wise combine acc OP= scr, chunked into <=255 repeats.
        for (uint32_t r = 0; r < totalRepeats;) {
            uint32_t chunk = totalRepeats - r;
            if (chunk > 255u) {
                chunk = 255u;
            }
            __ubuf__ T *a = acc + r * elemsPerRepeat;
            __ubuf__ T *s = scr + r * elemsPerRepeat;
            if constexpr (Op == pto::comm::ReduceOp::Sum) {
                vadd(a, a, s, static_cast<uint8_t>(chunk), 1, 1, 1, 8, 8, 8);
            } else if constexpr (Op == pto::comm::ReduceOp::Max) {
                vmax(a, a, s, static_cast<uint8_t>(chunk), 1, 1, 1, 8, 8, 8);
            } else {
                vmin(a, a, s, static_cast<uint8_t>(chunk), 1, 1, 1, 8, 8, 8);
            }
            r += chunk;
        }
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL); // V (combine) -> next MTE2 (GM->UB into scratch)
#endif
    }
#endif
}

} // namespace pto

#endif // PTO_A2A3_GRID_CCE_INTRINSIC_HPP
