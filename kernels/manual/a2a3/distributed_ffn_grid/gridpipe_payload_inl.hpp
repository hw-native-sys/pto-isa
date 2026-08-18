/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Per-demo payload + address-resolution adaptor for the GridPipe A2/A3 backend.
//
// GridTPush.hpp / GridTPop.hpp forward-declare a handful of functions in
// `pto::a2a3_grid_payload` that have no header-level definition, because they
// need the concrete tile type and the HCCL device-context window layout that
// live in this demo (not in the public header tree).  We define them here once
// per kernel translation unit.  These are plain runtime helpers -- address
// plumbing and tile<->L1 adapters -- NOT an intrinsic layer: the actual cross-
// core handshake runs through the V8 CCE facades in grid_cce_intrinsic.hpp
// (copy_l1_to_peer_l1 / sync_hscb / wait_ipc_scb).
//
//   ResolvePeerSlotAddr(...)      -> resolve an address in our own window to the
//                                    same byte offset in peerBlockId's window (mock:
//                                    contiguous GM windows + CommRemotePtr).
//   RemoteScbPtr(...)             -> same, for a scoreboard word (sync_hscb dst).
//   StageTileToProducerSramSlot<T> -> copy the tile into the pipe's isolated
//                                     producer L1 staging slot.
//   CopyProducerSramToPeerSlot -> copy that local L1 source to the resolved
//                                     peer receive ring (V8 COPY_L1_TO_PEER).
//   CopyLocalSlotToTile<T>        -> drain this core's local GM slot into the
//                                    tile with the existing local copy (V8 TPOP
//                                    local read; no cross-core read of payload).
//   PopSlotIsLocal(...)           -> mock read-locality guard against GmSramArena.

#ifndef DISTRIBUTED_FFN_GRID_PAYLOAD_INL_HPP
#define DISTRIBUTED_FFN_GRID_PAYLOAD_INL_HPP

#include <cstdint>

#include <pto/npu/a2a3/grid_intrinsic.hpp>

#include "common.hpp" // CommRemotePtr, CommDeviceContext

namespace pto {
namespace a2a3_grid_payload {

// Resolve a GM address inside this core's window to the same byte offset in the
// window of logical block `peerBlockId`.  The host lays the per-cell windows out
// contiguously (windowsIn[i] == windowsIn[0] + i*winSize), so windowsIn is indexed
// by LOGICAL BLOCK ID and window i stands in for the private SRAM of core i; a
// cross-window offset is a cross-core write.  (CommDeviceContext calls the index a
// "rank" because it is a shared HCCL struct; on this single-device mesh there is no
// multi-card rank and the index is the block id.)
AICORE inline uint64_t ResolvePeerWindowAddress(__gm__ void* runtimeCtx, uint64_t localAddr, int peerBlockId)
{
    auto* ctx = reinterpret_cast<__gm__ CommDeviceContext*>(runtimeCtx);
    for (uint32_t i = 0; i < ctx->rankNum && i < HCCL_MAX_RANK_NUM; ++i) {
        uint64_t base = ctx->windowsIn[i];
        if (localAddr >= base && localAddr < base + ctx->winSize) {
            return ctx->windowsIn[peerBlockId] + (localAddr - base);
        }
    }
    return reinterpret_cast<uint64_t>(CommRemotePtr(ctx, reinterpret_cast<__gm__ void*>(localAddr), peerBlockId));
}

// View the demo's per-cell GM windows as the GmSramArena address-segment model of
// per-core SRAM.  The TPOP guard uses this as the single source of truth for
// "which core owns this address".
AICORE inline GmSramArena SramArenaFromCtx(__gm__ void* runtimeCtx)
{
    auto* ctx = reinterpret_cast<__gm__ CommDeviceContext*>(runtimeCtx);
    GmSramArena arena;
    arena.base = ctx->windowsIn[0];
    arena.segBytes = ctx->winSize;
    arena.numSegs = ctx->rankNum;
    return arena;
}

AICORE inline __gm__ uint8_t* ResolvePeerSlotAddr(__gm__ void* runtimeCtx, __gm__ uint8_t* localSlot, int peerBlockId)
{
    uint64_t peer = ResolvePeerWindowAddress(runtimeCtx, reinterpret_cast<uint64_t>(localSlot), peerBlockId);
    return reinterpret_cast<__gm__ uint8_t*>(peer);
}

AICORE inline __gm__ uint32_t* RemoteScbPtr(__gm__ void* runtimeCtx, __gm__ uint32_t* localScb, int peerBlockId)
{
    uint64_t remoteAddr = ResolvePeerWindowAddress(runtimeCtx, reinterpret_cast<uint64_t>(localScb), peerBlockId);
    return reinterpret_cast<__gm__ uint32_t*>(remoteAddr);
}

template <typename TileT>
__tf__ AICORE inline void StageTileToProducerSramSlot(__gm__ uint8_t* localProducerSlot, TileT& tile, int slotBytes)
{
    // The Tile API still exposes the compiler's historical __ubuf__ view.  Real
    // WSE physically backs it with unified L1; the mock makes that fact explicit
    // by first materialising the bytes in the pipe's dedicated producer range.
    auto* srcUb = reinterpret_cast<__ubuf__ uint8_t*>(__cce_get_tile_ptr(tile.data()));
#if defined(PTO_GRID_CCE_NATIVE)
    auto* dstUnifiedL1 = reinterpret_cast<__ubuf__ uint8_t*>(reinterpret_cast<uint64_t>(localProducerSlot));
    for (uint32_t i = 0; i < static_cast<uint32_t>(slotBytes); ++i) {
        dstUnifiedL1[i] = srcUb[i];
    }
#elif defined(__CPU_SIM)
    auto* dstBytes = reinterpret_cast<uint8_t*>(localProducerSlot);
    const auto* srcBytes = reinterpret_cast<const uint8_t*>(srcUb);
    for (uint32_t i = 0; i < static_cast<uint32_t>(slotBytes); ++i) {
        dstBytes[i] = srcBytes[i];
    }
#else
    constexpr uint32_t kChunkBytes = 256;
    const uint32_t bytes = static_cast<uint32_t>(slotBytes);
    uint32_t offset = 0;
    while (offset < bytes) {
        const uint32_t chunk = (bytes - offset > kChunkBytes) ? kChunkBytes : (bytes - offset);
        copy_ubuf_to_gm_align_b8(localProducerSlot + offset, srcUb + offset, 0, 1, chunk, 0, 0, 0, 0);
        offset += chunk;
    }
#endif
}

template <typename TileT>
__tf__ AICORE inline void StageTileToProducerSramSlot2D(
    __gm__ uint8_t* localProducerSlot, TileT& tile, uint32_t rowBytes, uint32_t rowCount, uint32_t tileStride,
    uint32_t producerStride)
{
    for (uint32_t r = 0; r < rowCount; ++r) {
        auto* srcUb = reinterpret_cast<__ubuf__ uint8_t*>(__cce_get_tile_ptr(tile.data()));
#if defined(PTO_GRID_CCE_NATIVE)
        auto* dstUnifiedL1 =
            reinterpret_cast<__ubuf__ uint8_t*>(reinterpret_cast<uint64_t>(localProducerSlot + r * producerStride));
        for (uint32_t i = 0; i < rowBytes; ++i) {
            dstUnifiedL1[i] = srcUb[r * tileStride + i];
        }
#elif defined(__CPU_SIM)
        auto* dstBytes = reinterpret_cast<uint8_t*>(localProducerSlot + r * producerStride);
        const auto* srcBytes = reinterpret_cast<const uint8_t*>(srcUb + r * tileStride);
        for (uint32_t i = 0; i < rowBytes; ++i) {
            dstBytes[i] = srcBytes[i];
        }
#else
        constexpr uint32_t kChunkBytes = 256;
        uint32_t offset = 0;
        while (offset < rowBytes) {
            const uint32_t chunk = (rowBytes - offset > kChunkBytes) ? kChunkBytes : (rowBytes - offset);
            copy_ubuf_to_gm_align_b8(
                localProducerSlot + r * producerStride + offset, srcUb + r * tileStride + offset, 0, 1, chunk, 0, 0, 0,
                0);
            offset += chunk;
        }
#endif
    }
}

template <typename TileT>
__tf__ AICORE inline void CopyProducerSramToPeerSlot(
    __gm__ uint8_t* dstPeerSlot, __gm__ uint8_t* localProducerSlot, TileT& transferScratch, int slotBytes)
{
    auto* scratchUb = reinterpret_cast<__ubuf__ void*>(__cce_get_tile_ptr(transferScratch.data()));
    copy_l1_to_peer_l1(
        reinterpret_cast<__gm__ void*>(dstPeerSlot), reinterpret_cast<__gm__ const void*>(localProducerSlot), scratchUb,
        static_cast<uint32_t>(slotBytes));
}

// A 2-D payload is staged using the receive slot's row layout, then copied one
// contiguous row at a time.  The source and destination strides are both L1
// strides; tileStride is used only to locate the mock DMA scratch row.
template <typename TileT>
__tf__ AICORE inline void CopyProducerSramToPeerSlot2D(
    __gm__ uint8_t* dstPeerSlot, __gm__ uint8_t* localProducerSlot, TileT& transferScratch, uint32_t rowBytes,
    uint32_t rowCount, uint32_t producerStride, uint32_t dstStride, uint32_t tileStride)
{
    auto* scratchUb = reinterpret_cast<__ubuf__ uint8_t*>(__cce_get_tile_ptr(transferScratch.data()));
    for (uint32_t r = 0; r < rowCount; ++r) {
        copy_l1_to_peer_l1(
            reinterpret_cast<__gm__ void*>(dstPeerSlot + r * dstStride),
            reinterpret_cast<__gm__ const void*>(localProducerSlot + r * producerStride),
            reinterpret_cast<__ubuf__ void*>(scratchUb + r * tileStride), rowBytes);
    }
}

template <typename TileT>
__tf__ AICORE inline void CopyLocalSlotToTile(TileT& tile, __gm__ uint8_t* localSlot, int slotBytes)
{
    // Consumer-side local-slot drain (V8 TPOP local read; no cross-core read of
    // payload): GM slot window -> local UB (tile), chunked with the existing
    // local copy.
    auto* dstUb = reinterpret_cast<__ubuf__ uint8_t*>(__cce_get_tile_ptr(tile.data()));
    constexpr uint32_t kChunkBytes = 256;
    uint32_t bytes = static_cast<uint32_t>(slotBytes);
    uint32_t offset = 0;
    while (offset < bytes) {
        uint32_t chunk = (bytes - offset > kChunkBytes) ? kChunkBytes : (bytes - offset);
        copy_gm_to_ubuf_align_b8(dstUb + offset, localSlot + offset, 0, 1, chunk, 0, 0, 0, 0);
        offset += chunk;
    }
}

// 2-D counterpart of CopyLocalSlotToTile: drain `rowCount` runs of `rowBytes`
// from the slot into the tile.  Strides are named by which buffer they walk, so
// the argument order matches the push helper even though the data flows the
// other way.
template <typename TileT>
__tf__ AICORE inline void CopyLocalSlotToTile2D(
    TileT& tile, __gm__ uint8_t* localSlot, uint32_t rowBytes, uint32_t rowCount, uint32_t slotStride,
    uint32_t tileStride)
{
    auto* dstUb = reinterpret_cast<__ubuf__ uint8_t*>(__cce_get_tile_ptr(tile.data()));
    constexpr uint32_t kChunkBytes = 256;
    for (uint32_t r = 0; r < rowCount; ++r) {
        __ubuf__ uint8_t* rowDst = dstUb + r * tileStride;
        __gm__ uint8_t* rowSrc = localSlot + r * slotStride;
        uint32_t offset = 0;
        while (offset < rowBytes) {
            uint32_t chunk = (rowBytes - offset > kChunkBytes) ? kChunkBytes : (rowBytes - offset);
            copy_gm_to_ubuf_align_b8(rowDst + offset, rowSrc + offset, 0, 1, chunk, 0, 0, 0, 0);
            offset += chunk;
        }
    }
}

// Extract a tile's UB pointer (__ubuf__ void*) for the unified group-collective
// CCE intrinsic (mov_ubuf_group), which takes a raw __ubuf__ pointer (the dtype
// travels as the runtime `eltype` operand rather than a tile object).  Mirrors
// the extraction inside the producer-stage / local-drain helpers,
// factored out so the public GridTBroadcast.hpp / GridTReduce.hpp facades (which
// are tile-agnostic and cannot call __cce_get_tile_ptr directly) can hand the
// intrinsic a UB pointer.
template <typename TileT>
__tf__ AICORE inline __ubuf__ void* TileUbPtr(TileT& tile)
{
    return reinterpret_cast<__ubuf__ void*>(__cce_get_tile_ptr(tile.data()));
}

// Source word for the group collectives' atomic-add credit (grid_cce_intrinsic.hpp
// atom_add_hscb).  A2/A3 has no scalar cross-core atomic, so the accumulate rides
// MTE3's atomic-add mode, which needs a LOCAL UB word to send; native's
// ATOM_ADD_HSCB carries the delta as an operand and never calls this.
//
// The address is a fixed reservation at the very top of the 192 KB UB, chosen
// because every kernel in this demo lays its tiles out from 0 upward and the
// highest one ends far below it.  A kernel that grows past 0x2FF00 must move this
// (or its tiles): the collision would be silent, since a stray +1 landing in a
// tile is just data corruption.
constexpr int kGridCreditScratchUbOffset = 0x2FF00; // 192 KB - 256 B

AICORE inline __ubuf__ uint32_t* GridCreditScratchUb()
{
    return reinterpret_cast<__ubuf__ uint32_t*>(static_cast<uint64_t>(kGridCreditScratchUbOffset));
}

AICORE inline bool PopSlotIsLocal(__gm__ void* runtimeCtx, __gm__ uint8_t* localSlot, uint32_t bytes, int callerBlockId)
{
    // Enforce the NoC "TPOP only pops local SRAM" rule: the read is legal only
    // when the whole slot lies inside the caller core's own arena segment.
    GmSramArena arena = SramArenaFromCtx(runtimeCtx);
    return arena.InSegment(callerBlockId, reinterpret_cast<uint64_t>(localSlot), static_cast<uint64_t>(bytes));
}

} // namespace a2a3_grid_payload
} // namespace pto

#endif // DISTRIBUTED_FFN_GRID_PAYLOAD_INL_HPP
