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
// plumbing and tile<->UB adapters -- NOT an intrinsic layer: the actual cross-
// core handshake runs through the V8 CCE facades in grid_cce_intrinsic.hpp
// (copy_ubuf_to_neighbor_ubuf / sync_hscb / wait_ipc_scb).
//
//   ResolvePeerSlotAddr(...)      -> resolve an address in our own window to the
//                                    same byte offset in peerBlockId's window (mock:
//                                    contiguous GM windows + CommRemotePtr).
//   RemoteScbPtr(...)             -> same, for a scoreboard word (sync_hscb dst).
//   CopyTileToNeighborSramSlot<T> -> extract the tile UB pointer, then call the
//                                    copy_ubuf_to_neighbor_ubuf facade (V8
//                                    COPY_UBUF_TO_NBR).
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

// Resolve a GM address inside this block's window to the same byte offset in
// the window of the block `peerBlockId` names.  The host lays the per-cell
// windows out contiguously (windowsIn[i] == windowsIn[0] + i*winSize) and keys
// them by LOGICAL BLOCK ID, so each window stands in for the private SRAM of
// block i; a cross-window offset is a cross-core write.  The ctx field is
// spelled `rankNum` because it is the HCCL struct -- here every one of those is
// a block of this single launch, there is no second device.
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
__tf__ AICORE inline void CopyTileToNeighborSramSlot(__gm__ uint8_t* dstNeighborSlot, TileT& tile, int slotBytes)
{
    // Producer-side cross-core payload write: extract the tile UB pointer, then
    // call the copy_ubuf_to_neighbor_ubuf CCE facade (V8 COPY_UBUF_TO_NBR).
    auto* srcUb = reinterpret_cast<__ubuf__ uint8_t*>(__cce_get_tile_ptr(tile.data()));
    copy_ubuf_to_neighbor_ubuf(
        reinterpret_cast<__gm__ void*>(dstNeighborSlot), reinterpret_cast<__ubuf__ void*>(srcUb),
        static_cast<uint32_t>(slotBytes));
}

// 2-D form of the above: move `rowCount` runs of `rowBytes`, striding the tile by
// `srcStride` and the slot by `dstStride`.  COPY_UBUF_TO_NBR carries a single
// `bytes` operand, so a 2-D sub-block lowers to one burst per row; the caller
// (GRID_TRY_TPUSH_IMPL) still fires exactly ONE ready doorbell afterwards, so the
// handshake cost per TPUSH does not change.  rowCount == 1 is the 1-D case.
template <typename TileT>
__tf__ AICORE inline void CopyTileToNeighborSramSlot2D(
    __gm__ uint8_t* dstNeighborSlot, TileT& tile, uint32_t rowBytes, uint32_t rowCount, uint32_t tileStride,
    uint32_t slotStride)
{
    auto* srcUb = reinterpret_cast<__ubuf__ uint8_t*>(__cce_get_tile_ptr(tile.data()));
    for (uint32_t r = 0; r < rowCount; ++r) {
        copy_ubuf_to_neighbor_ubuf(
            reinterpret_cast<__gm__ void*>(dstNeighborSlot + r * slotStride),
            reinterpret_cast<__ubuf__ void*>(srcUb + r * tileStride), rowBytes);
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
// the extraction inside CopyTileToNeighborSramSlot / CopyLocalSlotToTile,
// factored out so the public GridTBroadcast.hpp / GridTReduce.hpp facades (which
// are tile-agnostic and cannot call __cce_get_tile_ptr directly) can hand the
// intrinsic a UB pointer.
template <typename TileT>
__tf__ AICORE inline __ubuf__ void* TileUbPtr(TileT& tile)
{
    return reinterpret_cast<__ubuf__ void*>(__cce_get_tile_ptr(tile.data()));
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
