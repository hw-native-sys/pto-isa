/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Per-demo payload + neighbor-SRAM adaptor for the GridPipe A2/A3 backend.
//
// GridTPush.hpp / GridTPop.hpp declare three forward functions in
// `pto::a2a3_grid_payload` that intentionally have no header-level definition,
// because they need a concrete tile type and the HCCL device context layout
// that lives in this demo (not in the public header tree).  We provide those
// definitions here once per kernel translation unit.
//
//   get_neighbor_sram_addr(...)
//     -> resolves an address in our own SRAM window to the same byte offset in
//        peerRank's window.  On A2/A3 this SRAM window is mocked by local GM
//        windows and HcclRemotePtr.
//
//   CopyTileToNeighborUbufSlot<TileT>(remoteSlot, tile, slotBytes)
//     -> Tile-to-intrinsic adapter: extract the Tile UB pointer, then call
//        copy_ubuf_to_neighbor_ubuf(...).
//
//   CopyNeighborUbufSlotToTile<TileT>(tile, localSlot, slotBytes)
//     -> Tile-to-intrinsic adapter: extract the Tile UB pointer, then call
//        copy_neighbor_ubuf_to_ubuf(...).
//
// The skeleton implementation converts Tile descriptors to UB pointers, then
// lets grid_sram_intrinsic.hpp choose native builtin vs A2/A3 mock lowering.
// Keeping this adapter outside GridTPush/GridTPop preserves the generic payload
// protocol while this demo owns the concrete PTO Tile representation.

#ifndef DISTRIBUTED_FFN_GRID_PAYLOAD_INL_HPP
#define DISTRIBUTED_FFN_GRID_PAYLOAD_INL_HPP

#include <cstdint>

#include <pto/common/grid_sram_intrinsic.hpp>

#include "common.hpp" // HcclRemotePtr, HcclDeviceContext

namespace pto {
namespace a2a3_grid_payload {

AICORE inline uint64_t ResolvePeerWindowAddress(__gm__ void *runtimeCtx, uint64_t localAddr, int peerRank)
{
    auto *ctx = reinterpret_cast<__gm__ HcclDeviceContext *>(runtimeCtx);
    for (uint32_t i = 0; i < ctx->rankNum && i < HCCL_MAX_RANK_NUM; ++i) {
        uint64_t base = ctx->windowsIn[i];
        if (localAddr >= base && localAddr < base + ctx->winSize) {
            return ctx->windowsIn[peerRank] + (localAddr - base);
        }
    }
    return reinterpret_cast<uint64_t>(HcclRemotePtr(ctx, reinterpret_cast<__gm__ void *>(localAddr), peerRank));
}

AICORE inline neighbor_sram_addr LocalSramAddr(__gm__ uint8_t *localSlot)
{
    return neighbor_sram_addr{reinterpret_cast<uint64_t>(localSlot)};
}

AICORE inline neighbor_ubuf_addr LocalUbufAddr(__gm__ uint8_t *localSlot)
{
    return neighbor_ubuf_addr{reinterpret_cast<uint64_t>(localSlot)};
}

AICORE inline __gm__ uint32_t *RemoteCounterPtr(__gm__ void *runtimeCtx, __gm__ uint32_t *localCounter, int peerRank)
{
    uint64_t remoteAddr = ResolvePeerWindowAddress(runtimeCtx, reinterpret_cast<uint64_t>(localCounter), peerRank);
    return reinterpret_cast<__gm__ uint32_t *>(remoteAddr);
}

template <typename TileT>
__tf__ AICORE inline void CopyTileToNeighborUbufSlot(neighbor_ubuf_addr remoteSlot, TileT &tile, int slotBytes)
{
    __ubuf__ void *src = reinterpret_cast<__ubuf__ void *>(__cce_get_tile_ptr(tile.data()));
    // Producer-side cross-core payload transfer in CCE-intrinsic form.
    copy_ubuf_to_neighbor_ubuf(remoteSlot, src, static_cast<uint32_t>(slotBytes), 0);
}

template <typename TileT>
__tf__ AICORE inline void CopyNeighborUbufSlotToTile(TileT &tile, neighbor_ubuf_addr localSlot, int slotBytes)
{
    __ubuf__ void *dst = reinterpret_cast<__ubuf__ void *>(__cce_get_tile_ptr(tile.data()));
    // Consumer-side counterpart of the same intrinsic-style payload transfer.
    copy_neighbor_ubuf_to_ubuf(dst, localSlot, static_cast<uint32_t>(slotBytes), 0);
}

} // namespace a2a3_grid_payload

namespace grid_sram_mock {

AICORE inline uint64_t MockGetNeighborSramAddr(__gm__ void *runtimeCtx, uint64_t localSramAddr, int32_t peerRank)
{
    return a2a3_grid_payload::ResolvePeerWindowAddress(runtimeCtx, localSramAddr, peerRank);
}

AICORE inline void MockCopyUbufToNeighborUbuf(neighbor_ubuf_addr dst, __ubuf__ void *src, uint32_t bytes,
                                              uint64_t config)
{
    (void)config;
    // A2/A3 mock lowering of copy_ubuf_to_neighbor_ubuf:
    // UB(local core) -> GM-backed peer slot window.
    constexpr uint32_t kChunkBytes = 256;
    auto *dstBytes = reinterpret_cast<__gm__ uint8_t *>(dst.value);
    auto *srcBytes = reinterpret_cast<__ubuf__ uint8_t *>(src);
    uint32_t offset = 0;
    while (offset < bytes) {
        uint32_t chunk = (bytes - offset > kChunkBytes) ? kChunkBytes : (bytes - offset);
        copy_ubuf_to_gm_align_b8(dstBytes + offset, srcBytes + offset, 0, 1, chunk, 0, 0, 0, 0);
        offset += chunk;
    }
}

AICORE inline void MockCopyUbufToNeighborCbuf(neighbor_cbuf_addr dst, __ubuf__ void *src, uint32_t bytes,
                                              uint64_t config)
{
    (void)config;
    constexpr uint32_t kChunkBytes = 256;
    auto *dstBytes = reinterpret_cast<__gm__ uint8_t *>(dst.value);
    auto *srcBytes = reinterpret_cast<__ubuf__ uint8_t *>(src);
    uint32_t offset = 0;
    while (offset < bytes) {
        uint32_t chunk = (bytes - offset > kChunkBytes) ? kChunkBytes : (bytes - offset);
        copy_ubuf_to_gm_align_b8(dstBytes + offset, srcBytes + offset, 0, 1, chunk, 0, 0, 0, 0);
        offset += chunk;
    }
}

AICORE inline void MockCopyCbufToNeighborUbuf(neighbor_ubuf_addr dst, __cbuf__ void *src, uint32_t bytes,
                                              uint64_t config)
{
    (void)config;
    copy_cbuf_to_gm(reinterpret_cast<__gm__ uint8_t *>(dst.value), reinterpret_cast<__cbuf__ uint8_t *>(src), 0, 1,
                    static_cast<uint16_t>(bytes), 0, 0);
}

AICORE inline void MockCopyCbufToNeighborCbuf(neighbor_cbuf_addr dst, __cbuf__ void *src, uint32_t bytes,
                                              uint64_t config)
{
    (void)config;
    copy_cbuf_to_gm(reinterpret_cast<__gm__ uint8_t *>(dst.value), reinterpret_cast<__cbuf__ uint8_t *>(src), 0, 1,
                    static_cast<uint16_t>(bytes), 0, 0);
}

AICORE inline void MockCopyNeighborUbufToUbuf(__ubuf__ void *dst, neighbor_ubuf_addr src, uint32_t bytes,
                                              uint64_t config)
{
    (void)config;
    // A2/A3 mock lowering of copy_neighbor_ubuf_to_ubuf:
    // GM-backed local slot window -> UB(local core).
    constexpr uint32_t kChunkBytes = 256;
    auto *dstBytes = reinterpret_cast<__ubuf__ uint8_t *>(dst);
    auto *srcBytes = reinterpret_cast<__gm__ uint8_t *>(src.value);
    uint32_t offset = 0;
    while (offset < bytes) {
        uint32_t chunk = (bytes - offset > kChunkBytes) ? kChunkBytes : (bytes - offset);
        copy_gm_to_ubuf_align_b8(dstBytes + offset, srcBytes + offset, 0, 1, chunk, 0, 0, 0, 0);
        offset += chunk;
    }
}

} // namespace grid_sram_mock
} // namespace pto

#endif // DISTRIBUTED_FFN_GRID_PAYLOAD_INL_HPP
