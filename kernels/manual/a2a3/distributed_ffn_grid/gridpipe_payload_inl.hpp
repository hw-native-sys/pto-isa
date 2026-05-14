/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Per-demo payload + remote-ptr adaptor for the GridPipe A2/A3 backend.
//
// GridTPush.hpp / GridTPop.hpp declare three forward functions in
// `pto::a2a3_grid_payload` that intentionally have no header-level definition,
// because they need a concrete tile type and the HCCL device context layout
// that lives in this demo (not in the public header tree).  We provide those
// definitions here once per kernel translation unit.
//
//   RemotePtr<T>(ctx, localPtr, peerRank)
//     -> resolves a pointer into our own GM window to the same byte offset in
//        peerRank's window via HcclRemotePtr.  Reused by both push and pop for
//        the cross-rank ready/free flag writes.
//
//   CopyTileToGmSlot<TileT>(remoteSlot, tile, slotBytes)
//     -> moves a UB-resident tile into a GM slot in the *neighbor's* window.
//        Implements design doc 5.3 step 3 producer side: `tmov [r_slot], tile`.
//
//   CopyGmSlotToTile<TileT>(tile, localSlot, slotBytes)
//     -> moves bytes from a GM slot in our own window into a UB-resident tile.
//        Implements design doc 5.3 step 3 consumer side.
//
// The skeleton implementation uses AscendC byte-level DataCopy through
// GlobalTensor/LocalTensor in/out of the tile's underlying address.  Real
// silicon LPU WSE would issue a single TMOV against the slot region.

#ifndef DISTRIBUTED_FFN_GRID_PAYLOAD_INL_HPP
#define DISTRIBUTED_FFN_GRID_PAYLOAD_INL_HPP

#include <cstdint>

#include "common.hpp" // HcclRemotePtr, HcclDeviceContext

namespace pto {
namespace a2a3_grid_payload {

// ---------------------------------------------------------------------------
// RemotePtr: trivial wrapper around HcclRemotePtr, but keeps the GridTPush /
// GridTPop callsites isolated from the HCCL header (they only see the void*
// runtimeCtx).
// ---------------------------------------------------------------------------
template <typename PtrT>
AICORE inline PtrT *RemotePtr(__gm__ void *runtimeCtx, PtrT *localPtr, int peerRank)
{
    auto *ctx = reinterpret_cast<__gm__ HcclDeviceContext *>(runtimeCtx);
    uint64_t localAddr = reinterpret_cast<uint64_t>(localPtr);
    for (uint32_t i = 0; i < ctx->rankNum && i < HCCL_MAX_RANK_NUM; ++i) {
        uint64_t base = ctx->windowsIn[i];
        if (localAddr >= base && localAddr < base + ctx->winSize) {
            return reinterpret_cast<__gm__ PtrT *>(ctx->windowsIn[peerRank] + (localAddr - base));
        }
    }
    return HcclRemotePtr(ctx, localPtr, peerRank);
}

// ---------------------------------------------------------------------------
// Payload hooks.
//
// M3-T1: wire to TSTORE / TLOAD against a 5D ND GlobalTensor view of the
// remote / local slot.  Tile must be UB-resident (caller TASSIGNs it before
// the hook fires).  The Shape/Stride convention matches topk_kernel.cpp and
// allgather_gemm_compute_kernel.cpp: dim0..2 = whole-tile element count,
// dim3 = Cols, dim4 = 1.
// ---------------------------------------------------------------------------
template <typename TileT>
AICORE inline void CopyTileToGmSlot(__gm__ uint8_t *remoteSlot, TileT &tile, int slotBytes)
{
    using Elem = typename TileT::DType;
    static constexpr int kRows = TileT::Rows;
    static constexpr int kCols = TileT::Cols;
    static constexpr int kNumel = kRows * kCols;
    using ShapeT = pto::Shape<1, 1, 1, kRows, kCols>;
    using StrideT = pto::Stride<kNumel, kNumel, kNumel, kCols, 1>;
    using GT = pto::GlobalTensor<Elem, ShapeT, StrideT, pto::Layout::ND>;
    GT dstG(reinterpret_cast<__gm__ Elem *>(remoteSlot));
    TSTORE(dstG, tile);
    (void)slotBytes;
}

template <typename TileT>
AICORE inline void CopyGmSlotToTile(TileT &tile, __gm__ uint8_t *localSlot, int slotBytes)
{
    using Elem = typename TileT::DType;
    static constexpr int kRows = TileT::Rows;
    static constexpr int kCols = TileT::Cols;
    static constexpr int kNumel = kRows * kCols;
    using ShapeT = pto::Shape<1, 1, 1, kRows, kCols>;
    using StrideT = pto::Stride<kNumel, kNumel, kNumel, kCols, 1>;
    using GT = pto::GlobalTensor<Elem, ShapeT, StrideT, pto::Layout::ND>;
    GT srcG(reinterpret_cast<__gm__ Elem *>(localSlot));
    TLOAD(tile, srcG);
    (void)slotBytes;
}

} // namespace a2a3_grid_payload
} // namespace pto

#endif // DISTRIBUTED_FFN_GRID_PAYLOAD_INL_HPP
