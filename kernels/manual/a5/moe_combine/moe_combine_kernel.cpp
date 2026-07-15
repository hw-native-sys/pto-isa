/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>

#ifndef PIPE_FIX
#define PIPE_FIX static_cast<pipe_t>(10)
#endif

#include <pto/comm/pto_comm_inst.hpp>
#include <pto/pto-inst.hpp>

#include "common.h"
#include "kernel_launchers.h"

using moe_combine::CombineRouteMetaLayout;
using moe_combine::CommDeviceContext;
using moe_combine::MoeCombineShape;
using moe_combine::PeerWindowLayout;
using moe_combine::WorkspaceLayout;

#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t*
#endif

namespace {

constexpr int kDefaultTileCols = static_cast<int>(moe_combine::kMoeCombineTileCols);
constexpr uint32_t kRouteCacheMax = 16;
constexpr uint64_t kPingUbAddr = 0x0;
constexpr uint64_t kPongUbAddr = 0x1000;
constexpr uint64_t kSoftSyncUbAddr = 0x5000;

using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;

template <typename T>
using GlobalNd = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

template <typename T, int kCols = kDefaultTileCols>
using VecTile = pto::Tile<pto::TileType::Vec, T, 1, kCols, pto::BLayout::RowMajor, -1, -1>;

AICORE inline uint64_t Align64Device(uint64_t value) { return ((value + 63) / 64) * 64; }

AICORE inline uint64_t AppendFieldDevice(uint64_t& offset, uint64_t bytes)
{
    offset = Align64Device(offset);
    uint64_t fieldOffset = offset;
    offset += bytes;
    return fieldOffset;
}

struct LocalWorkspaceView {
    GM_ADDR base;
    __gm__ int32_t* localSync;
};

struct LocalRouteMetaView {
    GM_ADDR base;
    __gm__ int32_t* peerTokenPerExpert;
    __gm__ int32_t* expandedRowIdx;
    __gm__ int32_t* cumsumPerExpert;
    __gm__ int32_t* dispatchOffset;
    __gm__ int32_t* prevSumBeforeRank;
};

struct LocalPeerWindowView {
    GM_ADDR base;
    __gm__ half* ptrD;
    __gm__ int32_t* countReadySignal;
    __gm__ int32_t* combineDoneSignal;
};

AICORE inline WorkspaceLayout MakeWorkspaceLayout(MoeCombineShape shape)
{
    const uint64_t i32 = 4;
    uint64_t expertNumPadded = ((static_cast<uint64_t>(shape.expertNum) + moe_combine::kMoeCombineMetadataPad - 1) /
                                moe_combine::kMoeCombineMetadataPad) *
                               moe_combine::kMoeCombineMetadataPad;
    uint64_t aivBlocks = shape.aivBlocks == 0 ? 1 : shape.aivBlocks;
    uint64_t offset = 0;

    WorkspaceLayout layout{};
    uint64_t syncSlots = aivBlocks * (8 + expertNumPadded);
    syncSlots = syncSlots < 64 ? 64 : syncSlots;
    layout.localSync = AppendFieldDevice(offset, syncSlots * i32);
    layout.totalBytes = Align64Device(offset);
    return layout;
}

AICORE inline CombineRouteMetaLayout MakeCombineRouteMetaLayout(MoeCombineShape shape)
{
    const uint64_t i32 = 4;
    uint64_t expertNumPadded = ((static_cast<uint64_t>(shape.expertNum) + moe_combine::kMoeCombineMetadataPad - 1) /
                                moe_combine::kMoeCombineMetadataPad) *
                               moe_combine::kMoeCombineMetadataPad;
    uint64_t expandedRows = static_cast<uint64_t>(shape.m) * shape.topK;
    uint64_t offset = 0;

    CombineRouteMetaLayout layout{};
    layout.peerTokenPerExpert = AppendFieldDevice(offset, static_cast<uint64_t>(shape.ep) * expertNumPadded * i32);
    layout.expandedRowIdx = AppendFieldDevice(offset, expandedRows * i32);
    layout.cumsumPerExpert = AppendFieldDevice(offset, static_cast<uint64_t>(shape.ep) * expertNumPadded * i32);
    layout.dispatchOffset = AppendFieldDevice(offset, static_cast<uint64_t>(shape.expertPerRank) * i32);
    layout.prevSumBeforeRank = AppendFieldDevice(offset, static_cast<uint64_t>(shape.ep) * shape.expertPerRank * i32);
    layout.totalBytes = Align64Device(offset);
    return layout;
}

AICORE inline PeerWindowLayout MakePeerWindowLayout(MoeCombineShape shape)
{
    const uint64_t i32 = 4;
    const uint64_t f16 = 2;
    uint64_t expandedRows = static_cast<uint64_t>(shape.m) * shape.topK;
    uint64_t offset = 0;

    PeerWindowLayout layout{};
    layout.ptrD = AppendFieldDevice(offset, expandedRows * shape.k * f16);
    layout.countReadySignal = AppendFieldDevice(offset, static_cast<uint64_t>(shape.ep) * i32);
    layout.combineDoneSignal = AppendFieldDevice(offset, static_cast<uint64_t>(shape.ep) * i32);
    layout.totalBytes = Align64Device(offset);
    return layout;
}

AICORE inline LocalWorkspaceView MakeLocalWorkspaceView(GM_ADDR workspaceBase, const WorkspaceLayout& layout)
{
    LocalWorkspaceView view{};
    view.base = workspaceBase;
    view.localSync = reinterpret_cast<__gm__ int32_t*>(workspaceBase + layout.localSync);
    return view;
}

AICORE inline LocalRouteMetaView MakeLocalRouteMetaView(GM_ADDR routeMetaBase, const CombineRouteMetaLayout& layout)
{
    LocalRouteMetaView view{};
    view.base = routeMetaBase;
    view.peerTokenPerExpert = reinterpret_cast<__gm__ int32_t*>(routeMetaBase + layout.peerTokenPerExpert);
    view.expandedRowIdx = reinterpret_cast<__gm__ int32_t*>(routeMetaBase + layout.expandedRowIdx);
    view.cumsumPerExpert = reinterpret_cast<__gm__ int32_t*>(routeMetaBase + layout.cumsumPerExpert);
    view.dispatchOffset = reinterpret_cast<__gm__ int32_t*>(routeMetaBase + layout.dispatchOffset);
    view.prevSumBeforeRank = reinterpret_cast<__gm__ int32_t*>(routeMetaBase + layout.prevSumBeforeRank);
    return view;
}

AICORE inline LocalPeerWindowView MakeLocalPeerWindowView(GM_ADDR peerWindowBase, const PeerWindowLayout& layout)
{
    LocalPeerWindowView view{};
    view.base = peerWindowBase;
    view.ptrD = reinterpret_cast<__gm__ half*>(peerWindowBase + layout.ptrD);
    view.countReadySignal = reinterpret_cast<__gm__ int32_t*>(peerWindowBase + layout.countReadySignal);
    view.combineDoneSignal = reinterpret_cast<__gm__ int32_t*>(peerWindowBase + layout.combineDoneSignal);
    return view;
}

template <typename T>
AICORE inline __gm__ T* RemotePtr(__gm__ CommDeviceContext* ctx, __gm__ T* localPtr, uint32_t peerRank)
{
    uint64_t localBase = ctx->windowsIn[ctx->rankId];
    uint64_t offset = reinterpret_cast<uint64_t>(localPtr) - localBase;
    return reinterpret_cast<__gm__ T*>(ctx->windowsIn[peerRank] + offset);
}

AICORE inline LocalPeerWindowView MakeRemotePeerWindowView(
    __gm__ CommDeviceContext* ctx, GM_ADDR localPeerWindowBase, uint32_t peerRank, const PeerWindowLayout& layout)
{
    GM_ADDR remoteBase = RemotePtr<uint8_t>(ctx, localPeerWindowBase, peerRank);
    return MakeLocalPeerWindowView(remoteBase, layout);
}

template <typename T>
AICORE inline GlobalNd<T> MakeGlobal1D(__gm__ T* ptr, int32_t elems)
{
    ShapeDyn shape(1, 1, 1, 1, elems);
    StrideDyn stride(elems, elems, elems, elems, 1);
    return GlobalNd<T>(ptr, shape, stride);
}

template <typename T>
AICORE inline GlobalNd<T> MakeGlobal2D(__gm__ T* ptr, int32_t rows, int32_t cols, int32_t rowStride)
{
    ShapeDyn shape(1, 1, 1, rows, cols);
    StrideDyn stride(rowStride * rows, rowStride * rows, rowStride * rows, rowStride, 1);
    return GlobalNd<T>(ptr, shape, stride);
}

AICORE inline pto::comm::Signal MakeSignal(__gm__ int32_t* ptr) { return pto::comm::Signal(ptr); }

AICORE inline void WaitStoreTileReusable()
{
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
}

AICORE inline void DcciGmRangeNoFence(__gm__ void* ptr, uint64_t bytes)
{
    if (bytes == 0) {
        return;
    }
    constexpr uint64_t cacheLineBytes = 64;
    uint64_t start = reinterpret_cast<uint64_t>(ptr) & ~(cacheLineBytes - 1);
    uint64_t end = (reinterpret_cast<uint64_t>(ptr) + bytes + cacheLineBytes - 1) & ~(cacheLineBytes - 1);
    for (uint64_t addr = start; addr < end; addr += cacheLineBytes) {
        dcci(reinterpret_cast<__gm__ void*>(addr), SINGLE_CACHE_LINE);
    }
}

AICORE inline void DcciGmRange(__gm__ void* ptr, uint64_t bytes)
{
    if (bytes == 0) {
        return;
    }
    DcciGmRangeNoFence(ptr, bytes);
    dsb(DSB_DDR);
}

AICORE inline void AcquireGmRangeBeforeRead(__gm__ void* ptr, uint64_t bytes)
{
    pipe_barrier(PIPE_ALL);
    DcciGmRange(ptr, bytes);
}

AICORE inline uint32_t ExpertNumPaddedDevice(MoeCombineShape shape)
{
    return ((shape.expertNum + moe_combine::kMoeCombineMetadataPad - 1) / moe_combine::kMoeCombineMetadataPad) *
           moe_combine::kMoeCombineMetadataPad;
}

AICORE inline uint32_t TokenShardBegin(uint32_t totalTokens, uint32_t blockId, uint32_t blockNum)
{
    if (blockNum == 0) {
        return 0;
    }
    uint32_t base = totalTokens / blockNum;
    uint32_t rem = totalTokens % blockNum;
    return blockId * base + (blockId < rem ? blockId : rem);
}

AICORE inline uint32_t TokenShardEnd(uint32_t totalTokens, uint32_t blockId, uint32_t blockNum)
{
    return TokenShardBegin(totalTokens, blockId + 1, blockNum);
}

AICORE inline void SoftSyncAiv(__gm__ int32_t* gmWorkspace, uint32_t blockNum)
{
    pto::Tile<pto::TileType::Vec, int32_t, 1, pto::SYNCALL_SOFT_SLOT_INT32, pto::BLayout::RowMajor, -1, -1> syncTile(
        1, pto::SYNCALL_SOFT_SLOT_INT32);
#ifndef __PTO_AUTO__
    syncTile.data() = reinterpret_cast<__ubuf__ int32_t*>(kSoftSyncUbAddr);
#endif
    GlobalNd<int32_t> syncGlobal =
        MakeGlobal1D(gmWorkspace, static_cast<int32_t>(blockNum * pto::SYNCALL_SOFT_SLOT_INT32));
    pto::SYNCALL<pto::SyncAllMode::Soft>(syncGlobal, syncTile, static_cast<int32_t>(blockNum));
}

AICORE inline uint32_t EffectiveRowChunk(MoeCombineShape shape)
{
    (void)shape;
    return moe_combine::kMoeCombineRowChunk;
}

AICORE inline void WaitCombinePhase(
    MoeCombineShape shape, LocalPeerWindowView localPeer, uint32_t blockId, uint32_t blockNum, int32_t value)
{
    if (blockNum == 0) {
        return;
    }
    for (uint32_t peer = blockId; peer < shape.ep; peer += blockNum) {
        pto::comm::Signal sig = MakeSignal(localPeer.combineDoneSignal + peer);
        pto::comm::TWAIT(sig, value, pto::comm::WaitCmp::GE);
    }
}

struct ReturnSegment {
    uint32_t src;
    uint32_t localExpert;
    uint32_t globalExpert;
    int32_t rows;
    int32_t srcStart;
    int32_t dstStart;
    uint32_t chunkCount;
};

struct ReturnContext {
    MoeCombineShape shape;
    LocalRouteMetaView routeMeta;
    LocalPeerWindowView localPeer;
    __gm__ CommDeviceContext* ctx;
    GM_ADDR peerWindow;
    __gm__ half* localExpertOutput;
    uint32_t myRank;
    uint32_t expertNumPadded;
    uint32_t rowChunk;
    const PeerWindowLayout* peerWindowLayout;
};

struct ReturnChunk {
    uint32_t rowBegin;
    uint32_t rowsThisChunk;
};

struct TileCopyRange {
    int32_t srcRow;
    int32_t dstRow;
    int32_t col;
    int32_t cols;
};

struct RestoreRouteCache {
    int32_t rows[kRouteCacheMax];
    float probs[kRouteCacheMax];
    uint32_t count;
    bool enabled;
};

struct RestoreTileContext {
    MoeCombineShape shape;
    LocalRouteMetaView routeMeta;
    LocalPeerWindowView localPeer;
    __gm__ float* probValues;
    __gm__ half* output;
    uint32_t token;
    int32_t col;
    int32_t cols;
    const RestoreRouteCache* cache;
};

AICORE inline void AcquireReturnMetadata(
    const MoeCombineShape& shape, LocalRouteMetaView routeMeta, uint32_t expertNumPadded)
{
    AcquireGmRangeBeforeRead(
        routeMeta.peerTokenPerExpert, static_cast<uint32_t>(shape.ep * expertNumPadded * sizeof(int32_t)));
    AcquireGmRangeBeforeRead(
        routeMeta.cumsumPerExpert, static_cast<uint32_t>(shape.ep * expertNumPadded * sizeof(int32_t)));
    AcquireGmRangeBeforeRead(routeMeta.dispatchOffset, static_cast<uint32_t>(shape.expertPerRank * sizeof(int32_t)));
    AcquireGmRangeBeforeRead(
        routeMeta.prevSumBeforeRank, static_cast<uint32_t>(shape.ep * shape.expertPerRank * sizeof(int32_t)));
}

AICORE inline bool LoadReturnSegment(const ReturnContext& ctx, uint32_t segment, ReturnSegment* out)
{
    out->src = segment / ctx.shape.expertPerRank;
    out->localExpert = segment % ctx.shape.expertPerRank;
    out->globalExpert = ctx.myRank * ctx.shape.expertPerRank + out->localExpert;
    out->rows =
        *(ctx.routeMeta.peerTokenPerExpert + static_cast<uint64_t>(out->src) * ctx.expertNumPadded + out->globalExpert);
    if (out->rows <= 0) {
        return false;
    }
    out->srcStart = *(ctx.routeMeta.dispatchOffset + out->localExpert) +
                    *(ctx.routeMeta.prevSumBeforeRank + static_cast<uint64_t>(out->src) * ctx.shape.expertPerRank +
                      out->localExpert);
    out->dstStart = out->globalExpert == 0 ?
                        0 :
                        *(ctx.routeMeta.cumsumPerExpert + static_cast<uint64_t>(out->src) * ctx.expertNumPadded +
                          out->globalExpert - 1);
    out->chunkCount = (static_cast<uint32_t>(out->rows) + ctx.rowChunk - 1) / ctx.rowChunk;
    return true;
}

AICORE inline ReturnChunk MakeReturnChunk(const ReturnContext& ctx, const ReturnSegment& segment, uint32_t chunk)
{
    ReturnChunk out{};
    out.rowBegin = chunk * ctx.rowChunk;
    out.rowsThisChunk = static_cast<uint32_t>(segment.rows) - out.rowBegin;
    out.rowsThisChunk = out.rowsThisChunk < ctx.rowChunk ? out.rowsThisChunk : ctx.rowChunk;
    return out;
}

AICORE inline void CopyLocalTileToPeerWindow(const ReturnContext& ctx, const TileCopyRange& range)
{
    VecTile<half, kDefaultTileCols> ping(1, range.cols);
    VecTile<half, kDefaultTileCols> pong(1, range.cols);
    TASSIGN(ping, kPingUbAddr);
    TASSIGN(pong, kPongUbAddr);
    VecTile<half, kDefaultTileCols>& tile = ((range.col / kDefaultTileCols) & 1) == 0 ? ping : pong;
    event_t event = ((range.col / kDefaultTileCols) & 1) == 0 ? EVENT_ID0 : EVENT_ID1;
    GlobalNd<half> srcGlobal = MakeGlobal2D(
        ctx.localExpertOutput + static_cast<int64_t>(range.srcRow) * static_cast<int32_t>(ctx.shape.k) + range.col, 1,
        range.cols, static_cast<int32_t>(ctx.shape.k));
    GlobalNd<half> dstGlobal = MakeGlobal2D(
        ctx.localPeer.ptrD + static_cast<int64_t>(range.dstRow) * static_cast<int32_t>(ctx.shape.k) + range.col, 1,
        range.cols, static_cast<int32_t>(ctx.shape.k));
    TLOAD(tile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_MTE3, event);
    wait_flag(PIPE_MTE2, PIPE_MTE3, event);
    TSTORE(dstGlobal, tile);
    WaitStoreTileReusable();
}

AICORE inline void CopyLocalRowsToPeerWindow(
    const ReturnContext& ctx, const ReturnSegment& segment, const ReturnChunk& chunk)
{
    for (uint32_t row = 0; row < chunk.rowsThisChunk; ++row) {
        int32_t dstRow = segment.dstStart + static_cast<int32_t>(chunk.rowBegin + row);
        int32_t srcRow = segment.srcStart + static_cast<int32_t>(chunk.rowBegin + row);
        for (int32_t col = 0; col < static_cast<int32_t>(ctx.shape.k); col += kDefaultTileCols) {
            int32_t cols = static_cast<int32_t>(ctx.shape.k) - col < kDefaultTileCols ?
                               static_cast<int32_t>(ctx.shape.k) - col :
                               kDefaultTileCols;
            CopyLocalTileToPeerWindow(ctx, TileCopyRange{srcRow, dstRow, col, cols});
        }
    }
}

AICORE inline void PutRemoteRowsToOwner(
    const ReturnContext& ctx, const ReturnSegment& segment, const ReturnChunk& chunk)
{
    LocalPeerWindowView remotePeer =
        MakeRemotePeerWindowView(ctx.ctx, ctx.peerWindow, segment.src, *ctx.peerWindowLayout);
    GlobalNd<half> remoteDst = MakeGlobal2D(
        remotePeer.ptrD + static_cast<int64_t>(segment.dstStart + static_cast<int32_t>(chunk.rowBegin)) *
                              static_cast<int32_t>(ctx.shape.k),
        static_cast<int32_t>(chunk.rowsThisChunk), static_cast<int32_t>(ctx.shape.k),
        static_cast<int32_t>(ctx.shape.k));
    GlobalNd<half> localSrc = MakeGlobal2D(
        ctx.localExpertOutput + static_cast<int64_t>(segment.srcStart + static_cast<int32_t>(chunk.rowBegin)) *
                                    static_cast<int32_t>(ctx.shape.k),
        static_cast<int32_t>(chunk.rowsThisChunk), static_cast<int32_t>(ctx.shape.k),
        static_cast<int32_t>(ctx.shape.k));
    VecTile<half, kDefaultTileCols> ping(1, kDefaultTileCols);
    VecTile<half, kDefaultTileCols> pong(1, kDefaultTileCols);
    TASSIGN(ping, kPingUbAddr);
    TASSIGN(pong, kPongUbAddr);
    pto::comm::TPUT(remoteDst, localSrc, ping, pong);
}

AICORE inline void ReturnRowsChunk(const ReturnContext& ctx, const ReturnSegment& segment, const ReturnChunk& chunk)
{
    if (segment.src == ctx.myRank) {
        CopyLocalRowsToPeerWindow(ctx, segment, chunk);
        return;
    }
    PutRemoteRowsToOwner(ctx, segment, chunk);
}

AICORE inline void NotifyCombineOwners(const ReturnContext& ctx, uint32_t blockId, uint32_t blockNum)
{
    for (uint32_t src = blockId; src < ctx.shape.ep; src += blockNum) {
        LocalPeerWindowView remotePeer = MakeRemotePeerWindowView(ctx.ctx, ctx.peerWindow, src, *ctx.peerWindowLayout);
        pipe_barrier(PIPE_ALL);
        pto::comm::Signal sig = MakeSignal(remotePeer.combineDoneSignal + ctx.myRank);
        pto::comm::TNOTIFY(sig, 1, pto::comm::NotifyOp::AtomicAdd);
    }
}

AICORE inline void ReturnExpertRowsToOwners(
    MoeCombineShape shape, LocalWorkspaceView workspaceView, LocalRouteMetaView routeMeta,
    LocalPeerWindowView localPeer, __gm__ CommDeviceContext* ctx, GM_ADDR peerWindow, GM_ADDR expertOutput,
    uint32_t myRank, uint32_t blockId, uint32_t blockNum, const PeerWindowLayout& peerWindowLayout)
{
    if (blockNum == 0) {
        return;
    }
    ReturnContext returnCtx{
        shape,
        routeMeta,
        localPeer,
        ctx,
        peerWindow,
        reinterpret_cast<__gm__ half*>(expertOutput),
        myRank,
        ExpertNumPaddedDevice(shape),
        EffectiveRowChunk(shape),
        &peerWindowLayout};
    uint32_t segmentCount = shape.ep * shape.expertPerRank;
    AcquireReturnMetadata(shape, routeMeta, returnCtx.expertNumPadded);

    uint32_t chunkBase = 0;
    for (uint32_t segment = 0; segment < segmentCount; ++segment) {
        ReturnSegment segmentInfo{};
        if (!LoadReturnSegment(returnCtx, segment, &segmentInfo)) {
            continue;
        }
        for (uint32_t chunk = 0; chunk < segmentInfo.chunkCount; ++chunk) {
            if (((chunkBase + chunk) % blockNum) != blockId) {
                continue;
            }
            ReturnRowsChunk(returnCtx, segmentInfo, MakeReturnChunk(returnCtx, segmentInfo, chunk));
        }
        chunkBase += segmentInfo.chunkCount;
    }
    SoftSyncAiv(workspaceView.localSync, blockNum);
    NotifyCombineOwners(returnCtx, blockId, blockNum);
}

AICORE inline void InitRestoreRouteCache(bool enabled, RestoreRouteCache* cache)
{
    cache->count = 0;
    cache->enabled = enabled;
}

AICORE inline void PrepareRestoreRouteReads(
    MoeCombineShape shape, LocalRouteMetaView routeMeta, LocalPeerWindowView localPeer, __gm__ float* probValues,
    uint32_t token, RestoreRouteCache* cache)
{
    for (uint32_t slot = 0; slot < shape.topK; ++slot) {
        uint32_t routeIndex = token * shape.topK + slot;
        int32_t ptrDRow = *(routeMeta.expandedRowIdx + routeIndex);
        if (ptrDRow < 0) {
            continue;
        }
        __gm__ half* ptrRow = localPeer.ptrD + static_cast<int64_t>(ptrDRow) * static_cast<int32_t>(shape.k);
        if (!cache->enabled) {
            AcquireGmRangeBeforeRead(ptrRow, static_cast<uint64_t>(shape.k) * sizeof(half));
            continue;
        }
        if (cache->count == 0) {
            pipe_barrier(PIPE_ALL);
        }
        cache->rows[cache->count] = ptrDRow;
        cache->probs[cache->count] = probValues[routeIndex];
        ++cache->count;
        DcciGmRangeNoFence(ptrRow, static_cast<uint64_t>(shape.k) * sizeof(half));
    }
    if (cache->count != 0) {
        dsb(DSB_DDR);
    }
}

AICORE inline bool LoadRestoreRoute(const RestoreTileContext& ctx, uint32_t route, int32_t* ptrDRow, float* prob)
{
    uint32_t routeIndex = ctx.token * ctx.shape.topK + route;
    *ptrDRow = ctx.cache->enabled ? ctx.cache->rows[route] : *(ctx.routeMeta.expandedRowIdx + routeIndex);
    if (*ptrDRow < 0) {
        return false;
    }
    *prob = ctx.cache->enabled ? ctx.cache->probs[route] : ctx.probValues[routeIndex];
    return true;
}

AICORE inline void AccumulateRestoreTile(
    const RestoreTileContext& ctx, VecTile<half, kDefaultTileCols>* outTile, VecTile<half, kDefaultTileCols>* ptrTile)
{
    uint32_t combineCount = ctx.cache->enabled ? ctx.cache->count : ctx.shape.topK;
    pto::Event<pto::Op::TAXPY, pto::Op::TLOAD> axpyToNextLoad;
    bool waitForPreviousAxpy = false;
    for (uint32_t route = 0; route < combineCount; ++route) {
        int32_t ptrDRow = 0;
        float prob = 0.0f;
        if (!LoadRestoreRoute(ctx, route, &ptrDRow, &prob)) {
            continue;
        }
        __gm__ half* ptrRow = ctx.localPeer.ptrD + static_cast<int64_t>(ptrDRow) * static_cast<int32_t>(ctx.shape.k);
        GlobalNd<half> ptrGlobal = MakeGlobal2D(ptrRow + ctx.col, 1, ctx.cols, static_cast<int32_t>(ctx.shape.k));
        pto::Event<pto::Op::TLOAD, pto::Op::TAXPY> loadToAxpy;
        if (waitForPreviousAxpy) {
            axpyToNextLoad.Wait();
        }
        loadToAxpy = TLOAD(*ptrTile, ptrGlobal);
        axpyToNextLoad = TAXPY(*outTile, *ptrTile, static_cast<half>(prob), loadToAxpy);
        waitForPreviousAxpy = true;
    }
    if (waitForPreviousAxpy) {
        axpyToNextLoad.Wait();
    }
}

AICORE inline void RestoreColumnTile(const RestoreTileContext& ctx)
{
    VecTile<half, kDefaultTileCols> outTile(1, ctx.cols);
    VecTile<half, kDefaultTileCols> ptrTile(1, ctx.cols);
    TASSIGN(outTile, kPingUbAddr);
    TASSIGN(ptrTile, kPongUbAddr);
    GlobalNd<half> outGlobal = MakeGlobal2D(
        ctx.output + static_cast<int64_t>(ctx.token) * static_cast<int32_t>(ctx.shape.k) + ctx.col, 1, ctx.cols,
        static_cast<int32_t>(ctx.shape.k));
    TEXPANDS(outTile, static_cast<half>(0.0));
    pipe_barrier(PIPE_ALL);
    AccumulateRestoreTile(ctx, &outTile, &ptrTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(outGlobal, outTile);
    WaitStoreTileReusable();
}

AICORE inline void RestoreOutputRows(
    MoeCombineShape shape, LocalRouteMetaView routeMeta, LocalPeerWindowView localPeer, GM_ADDR probs, GM_ADDR outputC,
    uint32_t blockId, uint32_t blockNum)
{
    if (blockNum == 0) {
        return;
    }
    __gm__ float* probValues = reinterpret_cast<__gm__ float*>(probs);
    __gm__ half* output = reinterpret_cast<__gm__ half*>(outputC);
    uint32_t tokenBegin = TokenShardBegin(shape.m, blockId, blockNum);
    uint32_t tokenEnd = TokenShardEnd(shape.m, blockId, blockNum);
    for (uint32_t token = tokenBegin; token < tokenEnd; ++token) {
        RestoreRouteCache cache{};
        InitRestoreRouteCache(shape.topK <= kRouteCacheMax, &cache);
        PrepareRestoreRouteReads(shape, routeMeta, localPeer, probValues, token, &cache);
        for (int32_t col = 0; col < static_cast<int32_t>(shape.k); col += kDefaultTileCols) {
            int32_t cols = static_cast<int32_t>(shape.k) - col < kDefaultTileCols ?
                               static_cast<int32_t>(shape.k) - col :
                               kDefaultTileCols;
            RestoreTileContext tileCtx{shape, routeMeta, localPeer, probValues, output, token, col, cols, &cache};
            RestoreColumnTile(tileCtx);
        }
    }
}

} // namespace

__global__ AICORE void MoeCombineKernel(
    MoeCombineShape shape, uint32_t myRank, GM_ADDR expertOutput, GM_ADDR probs, GM_ADDR outputC, GM_ADDR routeMeta,
    GM_ADDR peerWindow, GM_ADDR hcclCtx, GM_ADDR workspace)
{
    WorkspaceLayout workspaceLayout = MakeWorkspaceLayout(shape);
    CombineRouteMetaLayout routeMetaLayout = MakeCombineRouteMetaLayout(shape);
    PeerWindowLayout peerWindowLayout = MakePeerWindowLayout(shape);
    LocalWorkspaceView workspaceView = MakeLocalWorkspaceView(workspace, workspaceLayout);
    LocalRouteMetaView localRouteMeta = MakeLocalRouteMetaView(routeMeta, routeMetaLayout);
    LocalPeerWindowView localPeer = MakeLocalPeerWindowView(peerWindow, peerWindowLayout);
    __gm__ CommDeviceContext* ctx = reinterpret_cast<__gm__ CommDeviceContext*>(hcclCtx);
    uint32_t blockId = static_cast<uint32_t>(get_block_idx());
    uint32_t blockNum = shape.aivBlocks == 0 ? 1 : shape.aivBlocks;
    if (shape.ep == 0 || shape.m == 0 || shape.k == 0 || shape.topK == 0 || shape.expertPerRank == 0 ||
        shape.expertNum == 0 || blockId >= blockNum) {
        return;
    }

    ReturnExpertRowsToOwners(
        shape, workspaceView, localRouteMeta, localPeer, ctx, peerWindow, expertOutput, myRank, blockId, blockNum,
        peerWindowLayout);
    WaitCombinePhase(shape, localPeer, blockId, blockNum, static_cast<int32_t>(moe_combine::kMoeCombineSignalValue));
    SoftSyncAiv(workspaceView.localSync, blockNum);
    RestoreOutputRows(shape, localRouteMeta, localPeer, probs, outputC, blockId, blockNum);
    SoftSyncAiv(workspaceView.localSync, blockNum);
}

namespace moe_combine {

void LaunchMoeCombineKernel(
    MoeCombineShape shape, uint32_t myRank, uint8_t* expertOutput, uint8_t* probs, uint8_t* outputC, uint8_t* routeMeta,
    uint8_t* peerWindow, uint8_t* hcclCtx, uint8_t* workspace, void* stream, uint32_t launchBlockCount)
{
    MoeCombineKernel<<<launchBlockCount, nullptr, stream>>>(
        shape, myRank, expertOutput, probs, outputC, routeMeta, peerWindow, hcclCtx, workspace);
}

} // namespace moe_combine
