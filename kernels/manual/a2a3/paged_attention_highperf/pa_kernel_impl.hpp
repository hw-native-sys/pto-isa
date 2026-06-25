/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_PAGED_ATTENTION_HIGHPERF_IMPL_HPP
#define PTO_PAGED_ATTENTION_HIGHPERF_IMPL_HPP

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

#include "pa_tiling_struct.hpp"

using namespace pto;

constexpr int32_t TILING_BATCH = 0;
constexpr int32_t TILING_NUMHEADS = 1;
constexpr int32_t TILING_HEADDIM = 2;
constexpr int32_t TILING_BLOCKSIZE = 4;
constexpr int32_t TILING_MAXBLOCKS = 5;
constexpr int32_t TILING_KVHEADS = 7;
constexpr int32_t TILING_FORMER_BATCH = 8;
constexpr int32_t TILING_FORMER_HEAD = 9;
constexpr int32_t TILING_TAIL_BATCH = 10;
constexpr int32_t TILING_TAIL_HEAD = 11;
constexpr int32_t TILING_HEADNUM_MOVE = 12;
constexpr int32_t TILING_KEY = 16;
constexpr int32_t TILING_HEADSIZE = 17;
constexpr int32_t TILING_PARASIZE = 18;
constexpr int32_t TILING_GROUPNUM = 19;
constexpr int32_t TILING_FORMER_GROUP_MOVE = 20;
constexpr int32_t TILING_TAIL_GROUP_MOVE = 21;
constexpr int32_t TILING_MAX_KVSEQLEN = 22;
constexpr int32_t TILING_KVSPLIT = 23;
constexpr int32_t TILING_KVCORENUM = 24;
constexpr int32_t TILING_BLOCKSIZE_CALC = 25;
constexpr int32_t TILING_DECODER_BS = 28;
constexpr int32_t TILING_HEADDIM_V = 29;

constexpr int32_t kParaKvSeqLen = 1;
constexpr int32_t kParaBatchIndex = 13;

AICORE inline int32_t LoadTilingI32(__gm__ uint8_t *tiling, int32_t index)
{
    return *(reinterpret_cast<__gm__ int32_t *>(tiling) + index);
}

AICORE inline int32_t LoadBlockTable(__gm__ uint8_t *blockTablesGm, int64_t offset)
{
    return *(reinterpret_cast<__gm__ int32_t *>(blockTablesGm) + offset);
}

AICORE inline float LoadFp16(__gm__ uint8_t *gm, int64_t offset)
{
    __gm__ half *ptr = reinterpret_cast<__gm__ half *>(gm);
    return static_cast<float>(ptr[offset]);
}

AICORE inline void StoreOutputFp16(__gm__ uint8_t *oGm, int64_t offset, float value)
{
    __gm__ half *out = reinterpret_cast<__gm__ half *>(oGm);
    out[offset] = static_cast<half>(value);
}

AICORE inline float LoadScale(__gm__ uint8_t *tiling)
{
    union {
        int32_t i;
        float f;
    } scale;
    scale.i = LoadTilingI32(tiling, 6);
    return scale.f;
}

struct PaTilingContext {
    int32_t batch;
    int32_t decoderBatch;
    int32_t numHeads;
    int32_t kvHeads;
    int32_t headDim;
    int32_t headDimV;
    int32_t blockSize;
    int32_t maxBlocksPerQuery;
    int32_t maxKvSeqLen;
    int32_t formerBatch;
    int32_t formerHeadSplit;
    int32_t tailBatch;
    int32_t tailHeadSplit;
    int32_t headNumMove;
    int32_t groupNum;
    int32_t formerGroupMove;
    int32_t tailGroupMove;
    int32_t kvSplitPerCore;
    int32_t kvSplitCoreNum;
    int32_t blockSizeCalc;
    int32_t headSize;
    int32_t paraSize;
    float scale;
};

AICORE inline PaTilingContext LoadPaTilingContext(__gm__ uint8_t *tiling)
{
    PaTilingContext ctx{};
    ctx.batch = LoadTilingI32(tiling, TILING_BATCH);
    ctx.decoderBatch = LoadTilingI32(tiling, TILING_DECODER_BS);
    ctx.numHeads = LoadTilingI32(tiling, TILING_NUMHEADS);
    ctx.kvHeads = LoadTilingI32(tiling, TILING_KVHEADS);
    ctx.headDim = LoadTilingI32(tiling, TILING_HEADDIM);
    ctx.headDimV = LoadTilingI32(tiling, TILING_HEADDIM_V);
    ctx.blockSize = LoadTilingI32(tiling, TILING_BLOCKSIZE);
    ctx.maxBlocksPerQuery = LoadTilingI32(tiling, TILING_MAXBLOCKS);
    ctx.maxKvSeqLen = LoadTilingI32(tiling, TILING_MAX_KVSEQLEN);
    ctx.formerBatch = LoadTilingI32(tiling, TILING_FORMER_BATCH);
    ctx.formerHeadSplit = LoadTilingI32(tiling, TILING_FORMER_HEAD);
    ctx.tailBatch = LoadTilingI32(tiling, TILING_TAIL_BATCH);
    ctx.tailHeadSplit = LoadTilingI32(tiling, TILING_TAIL_HEAD);
    ctx.headNumMove = LoadTilingI32(tiling, TILING_HEADNUM_MOVE);
    ctx.groupNum = LoadTilingI32(tiling, TILING_GROUPNUM);
    ctx.formerGroupMove = LoadTilingI32(tiling, TILING_FORMER_GROUP_MOVE);
    ctx.tailGroupMove = LoadTilingI32(tiling, TILING_TAIL_GROUP_MOVE);
    ctx.kvSplitPerCore = LoadTilingI32(tiling, TILING_KVSPLIT);
    ctx.kvSplitCoreNum = LoadTilingI32(tiling, TILING_KVCORENUM);
    ctx.blockSizeCalc = LoadTilingI32(tiling, TILING_BLOCKSIZE_CALC);
    ctx.headSize = LoadTilingI32(tiling, TILING_HEADSIZE);
    ctx.paraSize = LoadTilingI32(tiling, TILING_PARASIZE);
    ctx.scale = LoadScale(tiling);
    return ctx;
}

template <typename ScalarTile>
AICORE inline float PtoExpScalar(ScalarTile &tile, float value)
{
    tile.data()[0] = value;
    TEXP(tile, tile);
    pipe_barrier(PIPE_V);
    return tile.data()[0];
}

template <typename ScalarTile>
AICORE inline float PtoLogScalar(ScalarTile &tile, float value)
{
    if (value <= 0.0f) {
        return -3.4028234663852886e38f;
    }
    tile.data()[0] = value;
    TLOG(tile, tile);
    pipe_barrier(PIPE_V);
    return tile.data()[0];
}

AICORE inline float LoadPagedKByBlock(
    __gm__ uint8_t *kGm,
    int32_t blockId,
    int32_t offsetInBlock,
    int32_t blockSize,
    int32_t kvHeads,
    int32_t kvHead,
    int32_t headDim,
    int32_t dim)
{
    const int64_t offset = (((static_cast<int64_t>(blockId) * blockSize + offsetInBlock) * kvHeads + kvHead) * headDim + dim);
    return LoadFp16(kGm, offset);
}

AICORE inline float LoadPagedVByBlock(
    __gm__ uint8_t *vGm,
    int32_t blockId,
    int32_t offsetInBlock,
    int32_t blockSize,
    int32_t kvHeads,
    int32_t kvHead,
    int32_t headDim,
    int32_t dim)
{
    const int64_t offset = (((static_cast<int64_t>(blockId) * blockSize + offsetInBlock) * kvHeads + kvHead) * headDim + dim);
    return LoadFp16(vGm, offset);
}

AICORE inline void ResolvePagedPosition(
    __gm__ uint8_t *blockTablesGm,
    int32_t batchIndex,
    int32_t maxBlocksPerQuery,
    int32_t pos,
    int32_t blockSize,
    int32_t &blockId,
    int32_t &offsetInBlock)
{
    const int32_t tableCol = pos / blockSize;
    offsetInBlock = pos - tableCol * blockSize;
    blockId = LoadBlockTable(blockTablesGm, static_cast<int64_t>(batchIndex) * maxBlocksPerQuery + tableCol);
}

AICORE inline float ComputeScoreByBlock(
    const float *qValues,
    __gm__ uint8_t *kGm,
    int32_t blockId,
    int32_t offsetInBlock,
    int32_t blockSize,
    int32_t kvHead,
    int32_t headDim,
    int32_t kvHeads,
    float scale)
{
    float score = 0.0f;
    for (int32_t dim = 0; dim < headDim; ++dim) {
        const float k = LoadPagedKByBlock(kGm, blockId, offsetInBlock, blockSize, kvHeads, kvHead, headDim, dim);
        score += qValues[dim] * k;
    }
    return score * scale;
}



constexpr int32_t PA_TILE_TOKENS = 128;
constexpr uint8_t PA_QK_FIFO_FLAG = 0;
constexpr uint8_t PA_P_FIFO_FLAG = 2;
constexpr uint8_t PA_PV_FIFO_FLAG = 4;
constexpr uint32_t PA_FIFO_DEPTH = 2;
constexpr uint8_t PTO_PA_REDUCE_READY_DECODER = static_cast<uint8_t>(SYNC_AIV_ONLY_ALL);
constexpr uint8_t PTO_PA_RAW_QK_READY = 0;
constexpr uint8_t PTO_PA_RAW_QK_FREE = 2;
constexpr uint8_t PTO_PA_RAW_P_READY = 4;
constexpr uint8_t PTO_PA_RAW_P_FREE = 6;
constexpr uint8_t PTO_PA_RAW_PV_READY = 8;
constexpr uint8_t PTO_PA_RAW_PV_FREE = 10;

AICORE inline uint8_t PtoPaSlotFlag(uint8_t baseFlag, uint8_t slot)
{
    return static_cast<uint8_t>(baseFlag + slot);
}

AICORE inline uint16_t PtoPaGetFftsMsg(uint16_t mode, uint16_t eventId, uint16_t baseConst = 0x1)
{
    return ((baseConst & 0xf) + ((mode & 0x3) << 4) + ((eventId & 0xf) << 8));
}

AICORE inline void PtoPaSignalFromCube(uint8_t flagId)
{
    pipe_barrier(PIPE_ALL);
    ffts_cross_core_sync(PIPE_FIX, PtoPaGetFftsMsg(0x2, flagId));
}

AICORE inline void PtoPaSignalFromVec(uint8_t flagId)
{
    pipe_barrier(PIPE_ALL);
    ffts_cross_core_sync(PIPE_MTE3, PtoPaGetFftsMsg(0x2, flagId));
}

AICORE inline void PtoPaSignalFreeFromVec(uint8_t flagId)
{
    pipe_barrier(PIPE_ALL);
    ffts_cross_core_sync(PIPE_MTE3, PtoPaGetFftsMsg(0x2, flagId));
}

AICORE inline void PtoPaSignalFreeFromCube(uint8_t flagId)
{
    pipe_barrier(PIPE_ALL);
    ffts_cross_core_sync(PIPE_FIX, PtoPaGetFftsMsg(0x2, flagId));
}

AICORE inline bool SupportsPtoPagedAttentionHighPerf(__gm__ uint8_t *tilingParaGm)
{
    const PaTilingContext ctx = LoadPaTilingContext(tilingParaGm);
    if (ctx.headDim != PA_TILE_TOKENS || ctx.headDimV != PA_TILE_TOKENS || ctx.blockSize != PA_TILE_TOKENS) {
        return false;
    }
    if (ctx.batch <= 0 || ctx.numHeads <= 0 || ctx.kvHeads <= 0 || ctx.numHeads % ctx.kvHeads != 0) {
        return false;
    }
    if (ctx.kvSplitCoreNum > 1) {
        return false;
    }
    return true;
}

AICORE inline bool SupportsPtoPagedAttentionRawSplitKV(__gm__ uint8_t *tilingParaGm)
{
    const PaTilingContext ctx = LoadPaTilingContext(tilingParaGm);
    if (ctx.headDim != PA_TILE_TOKENS || ctx.headDimV != PA_TILE_TOKENS || ctx.blockSize != PA_TILE_TOKENS) {
        return false;
    }
    if (ctx.batch <= 0 || ctx.numHeads <= 0 || ctx.kvHeads <= 0 || ctx.numHeads % ctx.kvHeads != 0) {
        return false;
    }
    if (ctx.kvSplitCoreNum <= 1) {
        return false;
    }
    const int32_t headsPerKv = ctx.numHeads / ctx.kvHeads;
    const int32_t formerHeadSplit = ctx.formerHeadSplit > 0 ? ctx.formerHeadSplit : 1;
    if (headsPerKv != 4 || formerHeadSplit % headsPerKv != 0 || formerHeadSplit < 16 || formerHeadSplit % 16 != 0) {
        return false;
    }
    return ctx.kvSplitPerCore <= 8192;
}

AICORE inline void DdrBarrierBeforePtoFfts()
{
#if defined(__CPU_SIM)
    dsb(0);
#else
    dsb(DSB_DDR);
#endif
    pipe_barrier(PIPE_ALL);
}

AICORE inline void DdrFenceBeforePtoAivReduce()
{
#if defined(__CPU_SIM)
    dsb(0);
#else
    dsb(DSB_DDR);
#endif
}

AICORE inline void PtoPaSetFloatVectorMask(uint32_t len)
{
    set_mask_norm();
    constexpr uint32_t kFloatVectorSize = 64;
    if (len >= kFloatVectorSize) {
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        return;
    }
    uint64_t mask = 0;
    for (uint32_t i = 0; i < len; ++i) {
        mask |= 1ULL << i;
    }
    set_vector_mask(0, mask);
}

AICORE inline void PtoPaStageSync()
{
    SYNCALL<SyncCoreType::Mix>();
}

#ifdef __DAV_C220_CUBE__
template <typename DstTileData, typename SrcTileData>
__tf__ AICORE void PtoPaLoadCbufToCaRaw(typename DstTileData::TileDType __out__ dst,
    typename SrcTileData::TileDType __in__ src, uint32_t srcElementOffset, uint16_t repeatTimes,
    uint16_t srcStride)
{
    using DataType = typename SrcTileData::DType;
    __ca__ DataType *dstAddr = reinterpret_cast<__ca__ DataType *>(__cce_get_tile_ptr(dst));
    __cbuf__ DataType *srcAddr = reinterpret_cast<__cbuf__ DataType *>(__cce_get_tile_ptr(src)) + srcElementOffset;
    load_cbuf_to_ca(dstAddr, srcAddr, 0, repeatTimes, srcStride, 0, 0, false, false, addr_cal_mode_t(0));
}

template <typename DstTileData, typename SrcTileData>
AICORE inline void PtoPaLoadCbufToCaRaw(DstTileData &dst, SrcTileData &src, uint32_t srcElementOffset,
    uint16_t repeatTimes, uint16_t srcStride)
{
    PtoPaLoadCbufToCaRaw<DstTileData, SrcTileData>(dst.data(), src.data(), srcElementOffset, repeatTimes, srcStride);
}

template <typename DstTileData, typename SrcTileData>
__tf__ AICORE void PtoPaLoadCbufToCbRaw(typename DstTileData::TileDType __out__ dst,
    typename SrcTileData::TileDType __in__ src, uint32_t srcElementOffset, uint16_t repeatTimes,
    uint16_t srcStride)
{
    using DataType = typename SrcTileData::DType;
    __cb__ DataType *dstAddr = reinterpret_cast<__cb__ DataType *>(__cce_get_tile_ptr(dst));
    __cbuf__ DataType *srcAddr = reinterpret_cast<__cbuf__ DataType *>(__cce_get_tile_ptr(src)) + srcElementOffset;
    load_cbuf_to_cb(dstAddr, srcAddr, 0, repeatTimes, srcStride, 0, 0, false, addr_cal_mode_t(0));
}

template <typename DstTileData, typename SrcTileData>
AICORE inline void PtoPaLoadCbufToCbRaw(DstTileData &dst, SrcTileData &src, uint32_t srcElementOffset,
    uint16_t repeatTimes, uint16_t srcStride)
{
    PtoPaLoadCbufToCbRaw<DstTileData, SrcTileData>(dst.data(), src.data(), srcElementOffset, repeatTimes, srcStride);
}

template <typename DstTileData, typename SrcTileData>
__tf__ AICORE void PtoPaLoadCbufToCbTranspose128Raw(typename DstTileData::TileDType __out__ dst,
    typename SrcTileData::TileDType __in__ src)
{
    using DataType = typename SrcTileData::DType;
    __cb__ DataType *dstAddr = reinterpret_cast<__cb__ DataType *>(__cce_get_tile_ptr(dst));
    __cbuf__ DataType *srcAddr = reinterpret_cast<__cbuf__ DataType *>(__cce_get_tile_ptr(src));
    constexpr uint32_t kBlock = 16;
    constexpr uint32_t kRows = 128;
    constexpr uint32_t kCols = 128;
    for (uint32_t idx = 0; idx < kCols / kBlock; ++idx) {
        load_cbuf_to_cb_transpose(dstAddr + idx * kRows * kBlock, srcAddr + idx * kBlock * kBlock, 0,
            kRows / kBlock, kCols / kBlock, 0, addr_cal_mode_t(0), 0);
    }
}

template <typename DstTileData, typename SrcTileData>
AICORE inline void PtoPaLoadCbufToCbTranspose128Raw(DstTileData &dst, SrcTileData &src)
{
    PtoPaLoadCbufToCbTranspose128Raw<DstTileData, SrcTileData>(dst.data(), src.data());
}
#endif

#ifdef __DAV_C220_CUBE__
AICORE inline void RunPtoPagedAttentionCubePipeline(__gm__ uint8_t *qGm, __gm__ uint8_t *kGm,
    __gm__ uint8_t *vGm, __gm__ uint8_t *blockTablesGm, __gm__ uint8_t *sGm, __gm__ uint8_t *pGm,
    __gm__ uint8_t *oTmpGm, __gm__ uint8_t *tilingParaGm, int64_t workerIdx, int64_t workerNum)
{
    constexpr int32_t kHeadDim = PA_TILE_TOKENS;
    constexpr int32_t kTileTokens = PA_TILE_TOKENS;
    constexpr int32_t kM = 16;
    constexpr int32_t kN = kTileTokens;
    constexpr int32_t kK = 256;

    const PaTilingContext ctx = LoadPaTilingContext(tilingParaGm);
    if (!SupportsPtoPagedAttentionHighPerf(tilingParaGm) || workerIdx < 0 || workerNum <= 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    const int32_t effectiveBatch = ctx.decoderBatch > 0 ? ctx.decoderBatch : ctx.batch;
    const int32_t maxBlocksPerQuery = ctx.maxBlocksPerQuery > 0 ? ctx.maxBlocksPerQuery :
                                      (ctx.maxKvSeqLen + ctx.blockSize - 1) / ctx.blockSize;
    const int32_t headsPerKv = ctx.numHeads / ctx.kvHeads;
    const int64_t totalRows = static_cast<int64_t>(effectiveBatch) * ctx.numHeads;

    using QKPipe = TPipe<PA_QK_FIFO_FLAG, Direction::DIR_C2V, 16 * kTileTokens * sizeof(float), PA_FIFO_DEPTH>;
    using PPipe = TPipe<PA_P_FIFO_FLAG, Direction::DIR_V2C, 256 * sizeof(half), PA_FIFO_DEPTH>;
    using PVPipe = TPipe<PA_PV_FIFO_FLAG, Direction::DIR_C2V, 16 * kHeadDim * sizeof(float), PA_FIFO_DEPTH>;
    using QGlobal = GlobalTensor<half, Shape<1, 1, 1, 1, kHeadDim>, Stride<kHeadDim, kHeadDim, kHeadDim, kHeadDim, 1>>;
    using KGlobal = GlobalTensor<half, Shape<1, 1, 1, kHeadDim, kTileTokens>,
        Stride<1, 1, 1, 1, 8 * kHeadDim>, Layout::DN>;
    using VGlobal = GlobalTensor<half, Shape<1, 1, 1, kTileTokens, kHeadDim>,
        Stride<kTileTokens * kHeadDim, kTileTokens * kHeadDim, kTileTokens * kHeadDim, 8 * kHeadDim, 1>>;
    using PGlobal = GlobalTensor<half, Shape<1, 1, 1, 1, kTileTokens>,
        Stride<kTileTokens, kTileTokens, kTileTokens, kTileTokens, 1>>;
    using ScoreGlobal = GlobalTensor<float, Shape<1, 1, 1, 1, kTileTokens>,
        Stride<kTileTokens, kTileTokens, kTileTokens, kTileTokens, 1>>;
    using OTmpGlobal = GlobalTensor<float, Shape<1, 1, 1, 1, kHeadDim>,
        Stride<kHeadDim, kHeadDim, kHeadDim, kHeadDim, 1>>;

    using QMatTile = Tile<TileType::Mat, half, 1, kK, BLayout::RowMajor, 1, kHeadDim>;
    using KMatTile = Tile<TileType::Mat, half, kK, kN, BLayout::RowMajor, kHeadDim, kTileTokens, SLayout::ColMajor, 512>;
    using PMatTile = Tile<TileType::Mat, half, 1, kK, BLayout::RowMajor, 1, kTileTokens>;
    using VMatTile = Tile<TileType::Mat, half, kK, kN, BLayout::ColMajor, kTileTokens, kHeadDim, SLayout::RowMajor, 512>;
    using LeftQTile = TileLeft<half, 1, kK, 1, kHeadDim>;
    using LeftPTile = TileLeft<half, 1, kK, 1, kTileTokens>;
    using RightTile = TileRight<half, kK, kN, kHeadDim, kTileTokens>;
    using AccTile = TileAcc<float, kM, kN, 1, kTileTokens>;

    QMatTile qMatTile;
    KMatTile kMatTile;
    PMatTile pMatTile;
    VMatTile vMatTile;
    LeftQTile qLeftTile;
    LeftPTile pLeftTile;
    RightTile rightTile;
    AccTile accTile;
    TASSIGN(qMatTile, 0x00000);
    TASSIGN(kMatTile, 0x20000);
    TASSIGN(pMatTile, 0x00000);
    TASSIGN(vMatTile, 0x20000);
    TASSIGN(qLeftTile, 0x00000);
    TASSIGN(pLeftTile, 0x00000);
    TASSIGN(rightTile, 0x00000);
    TASSIGN(accTile, 0x00000);

    __gm__ uint8_t *scoreBase = sGm + workerIdx * QKPipe::RingFiFo::SLOT_SIZE * QKPipe::RingFiFo::SLOT_NUM;
    __gm__ uint8_t *probBase = pGm + workerIdx * PPipe::RingFiFo::SLOT_SIZE * PPipe::RingFiFo::SLOT_NUM;
    __gm__ uint8_t *outBase = oTmpGm + workerIdx * PVPipe::RingFiFo::SLOT_SIZE * PVPipe::RingFiFo::SLOT_NUM;
    QKPipe qkPipe(reinterpret_cast<__gm__ void *>(scoreBase), 0, 0);
    PPipe pPipe(reinterpret_cast<__gm__ void *>(probBase), 0, 0);
    PVPipe pvPipe(reinterpret_cast<__gm__ void *>(outBase), 0, 0);

    for (int64_t row = workerIdx; row < totalRows; row += workerNum) {
        const int32_t head = static_cast<int32_t>(row % ctx.numHeads);
        const int32_t batchSlot = static_cast<int32_t>(row / ctx.numHeads);
        const int32_t paraBase = ctx.headSize + batchSlot * ctx.paraSize;
        const int32_t kvSeqLen = LoadTilingI32(tilingParaGm, paraBase + kParaKvSeqLen);
        const int32_t batchIndex = LoadTilingI32(tilingParaGm, paraBase + kParaBatchIndex);
        const int32_t kvHead = head / headsPerKv;
        const int32_t tileCount = (kvSeqLen + kTileTokens - 1) / kTileTokens;
        const int64_t qBase = (static_cast<int64_t>(batchIndex) * ctx.numHeads + head) * ctx.headDim;
        QGlobal qGlobal(reinterpret_cast<__gm__ half *>(qGm) + qBase);
        TLOAD(qMatTile, qGlobal);
        pipe_barrier(PIPE_ALL);
        TEXTRACT(qLeftTile, qMatTile, 0, 0);
        pipe_barrier(PIPE_ALL);

        for (int32_t tile = 0; tile < tileCount; ++tile) {
            const int32_t blockId = LoadBlockTable(blockTablesGm, static_cast<int64_t>(batchIndex) * maxBlocksPerQuery + tile);
            const int64_t kvBase = (static_cast<int64_t>(blockId) * ctx.blockSize * ctx.kvHeads + kvHead) * ctx.headDim;


            KGlobal kGlobal(reinterpret_cast<__gm__ half *>(kGm) + kvBase);
            TLOAD(kMatTile, kGlobal);
            pipe_barrier(PIPE_ALL);
            TMOV(rightTile, kMatTile);
            pipe_barrier(PIPE_ALL);
            TGEMV(accTile, qLeftTile, rightTile);
            pipe_barrier(PIPE_ALL);
            DdrBarrierBeforePtoFfts();
            TPUSH<QKPipe, AccTile, TileSplitAxis::TILE_NO_SPLIT>(qkPipe, accTile);

            TPOP<PPipe, PMatTile, TileSplitAxis::TILE_NO_SPLIT>(pPipe, pMatTile);
            VGlobal vGlobal(reinterpret_cast<__gm__ half *>(vGm) + kvBase);

            TLOAD(vMatTile, vGlobal);
            pipe_barrier(PIPE_ALL);
            TEXTRACT(pLeftTile, pMatTile, 0, 0);
            TMOV(rightTile, vMatTile);
            pipe_barrier(PIPE_ALL);
            TGEMV(accTile, pLeftTile, rightTile);
            pipe_barrier(PIPE_ALL);
            DdrBarrierBeforePtoFfts();
            TPUSH<PVPipe, AccTile, TileSplitAxis::TILE_NO_SPLIT>(pvPipe, accTile);
        }
    }
    pipe_barrier(PIPE_ALL);
}
#endif

#ifdef __DAV_C220_VEC__
AICORE inline void RunPtoPagedAttentionVecPipeline(__gm__ uint8_t *oGm, __gm__ uint8_t *sGm,
    __gm__ uint8_t *pGm, __gm__ uint8_t *oTmpGm, __gm__ uint8_t *tilingParaGm, int64_t workerIdx, int64_t workerNum,
    uint32_t subBlockId)
{
    constexpr int32_t kHeadDim = PA_TILE_TOKENS;
    constexpr int32_t kTileTokens = PA_TILE_TOKENS;
    if (!SupportsPtoPagedAttentionHighPerf(tilingParaGm) || workerIdx < 0 || workerNum <= 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    const PaTilingContext ctx = LoadPaTilingContext(tilingParaGm);
    const int32_t effectiveBatch = ctx.decoderBatch > 0 ? ctx.decoderBatch : ctx.batch;
    const int64_t totalRows = static_cast<int64_t>(effectiveBatch) * ctx.numHeads;

    using QKPipe = TPipe<PA_QK_FIFO_FLAG, Direction::DIR_C2V, 16 * kTileTokens * sizeof(float), PA_FIFO_DEPTH>;
    using PPipe = TPipe<PA_P_FIFO_FLAG, Direction::DIR_V2C, 256 * sizeof(half), PA_FIFO_DEPTH>;
    using PVPipe = TPipe<PA_PV_FIFO_FLAG, Direction::DIR_C2V, 16 * kHeadDim * sizeof(float), PA_FIFO_DEPTH>;
    using VecFloat128 = Tile<TileType::Vec, float, 1, kHeadDim, BLayout::RowMajor, 1, kHeadDim>;
    using VecHalf128 = Tile<TileType::Vec, half, 1, kHeadDim, BLayout::RowMajor, 1, kHeadDim>;
    using VecHalf256 = Tile<TileType::Vec, half, 1, 256, BLayout::RowMajor, 1, kHeadDim>;
    using VecFloat8 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8>;
    using GlobalFloat128 = GlobalTensor<float, Shape<1, 1, 1, 1, kHeadDim>, Stride<kHeadDim, kHeadDim, kHeadDim, kHeadDim, 1>>;
    using GlobalHalf128 = GlobalTensor<half, Shape<1, 1, 1, 1, kHeadDim>, Stride<kHeadDim, kHeadDim, kHeadDim, kHeadDim, 1>>;

    VecFloat128 weightedTile;
    VecFloat128 scoreTile;
    VecFloat128 pvTile;
    VecHalf256 probTile;
    VecHalf128 outHalfTile;
    VecFloat8 scalarMathTile;
    TASSIGN(weightedTile, 0x0000);
    TASSIGN(scoreTile, 0x0800);
    TASSIGN(pvTile, 0x1000);
    TASSIGN(probTile, 0x1800);
    TASSIGN(outHalfTile, 0x2000);
    TASSIGN(scalarMathTile, 0x2800);

    __gm__ uint8_t *scoreBase = sGm + workerIdx * QKPipe::RingFiFo::SLOT_SIZE * QKPipe::RingFiFo::SLOT_NUM;
    __gm__ uint8_t *probBase = pGm + workerIdx * PPipe::RingFiFo::SLOT_SIZE * PPipe::RingFiFo::SLOT_NUM;
    __gm__ uint8_t *outTmpBase = oTmpGm + workerIdx * PVPipe::RingFiFo::SLOT_SIZE * PVPipe::RingFiFo::SLOT_NUM;
    QKPipe qkPipe(reinterpret_cast<__gm__ void *>(scoreBase), 0, 0);
    PPipe pPipe(reinterpret_cast<__gm__ void *>(probBase), 0, 0);
    PVPipe pvPipe(reinterpret_cast<__gm__ void *>(outTmpBase), 0, 0);

    for (int64_t row = workerIdx; row < totalRows; row += workerNum) {
        const int32_t head = static_cast<int32_t>(row % ctx.numHeads);
        const int32_t batchSlot = static_cast<int32_t>(row / ctx.numHeads);
        const int32_t paraBase = ctx.headSize + batchSlot * ctx.paraSize;
        const int32_t kvSeqLen = LoadTilingI32(tilingParaGm, paraBase + kParaKvSeqLen);
        const int32_t batchIndex = LoadTilingI32(tilingParaGm, paraBase + kParaBatchIndex);
        const int32_t tileCount = (kvSeqLen + kTileTokens - 1) / kTileTokens;
        const bool doWork = subBlockId == 0;
        float maxScore = -3.4028234663852886e38f;
        float sumExp = 0.0f;
        TEXPANDS(weightedTile, 0.0f);
        pipe_barrier(PIPE_ALL);

        for (int32_t tile = 0; tile < tileCount; ++tile) {
            const int32_t validTokens = ((tile + 1) * kTileTokens <= kvSeqLen) ? kTileTokens : (kvSeqLen - tile * kTileTokens);
            TPOP<QKPipe, VecFloat128, TileSplitAxis::TILE_NO_SPLIT>(qkPipe, scoreTile);
            float tileMax = -3.4028234663852886e38f;
            for (int32_t pos = 0; pos < validTokens; ++pos) {
                const float score = scoreTile.data()[pos] * ctx.scale;
                tileMax = score > tileMax ? score : tileMax;
            }
            const float newMax = tileMax > maxScore ? tileMax : maxScore;
            const float oldScale = (tile == 0) ? 0.0f : PtoExpScalar(scalarMathTile, maxScore - newMax);
            float tileSum = 0.0f;
            TEXPANDS(probTile, static_cast<half>(0.0));
            for (int32_t pos = 0; pos < kTileTokens; ++pos) {
                float prob = 0.0f;
                if (pos < validTokens) {
                    prob = PtoExpScalar(scalarMathTile, scoreTile.data()[pos] * ctx.scale - newMax);
                    tileSum += prob;
                }
                probTile.data()[pos] = static_cast<half>(prob);
            }
            sumExp = sumExp * oldScale + tileSum;
            TMULS(weightedTile, weightedTile, oldScale);
            pipe_barrier(PIPE_ALL);
            maxScore = newMax;
            DdrBarrierBeforePtoFfts();
            TPUSH<PPipe, VecHalf256, TileSplitAxis::TILE_NO_SPLIT>(pPipe, probTile);

            TPOP<PVPipe, VecFloat128, TileSplitAxis::TILE_NO_SPLIT>(pvPipe, pvTile);
            if (doWork) {
                pipe_barrier(PIPE_ALL);
                TAXPY(weightedTile, pvTile, 1.0f);
                pipe_barrier(PIPE_ALL);
            }
        }

        if (!doWork) {
            continue;
        }
        const float invSum = sumExp > 0.0f ? 1.0f / sumExp : 0.0f;
        const int64_t outBase = (static_cast<int64_t>(batchIndex) * ctx.numHeads + head) * ctx.headDim;
        GlobalHalf128 outGlobal(reinterpret_cast<__gm__ half *>(oGm) + outBase);
        TMULS(weightedTile, weightedTile, invSum);
        pipe_barrier(PIPE_ALL);
        TCVT(outHalfTile, weightedTile, RoundMode::CAST_RINT);
        pipe_barrier(PIPE_ALL);
        TSTORE(outGlobal, outHalfTile);
        pipe_barrier(PIPE_ALL);
    }
    pipe_barrier(PIPE_ALL);
}
#endif


AICORE inline uint64_t LoadTilingOffset64(__gm__ uint8_t *tiling, int32_t base, int32_t highIdx, int32_t lowIdx)
{
    const uint32_t high = static_cast<uint32_t>(LoadTilingI32(tiling, base + highIdx));
    const uint32_t low = static_cast<uint32_t>(LoadTilingI32(tiling, base + lowIdx));
    return (static_cast<uint64_t>(high) << 32) | static_cast<uint64_t>(low);
}

#ifdef __DAV_C220_CUBE__
AICORE inline void RunPtoPagedAttentionCubePipelineSplitKV(__gm__ uint8_t *qGm, __gm__ uint8_t *kGm,
    __gm__ uint8_t *vGm, __gm__ uint8_t *blockTablesGm, __gm__ uint8_t *sGm, __gm__ uint8_t *pGm,
    __gm__ uint8_t *oTmpGm, __gm__ uint8_t *tilingParaGm, int64_t workerIdx, int64_t workerNum)
{
    constexpr int32_t kHeadDim = PA_TILE_TOKENS;
    constexpr int32_t kTileTokens = PA_TILE_TOKENS;
    constexpr int32_t kM = 16;
    constexpr int32_t kMValid = 4;
    constexpr int32_t kN = kTileTokens;
    constexpr int32_t kK = 256;
    constexpr int32_t kHeadGroup = 16;

    const PaTilingContext ctx = LoadPaTilingContext(tilingParaGm);
    if (workerIdx < 0 || workerNum <= 0 || ctx.headDim != kHeadDim || ctx.headDimV != kHeadDim ||
        ctx.blockSize != kTileTokens || ctx.kvSplitCoreNum <= 1 || ctx.numHeads <= 0 || ctx.kvHeads <= 0 ||
        ctx.numHeads % ctx.kvHeads != 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    const int32_t maxBlocksPerQuery = ctx.maxBlocksPerQuery > 0 ? ctx.maxBlocksPerQuery :
        (ctx.maxKvSeqLen + ctx.blockSize - 1) / ctx.blockSize;
    const int32_t headsPerKv = ctx.numHeads / ctx.kvHeads;
    const int32_t formerHeadSplit = ctx.formerHeadSplit > 0 ? ctx.formerHeadSplit : 1;
    const int32_t maxHeadGroups = (formerHeadSplit + kHeadGroup - 1) / kHeadGroup;
    const int32_t corePerBatch = (ctx.numHeads + formerHeadSplit - 1) / formerHeadSplit;
    const int64_t processNum = static_cast<int64_t>(ctx.formerBatch) * corePerBatch * ctx.kvSplitCoreNum;

    using QGlobal = GlobalTensor<half, Shape<1, 1, 1, kM, kHeadDim>,
        Stride<kM * kHeadDim, kM * kHeadDim, kM * kHeadDim, kHeadDim, 1>, Layout::ND>;
    using KGlobal = GlobalTensor<half, Shape<1, 1, 1, kHeadDim, kTileTokens>,
        Stride<1, 1, 1, 1, 8 * kHeadDim>, Layout::DN>;
    using VGlobal = GlobalTensor<half, Shape<1, 1, 1, kTileTokens, kHeadDim>,
        Stride<kTileTokens * kHeadDim, kTileTokens * kHeadDim, kTileTokens * kHeadDim, 8 * kHeadDim, 1>>;

    using QMatTile = Tile<TileType::Mat, half, kM, kHeadDim, BLayout::ColMajor, kM, kHeadDim, SLayout::RowMajor>;
    using KMatTile = Tile<TileType::Mat, half, kK, kN, BLayout::RowMajor, kHeadDim, kTileTokens, SLayout::ColMajor, 512>;
    using PMatTile = Tile<TileType::Mat, half, kM, kTileTokens, BLayout::ColMajor, kM, kTileTokens, SLayout::RowMajor>;
    using VMatTile = Tile<TileType::Mat, half, kK, kN, BLayout::ColMajor, kTileTokens, kHeadDim, SLayout::RowMajor, 512>;
    using LeftQTile = TileLeft<half, kM, kHeadDim, kMValid, kHeadDim>;
    using LeftPTile = TileLeft<half, kM, kTileTokens, kMValid, kTileTokens>;
    using RightTile = TileRight<half, kK, kN, kHeadDim, kTileTokens>;
    using AccTile = TileAcc<float, kM, kN, kMValid, kTileTokens>;

    QMatTile qMatTile;
    KMatTile kMatTile;
    PMatTile pMatTile;
    VMatTile vMatTile;
    LeftQTile qLeftTile;
    LeftPTile pLeftTile;
    RightTile rightTile;
    AccTile accTile;
    TASSIGN(qMatTile, 0x00000);
    TASSIGN(kMatTile, 0x20000);
    TASSIGN(pMatTile, 0x00000);
    TASSIGN(vMatTile, 0x20000);
    TASSIGN(qLeftTile, 0x00000);
    TASSIGN(pLeftTile, 0x00000);
    TASSIGN(rightTile, 0x00000);
    TASSIGN(accTile, 0x00000);

    using ScoreGlobal = GlobalTensor<float, Shape<1, 1, 1, kMValid, kN>,
        Stride<kMValid * kN, kMValid * kN, kMValid * kN, kN, 1>>;
    using ProbGlobal = GlobalTensor<half, Shape<1, 1, 1, kM, kTileTokens>,
        Stride<kM * kTileTokens, kM * kTileTokens, kM * kTileTokens, kTileTokens, 1>, Layout::ND>;
    using OutGlobal = GlobalTensor<float, Shape<1, 1, 1, kMValid, kN>,
        Stride<kMValid * kN, kMValid * kN, kMValid * kN, kN, 1>>;

    constexpr int64_t scoreHeadBytes = kMValid * kTileTokens * sizeof(float);
    constexpr int64_t probHeadBytes = 256 * sizeof(half);
    constexpr int64_t outHeadBytes = kMValid * kHeadDim * sizeof(float);
    constexpr int64_t scoreGroupBytes = kHeadGroup * scoreHeadBytes;
    constexpr int64_t probGroupBytes = kHeadGroup * probHeadBytes;
    constexpr int64_t outGroupBytes = kHeadGroup * outHeadBytes;
    __gm__ uint8_t *scoreBase = sGm + workerIdx * scoreGroupBytes * 2;
    __gm__ uint8_t *probBase = pGm + workerIdx * probGroupBytes * 2;
    __gm__ uint8_t *outBase = oTmpGm + workerIdx * outGroupBytes * 2;

    const int64_t processRounds = (processNum + workerNum - 1) / workerNum;
    const int32_t stageTileCount = (ctx.kvSplitPerCore + kTileTokens - 1) / kTileTokens;
    for (int64_t processRound = 0; processRound < processRounds; ++processRound) {
        const int64_t process = processRound * workerNum + workerIdx;
        bool validProcess = process < processNum;
        int32_t batchIndex = 0;
        int32_t curHeadNum = 0;
        int32_t startHead = 0;
        int32_t startTile = 0;
        int32_t tileCount = 0;
        int32_t curKvSeqLen = 0;
        if (validProcess) {
            int32_t curBatchSlot = static_cast<int32_t>(process / (corePerBatch * ctx.kvSplitCoreNum));
            int32_t paraBase = ctx.headSize + curBatchSlot * ctx.paraSize;
            const int32_t sortedBatch = LoadTilingI32(tilingParaGm, paraBase + kParaBatchIndex);
            paraBase = ctx.headSize + sortedBatch * ctx.paraSize;
            batchIndex = LoadTilingI32(tilingParaGm, paraBase + 8);
            const int32_t kvSeqLen = LoadTilingI32(tilingParaGm, paraBase + kParaKvSeqLen);
            const int32_t kvSeqLenAlign = ((kvSeqLen + ctx.blockSize - 1) / ctx.blockSize) * ctx.blockSize;
            const int32_t kvLoop = (kvSeqLenAlign + ctx.kvSplitPerCore - 1) / ctx.kvSplitPerCore;
            const int32_t curSplit = static_cast<int32_t>(process % ctx.kvSplitCoreNum);
            validProcess = kvSeqLen > 0 && curSplit < kvLoop;
            if (validProcess) {
                const int32_t curHeadBlock = static_cast<int32_t>((process / ctx.kvSplitCoreNum) % corePerBatch);
                startHead = curHeadBlock * formerHeadSplit;
                curHeadNum = formerHeadSplit;
                if (curHeadBlock == corePerBatch - 1) {
                    curHeadNum = ctx.numHeads - curHeadBlock * formerHeadSplit;
                }
                const int32_t startKv = curSplit * ctx.kvSplitPerCore;
                curKvSeqLen = ctx.kvSplitPerCore;
                if (curSplit == kvLoop - 1) {
                    curKvSeqLen = kvSeqLen - startKv;
                }
                tileCount = (curKvSeqLen + kTileTokens - 1) / kTileTokens;
                startTile = startKv / kTileTokens;
            }
        }

        for (int32_t headGroup = 0; headGroup < maxHeadGroups; ++headGroup) {
            const int32_t groupHeadBase = headGroup * kHeadGroup;
            const bool validGroup = validProcess && groupHeadBase < curHeadNum;
            if (validGroup) {
                const int32_t firstHead = startHead + groupHeadBase;
                const int64_t qBase = (static_cast<int64_t>(batchIndex) * ctx.numHeads + firstHead) * ctx.headDim;
                QGlobal qGlobal(reinterpret_cast<__gm__ half *>(qGm) + qBase);
                TLOAD(qMatTile, qGlobal);
                pipe_barrier(PIPE_ALL);
            }
            for (int32_t tile = 0; tile < stageTileCount; ++tile) {
                const bool activeTile = validProcess && tile < tileCount;
                const uint8_t slot = static_cast<uint8_t>(tile & 1);
                const bool activeGroup = validGroup && activeTile;
                if (activeGroup) {
                    const int32_t blockId = LoadBlockTable(blockTablesGm,
                        static_cast<int64_t>(batchIndex) * maxBlocksPerQuery + startTile + tile);
                    for (int32_t headInGroupBase = 0; headInGroupBase < kHeadGroup; headInGroupBase += headsPerKv) {
                        const int32_t baseHeadLocal = groupHeadBase + headInGroupBase;
                        if (baseHeadLocal >= curHeadNum) {
                            break;
                        }
                        const int32_t baseHead = startHead + baseHeadLocal;
                        const int32_t kvHead = baseHead / headsPerKv;
                        const int64_t kvBase = (static_cast<int64_t>(blockId) * ctx.blockSize * ctx.kvHeads + kvHead) *
                            ctx.headDim;
                        PtoPaLoadCbufToCaRaw(qLeftTile, qMatTile, static_cast<uint32_t>(headInGroupBase * 16),
                            static_cast<uint16_t>(kHeadDim / 16), 1);
                        KGlobal kGlobal(reinterpret_cast<__gm__ half *>(kGm) + kvBase);
                        TLOAD(kMatTile, kGlobal);
                        auto matmulEvent = EVENT_ID1;
                        set_flag(PIPE_FIX, PIPE_M, matmulEvent);
                        wait_flag(PIPE_FIX, PIPE_M, matmulEvent);
                        set_flag(PIPE_MTE2, PIPE_MTE1, matmulEvent);
                        wait_flag(PIPE_MTE2, PIPE_MTE1, matmulEvent);
                        set_flag(PIPE_M, PIPE_MTE1, matmulEvent);
                        wait_flag(PIPE_M, PIPE_MTE1, matmulEvent);
                        PtoPaLoadCbufToCbRaw(rightTile, kMatTile, 0,
                            static_cast<uint16_t>((kHeadDim * kTileTokens) / 256), 1);
                        set_flag(PIPE_MTE1, PIPE_M, matmulEvent);
                        wait_flag(PIPE_MTE1, PIPE_M, matmulEvent);
                        TMATMUL(accTile, qLeftTile, rightTile);
                        set_flag(PIPE_MTE1, PIPE_MTE2, matmulEvent);
                        wait_flag(PIPE_MTE1, PIPE_MTE2, matmulEvent);
                        set_flag(PIPE_M, PIPE_FIX, matmulEvent);
                        wait_flag(PIPE_M, PIPE_FIX, matmulEvent);
                        ScoreGlobal scoreGlobal(reinterpret_cast<__gm__ float *>(scoreBase +
                            static_cast<int64_t>(slot) * scoreGroupBytes +
                            static_cast<int64_t>(headInGroupBase) * scoreHeadBytes));
                        TSTORE(scoreGlobal, accTile);
                    }
                    DdrFenceBeforePtoAivReduce();
                }
                PtoPaSignalFromCube(PtoPaSlotFlag(PTO_PA_RAW_QK_READY, slot));
                wait_flag_dev(PtoPaSlotFlag(PTO_PA_RAW_P_READY, slot));
                if (tile >= 2) {
                    wait_flag_dev(PtoPaSlotFlag(PTO_PA_RAW_PV_FREE, slot));
                }
                if (activeGroup) {
                    const int32_t blockId = LoadBlockTable(blockTablesGm,
                        static_cast<int64_t>(batchIndex) * maxBlocksPerQuery + startTile + tile);
                    ProbGlobal probGlobal(reinterpret_cast<__gm__ half *>(probBase +
                        static_cast<int64_t>(slot) * probGroupBytes));
                    TLOAD(pMatTile, probGlobal);
                    pipe_barrier(PIPE_ALL);
                    for (int32_t headInGroupBase = 0; headInGroupBase < kHeadGroup; headInGroupBase += headsPerKv) {
                        const int32_t baseHeadLocal = groupHeadBase + headInGroupBase;
                        if (baseHeadLocal >= curHeadNum) {
                            break;
                        }
                        const int32_t baseHead = startHead + baseHeadLocal;
                        const int32_t kvHead = baseHead / headsPerKv;
                        const int64_t kvBase = (static_cast<int64_t>(blockId) * ctx.blockSize * ctx.kvHeads + kvHead) *
                            ctx.headDim;
                        PtoPaLoadCbufToCaRaw(pLeftTile, pMatTile, static_cast<uint32_t>(headInGroupBase * 16),
                            static_cast<uint16_t>(kTileTokens / 16), 1);
                        VGlobal vGlobal(reinterpret_cast<__gm__ half *>(vGm) + kvBase);
                        TLOAD(vMatTile, vGlobal);
                        auto matmulEvent = EVENT_ID1;
                        set_flag(PIPE_FIX, PIPE_M, matmulEvent);
                        wait_flag(PIPE_FIX, PIPE_M, matmulEvent);
                        set_flag(PIPE_MTE2, PIPE_MTE1, matmulEvent);
                        wait_flag(PIPE_MTE2, PIPE_MTE1, matmulEvent);
                        set_flag(PIPE_M, PIPE_MTE1, matmulEvent);
                        wait_flag(PIPE_M, PIPE_MTE1, matmulEvent);
                        PtoPaLoadCbufToCbTranspose128Raw(rightTile, vMatTile);
                        set_flag(PIPE_MTE1, PIPE_M, matmulEvent);
                        wait_flag(PIPE_MTE1, PIPE_M, matmulEvent);
                        TMATMUL(accTile, pLeftTile, rightTile);
                        set_flag(PIPE_MTE1, PIPE_MTE2, matmulEvent);
                        wait_flag(PIPE_MTE1, PIPE_MTE2, matmulEvent);
                        set_flag(PIPE_M, PIPE_FIX, matmulEvent);
                        wait_flag(PIPE_M, PIPE_FIX, matmulEvent);
                        OutGlobal outGlobal(reinterpret_cast<__gm__ float *>(outBase +
                            static_cast<int64_t>(slot) * outGroupBytes +
                            static_cast<int64_t>(headInGroupBase) * outHeadBytes));
                        TSTORE(outGlobal, accTile);
                    }
                    DdrFenceBeforePtoAivReduce();
                }
                PtoPaSignalFromCube(PtoPaSlotFlag(PTO_PA_RAW_PV_READY, slot));
            }
        }
    }
    pipe_barrier(PIPE_ALL);
}
#endif

#ifdef __DAV_C220_VEC__
AICORE inline void RunPtoPagedAttentionVecPipelineSplitKV(__gm__ uint8_t *oGm, __gm__ uint8_t *sGm,
    __gm__ uint8_t *pGm, __gm__ uint8_t *oTmpGm, __gm__ uint8_t *oCoreTmpGm, __gm__ uint8_t *lGm,
    __gm__ uint8_t *tilingParaGm, int64_t workerIdx, int64_t workerNum, uint32_t subBlockId)
{
    constexpr int32_t kHeadDim = PA_TILE_TOKENS;
    constexpr int32_t kTileTokens = PA_TILE_TOKENS;
    constexpr int32_t kHeadGroup = 16;
    constexpr int32_t kMaxHeadsPerProcess = 32;
    const PaTilingContext ctx = LoadPaTilingContext(tilingParaGm);
    if (workerIdx < 0 || workerNum <= 0 || ctx.headDim != kHeadDim || ctx.headDimV != kHeadDim ||
        ctx.blockSize != kTileTokens || ctx.kvSplitCoreNum <= 1 || ctx.numHeads <= 0 || ctx.kvHeads <= 0 ||
        ctx.numHeads % ctx.kvHeads != 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    const bool activeSubBlock = subBlockId < 2;
    const bool combineSubBlock = subBlockId == 0;
    const int32_t formerHeadSplit = ctx.formerHeadSplit > 0 ? ctx.formerHeadSplit : 1;
    const int32_t maxHeadGroups = (formerHeadSplit + kHeadGroup - 1) / kHeadGroup;
    const int32_t headsPerKv = ctx.numHeads / ctx.kvHeads;
    const int32_t corePerBatch = (ctx.numHeads + formerHeadSplit - 1) / formerHeadSplit;
    const int64_t processNum = static_cast<int64_t>(ctx.formerBatch) * corePerBatch * ctx.kvSplitCoreNum;
    __gm__ float *partialOut = reinterpret_cast<__gm__ float *>(oCoreTmpGm);
    __gm__ float *partialL = reinterpret_cast<__gm__ float *>(lGm);

    using VecFloat128 = Tile<TileType::Vec, float, 1, kHeadDim, BLayout::RowMajor, 1, kHeadDim>;
    using VecHalf128 = Tile<TileType::Vec, half, 1, kHeadDim, BLayout::RowMajor, 1, kHeadDim>;
    using VecHalf256 = Tile<TileType::Vec, half, 1, 256, BLayout::RowMajor, 1, kHeadDim>;
    using VecFloat8 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8>;
    using VecFloat4x128 = Tile<TileType::Vec, float, 4, kHeadDim, BLayout::RowMajor, 4, kHeadDim>;
    using VecFloat4x1 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, 4, 1>;
    using VecFloat1x8 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8>;
    using ScoreGlobal = GlobalTensor<float, Shape<1, 1, 1, 1, kHeadDim>,
        Stride<kHeadDim, kHeadDim, kHeadDim, kHeadDim, 1>>;
    using ScoreRowsGlobal = GlobalTensor<float, Shape<1, 1, 1, 4, kHeadDim>,
        Stride<1, 1, 1, kHeadDim, 1>>;
    using ProbGlobal = GlobalTensor<half, Shape<1, 1, 1, 1, 256>, Stride<256, 256, 256, 256, 1>>;
    using ProbRowGlobal = GlobalTensor<half, Shape<1, 1, 1, 1, kHeadDim>,
        Stride<kHeadDim, kHeadDim, kHeadDim, kHeadDim, 1>>;
    using OutGlobal = GlobalTensor<float, Shape<1, 1, 1, 1, kHeadDim>,
        Stride<kHeadDim, kHeadDim, kHeadDim, kHeadDim, 1>>;
    using OutputGlobal = GlobalTensor<half, Shape<1, 1, 1, 1, kHeadDim>,
        Stride<kHeadDim, kHeadDim, kHeadDim, kHeadDim, 1>>;
    using OutRowsGlobal = GlobalTensor<float, Shape<1, 1, 1, 4, kHeadDim>,
        Stride<1, 1, 1, kHeadDim, 1>>;

    VecFloat128 weightedTile;
    VecFloat128 scoreTile;
    VecFloat128 scoreWorkTile;
    VecFloat128 pvTile;
    VecHalf128 probHalfTile;
    VecHalf128 outHalfTile;
    VecHalf256 probTile;
    VecFloat8 rowMaxTile;
    VecFloat8 rowSumTile;
    VecFloat8 scalarMathTile;
    VecFloat4x128 scoreRowsTile;
    VecFloat4x128 scoreRowsWorkTile;
    VecFloat4x128 pvRowsTile;
    VecFloat4x1 rowMaxRowsTile;
    VecFloat4x1 maxStateRowsTile;
    VecFloat4x1 newMaxRowsTile;
    VecFloat4x1 oldScaleRowsTile;
    VecFloat4x1 rowSumRowsTile;
    VecFloat4x1 sumStateRowsTile;
    VecFloat1x8 rowMaxRowsView;
    VecFloat1x8 maxStateRowsView;
    VecFloat1x8 newMaxRowsView;
    VecFloat1x8 oldScaleRowsView;
    VecFloat1x8 rowSumRowsView;
    VecFloat1x8 sumStateRowsView;
    TASSIGN(weightedTile, 0x0000);
    TASSIGN(scoreTile, 0x0800);
    TASSIGN(scoreWorkTile, 0x1000);
    TASSIGN(pvTile, 0x1800);
    TASSIGN(probHalfTile, 0x2000);
    TASSIGN(outHalfTile, 0x2000);
    TASSIGN(probTile, 0x2800);
    TASSIGN(rowMaxTile, 0x3000);
    TASSIGN(rowSumTile, 0x3040);
    TASSIGN(scalarMathTile, 0x3080);
    TASSIGN(scoreRowsTile, 0x0800);
    TASSIGN(scoreRowsWorkTile, 0x1000);
    TASSIGN(pvRowsTile, 0x1800);
    TASSIGN(rowMaxRowsTile, 0x3000);
    TASSIGN(maxStateRowsTile, 0x3020);
    TASSIGN(newMaxRowsTile, 0x3040);
    TASSIGN(oldScaleRowsTile, 0x3060);
    TASSIGN(rowSumRowsTile, 0x3080);
    TASSIGN(sumStateRowsTile, 0x30a0);
    TRESHAPE(rowMaxRowsView, rowMaxRowsTile);
    TRESHAPE(maxStateRowsView, maxStateRowsTile);
    TRESHAPE(newMaxRowsView, newMaxRowsTile);
    TRESHAPE(oldScaleRowsView, oldScaleRowsTile);
    TRESHAPE(rowSumRowsView, rowSumRowsTile);
    TRESHAPE(sumStateRowsView, sumStateRowsTile);

    constexpr uint32_t kAccumUbBase = 0x4000;
    constexpr uint32_t kAccumHeadBytes = kHeadDim * sizeof(float);
    constexpr int64_t scoreHeadBytes = 4 * kTileTokens * sizeof(float);
    constexpr int64_t probHeadBytes = 256 * sizeof(half);
    constexpr int64_t outHeadBytes = 4 * kHeadDim * sizeof(float);
    constexpr int64_t scoreGroupBytes = kHeadGroup * scoreHeadBytes;
    constexpr int64_t probGroupBytes = kHeadGroup * probHeadBytes;
    constexpr int64_t outGroupBytes = kHeadGroup * outHeadBytes;
    __gm__ uint8_t *scoreBase = sGm + workerIdx * scoreGroupBytes * 2;
    __gm__ uint8_t *probBase = pGm + workerIdx * probGroupBytes * 2;
    __gm__ uint8_t *outTmpBase = oTmpGm + workerIdx * outGroupBytes * 2;

    const int64_t processRounds = (processNum + workerNum - 1) / workerNum;
    const int32_t stageTileCount = (ctx.kvSplitPerCore + kTileTokens - 1) / kTileTokens;
    for (int64_t processRound = 0; processRound < processRounds; ++processRound) {
        const int64_t process = processRound * workerNum + workerIdx;
        bool validProcess = process < processNum;
        int32_t batchIndex = 0;
        int32_t curHeadNum = 0;
        int32_t startHead = 0;
        int32_t tileCount = 0;
        int32_t curKvSeqLen = 0;
        int32_t curSplit = 0;
        uint64_t lBase = 0;
        uint64_t oFdBase = 0;
        if (validProcess) {
            int32_t curBatchSlot = static_cast<int32_t>(process / (corePerBatch * ctx.kvSplitCoreNum));
            int32_t paraBase = ctx.headSize + curBatchSlot * ctx.paraSize;
            const int32_t sortedBatch = LoadTilingI32(tilingParaGm, paraBase + kParaBatchIndex);
            paraBase = ctx.headSize + sortedBatch * ctx.paraSize;
            batchIndex = LoadTilingI32(tilingParaGm, paraBase + 8);
            const int32_t kvSeqLen = LoadTilingI32(tilingParaGm, paraBase + kParaKvSeqLen);
            const int32_t kvSeqLenAlign = ((kvSeqLen + ctx.blockSize - 1) / ctx.blockSize) * ctx.blockSize;
            const int32_t kvLoop = (kvSeqLenAlign + ctx.kvSplitPerCore - 1) / ctx.kvSplitPerCore;
            curSplit = static_cast<int32_t>(process % ctx.kvSplitCoreNum);
            validProcess = kvSeqLen > 0 && curSplit < kvLoop;
            if (validProcess) {
                const int32_t curHeadBlock = static_cast<int32_t>((process / ctx.kvSplitCoreNum) % corePerBatch);
                startHead = curHeadBlock * formerHeadSplit;
                curHeadNum = formerHeadSplit;
                if (curHeadBlock == corePerBatch - 1) {
                    curHeadNum = ctx.numHeads - curHeadBlock * formerHeadSplit;
                }
                const int32_t startKv = curSplit * ctx.kvSplitPerCore;
                curKvSeqLen = ctx.kvSplitPerCore;
                if (curSplit == kvLoop - 1) {
                    curKvSeqLen = kvSeqLen - startKv;
                }
                tileCount = (curKvSeqLen + kTileTokens - 1) / kTileTokens;
                lBase = LoadTilingOffset64(tilingParaGm, paraBase, 11, 12);
                oFdBase = LoadTilingOffset64(tilingParaGm, paraBase, 15, 16);
            }
        }

        const int32_t subHeadBegin = subBlockId == 0 ? 0 : curHeadNum / 2;
        const int32_t subHeadEnd = subBlockId == 0 ? curHeadNum / 2 : curHeadNum;
        float maxScore[kMaxHeadsPerProcess];
        float sumExp[kMaxHeadsPerProcess];
        float oldScaleByHead[kMaxHeadsPerProcess];
        if (activeSubBlock && validProcess) {
            for (int32_t headLocal = subHeadBegin; headLocal < subHeadEnd; ++headLocal) {
                maxScore[headLocal] = -3.4028234663852886e38f;
                sumExp[headLocal] = 0.0f;
                oldScaleByHead[headLocal] = 0.0f;
                TASSIGN(weightedTile, kAccumUbBase + static_cast<uint32_t>(headLocal) * kAccumHeadBytes);
                TEXPANDS(weightedTile, 0.0f);
                pipe_barrier(PIPE_V);
            }
        }

        for (int32_t headGroup = 0; headGroup < maxHeadGroups; ++headGroup) {
            const int32_t groupHeadBase = headGroup * kHeadGroup;
            const bool validGroup = validProcess && groupHeadBase < curHeadNum;
            for (int32_t tile = 0; tile < stageTileCount; ++tile) {
                const bool activeTile = validProcess && tile < tileCount;
                const int32_t validTokens = activeTile ? (((tile + 1) * kTileTokens <= curKvSeqLen) ? kTileTokens :
                    (curKvSeqLen - tile * kTileTokens)) : 0;
                (void)validTokens;
                const uint8_t slot = static_cast<uint8_t>(tile & 1);
                const bool activeGroup = validGroup && activeTile;
                wait_flag_dev(PtoPaSlotFlag(PTO_PA_RAW_QK_READY, slot));
                if (activeSubBlock && activeGroup) {
                    for (int32_t headInGroupBase = 0; headInGroupBase < kHeadGroup; headInGroupBase += headsPerKv) {
                        const int32_t headLocal = groupHeadBase + headInGroupBase;
                        if (headLocal >= curHeadNum) {
                            break;
                        }
                        if (headsPerKv != 4 || headLocal < subHeadBegin || headLocal + 4 > subHeadEnd) {
                            continue;
                        }
                        ScoreRowsGlobal scoreGlobal(reinterpret_cast<__gm__ float *>(scoreBase +
                            static_cast<int64_t>(slot) * scoreGroupBytes +
                            static_cast<int64_t>(headInGroupBase) * scoreHeadBytes));
                        TLOAD(scoreRowsTile, scoreGlobal);
                        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                        TMULS(scoreRowsTile, scoreRowsTile, ctx.scale);
                        pipe_barrier(PIPE_V);
                        for (int32_t row = 0; row < 4; ++row) {
                            maxStateRowsTile.data()[row] = maxScore[headLocal + row];
                            sumStateRowsTile.data()[row] = sumExp[headLocal + row];
                        }
                        TROWMAX(rowMaxRowsTile, scoreRowsTile, scoreRowsWorkTile);
                        pipe_barrier(PIPE_V);
                        TMAX(newMaxRowsView, rowMaxRowsView, maxStateRowsView);
                        pipe_barrier(PIPE_V);
                        TSUB(oldScaleRowsView, maxStateRowsView, newMaxRowsView);
                        pipe_barrier(PIPE_V);
                        TEXP(oldScaleRowsView, oldScaleRowsView);
                        pipe_barrier(PIPE_V);
                        if (tile == 0) {
                            TEXPANDS(oldScaleRowsView, 0.0f);
                            pipe_barrier(PIPE_V);
                        }
                        TROWEXPANDSUB(scoreRowsWorkTile, scoreRowsTile, newMaxRowsTile);
                        pipe_barrier(PIPE_V);
                        TEXP(scoreRowsWorkTile, scoreRowsWorkTile);
                        pipe_barrier(PIPE_V);
                        TROWSUM(rowSumRowsTile, scoreRowsWorkTile, scoreRowsTile);
                        pipe_barrier(PIPE_V);
                        TMUL(sumStateRowsView, sumStateRowsView, oldScaleRowsView);
                        pipe_barrier(PIPE_V);
                        TADD(sumStateRowsView, sumStateRowsView, rowSumRowsView);
                        pipe_barrier(PIPE_V);
                        __gm__ half *probScratch = reinterpret_cast<__gm__ half *>(probBase +
                            static_cast<int64_t>(slot) * probGroupBytes);
                        __ubuf__ half *probHalf = reinterpret_cast<__ubuf__ half *>(probHalfTile.data());
                        for (int32_t row = 0; row < 4; ++row) {
                            maxScore[headLocal + row] = newMaxRowsTile.data()[row];
                            sumExp[headLocal + row] = sumStateRowsTile.data()[row];
                            oldScaleByHead[headLocal + row] = oldScaleRowsTile.data()[row];
                            __ubuf__ float *probFloat = reinterpret_cast<__ubuf__ float *>(
                                scoreRowsWorkTile.data() + row * kTileTokens);
                            vconv_f322f16a(probHalf, probFloat, 2, 1, 1, 4, 8);
                            pipe_barrier(PIPE_V);
                            ProbRowGlobal probRowGlobal(probScratch +
                                static_cast<int64_t>(headInGroupBase + row) * kTileTokens);
                            TSTORE(probRowGlobal, probHalfTile);
                        }
                    }
                    DdrFenceBeforePtoAivReduce();
                }
                PtoPaSignalFromVec(PtoPaSlotFlag(PTO_PA_RAW_P_READY, slot));
                wait_flag_dev(PtoPaSlotFlag(PTO_PA_RAW_PV_READY, slot));
                if (activeSubBlock && activeGroup) {
                    for (int32_t headInGroupBase = 0; headInGroupBase < kHeadGroup; headInGroupBase += headsPerKv) {
                        const int32_t headLocal = groupHeadBase + headInGroupBase;
                        if (headLocal >= curHeadNum) {
                            break;
                        }
                        if (headsPerKv != 4 || headLocal < subHeadBegin || headLocal + 4 > subHeadEnd) {
                            continue;
                        }
                        for (int32_t row = 0; row < 4; ++row) {
                            oldScaleRowsTile.data()[row] = oldScaleByHead[headLocal + row];
                        }
                        OutRowsGlobal outGlobal(reinterpret_cast<__gm__ float *>(outTmpBase +
                            static_cast<int64_t>(slot) * outGroupBytes +
                            static_cast<int64_t>(headInGroupBase) * outHeadBytes));
                        TLOAD(pvRowsTile, outGlobal);
                        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                        VecFloat4x128 weightedRowsTile;
                        TASSIGN(weightedRowsTile, kAccumUbBase + static_cast<uint32_t>(headLocal) * kAccumHeadBytes);
                        TROWEXPANDMUL(weightedRowsTile, weightedRowsTile, oldScaleRowsTile);
                        pipe_barrier(PIPE_V);
                        TADD(weightedRowsTile, weightedRowsTile, pvRowsTile);
                        pipe_barrier(PIPE_V);
                    }
                }
                PtoPaSignalFreeFromVec(PtoPaSlotFlag(PTO_PA_RAW_PV_FREE, slot));
            }
        }

        if (activeSubBlock && validProcess) {
            for (int32_t headLocal = subHeadBegin; headLocal < subHeadEnd; ++headLocal) {
                const int32_t head = startHead + headLocal;
                const float invSum = sumExp[headLocal] > 0.0f ? 1.0f / sumExp[headLocal] : 0.0f;
                const uint64_t outOffset = oFdBase * ctx.kvSplitCoreNum +
                    static_cast<uint64_t>(head) * ctx.headDim * ctx.kvSplitCoreNum +
                    static_cast<uint64_t>(curSplit) * ctx.headDim;
                const uint64_t lOffset = lBase + static_cast<uint64_t>(head) * ctx.kvSplitCoreNum + curSplit;
                partialL[lOffset] = maxScore[headLocal] + PtoLogScalar(scalarMathTile, sumExp[headLocal]);
                TASSIGN(weightedTile, kAccumUbBase + static_cast<uint32_t>(headLocal) * kAccumHeadBytes);
                TMULS(weightedTile, weightedTile, invSum);
                pipe_barrier(PIPE_V);
                OutGlobal weightedGlobal(reinterpret_cast<__gm__ float *>(partialOut + outOffset));
                TSTORE(weightedGlobal, weightedTile);
            }
        }
    }

    DdrFenceBeforePtoAivReduce();
    ffts_cross_core_sync(PIPE_MTE3, PtoPaGetFftsMsg(0x0, PTO_PA_REDUCE_READY_DECODER));
    wait_flag_dev(PTO_PA_REDUCE_READY_DECODER);
    if (!combineSubBlock) {
        pipe_barrier(PIPE_ALL);
        return;
    }
    const int32_t effectiveBatch = ctx.decoderBatch > 0 ? ctx.decoderBatch : ctx.batch;
    const int64_t totalRows = static_cast<int64_t>(effectiveBatch) * ctx.numHeads;
    const int64_t combineWorkerIdx = workerIdx;
    const int64_t combineWorkerNum = workerNum;
    for (int64_t row = combineWorkerIdx; row < totalRows; row += combineWorkerNum) {
        const int32_t head = static_cast<int32_t>(row % ctx.numHeads);
        const int32_t batchSlot = static_cast<int32_t>(row / ctx.numHeads);
        int32_t paraBase = ctx.headSize + batchSlot * ctx.paraSize;
        const int32_t sortedBatch = LoadTilingI32(tilingParaGm, paraBase + kParaBatchIndex);
        paraBase = ctx.headSize + sortedBatch * ctx.paraSize;
        const int32_t batchIndex = LoadTilingI32(tilingParaGm, paraBase + 8);
        const int32_t kvSeqLen = LoadTilingI32(tilingParaGm, paraBase + kParaKvSeqLen);
        if (kvSeqLen <= 0) {
            continue;
        }
        const int32_t kvSeqLenAlign = ((kvSeqLen + ctx.blockSize - 1) / ctx.blockSize) * ctx.blockSize;
        const int32_t kvLoop = (kvSeqLenAlign + ctx.kvSplitPerCore - 1) / ctx.kvSplitPerCore;
        const uint64_t lBase = LoadTilingOffset64(tilingParaGm, paraBase, 11, 12);
        const uint64_t oFdBase = LoadTilingOffset64(tilingParaGm, paraBase, 15, 16);
        __ubuf__ float *splitScale = reinterpret_cast<__ubuf__ float *>((uintptr_t)0x3200);
        __ubuf__ float *splitReduce = reinterpret_cast<__ubuf__ float *>((uintptr_t)0x3400);
        float lMax = -3.4028234663852886e38f;
        for (int32_t split = 0; split < kvLoop; ++split) {
            const float lValue = partialL[lBase + static_cast<uint64_t>(head) * ctx.kvSplitCoreNum + split];
            splitScale[split] = lValue;
            lMax = lValue > lMax ? lValue : lMax;
        }
        set_flag(PIPE_S, PIPE_V, EVENT_ID2);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID2);
        PtoPaSetFloatVectorMask(static_cast<uint32_t>(kvLoop));
        vadds(splitScale, splitScale, -lMax, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vexp(splitScale, splitScale, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vcadd(splitReduce, splitScale, 1, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        const float denom = splitReduce[0];
        float invDenom = denom > 0.0f ? 1.0f / denom : 0.0f;
        PtoPaSetFloatVectorMask(static_cast<uint32_t>(kvLoop));
        vmuls(splitScale, splitScale, static_cast<float>(invDenom), 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        TASSIGN(weightedTile, 0x0000);
        TEXPANDS(weightedTile, 0.0f);
        pipe_barrier(PIPE_V);
        for (int32_t split = 0; split < kvLoop; ++split) {
            const uint64_t outOffset = oFdBase * ctx.kvSplitCoreNum +
                static_cast<uint64_t>(head) * ctx.headDim * ctx.kvSplitCoreNum +
                static_cast<uint64_t>(split) * ctx.headDim;
            OutGlobal splitGlobal(partialOut + outOffset);
            TLOAD(pvTile, splitGlobal);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TMULS(pvTile, pvTile, splitScale[split]);
            pipe_barrier(PIPE_V);
            TADD(weightedTile, weightedTile, pvTile);
            pipe_barrier(PIPE_V);
        }
        const int64_t outBase = (static_cast<int64_t>(batchIndex) * ctx.numHeads + head) * ctx.headDim;
        for (int32_t dim = 0; dim < ctx.headDim; ++dim) {
            StoreOutputFp16(oGm, outBase + dim, weightedTile.data()[dim]);
        }
    }
    pipe_barrier(PIPE_ALL);
}
#endif

#ifdef __DAV_C220_VEC__
AICORE inline void RunPtoPagedAttentionDecodeSplitKV(
    __gm__ uint8_t *qGm,
    __gm__ uint8_t *kGm,
    __gm__ uint8_t *vGm,
    __gm__ uint8_t *blockTablesGm,
    __gm__ uint8_t *oGm,
    __gm__ uint8_t *oCoreTmpGm,
    __gm__ uint8_t *lGm,
    __gm__ uint8_t *tilingParaGm,
    int64_t workerIdx,
    int64_t workerNum,
    uint32_t subBlockId)
{
    constexpr int32_t kHeadDim = 128;
    const PaTilingContext ctx = LoadPaTilingContext(tilingParaGm);
    if (workerIdx < 0 || workerNum <= 0 || ctx.headDim != kHeadDim || ctx.headDimV != kHeadDim ||
        ctx.blockSize != PA_TILE_TOKENS || ctx.kvSplitCoreNum <= 1 || ctx.numHeads <= 0 || ctx.kvHeads <= 0 ||
        ctx.numHeads % ctx.kvHeads != 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    const int32_t maxBlocksPerQuery = ctx.maxBlocksPerQuery > 0 ? ctx.maxBlocksPerQuery :
        (ctx.maxKvSeqLen + ctx.blockSize - 1) / ctx.blockSize;
    const int32_t headsPerKv = ctx.numHeads / ctx.kvHeads;
    const int32_t formerHeadSplit = ctx.formerHeadSplit > 0 ? ctx.formerHeadSplit : 1;
    const int32_t corePerBatch = (ctx.numHeads + formerHeadSplit - 1) / formerHeadSplit;
    const int64_t processNum = static_cast<int64_t>(ctx.formerBatch) * corePerBatch * ctx.kvSplitCoreNum;
    __gm__ float *partialOut = reinterpret_cast<__gm__ float *>(oCoreTmpGm);
    __gm__ float *partialL = reinterpret_cast<__gm__ float *>(lGm);

    using VecHalf128 = Tile<TileType::Vec, half, 1, kHeadDim, BLayout::RowMajor, 1, kHeadDim>;
    using VecFloat128 = Tile<TileType::Vec, float, 1, kHeadDim, BLayout::RowMajor, 1, kHeadDim>;
    using VecFloat8 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8>;
    using GlobalHalf128 =
        GlobalTensor<half, Shape<1, 1, 1, 1, kHeadDim>, Stride<1, 1, 1, kHeadDim, 1>>;

    VecHalf128 qHalfTile;
    VecFloat128 qFloatTile;
    VecHalf128 kHalfTile;
    VecFloat128 kFloatTile;
    VecFloat128 qkProductTile;
    VecFloat8 scoreTile;
    VecFloat128 reduceTmpTile;
    VecHalf128 vHalfTile;
    VecFloat128 vFloatTile;
    VecFloat128 weightedTile;
    VecFloat8 scalarMathTile;
    TASSIGN(qHalfTile, 0x0800);
    TASSIGN(qFloatTile, 0x1000);
    TASSIGN(kHalfTile, 0x1800);
    TASSIGN(kFloatTile, 0x2000);
    TASSIGN(qkProductTile, 0x2800);
    TASSIGN(scoreTile, 0x3000);
    TASSIGN(reduceTmpTile, 0x3800);
    TASSIGN(vHalfTile, 0x4000);
    TASSIGN(vFloatTile, 0x4800);
    TASSIGN(weightedTile, 0x5000);
    TASSIGN(scalarMathTile, 0x5800);

    for (int64_t process = workerIdx; process < processNum; process += workerNum) {
        int32_t curBatchSlot = static_cast<int32_t>(process / (corePerBatch * ctx.kvSplitCoreNum));
        int32_t paraBase = ctx.headSize + curBatchSlot * ctx.paraSize;
        const int32_t sortedBatch = LoadTilingI32(tilingParaGm, paraBase + kParaBatchIndex);
        paraBase = ctx.headSize + sortedBatch * ctx.paraSize;
        const int32_t batchIndex = LoadTilingI32(tilingParaGm, paraBase + 8);
        const int32_t kvSeqLen = LoadTilingI32(tilingParaGm, paraBase + kParaKvSeqLen);
        if (kvSeqLen <= 0) {
            continue;
        }

        const int32_t kvSeqLenAlign = ((kvSeqLen + ctx.blockSize - 1) / ctx.blockSize) * ctx.blockSize;
        const int32_t kvLoop = (kvSeqLenAlign + ctx.kvSplitPerCore - 1) / ctx.kvSplitPerCore;
        const int32_t curSplit = static_cast<int32_t>(process % ctx.kvSplitCoreNum);
        if (curSplit >= kvLoop) {
            continue;
        }

        const int32_t curHeadBlock = static_cast<int32_t>((process / ctx.kvSplitCoreNum) % corePerBatch);
        const int32_t startHead = curHeadBlock * formerHeadSplit;
        int32_t curHeadNum = formerHeadSplit;
        if (curHeadBlock == corePerBatch - 1) {
            curHeadNum = ctx.numHeads - curHeadBlock * formerHeadSplit;
        }
        const int32_t startKv = curSplit * ctx.kvSplitPerCore;
        int32_t curKvSeqLen = ctx.kvSplitPerCore;
        if (curSplit == kvLoop - 1) {
            curKvSeqLen = kvSeqLen - startKv;
        }

        const uint64_t lBase = LoadTilingOffset64(tilingParaGm, paraBase, 11, 12);
        const uint64_t oFdBase = LoadTilingOffset64(tilingParaGm, paraBase, 15, 16);
        const int32_t headBegin = subBlockId == 0 ? 0 : curHeadNum / 2;
        const int32_t headEnd = subBlockId == 0 ? curHeadNum / 2 : curHeadNum;
        for (int32_t headLocal = headBegin; headLocal < headEnd; ++headLocal) {
            const int32_t head = startHead + headLocal;
            const int32_t kvHead = head / headsPerKv;
            const int64_t qBase = (static_cast<int64_t>(batchIndex) * ctx.numHeads + head) * ctx.headDim;
            GlobalHalf128 qGlobal(reinterpret_cast<__gm__ half *>(qGm) + qBase);
            TLOAD(qHalfTile, qGlobal);
            pipe_barrier(PIPE_ALL);
            TCVT(qFloatTile, qHalfTile, RoundMode::CAST_NONE);
            pipe_barrier(PIPE_V);

            float maxScore = -3.4028234663852886e38f;
            float sumExp = 0.0f;
            TEXPANDS(weightedTile, 0.0f);
            pipe_barrier(PIPE_V);
            for (int32_t relPos = 0; relPos < curKvSeqLen; ++relPos) {
                const int32_t pos = startKv + relPos;
                int32_t blockId = 0;
                int32_t offsetInBlock = 0;
                ResolvePagedPosition(blockTablesGm, batchIndex, maxBlocksPerQuery, pos, ctx.blockSize, blockId,
                    offsetInBlock);
                const int64_t kvOffset = (((static_cast<int64_t>(blockId) * ctx.blockSize + offsetInBlock) *
                    ctx.kvHeads + kvHead) * ctx.headDim);

                GlobalHalf128 kGlobal(reinterpret_cast<__gm__ half *>(kGm) + kvOffset);
                TLOAD(kHalfTile, kGlobal);
                pipe_barrier(PIPE_ALL);
                TCVT(kFloatTile, kHalfTile, RoundMode::CAST_NONE);
                pipe_barrier(PIPE_V);
                TMUL(qkProductTile, qFloatTile, kFloatTile);
                pipe_barrier(PIPE_V);
                TROWSUM(scoreTile, qkProductTile, reduceTmpTile);
                pipe_barrier(PIPE_V);
                const float score = scoreTile.data()[0] * ctx.scale;
                const float newMax = score > maxScore ? score : maxScore;
                float oldScale = 0.0f;
                if (relPos != 0) {
                    scalarMathTile.data()[0] = maxScore - newMax;
                    TEXP(scalarMathTile, scalarMathTile);
                    pipe_barrier(PIPE_V);
                    oldScale = scalarMathTile.data()[0];
                }
                scalarMathTile.data()[0] = score - newMax;
                TEXP(scalarMathTile, scalarMathTile);
                pipe_barrier(PIPE_V);
                const float probUnnorm = scalarMathTile.data()[0];
                sumExp = sumExp * oldScale + probUnnorm;

                GlobalHalf128 vGlobal(reinterpret_cast<__gm__ half *>(vGm) + kvOffset);
                TMULS(weightedTile, weightedTile, oldScale);
                pipe_barrier(PIPE_V);
                TLOAD(vHalfTile, vGlobal);
                pipe_barrier(PIPE_ALL);
                TCVT(vFloatTile, vHalfTile, RoundMode::CAST_NONE);
                pipe_barrier(PIPE_V);
                TAXPY(weightedTile, vFloatTile, probUnnorm);
                pipe_barrier(PIPE_V);
                maxScore = newMax;
            }

            const float invSum = sumExp > 0.0f ? 1.0f / sumExp : 0.0f;
            const uint64_t outOffset = oFdBase * ctx.kvSplitCoreNum +
                static_cast<uint64_t>(head) * ctx.headDim * ctx.kvSplitCoreNum +
                static_cast<uint64_t>(curSplit) * ctx.headDim;
            const uint64_t lOffset = lBase + static_cast<uint64_t>(head) * ctx.kvSplitCoreNum + curSplit;
            float logSumExp = -3.4028234663852886e38f;
            if (sumExp > 0.0f) {
                scalarMathTile.data()[0] = sumExp;
                TLOG(scalarMathTile, scalarMathTile);
                pipe_barrier(PIPE_V);
                logSumExp = scalarMathTile.data()[0];
            }
            partialL[lOffset] = maxScore + logSumExp;
            for (int32_t dim = 0; dim < kHeadDim; ++dim) {
                partialOut[outOffset + dim] = weightedTile.data()[dim] * invSum;
            }
        }
    }

    DdrFenceBeforePtoAivReduce();
    ffts_cross_core_sync(PIPE_MTE3, PtoPaGetFftsMsg(0x0, PTO_PA_REDUCE_READY_DECODER));
    wait_flag_dev(PTO_PA_REDUCE_READY_DECODER);

    const int32_t effectiveBatch = ctx.decoderBatch > 0 ? ctx.decoderBatch : ctx.batch;
    const int64_t totalRows = static_cast<int64_t>(effectiveBatch) * ctx.numHeads;
    const int64_t combineWorkerIdx = workerIdx * 2 + static_cast<int64_t>(subBlockId);
    const int64_t combineWorkerNum = workerNum * 2;
    for (int64_t row = combineWorkerIdx; row < totalRows; row += combineWorkerNum) {
        const int32_t head = static_cast<int32_t>(row % ctx.numHeads);
        const int32_t batchSlot = static_cast<int32_t>(row / ctx.numHeads);
        int32_t paraBase = ctx.headSize + batchSlot * ctx.paraSize;
        const int32_t sortedBatch = LoadTilingI32(tilingParaGm, paraBase + kParaBatchIndex);
        paraBase = ctx.headSize + sortedBatch * ctx.paraSize;
        const int32_t batchIndex = LoadTilingI32(tilingParaGm, paraBase + 8);
        const int32_t kvSeqLen = LoadTilingI32(tilingParaGm, paraBase + kParaKvSeqLen);
        if (kvSeqLen <= 0) {
            continue;
        }

        const int32_t kvSeqLenAlign = ((kvSeqLen + ctx.blockSize - 1) / ctx.blockSize) * ctx.blockSize;
        const int32_t kvLoop = (kvSeqLenAlign + ctx.kvSplitPerCore - 1) / ctx.kvSplitPerCore;
        const uint64_t lBase = LoadTilingOffset64(tilingParaGm, paraBase, 11, 12);
        const uint64_t oFdBase = LoadTilingOffset64(tilingParaGm, paraBase, 15, 16);
        float lMax = -3.4028234663852886e38f;
        for (int32_t split = 0; split < kvLoop; ++split) {
            const float lValue = partialL[lBase + static_cast<uint64_t>(head) * ctx.kvSplitCoreNum + split];
            lMax = lValue > lMax ? lValue : lMax;
        }
        float denom = 0.0f;
        float splitScale[64];
        for (int32_t split = 0; split < kvLoop; ++split) {
            const float lValue = partialL[lBase + static_cast<uint64_t>(head) * ctx.kvSplitCoreNum + split];
            scalarMathTile.data()[0] = lValue - lMax;
            TEXP(scalarMathTile, scalarMathTile);
            pipe_barrier(PIPE_V);
            const float scale = scalarMathTile.data()[0];
            splitScale[split] = scale;
            denom += scale;
        }
        const float invDenom = denom > 0.0f ? 1.0f / denom : 0.0f;
        for (int32_t split = 0; split < kvLoop; ++split) {
            splitScale[split] *= invDenom;
        }
        const int64_t outBase = (static_cast<int64_t>(batchIndex) * ctx.numHeads + head) * ctx.headDim;
        for (int32_t dim = 0; dim < kHeadDim; ++dim) {
            float value = 0.0f;
            for (int32_t split = 0; split < kvLoop; ++split) {
                const uint64_t outOffset = oFdBase * ctx.kvSplitCoreNum +
                    static_cast<uint64_t>(head) * ctx.headDim * ctx.kvSplitCoreNum +
                    static_cast<uint64_t>(split) * ctx.headDim;
                value += partialOut[outOffset + dim] * splitScale[split];
            }
            StoreOutputFp16(oGm, outBase + dim, value);
        }
    }
    pipe_barrier(PIPE_ALL);
}
#endif

AICORE inline void RunPtoPagedAttentionDecode(
    __gm__ uint8_t *qGm,
    __gm__ uint8_t *kGm,
    __gm__ uint8_t *vGm,
    __gm__ uint8_t *blockTablesGm,
    __gm__ uint8_t *oGm,
    __gm__ uint8_t *tilingParaGm,
    int64_t workerIdx,
    int64_t workerNum)
{
    constexpr int32_t kMaxHeadDim = 128;
    const PaTilingContext ctx = LoadPaTilingContext(tilingParaGm);
    const int32_t effectiveBatch = ctx.decoderBatch > 0 ? ctx.decoderBatch : ctx.batch;
    const int32_t maxBlocksPerQuery = ctx.maxBlocksPerQuery > 0 ? ctx.maxBlocksPerQuery :
        (ctx.maxKvSeqLen + ctx.blockSize - 1) / ctx.blockSize;
    const int32_t headsPerKv = ctx.numHeads / ctx.kvHeads;
    const int64_t totalRows = static_cast<int64_t>(effectiveBatch) * ctx.numHeads;
    if (workerIdx < 0 || workerNum <= 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    if (ctx.headDim > kMaxHeadDim) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    using DecodeScalarTile = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8>;
    DecodeScalarTile decodeScalarMathTile;
    TASSIGN(decodeScalarMathTile, 0x5800);

    if (ctx.headDim == kMaxHeadDim) {
        constexpr uint64_t kWeightedUb = 0x0000;
        constexpr uint64_t kQHalfUb = 0x0800;
        constexpr uint64_t kQFloatUb = 0x1000;
        constexpr uint64_t kKHalfUb = 0x1800;
        constexpr uint64_t kKFloatUb = 0x2000;
        constexpr uint64_t kQKProductUb = 0x2800;
        constexpr uint64_t kScoreUb = 0x3000;
        constexpr uint64_t kReduceTmpUb = 0x3800;
        constexpr uint64_t kVHalfUb = 0x4000;
        constexpr uint64_t kVFloatUb = 0x4800;
        constexpr uint64_t kOutHalfUb = 0x5000;

        using VecHalf128 = Tile<TileType::Vec, half, 1, kMaxHeadDim, BLayout::RowMajor, 1, kMaxHeadDim>;
        using VecFloat128 = Tile<TileType::Vec, float, 1, kMaxHeadDim, BLayout::RowMajor, 1, kMaxHeadDim>;
        using VecFloat8 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8>;
        using GlobalHalf128 =
            GlobalTensor<half, Shape<1, 1, 1, 1, kMaxHeadDim>, Stride<1, 1, 1, kMaxHeadDim, 1>>;

        VecFloat128 weightedTile;
        VecHalf128 qHalfTile;
        VecFloat128 qFloatTile;
        VecHalf128 kHalfTile;
        VecFloat128 kFloatTile;
        VecFloat128 qkProductTile;
        VecFloat8 scoreTile;
        VecFloat128 reduceTmpTile;
        VecHalf128 vHalfTile;
        VecFloat128 vFloatTile;
        VecHalf128 outHalfTile;

        TASSIGN(weightedTile, kWeightedUb);
        TASSIGN(qHalfTile, kQHalfUb);
        TASSIGN(qFloatTile, kQFloatUb);
        TASSIGN(kHalfTile, kKHalfUb);
        TASSIGN(kFloatTile, kKFloatUb);
        TASSIGN(qkProductTile, kQKProductUb);
        TASSIGN(scoreTile, kScoreUb);
        TASSIGN(reduceTmpTile, kReduceTmpUb);
        TASSIGN(vHalfTile, kVHalfUb);
        TASSIGN(vFloatTile, kVFloatUb);
        TASSIGN(outHalfTile, kOutHalfUb);

        for (int64_t row = workerIdx; row < totalRows; row += workerNum) {
            const int32_t head = static_cast<int32_t>(row % ctx.numHeads);
            const int32_t batchSlot = static_cast<int32_t>(row / ctx.numHeads);
            const int32_t paraBase = ctx.headSize + batchSlot * ctx.paraSize;
            const int32_t kvSeqLen = LoadTilingI32(tilingParaGm, paraBase + kParaKvSeqLen);
            const int32_t batchIndex = LoadTilingI32(tilingParaGm, paraBase + kParaBatchIndex);
            const int32_t kvHead = head / headsPerKv;
            const int64_t qBase = (static_cast<int64_t>(batchIndex) * ctx.numHeads + head) * ctx.headDim;
            GlobalHalf128 qGlobal(reinterpret_cast<__gm__ half *>(qGm) + qBase);
            TLOAD(qHalfTile, qGlobal);
            pipe_barrier(PIPE_ALL);
            TCVT(qFloatTile, qHalfTile, RoundMode::CAST_NONE);
            pipe_barrier(PIPE_ALL);

            float maxScore = -3.4028234663852886e38f;
            float sumExp = 0.0f;
            TEXPANDS(weightedTile, 0.0f);
            pipe_barrier(PIPE_ALL);

            for (int32_t pos = 0; pos < kvSeqLen; ++pos) {
                int32_t blockId = 0;
                int32_t offsetInBlock = 0;
                ResolvePagedPosition(blockTablesGm, batchIndex, maxBlocksPerQuery, pos, ctx.blockSize, blockId,
                    offsetInBlock);
                const int64_t kvOffset = (((static_cast<int64_t>(blockId) * ctx.blockSize + offsetInBlock) *
                    ctx.kvHeads + kvHead) * ctx.headDim);

                GlobalHalf128 kGlobal(reinterpret_cast<__gm__ half *>(kGm) + kvOffset);
                TLOAD(kHalfTile, kGlobal);
                pipe_barrier(PIPE_ALL);
                TCVT(kFloatTile, kHalfTile, RoundMode::CAST_NONE);
                pipe_barrier(PIPE_ALL);
                TMUL(qkProductTile, qFloatTile, kFloatTile);
                pipe_barrier(PIPE_ALL);
                TROWSUM(scoreTile, qkProductTile, reduceTmpTile);
                pipe_barrier(PIPE_ALL);
                const float rawScore = scoreTile.data()[0];
                const float score = rawScore * ctx.scale;

                const float newMax = score > maxScore ? score : maxScore;
                const float oldScale = (pos == 0) ? 0.0f : PtoExpScalar(decodeScalarMathTile, maxScore - newMax);
                const float probUnnorm = PtoExpScalar(decodeScalarMathTile, score - newMax);
                sumExp = sumExp * oldScale + probUnnorm;

                GlobalHalf128 vGlobal(reinterpret_cast<__gm__ half *>(vGm) + kvOffset);
                TMULS(weightedTile, weightedTile, oldScale);
                pipe_barrier(PIPE_ALL);
                TLOAD(vHalfTile, vGlobal);
                pipe_barrier(PIPE_ALL);
                TCVT(vFloatTile, vHalfTile, RoundMode::CAST_NONE);
                pipe_barrier(PIPE_ALL);
                TAXPY(weightedTile, vFloatTile, probUnnorm);
                pipe_barrier(PIPE_ALL);
                maxScore = newMax;
            }

            const float invSum = sumExp > 0.0f ? 1.0f / sumExp : 0.0f;
            const int64_t outBase = (static_cast<int64_t>(batchIndex) * ctx.numHeads + head) * ctx.headDim;
            GlobalHalf128 outGlobal(reinterpret_cast<__gm__ half *>(oGm) + outBase);
            TMULS(weightedTile, weightedTile, invSum);
            pipe_barrier(PIPE_ALL);
            TCVT(outHalfTile, weightedTile, RoundMode::CAST_RINT);
            pipe_barrier(PIPE_ALL);
            TSTORE(outGlobal, outHalfTile);
            pipe_barrier(PIPE_ALL);
        }

        pipe_barrier(PIPE_ALL);
        return;
    }

    for (int64_t row = workerIdx; row < totalRows; row += workerNum) {
        const int32_t head = static_cast<int32_t>(row % ctx.numHeads);
        const int32_t batchSlot = static_cast<int32_t>(row / ctx.numHeads);
        const int32_t paraBase = ctx.headSize + batchSlot * ctx.paraSize;
        const int32_t kvSeqLen = LoadTilingI32(tilingParaGm, paraBase + kParaKvSeqLen);
        const int32_t batchIndex = LoadTilingI32(tilingParaGm, paraBase + kParaBatchIndex);
        const int32_t kvHead = head / headsPerKv;

        float qValues[kMaxHeadDim];
        const int64_t qBase = (static_cast<int64_t>(batchIndex) * ctx.numHeads + head) * ctx.headDim;
        for (int32_t dim = 0; dim < ctx.headDim; ++dim) {
            qValues[dim] = LoadFp16(qGm, qBase + dim);
        }

        float maxScore = -3.4028234663852886e38f;
        float sumExp = 0.0f;
        float weighted[kMaxHeadDim];
        for (int32_t dim = 0; dim < ctx.headDim; ++dim) {
            weighted[dim] = 0.0f;
        }

        for (int32_t pos = 0; pos < kvSeqLen; ++pos) {
            int32_t blockId = 0;
            int32_t offsetInBlock = 0;
            ResolvePagedPosition(blockTablesGm, batchIndex, maxBlocksPerQuery, pos, ctx.blockSize, blockId, offsetInBlock);
            const float score = ComputeScoreByBlock(qValues, kGm, blockId, offsetInBlock, ctx.blockSize, kvHead,
                ctx.headDim, ctx.kvHeads, ctx.scale);
            const bool updateMax = score > maxScore;
            const float newMax = updateMax ? score : maxScore;
            const float oldScale = (pos == 0) ? 0.0f : PtoExpScalar(decodeScalarMathTile, maxScore - newMax);
            const float probUnnorm = PtoExpScalar(decodeScalarMathTile, score - newMax);
            sumExp = sumExp * oldScale + probUnnorm;
            for (int32_t dim = 0; dim < ctx.headDim; ++dim) {
                const float value = LoadPagedVByBlock(vGm, blockId, offsetInBlock, ctx.blockSize, ctx.kvHeads, kvHead, ctx.headDim, dim);
                weighted[dim] = weighted[dim] * oldScale + probUnnorm * value;
            }
            maxScore = newMax;
        }

        const float invSum = sumExp > 0.0f ? 1.0f / sumExp : 0.0f;
        const int64_t outBase = (static_cast<int64_t>(batchIndex) * ctx.numHeads + head) * ctx.headDim;
        for (int32_t dim = 0; dim < ctx.headDim; ++dim) {
            StoreOutputFp16(oGm, outBase + dim, weighted[dim] * invSum);
        }
    }

    pipe_barrier(PIPE_ALL);
}

#endif
