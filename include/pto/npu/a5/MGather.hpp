/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MGATHER_HPP
#define MGATHER_HPP

#include <pto/common/utils.hpp>
#include <pto/common/constants.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/common/arch_cce_intrinsic.hpp>
#include "common.hpp"
#include "utils.hpp"

namespace pto {

#ifndef PTO_GATHER_EXEC_ENUM_DEFINED
#define PTO_GATHER_EXEC_ENUM_DEFINED
enum class GatherExec : uint8_t
{
    Scalar = 0,
    Simt = 1
};
#endif

template <typename T>
struct IsValidGatherDType {
    static constexpr bool value =
        std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int16_t> ||
        std::is_same_v<T, uint16_t> || std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
        std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float> ||
        std::is_same_v<T, hifloat8_t> || std::is_same_v<T, float8_e4m3_t> || std::is_same_v<T, float8_e5m2_t>;
};

namespace mgather_cfg {
constexpr uint32_t WARP_SIZE = 32u;
constexpr uint32_t MAX_WARPS = 32u;
constexpr uint32_t MAX_THREADS = WARP_SIZE * MAX_WARPS;
} // namespace mgather_cfg

template <GatherOOB Oob>
__simt_callee__ AICORE PTO_INLINE uint32_t gather_remap(uint32_t idx, uint32_t cap, uint32_t &doRead)
{
    if constexpr (Oob == GatherOOB::Clamp) {
        doRead = 1u;
        return (idx >= cap) ? (cap - 1u) : idx;
    } else if constexpr (Oob == GatherOOB::Undefined) {
        doRead = 1u;
        return idx;
    } else if constexpr (Oob == GatherOOB::Wrap) {
        doRead = 1u;
        return idx % cap;
    } else {
        doRead = (idx < cap) ? 1u : 0u;
        return idx;
    }
}

#ifndef PTO_TILE_OFFSET_2D_DEFINED
#define PTO_TILE_OFFSET_2D_DEFINED
template <typename Tile2D>
__simt_callee__ AICORE PTO_INLINE uint32_t tile_offset_2d(uint32_t r, uint32_t c)
{
    if constexpr (Tile2D::isRowMajor) {
        return r * static_cast<uint32_t>(Tile2D::Cols) + c;
    } else {
        return c * static_cast<uint32_t>(Tile2D::Rows) + r;
    }
}
#endif

template <typename T, typename TIdx, typename TileDst, GatherOOB Oob, uint32_t ValidRowsT, uint32_t ValidColsT,
          uint32_t TableRowsT>
AICORE __simt_vf__ LAUNCH_BOUND(1024) PTO_INLINE
    void simt_mgather_row_kernel(__ubuf__ T *__restrict__ dst, __gm__ const T *__restrict__ table,
                                 __ubuf__ const TIdx *__restrict__ indices, uint32_t validRowsRT, uint32_t validColsRT,
                                 uint32_t tableRowsRT)
{
    constexpr bool kStaticR = (ValidRowsT > 0u);
    constexpr bool kStaticC = (ValidColsT > 0u);
    constexpr bool kStaticTR = (TableRowsT > 0u);
    const uint32_t validRows = kStaticR ? ValidRowsT : validRowsRT;
    const uint32_t validCols = kStaticC ? ValidColsT : validColsRT;
    const uint32_t tableRows = kStaticTR ? TableRowsT : tableRowsRT;

    const uint32_t kRowWarps =
        (validRows == 0u) ? 1u : ((validRows < mgather_cfg::MAX_WARPS) ? validRows : mgather_cfg::MAX_WARPS);
    const uint32_t kFreeWarps = mgather_cfg::MAX_WARPS / kRowWarps;
    const uint32_t kColChunks = (validCols + mgather_cfg::WARP_SIZE - 1u) / mgather_cfg::WARP_SIZE;
    const uint32_t kWarpsPerRowRaw = (kFreeWarps < kColChunks) ? kFreeWarps : kColChunks;
    const uint32_t kWarpsPerRow = (kWarpsPerRowRaw == 0u) ? 1u : kWarpsPerRowRaw;
    const uint32_t kColStride = kWarpsPerRow * mgather_cfg::WARP_SIZE;

    const uint32_t tx = threadIdx.x;
    const uint32_t ty = threadIdx.y;
    const uint32_t rowWarp = ty % kRowWarps;
    const uint32_t colSeg = ty / kRowWarps;

#pragma unroll(1)
    for (uint32_t row = rowWarp; row < validRows; row += kRowWarps) {
        const uint32_t rawIdx = static_cast<uint32_t>(indices[row]);
        uint32_t doRead;
        const uint32_t safeIdx = gather_remap<Oob>(rawIdx, tableRows, doRead);
        __gm__ const T *srcRow = table + safeIdx * validCols;
#pragma unroll(4)
        for (uint32_t col = colSeg * mgather_cfg::WARP_SIZE + tx; col < validCols; col += kColStride) {
            const T val = doRead ? srcRow[col] : static_cast<T>(0);
            dst[tile_offset_2d<TileDst>(row, col)] = val;
        }
    }
}

template <typename T, typename TIdx, typename TileDst, typename TileIdx, GatherOOB Oob, uint32_t ValidRowsT,
          uint32_t ValidColsT, uint32_t TableSizeT>
AICORE __simt_vf__ LAUNCH_BOUND(1024) PTO_INLINE
    void simt_mgather_elem_kernel(__ubuf__ T *__restrict__ dst, __gm__ const T *__restrict__ table,
                                  __ubuf__ const TIdx *__restrict__ indices, uint32_t validRowsRT, uint32_t validColsRT,
                                  uint32_t tableSizeRT)
{
    constexpr bool kStaticR = (ValidRowsT > 0u);
    constexpr bool kStaticC = (ValidColsT > 0u);
    constexpr bool kStaticTS = (TableSizeT > 0u);
    const uint32_t validRows = kStaticR ? ValidRowsT : validRowsRT;
    const uint32_t validCols = kStaticC ? ValidColsT : validColsRT;
    const uint32_t tableSize = kStaticTS ? TableSizeT : tableSizeRT;

    const uint32_t totalElems = validRows * validCols;
    const uint32_t kNeededWarps = (totalElems + mgather_cfg::WARP_SIZE - 1u) / mgather_cfg::WARP_SIZE;
    const uint32_t kLaunchWarps =
        (kNeededWarps == 0u) ? 1u : ((kNeededWarps < mgather_cfg::MAX_WARPS) ? kNeededWarps : mgather_cfg::MAX_WARPS);
    const uint32_t kLaunchThreads = kLaunchWarps * mgather_cfg::WARP_SIZE;

    const uint32_t tx = threadIdx.x;
    const uint32_t ty = threadIdx.y;
    const uint32_t tid = ty * mgather_cfg::WARP_SIZE + tx;

#pragma unroll(1)
    for (uint32_t i = tid; i < totalElems; i += kLaunchThreads) {
        const uint32_t r = (validCols == 1u) ? i : (i / validCols);
        const uint32_t c = (validCols == 1u) ? 0u : (i - r * validCols);
        const uint32_t dstOff = tile_offset_2d<TileDst>(r, c);
        const uint32_t idxOff = tile_offset_2d<TileIdx>(r, c);
        const uint32_t rawIdx = static_cast<uint32_t>(indices[idxOff]);
        uint32_t doRead;
        const uint32_t safeIdx = gather_remap<Oob>(rawIdx, tableSize, doRead);
        dst[dstOff] = doRead ? table[safeIdx] : static_cast<T>(0);
    }
}

template <typename T, typename TIdx, GatherOOB Oob, uint32_t ValidRowsT, uint32_t ValidColsT, uint32_t TableRowsT,
          typename DstTileData, typename IdxTileData>
__tf__ AICORE void MGatherRowImpl(typename DstTileData::TileDType __out__ dst, __gm__ const T *__restrict__ tablePtr,
                                  typename IdxTileData::TileDType __in__ indices, uint32_t validRows,
                                  uint32_t validCols, uint32_t tableRows)
{
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ const TIdx *idxPtr = (__ubuf__ const TIdx *)__cce_get_tile_ptr(indices);

    const uint32_t rowWarps =
        (validRows == 0u) ? 1u : ((validRows < mgather_cfg::MAX_WARPS) ? validRows : mgather_cfg::MAX_WARPS);
    const uint32_t freeWarps = mgather_cfg::MAX_WARPS / rowWarps;
    const uint32_t colChunks = (validCols + mgather_cfg::WARP_SIZE - 1u) / mgather_cfg::WARP_SIZE;
    const uint32_t warpsPerRowRaw = (freeWarps < colChunks) ? freeWarps : colChunks;
    const uint32_t warpsPerRow = (warpsPerRowRaw == 0u) ? 1u : warpsPerRowRaw;
    const uint32_t launchWarps = rowWarps * warpsPerRow;

    cce::async_invoke<simt_mgather_row_kernel<T, TIdx, DstTileData, Oob, ValidRowsT, ValidColsT, TableRowsT>>(
        cce::dim3{mgather_cfg::WARP_SIZE, launchWarps}, dstPtr, tablePtr, idxPtr, validRows, validCols, tableRows);
}

template <typename T, typename TIdx, GatherOOB Oob, uint32_t ValidRowsT, uint32_t ValidColsT, uint32_t TableSizeT,
          typename DstTileData, typename IdxTileData>
__tf__ AICORE void MGatherElemImpl(typename DstTileData::TileDType __out__ dst, __gm__ const T *__restrict__ tablePtr,
                                   typename IdxTileData::TileDType __in__ indices, uint32_t validRows,
                                   uint32_t validCols, uint32_t tableSize)
{
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ const TIdx *idxPtr = (__ubuf__ const TIdx *)__cce_get_tile_ptr(indices);

    const uint32_t totalElems = validRows * validCols;
    const uint32_t needed = (totalElems + mgather_cfg::WARP_SIZE - 1u) / mgather_cfg::WARP_SIZE;
    const uint32_t launchWarps =
        (needed == 0u) ? 1u : ((needed < mgather_cfg::MAX_WARPS) ? needed : mgather_cfg::MAX_WARPS);
    cce::async_invoke<
        simt_mgather_elem_kernel<T, TIdx, DstTileData, IdxTileData, Oob, ValidRowsT, ValidColsT, TableSizeT>>(
        cce::dim3{mgather_cfg::WARP_SIZE, launchWarps}, dstPtr, tablePtr, idxPtr, validRows, validCols, tableSize);
}

template <typename T, typename TIdx, GatherOOB Oob, typename DstTileData, typename IdxTileData>
__tf__ AICORE void MGatherScalarImpl(typename DstTileData::TileDType __out__ dst, __gm__ const T *__restrict__ tablePtr,
                                     typename IdxTileData::TileDType __in__ indices, uint32_t tableSize)
{
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ const TIdx *idxPtr = (__ubuf__ const TIdx *)__cce_get_tile_ptr(indices);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    const uint32_t rawIdx = static_cast<uint32_t>(idxPtr[0]);
    uint32_t doRead;
    uint32_t safeIdx;
    if constexpr (Oob == GatherOOB::Undefined) {
        doRead = 1u;
        safeIdx = rawIdx;
    } else if constexpr (Oob == GatherOOB::Clamp) {
        doRead = 1u;
        safeIdx = (rawIdx >= tableSize) ? (tableSize - 1u) : rawIdx;
    } else if constexpr (Oob == GatherOOB::Wrap) {
        doRead = 1u;
        safeIdx = rawIdx % tableSize;
    } else {
        doRead = (rawIdx < tableSize) ? 1u : 0u;
        safeIdx = rawIdx;
    }
    dstPtr[0] = doRead ? tablePtr[safeIdx] : static_cast<T>(0);
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
}

template <typename Tile>
struct IsMGatherNZTile {
    static constexpr bool value =
        !Tile::isRowMajor && (Tile::SFractal == SLayout::RowMajor) && (Tile::SFractalSize == TileConfig::fractalABSize);
};

template <GatherOOB Oob>
AICORE PTO_INLINE uint32_t gather_remap_l1(uint32_t idx, uint32_t cap, uint32_t &doRead)
{
    if constexpr (Oob == GatherOOB::Undefined) {
        doRead = 1u;
        return idx;
    } else if constexpr (Oob == GatherOOB::Clamp) {
        doRead = 1u;
        return (idx >= cap) ? (cap - 1u) : idx;
    } else if constexpr (Oob == GatherOOB::Wrap) {
        doRead = 1u;
        return idx % cap;
    } else {
        doRead = (idx < cap) ? 1u : 0u;
        return idx;
    }
}

template <GatherOOB Oob, typename T, typename TIdx, typename DstTile>
__tf__ AICORE void MGatherGm2L1RowImpl(typename DstTile::TileDType __out__ dst, __gm__ const T *tablePtr,
                                       __gm__ const TIdx *idxPtr, uint32_t validRow, uint32_t validCol,
                                       uint32_t tableRows, uint32_t tableRowStride)
{
#if defined(__DAV_CUBE__)
    constexpr uint32_t kC0 = C0_SIZE_BYTE / sizeof(T);
    constexpr uint32_t kTileRows = DstTile::Rows;
    constexpr uint32_t kTileCols = DstTile::Cols;
    __cbuf__ T *dstPtr = (__cbuf__ T *)__cce_get_tile_ptr(dst);

    if constexpr (Oob == GatherOOB::Zero) {
        constexpr uint32_t kColBlocks = kTileCols / kC0;
        int64_t repeatConfig = (static_cast<int64_t>(kTileRows) << 16) | static_cast<int64_t>(kColBlocks);
        pto_create_cbuf_matrix((__cbuf__ uint16_t *)dstPtr, repeatConfig, 0);
    }

    constexpr uint16_t ndNum = 1;
    constexpr uint16_t loop2DstStride = 1;
    constexpr uint16_t loop3DstStride = kTileRows;
    constexpr uint16_t loop4DstStride = 0;
    uint64_t mte2NzPara = static_cast<uint64_t>(loop4DstStride) << 48;
    mte2NzPara |= static_cast<uint64_t>(loop3DstStride) << 32;
    mte2NzPara |= static_cast<uint64_t>(loop2DstStride) << 16;
    mte2NzPara |= static_cast<uint64_t>(ndNum);
    set_mte2_nz_para(mte2NzPara);

    const uint64_t loop1SrcStride = static_cast<uint64_t>(tableRowStride) * sizeof(T);
    for (uint32_t r = 0; r < validRow; r++) {
        uint32_t rawIdx = static_cast<uint32_t>(idxPtr[r]);
        uint32_t doRead;
        uint32_t safeIdx = gather_remap_l1<Oob>(rawIdx, tableRows, doRead);
        if (doRead) {
            __gm__ const T *srcRow = tablePtr + static_cast<uint64_t>(safeIdx) * tableRowStride;
            __cbuf__ T *dstRow = dstPtr + static_cast<uint64_t>(r) * kC0;
            pto_copy_gm_to_cbuf_multi_nd2nz<T>(dstRow, const_cast<__gm__ T *>(srcRow), 0, loop1SrcStride, 0, 1,
                                               static_cast<uint32_t>(validCol), 0);
        }
    }
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
#endif
}

template <GatherOOB Oob, typename T, typename TIdx, typename DstTile>
__tf__ AICORE void MGatherGm2L1ElemImpl(typename DstTile::TileDType __out__ dst, __gm__ const T *tablePtr,
                                        __gm__ const TIdx *idxPtr, __gm__ T *scratchPtr, uint32_t validRow,
                                        uint32_t validCol, uint32_t tableSize, uint32_t idxRowStride)
{
#if defined(__DAV_CUBE__)
    __cbuf__ T *dstPtr = (__cbuf__ T *)__cce_get_tile_ptr(dst);
    constexpr uint32_t kC0 = C0_SIZE_BYTE / sizeof(T);
    constexpr uint32_t kTileRows = DstTile::Rows;
    constexpr uint32_t kTileCols = DstTile::Cols;
    constexpr uint32_t kTileNumel = kTileRows * kTileCols;

    for (uint32_t i = 0; i < kTileNumel; i++) {
        scratchPtr[i] = static_cast<T>(0);
    }
    for (uint32_t r = 0; r < validRow; r++) {
        const uint32_t idxRowOff = r * idxRowStride;
        for (uint32_t c = 0; c < validCol; c++) {
            uint32_t rawIdx = static_cast<uint32_t>(idxPtr[idxRowOff + c]);
            uint32_t doRead;
            uint32_t safeIdx = gather_remap_l1<Oob>(rawIdx, tableSize, doRead);
            if (doRead == 1) {
                const uint32_t blockCol = c / kC0;
                const uint32_t colInBlock = c - blockCol * kC0;
                const uint64_t off =
                    static_cast<uint64_t>(blockCol) * static_cast<uint64_t>(kTileRows) * static_cast<uint64_t>(kC0) +
                    static_cast<uint64_t>(r) * static_cast<uint64_t>(kC0) + static_cast<uint64_t>(colInBlock);
                scratchPtr[off] = tablePtr[safeIdx];
            }
        }
    }
    constexpr uint32_t kCacheLineBytes = 64;
    const uint32_t totalBytes = kTileNumel * sizeof(T);
    const uint32_t numLines = (totalBytes + kCacheLineBytes - 1) / kCacheLineBytes;
    __gm__ uint8_t *flushPtr = reinterpret_cast<__gm__ uint8_t *>(scratchPtr);
    for (uint32_t i = 0; i < numLines; i++) {
        dcci(static_cast<__gm__ void *>(flushPtr + i * kCacheLineBytes), SINGLE_CACHE_LINE);
    }
    dsb(DSB_DDR);
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    const uint32_t lenBurst = kTileNumel * sizeof(T);
    pto_copy_gm_to_cbuf_align_v2<T>(dstPtr, scratchPtr, 0, 1, lenBurst, 0, 0, true, 0, 0, 0);
#endif
}

template <typename T, typename TIdx, GatherOOB Oob, uint32_t TileRowsT, uint32_t TileColsT, uint32_t ValidRowsT,
          uint32_t ValidColsT, uint32_t TableSizeT>
AICORE __simt_vf__ LAUNCH_BOUND(1024) PTO_INLINE
    void simt_mgather_l1_elem_kernel(__gm__ T *__restrict__ scratch, __gm__ const T *__restrict__ table,
                                     __gm__ const TIdx *__restrict__ indices, uint32_t validRowsRT,
                                     uint32_t validColsRT, uint32_t tableSizeRT, uint32_t idxRowStrideRT)
{
    constexpr uint32_t kC0 = C0_SIZE_BYTE / sizeof(T);
    constexpr uint32_t kTileRows = TileRowsT;
    constexpr uint32_t kTileNumel = TileRowsT * TileColsT;
    constexpr uint32_t kBlockSpan = kTileRows * kC0;

    const uint32_t validRows = (ValidRowsT > 0u) ? ValidRowsT : validRowsRT;
    const uint32_t validCols = (ValidColsT > 0u) ? ValidColsT : validColsRT;
    const uint32_t tableSize = (TableSizeT > 0u) ? TableSizeT : tableSizeRT;
    const uint32_t idxRowStride = idxRowStrideRT;

    const uint32_t kNeededWarps = (kTileNumel + mgather_cfg::WARP_SIZE - 1u) / mgather_cfg::WARP_SIZE;
    const uint32_t kLaunchWarps =
        (kNeededWarps == 0u) ? 1u : ((kNeededWarps < mgather_cfg::MAX_WARPS) ? kNeededWarps : mgather_cfg::MAX_WARPS);
    const uint32_t kLaunchThreads = kLaunchWarps * mgather_cfg::WARP_SIZE;

    const uint32_t tid = threadIdx.y * mgather_cfg::WARP_SIZE + threadIdx.x;

#pragma unroll(1)
    for (uint32_t off = tid; off < kTileNumel; off += kLaunchThreads) {
        const uint32_t blockCol = off / kBlockSpan;
        const uint32_t rem = off - blockCol * kBlockSpan;
        const uint32_t r = rem / kC0;
        const uint32_t colInBlock = rem - r * kC0;
        const uint32_t c = blockCol * kC0 + colInBlock;
        T val = static_cast<T>(0);
        if (r < validRows && c < validCols) {
            const uint32_t rawIdx = static_cast<uint32_t>(indices[r * idxRowStride + c]);
            uint32_t doRead;
            const uint32_t safeIdx = gather_remap<Oob>(rawIdx, tableSize, doRead);
            val = doRead ? table[safeIdx] : static_cast<T>(0);
        }
        scratch[off] = val;
    }
}

template <GatherOOB Oob, typename T, typename TIdx, typename DstTile, uint8_t SyncId>
__tf__ AICORE void MGatherGm2L1ElemSimtImpl(typename DstTile::TileDType __out__ dst, __gm__ const T *tablePtr,
                                            __gm__ const TIdx *idxPtr, __gm__ T *scratchPtr, uint32_t validRow,
                                            uint32_t validCol, uint32_t tableSize, uint32_t idxRowStride)
{
    constexpr uint32_t kTileRows = DstTile::Rows;
    constexpr uint32_t kTileCols = DstTile::Cols;
    constexpr uint32_t kTileNumel = kTileRows * kTileCols;
#if defined(__DAV_VEC__)
    if (get_subblockid() == 0) {
        const uint32_t needed = (kTileNumel + mgather_cfg::WARP_SIZE - 1u) / mgather_cfg::WARP_SIZE;
        const uint32_t launchWarps =
            (needed == 0u) ? 1u : ((needed < mgather_cfg::MAX_WARPS) ? needed : mgather_cfg::MAX_WARPS);
        cce::async_invoke<simt_mgather_l1_elem_kernel<T, TIdx, Oob, kTileRows, kTileCols, 0u, 0u, 0u>>(
            cce::dim3{mgather_cfg::WARP_SIZE, launchWarps}, scratchPtr, tablePtr, idxPtr, validRow, validCol, tableSize,
            idxRowStride);

        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

        constexpr uint32_t kCacheLineBytes = 64;
        const uint32_t totalBytes = kTileNumel * sizeof(T);
        const uint32_t numLines = (totalBytes + kCacheLineBytes - 1) / kCacheLineBytes;
        __gm__ uint8_t *flushPtr = reinterpret_cast<__gm__ uint8_t *>(scratchPtr);
        for (uint32_t i = 0; i < numLines; i++) {
            dcci(static_cast<__gm__ void *>(flushPtr + i * kCacheLineBytes), SINGLE_CACHE_LINE);
        }
        dsb(DSB_DDR);
        set_intra_block(PIPE_S, SyncId);
    }
#endif
#if defined(__DAV_CUBE__)
    __cbuf__ T *dstPtr = (__cbuf__ T *)__cce_get_tile_ptr(dst);
    wait_intra_block(PIPE_MTE2, SyncId);
    const uint32_t lenBurst = kTileNumel * sizeof(T);
    pto_copy_gm_to_cbuf_align_v2<T>(dstPtr, scratchPtr, 0, 1, lenBurst, 0, 0, true, 0, 0, 0);
    (void)tablePtr;
    (void)idxPtr;
    (void)validRow;
    (void)validCol;
    (void)tableSize;
    (void)idxRowStride;
#endif
}

template <Coalesce Mode, GatherOOB Oob, typename DstTile, typename GlobalTable, typename IdxSrc>
PTO_INTERNAL void MGatherCheckGm2L1()
{
    using T = typename DstTile::DType;

    static_assert(IsValidGatherDType<T>::value,
                  "MGATHER A5 GM->L1 data type must be int8/uint8/int16/uint16/int32/uint32/half/bfloat16/float/"
                  "hifloat8/float8_e4m3/float8_e5m2.");
    static_assert(std::is_same_v<typename GlobalTable::DType, __gm__ T>,
                  "MGATHER A5 GM->L1 table must be a GM GlobalTensor with element type matching the destination.");
    static_assert(std::is_same_v<typename IdxSrc::DType, __gm__ int32_t> ||
                      std::is_same_v<typename IdxSrc::DType, __gm__ uint32_t>,
                  "MGATHER A5 GM->L1 indices must be a GM int32_t/uint32_t GlobalTensor.");
    static_assert(DstTile::Loc == TileType::Mat, "MGATHER A5 GM->L1 destination must be a Mat tile (L1).");
    static_assert(IsMGatherNZTile<DstTile>::value,
                  "MGATHER A5 GM->L1 destination must be NZ (BLayout::ColMajor + SLayout::RowMajor + "
                  "SFractalSize=512).");
    static_assert(GlobalTable::layout == Layout::ND, "MGATHER A5 GM->L1 table must use Layout::ND.");
    static_assert(DstTile::Cols % (C0_SIZE_BYTE / sizeof(T)) == 0,
                  "MGATHER A5 GM->L1 destination tile Cols must be a multiple of C0 (= 32 / sizeof(T)).");
    static_assert(DstTile::Rows % FRACTAL_NZ_ROW == 0,
                  "MGATHER A5 GM->L1 destination tile Rows must be a multiple of FRACTAL_NZ_ROW (16).");
    if constexpr (Mode == Coalesce::Row) {
        static_assert(sizeof(T) <= 4, "MGATHER A5 GM->L1 Coalesce::Row supports b8/b16/b32 element types.");
    }
}

template <Coalesce Mode, typename TileDst, typename GlobalTable, typename TileIdx>
PTO_INTERNAL void MGatherCheck(const TileDst &dst, const GlobalTable &table, const TileIdx &indices)
{
    using T = typename TileDst::DType;
    using TIdx = typename TileIdx::DType;

    static_assert(IsValidGatherDType<T>::value,
                  "MGATHER data type must be int8/uint8/int16/uint16/int32/uint32/half/bfloat16/float/"
                  "hifloat8/float8_e4m3/float8_e5m2.");

    static_assert(std::is_same_v<TIdx, int32_t> || std::is_same_v<TIdx, uint32_t>,
                  "MGATHER index type must be int32_t or uint32_t.");

    static_assert(std::is_same_v<typename GlobalTable::DType, __gm__ T>,
                  "MGATHER source table must be a GM GlobalTensor with element type matching the destination tile.");

    static_assert(TileDst::Loc == TileType::Vec, "MGATHER destination must be a Vec tile (UB).");
    static_assert(TileIdx::Loc == TileType::Vec, "MGATHER indices must be a Vec tile (UB).");

    constexpr int kDstValidR = TileDst::ValidRow;
    constexpr int kDstValidC = TileDst::ValidCol;
    constexpr int kIdxValidR = TileIdx::ValidRow;
    constexpr int kIdxValidC = TileIdx::ValidCol;

    using ShapeType = typename GlobalTable::Shape;
    constexpr int64_t kTableCols = ShapeType::staticShape[4];

    if constexpr (Mode == Coalesce::Row) {
        if constexpr (kDstValidR > 0 && kDstValidC > 0) {
            static_assert(kDstValidR >= 1 && kDstValidC >= 1,
                          "MGATHER Coalesce::Row requires non-empty valid destination shape [R, C].");
        }
        if constexpr (kDstValidR > 0 && kIdxValidR > 0 && kIdxValidC > 0) {
            static_assert(
                (kIdxValidR == 1 && kIdxValidC == kDstValidR) || (kIdxValidR == kDstValidR && kIdxValidC == 1),
                "MGATHER Coalesce::Row requires index tile valid shape [1, R] or [R, 1] matching TileDst::ValidRow.");
        }
        if constexpr (kDstValidC > 0 && kTableCols > 0) {
            static_assert(kTableCols == kDstValidC,
                          "MGATHER Coalesce::Row requires GlobalTensor inner dim (Shape[4]) == TileDst::ValidCol.");
        }
    } else {
        if constexpr (kDstValidR > 0 && kIdxValidR > 0) {
            static_assert(kIdxValidR == kDstValidR,
                          "MGATHER Coalesce::Elem requires index tile ValidRow == destination tile ValidRow.");
        }
        if constexpr (kDstValidC > 0 && kIdxValidC > 0) {
            static_assert(kIdxValidC == kDstValidC,
                          "MGATHER Coalesce::Elem requires index tile ValidCol == destination tile ValidCol.");
        }
        if constexpr (kDstValidR > 0 && kDstValidC > 0) {
            static_assert(kDstValidR >= 1 && kDstValidC >= 1,
                          "MGATHER Coalesce::Elem requires non-empty valid destination shape.");
        }
    }
}

template <Coalesce Mode = Coalesce::Row, GatherOOB Oob = GatherOOB::Undefined, typename TileDst, typename GlobalTable,
          typename TileIdx>
PTO_INTERNAL void MGATHER_IMPL(TileDst &dst, GlobalTable &table, TileIdx &indices)
{
    using T = typename TileDst::DType;

    if constexpr (TileDst::Loc == TileType::Mat) {
        MGatherCheckGm2L1<Coalesce::Row, Oob, TileDst, GlobalTable, TileIdx>();
        using TIdx = std::conditional_t<std::is_same_v<typename TileIdx::DType, __gm__ uint32_t>, uint32_t, int32_t>;
        __gm__ const T *tablePtr = reinterpret_cast<__gm__ const T *>(table.data());
        __gm__ const TIdx *idxPtr = reinterpret_cast<__gm__ const TIdx *>(indices.data());
        const uint32_t validRow = dst.GetValidRow();
        const uint32_t validCol = dst.GetValidCol();
        const uint32_t tableRows =
            static_cast<uint32_t>(table.GetShape(GlobalTensorDim::DIM_0) * table.GetShape(GlobalTensorDim::DIM_1) *
                                  table.GetShape(GlobalTensorDim::DIM_2) * table.GetShape(GlobalTensorDim::DIM_3));
        const uint32_t tableRowStride = static_cast<uint32_t>(table.GetStride(GlobalTensorDim::DIM_3));
        MGatherGm2L1RowImpl<Oob, T, TIdx, TileDst>(dst.data(), tablePtr, idxPtr, validRow, validCol, tableRows,
                                                   tableRowStride);
        return;
    } else {
        using TIdx = typename TileIdx::DType;

        MGatherCheck<Mode>(dst, table, indices);

        __gm__ const T *tablePtr = reinterpret_cast<__gm__ const T *>(table.data());

        constexpr int kDstValidRowS = TileDst::ValidRow;
        constexpr int kDstValidColS = TileDst::ValidCol;
        constexpr uint32_t kValidRowsT = (kDstValidRowS > 0) ? static_cast<uint32_t>(kDstValidRowS) : 0u;
        constexpr uint32_t kValidColsT = (kDstValidColS > 0) ? static_cast<uint32_t>(kDstValidColS) : 0u;

        const uint32_t validRows = dst.GetValidRow();
        const uint32_t validCols = dst.GetValidCol();

        if constexpr (Mode == Coalesce::Row) {
            using TableShape = typename GlobalTable::Shape;
            constexpr int64_t kTableRowsS = TableShape::staticShape[3];
            constexpr uint32_t kTableRowsT = (kTableRowsS > 0) ? static_cast<uint32_t>(kTableRowsS) : 0u;
            const uint32_t tableRows = static_cast<uint32_t>(table.GetShape(GlobalTensorDim::DIM_3));
            MGatherRowImpl<T, TIdx, Oob, kValidRowsT, kValidColsT, kTableRowsT, TileDst, TileIdx>(
                dst.data(), tablePtr, indices.data(), validRows, validCols, tableRows);
        } else {
            using TableShape = typename GlobalTable::Shape;
            constexpr int64_t kTS0 = TableShape::staticShape[0];
            constexpr int64_t kTS1 = TableShape::staticShape[1];
            constexpr int64_t kTS2 = TableShape::staticShape[2];
            constexpr int64_t kTS3 = TableShape::staticShape[3];
            constexpr int64_t kTS4 = TableShape::staticShape[4];
            constexpr bool kAllStatic = (kTS0 > 0) && (kTS1 > 0) && (kTS2 > 0) && (kTS3 > 0) && (kTS4 > 0);
            constexpr uint32_t kTableSizeT = kAllStatic ? static_cast<uint32_t>(kTS0 * kTS1 * kTS2 * kTS3 * kTS4) : 0u;
            const uint32_t tableSize =
                static_cast<uint32_t>(table.GetShape(GlobalTensorDim::DIM_0) * table.GetShape(GlobalTensorDim::DIM_1) *
                                      table.GetShape(GlobalTensorDim::DIM_2) * table.GetShape(GlobalTensorDim::DIM_3) *
                                      table.GetShape(GlobalTensorDim::DIM_4));
            if constexpr (TileDst::ValidRow == 1 && TileDst::ValidCol == 1) {
                MGatherScalarImpl<T, TIdx, Oob, TileDst, TileIdx>(dst.data(), tablePtr, indices.data(), tableSize);
            } else {
                MGatherElemImpl<T, TIdx, Oob, kValidRowsT, kValidColsT, kTableSizeT, TileDst, TileIdx>(
                    dst.data(), tablePtr, indices.data(), validRows, validCols, tableSize);
            }
        }
    }
}

template <Coalesce Mode = Coalesce::Elem, GatherOOB Oob = GatherOOB::Undefined, typename TileDst, typename GlobalTable,
          typename IdxSrc, typename GlobalScratch>
PTO_INTERNAL void MGATHER_IMPL(TileDst &dst, GlobalTable &table, IdxSrc &indices, GlobalScratch &scratch)
{
    using T = typename TileDst::DType;

    MGatherCheckGm2L1<Coalesce::Elem, Oob, TileDst, GlobalTable, IdxSrc>();
    static_assert(std::is_same_v<typename GlobalScratch::DType, __gm__ T>,
                  "MGATHER A5 GM->L1 scratch must be a GM GlobalTensor with element type matching the destination.");

    using TIdx = std::conditional_t<std::is_same_v<typename IdxSrc::DType, __gm__ uint32_t>, uint32_t, int32_t>;
    __gm__ const T *tablePtr = reinterpret_cast<__gm__ const T *>(table.data());
    __gm__ const TIdx *idxPtr = reinterpret_cast<__gm__ const TIdx *>(indices.data());
    __gm__ T *scratchPtr = reinterpret_cast<__gm__ T *>(scratch.data());

    const uint32_t validRow = dst.GetValidRow();
    const uint32_t validCol = dst.GetValidCol();
    const uint32_t tableSize =
        static_cast<uint32_t>(table.GetShape(GlobalTensorDim::DIM_0) * table.GetShape(GlobalTensorDim::DIM_1) *
                              table.GetShape(GlobalTensorDim::DIM_2) * table.GetShape(GlobalTensorDim::DIM_3) *
                              table.GetShape(GlobalTensorDim::DIM_4));
    const uint32_t idxRowStride = static_cast<uint32_t>(indices.GetStride(GlobalTensorDim::DIM_3));

    MGatherGm2L1ElemImpl<Oob, T, TIdx, TileDst>(dst.data(), tablePtr, idxPtr, scratchPtr, validRow, validCol, tableSize,
                                                idxRowStride);
}

template <Coalesce Mode, GatherOOB Oob, GatherExec Exec, typename TileDst, typename GlobalTable, typename IdxSrc,
          typename GlobalScratch>
PTO_INTERNAL void MGATHER_IMPL(TileDst &dst, GlobalTable &table, IdxSrc &indices, GlobalScratch &scratch)
{
    using T = typename TileDst::DType;

    if constexpr (Exec == GatherExec::Scalar) {
        MGATHER_IMPL<Mode, Oob>(dst, table, indices, scratch);
    } else {
        static_assert(Mode == Coalesce::Elem, "MGATHER A5 GM->L1 SIMT executor is only supported for Coalesce::Elem.");
        MGatherCheckGm2L1<Coalesce::Elem, Oob, TileDst, GlobalTable, IdxSrc>();
        static_assert(std::is_same_v<typename GlobalScratch::DType, __gm__ T>,
                      "MGATHER A5 GM->L1 scratch need GM GlobalTensor with element type matching the destination");

        using TIdx1 = std::conditional_t<std::is_same_v<typename IdxSrc::DType, __gm__ uint32_t>, uint32_t, int32_t>;
        __gm__ const T *tablePtr = reinterpret_cast<__gm__ const T *>(table.data());
        __gm__ const TIdx1 *idxPtr = reinterpret_cast<__gm__ const TIdx1 *>(indices.data());
        __gm__ T *scratchPtr = reinterpret_cast<__gm__ T *>(scratch.data());

        const uint32_t validRow = dst.GetValidRow();
        const uint32_t validCol = dst.GetValidCol();
        const uint32_t tableSize =
            static_cast<uint32_t>(table.GetShape(GlobalTensorDim::DIM_0) * table.GetShape(GlobalTensorDim::DIM_1) *
                                  table.GetShape(GlobalTensorDim::DIM_2) * table.GetShape(GlobalTensorDim::DIM_3) *
                                  table.GetShape(GlobalTensorDim::DIM_4));
        const uint32_t idxRowStride = static_cast<uint32_t>(indices.GetStride(GlobalTensorDim::DIM_3));

        constexpr uint8_t kSimtSyncId = 2;
        MGatherGm2L1ElemSimtImpl<Oob, T, TIdx1, TileDst, kSimtSyncId>(dst.data(), tablePtr, idxPtr, scratchPtr,
                                                                      validRow, validCol, tableSize, idxRowStride);
    }
}

} // namespace pto

#endif // MGATHER_HPP
