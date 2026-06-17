/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MSCATTER_HPP
#define MSCATTER_HPP

#include <pto/common/utils.hpp>
#include <pto/common/constants.hpp>
#include "common.hpp"
#include "utils.hpp"

namespace pto {

template <typename T>
struct IsValidScatterDType {
    static constexpr bool value =
        std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int16_t> ||
        std::is_same_v<T, uint16_t> || std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
        std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float> ||
        std::is_same_v<T, hifloat8_t> || std::is_same_v<T, float8_e4m3_t> || std::is_same_v<T, float8_e5m2_t>;
};

template <typename T, ScatterAtomicOp Atomic>
struct IsValidScatterAtomic {
    static constexpr bool value =
        (Atomic == ScatterAtomicOp::None) ||
        ((Atomic == ScatterAtomicOp::Add) &&
         (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, float> ||
          std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>)) ||
        ((Atomic == ScatterAtomicOp::Max || Atomic == ScatterAtomicOp::Min) &&
         (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, float>));
};

namespace mscatter_cfg {
constexpr uint32_t WARP_SIZE = 32u;
constexpr uint32_t MAX_WARPS = 32u;
constexpr uint32_t MAX_THREADS = WARP_SIZE * MAX_WARPS;
} // namespace mscatter_cfg

template <ScatterOOB Oob>
__simt_callee__ AICORE PTO_INLINE uint32_t scatter_remap(uint32_t idx, uint32_t cap, uint32_t &doWrite)
{
    if constexpr (Oob == ScatterOOB::Undefined) {
        doWrite = 1u;
        return idx;
    } else if constexpr (Oob == ScatterOOB::Skip) {
        doWrite = (idx < cap) ? 1u : 0u;
        return idx;
    } else if constexpr (Oob == ScatterOOB::Clamp) {
        doWrite = 1u;
        return (idx >= cap) ? (cap - 1u) : idx;
    } else {
        doWrite = 1u;
        return idx % cap;
    }
}

template <ScatterAtomicOp Atomic, typename T>
__simt_callee__ AICORE PTO_INLINE void scatter_apply(__gm__ T *ptr, T val)
{
    if constexpr (Atomic == ScatterAtomicOp::None) {
        *ptr = val;
    } else if constexpr (Atomic == ScatterAtomicOp::Add) {
        atomicAdd(ptr, val);
    } else if constexpr (Atomic == ScatterAtomicOp::Max) {
        atomicMax(ptr, val);
    } else {
        atomicMin(ptr, val);
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

template <ScatterOOB Oob, typename TIdx, typename TileIdx>
__simt_callee__ AICORE PTO_INLINE bool last_owner_find_elem(__ubuf__ const TIdx *__restrict__ indices,
                                                            uint32_t targetSlot, uint32_t cap, uint32_t totalElems,
                                                            uint32_t validCols, uint32_t &winnerR, uint32_t &winnerC)
{
#pragma unroll(1)
    for (uint32_t k = 0u; k < totalElems; ++k) {
        const uint32_t j = totalElems - 1u - k;
        const uint32_t r = (validCols == 1u) ? j : (j / validCols);
        const uint32_t c = (validCols == 1u) ? 0u : (j - r * validCols);
        uint32_t doW;
        const uint32_t s = scatter_remap<Oob>(static_cast<uint32_t>(indices[tile_offset_2d<TileIdx>(r, c)]), cap, doW);
        if (doW && s == targetSlot) {
            winnerR = r;
            winnerC = c;
            return true;
        }
    }
    return false;
}

template <ScatterOOB Oob, typename TIdx>
__simt_callee__ AICORE PTO_INLINE bool last_owner_find_row(__ubuf__ const TIdx *__restrict__ indices,
                                                           uint32_t targetSlot, uint32_t cap, uint32_t numRows,
                                                           uint32_t &winnerRow)
{
#pragma unroll(1)
    for (uint32_t k = 0u; k < numRows; ++k) {
        const uint32_t row = numRows - 1u - k;
        uint32_t doW;
        const uint32_t s = scatter_remap<Oob>(static_cast<uint32_t>(indices[row]), cap, doW);
        if (doW && s == targetSlot) {
            winnerRow = row;
            return true;
        }
    }
    return false;
}

template <typename T, typename TIdx, typename TileSrc, ScatterAtomicOp Atomic, ScatterOOB Oob, uint32_t ValidRowsT,
          uint32_t ValidColsT, uint32_t TableRowsT>
AICORE __simt_vf__ LAUNCH_BOUND(1024) PTO_INLINE
    void simt_mscatter_row_kernel(__gm__ T *__restrict__ table, __ubuf__ const T *__restrict__ src,
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
        (validRows == 0u) ? 1u : ((validRows < mscatter_cfg::MAX_WARPS) ? validRows : mscatter_cfg::MAX_WARPS);
    const uint32_t kFreeWarps = mscatter_cfg::MAX_WARPS / kRowWarps;
    const uint32_t kColChunks = (validCols + mscatter_cfg::WARP_SIZE - 1u) / mscatter_cfg::WARP_SIZE;
    const uint32_t kWarpsPerRowRaw = (kFreeWarps < kColChunks) ? kFreeWarps : kColChunks;
    const uint32_t kWarpsPerRow = (kWarpsPerRowRaw == 0u) ? 1u : kWarpsPerRowRaw;
    const uint32_t kColStride = kWarpsPerRow * mscatter_cfg::WARP_SIZE;

    const uint32_t tx = threadIdx.x;
    const uint32_t ty = threadIdx.y;
    const uint32_t rowWarp = ty % kRowWarps;
    const uint32_t colSeg = ty / kRowWarps;

#pragma unroll(1)
    for (uint32_t row = rowWarp; row < validRows; row += kRowWarps) {
        const uint32_t rawIdx = static_cast<uint32_t>(indices[row]);
        uint32_t doWrite;
        const uint32_t safeIdx = scatter_remap<Oob>(rawIdx, tableRows, doWrite);
        if constexpr (Atomic != ScatterAtomicOp::None) {
            if (doWrite) {
                __gm__ T *dstRow = table + safeIdx * validCols;
#pragma unroll(4)
                for (uint32_t col = colSeg * mscatter_cfg::WARP_SIZE + tx; col < validCols; col += kColStride) {
                    scatter_apply<Atomic>(&dstRow[col], src[tile_offset_2d<TileSrc>(row, col)]);
                }
            }
        } else {
            if (doWrite) {
                __gm__ T *dstRow = table + safeIdx * validCols;
#pragma unroll(4)
                for (uint32_t col = colSeg * mscatter_cfg::WARP_SIZE + tx; col < validCols; col += kColStride) {
                    dstRow[col] = src[tile_offset_2d<TileSrc>(row, col)];
                }
            }
        }
    }
}

template <typename T, typename TIdx, typename TileSrc, ScatterOOB Oob, uint32_t ValidRowsT, uint32_t ValidColsT,
          uint32_t TableRowsT>
AICORE __simt_vf__ LAUNCH_BOUND(1024) PTO_INLINE
    void simt_mscatter_row_last_kernel(__gm__ T *__restrict__ table, __ubuf__ const T *__restrict__ src,
                                       __ubuf__ const TIdx *__restrict__ indices, uint32_t validRowsRT,
                                       uint32_t validColsRT, uint32_t tableRowsRT)
{
    constexpr bool kStaticR = (ValidRowsT > 0u);
    constexpr bool kStaticC = (ValidColsT > 0u);
    constexpr bool kStaticTR = (TableRowsT > 0u);
    const uint32_t validRows = kStaticR ? ValidRowsT : validRowsRT;
    const uint32_t validCols = kStaticC ? ValidColsT : validColsRT;
    const uint32_t tableRows = kStaticTR ? TableRowsT : tableRowsRT;

    const uint32_t tid = threadIdx.y * mscatter_cfg::WARP_SIZE + threadIdx.x;
    const uint32_t totalThreads = mscatter_cfg::MAX_THREADS;

#pragma unroll(1)
    for (uint32_t s = tid; s < tableRows; s += totalThreads) {
        uint32_t winnerRow = 0u;
        if (last_owner_find_row<Oob, TIdx>(indices, s, tableRows, validRows, winnerRow)) {
            __gm__ T *dstRow = table + s * validCols;
#pragma unroll(4)
            for (uint32_t col = 0u; col < validCols; ++col) {
                dstRow[col] = src[tile_offset_2d<TileSrc>(winnerRow, col)];
            }
        }
    }
}

template <typename T, typename TIdx, typename TileSrc, typename TileIdx, ScatterAtomicOp Atomic, ScatterOOB Oob,
          uint32_t ValidRowsT, uint32_t ValidColsT, uint32_t TableSizeT>
AICORE __simt_vf__ LAUNCH_BOUND(1024) PTO_INLINE
    void simt_mscatter_elem_kernel(__gm__ T *__restrict__ table, __ubuf__ const T *__restrict__ src,
                                   __ubuf__ const TIdx *__restrict__ indices, uint32_t validRowsRT,
                                   uint32_t validColsRT, uint32_t tableSizeRT)
{
    constexpr bool kStaticR = (ValidRowsT > 0u);
    constexpr bool kStaticC = (ValidColsT > 0u);
    constexpr bool kStaticTS = (TableSizeT > 0u);
    const uint32_t validRows = kStaticR ? ValidRowsT : validRowsRT;
    const uint32_t validCols = kStaticC ? ValidColsT : validColsRT;
    const uint32_t tableSize = kStaticTS ? TableSizeT : tableSizeRT;

    const uint32_t totalElems = validRows * validCols;
    const uint32_t kNeededWarps = (totalElems + mscatter_cfg::WARP_SIZE - 1u) / mscatter_cfg::WARP_SIZE;
    const uint32_t kLaunchWarps =
        (kNeededWarps == 0u) ? 1u : ((kNeededWarps < mscatter_cfg::MAX_WARPS) ? kNeededWarps : mscatter_cfg::MAX_WARPS);
    const uint32_t kLaunchThreads = kLaunchWarps * mscatter_cfg::WARP_SIZE;

    const uint32_t tid = threadIdx.y * mscatter_cfg::WARP_SIZE + threadIdx.x;

#pragma unroll(1)
    for (uint32_t i = tid; i < totalElems; i += kLaunchThreads) {
        const uint32_t r = (validCols == 1u) ? i : (i / validCols);
        const uint32_t c = (validCols == 1u) ? 0u : (i - r * validCols);
        const uint32_t srcOff = tile_offset_2d<TileSrc>(r, c);
        const uint32_t idxOff = tile_offset_2d<TileIdx>(r, c);
        const uint32_t rawIdx = static_cast<uint32_t>(indices[idxOff]);
        uint32_t doWrite;
        const uint32_t safeIdx = scatter_remap<Oob>(rawIdx, tableSize, doWrite);
        if constexpr (Atomic != ScatterAtomicOp::None) {
            if (doWrite) {
                scatter_apply<Atomic>(&table[safeIdx], src[srcOff]);
            }
        } else {
            if (doWrite) {
                table[safeIdx] = src[srcOff];
            }
        }
    }
}

template <typename T, typename TIdx, typename TileSrc, typename TileIdx, ScatterOOB Oob, uint32_t ValidRowsT,
          uint32_t ValidColsT, uint32_t TableSizeT>
AICORE __simt_vf__ LAUNCH_BOUND(1024) PTO_INLINE
    void simt_mscatter_elem_last_kernel(__gm__ T *__restrict__ table, __ubuf__ const T *__restrict__ src,
                                        __ubuf__ const TIdx *__restrict__ indices, uint32_t validRowsRT,
                                        uint32_t validColsRT, uint32_t tableSizeRT)
{
    constexpr bool kStaticR = (ValidRowsT > 0u);
    constexpr bool kStaticC = (ValidColsT > 0u);
    constexpr bool kStaticTS = (TableSizeT > 0u);
    const uint32_t validRows = kStaticR ? ValidRowsT : validRowsRT;
    const uint32_t validCols = kStaticC ? ValidColsT : validColsRT;
    const uint32_t tableSize = kStaticTS ? TableSizeT : tableSizeRT;

    const uint32_t totalElems = validRows * validCols;
    const uint32_t tid = threadIdx.y * mscatter_cfg::WARP_SIZE + threadIdx.x;
    const uint32_t totalThreads = mscatter_cfg::MAX_THREADS;

#pragma unroll(1)
    for (uint32_t s = tid; s < tableSize; s += totalThreads) {
        uint32_t winnerR = 0u;
        uint32_t winnerC = 0u;
        if (last_owner_find_elem<Oob, TIdx, TileIdx>(indices, s, tableSize, totalElems, validCols, winnerR, winnerC)) {
            const uint32_t srcOff = tile_offset_2d<TileSrc>(winnerR, winnerC);
            table[s] = src[srcOff];
        }
    }
}

template <typename T, typename TIdx, ScatterAtomicOp Atomic, ScatterOOB Oob, ScatterConflict Conflict,
          uint32_t ValidRowsT, uint32_t ValidColsT, uint32_t TableRowsT, typename SrcTileData, typename IdxTileData>
__tf__ AICORE void MScatterRowImpl(__gm__ T *__restrict__ tablePtr, typename SrcTileData::TileDType __in__ src,
                                   typename IdxTileData::TileDType __in__ indices, uint32_t validRows,
                                   uint32_t validCols, uint32_t tableRows)
{
    __ubuf__ const T *srcPtr = (__ubuf__ const T *)__cce_get_tile_ptr(src);
    __ubuf__ const TIdx *idxPtr = (__ubuf__ const TIdx *)__cce_get_tile_ptr(indices);

    constexpr bool kIsLast = (Atomic == ScatterAtomicOp::None) && (Conflict == ScatterConflict::Last);

    if constexpr (kIsLast) {
        const uint32_t slotWarps =
            (tableRows == 0u) ?
                1u :
                (((tableRows + mscatter_cfg::WARP_SIZE - 1u) / mscatter_cfg::WARP_SIZE) < mscatter_cfg::MAX_WARPS ?
                     ((tableRows + mscatter_cfg::WARP_SIZE - 1u) / mscatter_cfg::WARP_SIZE) :
                     mscatter_cfg::MAX_WARPS);
        cce::async_invoke<simt_mscatter_row_last_kernel<T, TIdx, SrcTileData, Oob, ValidRowsT, ValidColsT, TableRowsT>>(
            cce::dim3{mscatter_cfg::WARP_SIZE, slotWarps}, tablePtr, srcPtr, idxPtr, validRows, validCols, tableRows);
    } else {
        const uint32_t rowWarps =
            (validRows == 0u) ? 1u : ((validRows < mscatter_cfg::MAX_WARPS) ? validRows : mscatter_cfg::MAX_WARPS);
        const uint32_t freeWarps = mscatter_cfg::MAX_WARPS / rowWarps;
        const uint32_t colChunks = (validCols + mscatter_cfg::WARP_SIZE - 1u) / mscatter_cfg::WARP_SIZE;
        const uint32_t warpsPerRowRaw = (freeWarps < colChunks) ? freeWarps : colChunks;
        const uint32_t warpsPerRow = (warpsPerRowRaw == 0u) ? 1u : warpsPerRowRaw;
        const uint32_t launchWarps = rowWarps * warpsPerRow;

        cce::async_invoke<
            simt_mscatter_row_kernel<T, TIdx, SrcTileData, Atomic, Oob, ValidRowsT, ValidColsT, TableRowsT>>(
            cce::dim3{mscatter_cfg::WARP_SIZE, launchWarps}, tablePtr, srcPtr, idxPtr, validRows, validCols, tableRows);
    }
}

template <typename T, typename TIdx, ScatterAtomicOp Atomic, ScatterOOB Oob, ScatterConflict Conflict,
          uint32_t ValidRowsT, uint32_t ValidColsT, uint32_t TableSizeT, typename SrcTileData, typename IdxTileData>
__tf__ AICORE void MScatterElemImpl(__gm__ T *__restrict__ tablePtr, typename SrcTileData::TileDType __in__ src,
                                    typename IdxTileData::TileDType __in__ indices, uint32_t validRows,
                                    uint32_t validCols, uint32_t tableSize)
{
    __ubuf__ const T *srcPtr = (__ubuf__ const T *)__cce_get_tile_ptr(src);
    __ubuf__ const TIdx *idxPtr = (__ubuf__ const TIdx *)__cce_get_tile_ptr(indices);

    constexpr bool kIsLast = (Atomic == ScatterAtomicOp::None) && (Conflict == ScatterConflict::Last);

    if constexpr (kIsLast) {
        const uint32_t slotWarps =
            (tableSize == 0u) ?
                1u :
                (((tableSize + mscatter_cfg::WARP_SIZE - 1u) / mscatter_cfg::WARP_SIZE) < mscatter_cfg::MAX_WARPS ?
                     ((tableSize + mscatter_cfg::WARP_SIZE - 1u) / mscatter_cfg::WARP_SIZE) :
                     mscatter_cfg::MAX_WARPS);
        cce::async_invoke<
            simt_mscatter_elem_last_kernel<T, TIdx, SrcTileData, IdxTileData, Oob, ValidRowsT, ValidColsT, TableSizeT>>(
            cce::dim3{mscatter_cfg::WARP_SIZE, slotWarps}, tablePtr, srcPtr, idxPtr, validRows, validCols, tableSize);
    } else {
        const uint32_t totalElems = validRows * validCols;
        const uint32_t needed = (totalElems + mscatter_cfg::WARP_SIZE - 1u) / mscatter_cfg::WARP_SIZE;
        const uint32_t launchWarps =
            (needed == 0u) ? 1u : ((needed < mscatter_cfg::MAX_WARPS) ? needed : mscatter_cfg::MAX_WARPS);
        cce::async_invoke<simt_mscatter_elem_kernel<T, TIdx, SrcTileData, IdxTileData, Atomic, Oob, ValidRowsT,
                                                    ValidColsT, TableSizeT>>(
            cce::dim3{mscatter_cfg::WARP_SIZE, launchWarps}, tablePtr, srcPtr, idxPtr, validRows, validCols, tableSize);
    }
}

template <typename T, typename TIdx, ScatterAtomicOp Atomic, ScatterOOB Oob, typename SrcTileData, typename IdxTileData>
__tf__ AICORE void MScatterScalarImpl(__gm__ T *__restrict__ tablePtr, typename SrcTileData::TileDType __in__ src,
                                      typename IdxTileData::TileDType __in__ indices, uint32_t tableSize)
{
    __ubuf__ const T *srcPtr = (__ubuf__ const T *)__cce_get_tile_ptr(src);
    __ubuf__ const TIdx *idxPtr = (__ubuf__ const TIdx *)__cce_get_tile_ptr(indices);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    const uint32_t rawIdx = static_cast<uint32_t>(idxPtr[0]);
    uint32_t doWrite;
    uint32_t safeIdx;
    if constexpr (Oob == ScatterOOB::Undefined) {
        doWrite = 1u;
        safeIdx = rawIdx;
    } else if constexpr (Oob == ScatterOOB::Skip) {
        doWrite = (rawIdx < tableSize) ? 1u : 0u;
        safeIdx = rawIdx;
    } else if constexpr (Oob == ScatterOOB::Clamp) {
        doWrite = 1u;
        safeIdx = (rawIdx >= tableSize) ? (tableSize - 1u) : rawIdx;
    } else {
        doWrite = 1u;
        safeIdx = rawIdx % tableSize;
    }
    if (doWrite) {
        if constexpr (Atomic == ScatterAtomicOp::None) {
            tablePtr[safeIdx] = srcPtr[0];
        } else if constexpr (Atomic == ScatterAtomicOp::Add) {
            atomicAdd(&tablePtr[safeIdx], srcPtr[0]);
        } else if constexpr (Atomic == ScatterAtomicOp::Max) {
            atomicMax(&tablePtr[safeIdx], srcPtr[0]);
        } else {
            atomicMin(&tablePtr[safeIdx], srcPtr[0]);
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
}

template <Coalesce Mode, ScatterAtomicOp Atomic, typename GlobalTable, typename TileSrc, typename TileIdx>
PTO_INTERNAL void MScatterCheck(const GlobalTable &table, const TileSrc &src, const TileIdx &indices)
{
    using T = typename TileSrc::DType;
    using TIdx = typename TileIdx::DType;

    static_assert(IsValidScatterDType<T>::value,
                  "MSCATTER data type must be int8/uint8/int16/uint16/int32/uint32/half/bfloat16/float/"
                  "hifloat8/float8_e4m3/float8_e5m2.");

    static_assert(std::is_same_v<TIdx, int32_t> || std::is_same_v<TIdx, uint32_t>,
                  "MSCATTER index type must be int32_t or uint32_t.");

    static_assert(std::is_same_v<typename GlobalTable::DType, __gm__ T>,
                  "MSCATTER destination table must be a GM GlobalTensor with element type matching the source tile.");

    static_assert(TileSrc::Loc == TileType::Vec, "MSCATTER source must be a Vec tile (UB).");
    static_assert(TileIdx::Loc == TileType::Vec, "MSCATTER indices must be a Vec tile (UB).");

    static_assert(IsValidScatterAtomic<T, Atomic>::value,
                  "MSCATTER atomic operation: Add requires int32/uint32/float/half/bfloat16; "
                  "Max/Min requires int32/uint32/float.");

    constexpr int kSrcValidR = TileSrc::ValidRow;
    constexpr int kSrcValidC = TileSrc::ValidCol;
    constexpr int kIdxValidR = TileIdx::ValidRow;
    constexpr int kIdxValidC = TileIdx::ValidCol;

    using ShapeType = typename GlobalTable::Shape;
    constexpr int64_t kTableRows = ShapeType::staticShape[3];
    constexpr int64_t kTableCols = ShapeType::staticShape[4];

    if constexpr (Mode == Coalesce::Row) {
        if constexpr (kSrcValidR > 0 && kSrcValidC > 0) {
            static_assert(kSrcValidR >= 1 && kSrcValidC >= 1,
                          "MSCATTER Coalesce::Row requires non-empty valid source shape [R, C].");
        }
        if constexpr (kSrcValidR > 0 && kIdxValidR > 0 && kIdxValidC > 0) {
            static_assert(
                (kIdxValidR == 1 && kIdxValidC == kSrcValidR) || (kIdxValidR == kSrcValidR && kIdxValidC == 1),
                "MSCATTER Coalesce::Row requires index tile valid shape [1, R] or [R, 1] matching TileSrc::ValidRow.");
        }
        if constexpr (kSrcValidC > 0 && kTableCols > 0) {
            static_assert(kTableCols == kSrcValidC,
                          "MSCATTER Coalesce::Row requires GlobalTensor inner dim (Shape[4]) == TileSrc::ValidCol.");
        }
        if constexpr (kTableRows > 0) {
            static_assert(kTableRows >= 1, "MSCATTER Coalesce::Row requires GlobalTensor TableRows (Shape[3]) >= 1.");
        }
    } else {
        if constexpr (kSrcValidR > 0 && kIdxValidR > 0) {
            static_assert(kIdxValidR == kSrcValidR,
                          "MSCATTER Coalesce::Elem requires index tile ValidRow == source tile ValidRow.");
        }
        if constexpr (kSrcValidC > 0 && kIdxValidC > 0) {
            static_assert(kIdxValidC == kSrcValidC,
                          "MSCATTER Coalesce::Elem requires index tile ValidCol == source tile ValidCol.");
        }
        if constexpr (kSrcValidR > 0 && kSrcValidC > 0) {
            static_assert(kSrcValidR >= 1 && kSrcValidC >= 1,
                          "MSCATTER Coalesce::Elem requires non-empty valid source shape.");
        }
    }
}

template <Coalesce Mode = Coalesce::Row, ScatterAtomicOp Atomic = ScatterAtomicOp::None,
          ScatterOOB Oob = ScatterOOB::Undefined, ScatterConflict Conflict = ScatterConflict::Last,
          typename GlobalTable, typename TileSrc, typename TileIdx>
PTO_INTERNAL void MSCATTER_IMPL(GlobalTable &table, TileSrc &src, TileIdx &indices)
{
    using T = typename TileSrc::DType;
    using TIdx = typename TileIdx::DType;

    MScatterCheck<Mode, Atomic>(table, src, indices);

    __gm__ T *tablePtr = reinterpret_cast<__gm__ T *>(table.data());

    constexpr int kSrcValidRowS = TileSrc::ValidRow;
    constexpr int kSrcValidColS = TileSrc::ValidCol;
    constexpr uint32_t kValidRowsT = (kSrcValidRowS > 0) ? static_cast<uint32_t>(kSrcValidRowS) : 0u;
    constexpr uint32_t kValidColsT = (kSrcValidColS > 0) ? static_cast<uint32_t>(kSrcValidColS) : 0u;

    const uint32_t validRows = src.GetValidRow();
    const uint32_t validCols = src.GetValidCol();

    if constexpr (Mode == Coalesce::Row) {
        using TableShape = typename GlobalTable::Shape;
        constexpr int64_t kTableRowsS = TableShape::staticShape[3];
        constexpr uint32_t kTableRowsT = (kTableRowsS > 0) ? static_cast<uint32_t>(kTableRowsS) : 0u;
        const uint32_t tableRows = static_cast<uint32_t>(table.GetShape(GlobalTensorDim::DIM_3));
        MScatterRowImpl<T, TIdx, Atomic, Oob, Conflict, kValidRowsT, kValidColsT, kTableRowsT, TileSrc, TileIdx>(
            tablePtr, src.data(), indices.data(), validRows, validCols, tableRows);
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
        if constexpr (TileSrc::ValidRow == 1 && TileSrc::ValidCol == 1) {
            MScatterScalarImpl<T, TIdx, Atomic, Oob, TileSrc, TileIdx>(tablePtr, src.data(), indices.data(), tableSize);
        } else {
            MScatterElemImpl<T, TIdx, Atomic, Oob, Conflict, kValidRowsT, kValidColsT, kTableSizeT, TileSrc, TileIdx>(
                tablePtr, src.data(), indices.data(), validRows, validCols, tableSize);
        }
    }
}

} // namespace pto

#endif // MSCATTER_HPP
