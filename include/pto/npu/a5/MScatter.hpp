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

enum class ScatterAtomicOp : uint8_t
{
    None = 0,
    Add = 1,
    Max = 2,
    Min = 3
};

enum class ScatterOOB : uint8_t
{
    Undefined = 0,
    Skip = 1,
    Clamp = 2,
    Wrap = 3
};

enum class ScatterConflict : uint8_t
{
    Last = 0,
    First = 1
};

#ifndef PTO_COALESCE_ENUM_DEFINED
#define PTO_COALESCE_ENUM_DEFINED
enum class Coalesce : uint8_t
{
    Row = 0,
    Elem = 1
};
#endif

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

template <uint32_t TotalElems>
struct ElemLaunch {
    static constexpr uint32_t kWarpsNeeded = (TotalElems + WARP_SIZE - 1u) / WARP_SIZE;
    static constexpr uint32_t kLaunchWarps =
        (kWarpsNeeded == 0u) ? 1u : ((kWarpsNeeded < MAX_WARPS) ? kWarpsNeeded : MAX_WARPS);
};

template <uint32_t NumRows, uint32_t RowWidth>
struct RowLaunch {
    static constexpr uint32_t kRowWarps = (NumRows < MAX_WARPS) ? ((NumRows == 0u) ? 1u : NumRows) : MAX_WARPS;
    static constexpr uint32_t kFreeWarps = MAX_WARPS / kRowWarps;
    static constexpr uint32_t kColChunks = (RowWidth + WARP_SIZE - 1u) / WARP_SIZE;
    static constexpr uint32_t kWarpsPerRow = (kFreeWarps < kColChunks) ? kFreeWarps : kColChunks;
    static constexpr uint32_t kLaunchWarps = kRowWarps * ((kWarpsPerRow == 0u) ? 1u : kWarpsPerRow);
};
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

template <ScatterOOB Oob, typename T, typename TIdx, typename TileSrc, typename TileIdx, uint32_t ValidRows,
          uint32_t ValidCols>
__simt_callee__ AICORE PTO_INLINE T last_winner_value_elem(__ubuf__ const TIdx *__restrict__ indices,
                                                           __ubuf__ const T *__restrict__ src, uint32_t selfSafeIdx,
                                                           uint32_t cap, T fallback)
{
    T result = fallback;
#pragma unroll(1)
    for (uint32_t j = 0; j < ValidRows * ValidCols; ++j) {
        const uint32_t r = (ValidCols == 1u) ? j : (j / ValidCols);
        const uint32_t c = (ValidCols == 1u) ? 0u : (j - r * ValidCols);
        uint32_t doW;
        const uint32_t s = scatter_remap<Oob>(static_cast<uint32_t>(indices[tile_offset_2d<TileIdx>(r, c)]), cap, doW);
        if (doW && s == selfSafeIdx) {
            result = src[tile_offset_2d<TileSrc>(r, c)];
        }
    }
    return result;
}

template <ScatterOOB Oob, typename T, typename TIdx, typename TileSrc, uint32_t NumRows>
__simt_callee__ AICORE PTO_INLINE T last_winner_value_row(__ubuf__ const TIdx *__restrict__ indices,
                                                          __ubuf__ const T *__restrict__ src, uint32_t selfSafeIdx,
                                                          uint32_t col, uint32_t cap, T fallback)
{
    T result = fallback;
#pragma unroll(1)
    for (uint32_t row = 0; row < NumRows; ++row) {
        uint32_t doW;
        const uint32_t s = scatter_remap<Oob>(static_cast<uint32_t>(indices[row]), cap, doW);
        if (doW && s == selfSafeIdx) {
            result = src[tile_offset_2d<TileSrc>(row, col)];
        }
    }
    return result;
}

template <typename T, typename TIdx, typename TileSrc, ScatterAtomicOp Atomic, ScatterOOB Oob, ScatterConflict Conflict,
          uint32_t ValidRows, uint32_t ValidCols, uint32_t TableRows>
AICORE __simt_vf__ LAUNCH_BOUND(1024) PTO_INLINE
    void simt_mscatter_row_kernel(__gm__ T *__restrict__ table, __ubuf__ const T *__restrict__ src,
                                  __ubuf__ const TIdx *__restrict__ indices, uint32_t validRowsRT, uint32_t validColsRT,
                                  uint32_t tableRowsRT)
{
    constexpr bool kIsAtomic = (Atomic != ScatterAtomicOp::None);
    constexpr bool kIsLast = !kIsAtomic && (Conflict == ScatterConflict::Last);
    using Launch = mscatter_cfg::RowLaunch<ValidRows, ValidCols>;
    constexpr uint32_t kRowWarps = Launch::kRowWarps;
    constexpr uint32_t kWarpsPerRow = (Launch::kWarpsPerRow == 0u) ? 1u : Launch::kWarpsPerRow;
    constexpr uint32_t kColStride = kWarpsPerRow * mscatter_cfg::WARP_SIZE;

    const uint32_t tx = threadIdx.x;
    const uint32_t ty = threadIdx.y;
    const uint32_t rowWarp = ty % kRowWarps;
    const uint32_t colSeg = ty / kRowWarps;

#pragma unroll(1)
    for (uint32_t row = rowWarp; row < ValidRows; row += kRowWarps) {
        const uint32_t rawIdx = static_cast<uint32_t>(indices[row]);
        uint32_t doWrite;
        const uint32_t safeIdx = scatter_remap<Oob>(rawIdx, TableRows, doWrite);
        __gm__ T *dstRow = table + safeIdx * ValidCols;
        if constexpr (kIsAtomic) {
            if (doWrite) {
#pragma unroll(4)
                for (uint32_t col = colSeg * mscatter_cfg::WARP_SIZE + tx; col < ValidCols; col += kColStride) {
                    scatter_apply<Atomic>(&dstRow[col], src[tile_offset_2d<TileSrc>(row, col)]);
                }
            }
        } else {
            const uint32_t loadRow = (rawIdx < TableRows) ? rawIdx : 0u;
            __gm__ T *loadDst = table + loadRow * ValidCols;
#pragma unroll(4)
            for (uint32_t col = colSeg * mscatter_cfg::WARP_SIZE + tx; col < ValidCols; col += kColStride) {
                const T cur = loadDst[col];
                const T newval = kIsLast ? last_winner_value_row<Oob, T, TIdx, TileSrc, ValidRows>(
                                               indices, src, safeIdx, col, TableRows, cur) :
                                           src[tile_offset_2d<TileSrc>(row, col)];
                if (doWrite && newval != cur) {
                    dstRow[col] = newval;
                }
            }
        }
    }
}

template <typename T, typename TIdx, typename TileSrc, typename TileIdx, ScatterAtomicOp Atomic, ScatterOOB Oob,
          ScatterConflict Conflict, uint32_t ValidRows, uint32_t ValidCols, uint32_t TableSize>
AICORE __simt_vf__ LAUNCH_BOUND(1024) PTO_INLINE
    void simt_mscatter_row_last_kernel(__gm__ T *__restrict__ table, __ubuf__ const T *__restrict__ src,
                                       __ubuf__ const TIdx *__restrict__ indices, uint32_t validRowsRT,
                                       uint32_t validColsRT, uint32_t tableRowsRT)
{
    constexpr bool kIsAtomic = (Atomic != ScatterAtomicOp::None);
    constexpr bool kIsLast = !kIsAtomic && (Conflict == ScatterConflict::Last);
    constexpr uint32_t kTotalElems = ValidRows * ValidCols;
    constexpr uint32_t kLaunchThreads = mscatter_cfg::ElemLaunch<kTotalElems>::kLaunchWarps * mscatter_cfg::WARP_SIZE;

    const uint32_t tx = threadIdx.x;
    const uint32_t ty = threadIdx.y;
    const uint32_t tid = ty * mscatter_cfg::WARP_SIZE + tx;

#pragma unroll(1)
    for (uint32_t i = tid; i < kTotalElems; i += kLaunchThreads) {
        const uint32_t r = (ValidCols == 1u) ? i : (i / ValidCols);
        const uint32_t c = (ValidCols == 1u) ? 0u : (i - r * ValidCols);
        const uint32_t srcOff = tile_offset_2d<TileSrc>(r, c);
        const uint32_t idxOff = tile_offset_2d<TileIdx>(r, c);
        const uint32_t rawIdx = static_cast<uint32_t>(indices[idxOff]);
        uint32_t doWrite;
        const uint32_t safeIdx = scatter_remap<Oob>(rawIdx, TableSize, doWrite);
        if constexpr (kIsAtomic) {
            if (doWrite) {
                scatter_apply<Atomic>(&table[safeIdx], src[srcOff]);
            }
        } else {
            const uint32_t loadIdx = (rawIdx < TableSize) ? rawIdx : 0u;
            const T cur = table[loadIdx];
            const T newval = kIsLast ? last_winner_value_elem<Oob, T, TIdx, TileSrc, TileIdx, ValidRows, ValidCols>(
                                           indices, src, safeIdx, TableSize, cur) :
                                       src[srcOff];
            if (doWrite && newval != cur) {
                table[safeIdx] = newval;
            }
        }
    }
}

template <typename T, typename TIdx, ScatterAtomicOp Atomic, ScatterOOB Oob, ScatterConflict Conflict,
          typename SrcTileData, typename IdxTileData, uint32_t ValidRows, uint32_t ValidCols, uint32_t TableRows>
__tf__ AICORE void MScatterRowImpl(__gm__ T *__restrict__ tablePtr, typename SrcTileData::TileDType __in__ src,
                                   typename IdxTileData::TileDType __in__ indices, uint32_t validRows,
                                   uint32_t validCols, uint32_t tableRows)
{
    __ubuf__ const T *srcPtr = (__ubuf__ const T *)__cce_get_tile_ptr(src);
    __ubuf__ const TIdx *idxPtr = (__ubuf__ const TIdx *)__cce_get_tile_ptr(indices);

    constexpr uint32_t kLaunchWarps = mscatter_cfg::RowLaunch<ValidRows, ValidCols>::kLaunchWarps;
    cce::async_invoke<
        simt_mscatter_row_kernel<T, TIdx, SrcTileData, Atomic, Oob, Conflict, ValidRows, ValidCols, TableRows>>(
        cce::dim3{mscatter_cfg::WARP_SIZE, kLaunchWarps}, tablePtr, srcPtr, idxPtr);
}

template <typename T, typename TIdx, ScatterAtomicOp Atomic, ScatterOOB Oob, ScatterConflict Conflict,
          typename SrcTileData, typename IdxTileData, uint32_t ValidRows, uint32_t ValidCols, uint32_t TableSize>
__tf__ AICORE void MScatterElemImpl(__gm__ T *__restrict__ tablePtr, typename SrcTileData::TileDType __in__ src,
                                    typename IdxTileData::TileDType __in__ indices, uint32_t validRows,
                                    uint32_t validCols, uint32_t tableSize)
{
    __ubuf__ const T *srcPtr = (__ubuf__ const T *)__cce_get_tile_ptr(src);
    __ubuf__ const TIdx *idxPtr = (__ubuf__ const TIdx *)__cce_get_tile_ptr(indices);

    constexpr uint32_t kLaunchWarps = mscatter_cfg::ElemLaunch<ValidRows * ValidCols>::kLaunchWarps;
    cce::async_invoke<simt_mscatter_elem_kernel<T, TIdx, SrcTileData, IdxTileData, Atomic, Oob, Conflict, ValidRows,
                                                ValidCols, TableSize>>(cce::dim3{mscatter_cfg::WARP_SIZE, kLaunchWarps},
                                                                       tablePtr, srcPtr, idxPtr);
}

template <typename T, typename TIdx, ScatterAtomicOp Atomic, ScatterOOB Oob, typename SrcTileData, typename IdxTileData,
          uint32_t TableSize>
__tf__ AICORE void MScatterScalarImpl(__gm__ T *__restrict__ tablePtr, typename SrcTileData::TileDType __in__ src,
                                      typename IdxTileData::TileDType __in__ indices)
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
        doWrite = (rawIdx < TableSize) ? 1u : 0u;
        safeIdx = rawIdx;
    } else if constexpr (Oob == ScatterOOB::Clamp) {
        doWrite = 1u;
        safeIdx = (rawIdx >= TableSize) ? (TableSize - 1u) : rawIdx;
    } else {
        doWrite = 1u;
        safeIdx = rawIdx % TableSize;
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
                  "MSCATTER data type must be int8/uint8/int16/uint16/int32/uint32/half/bfloat16/float "
                  "(and on AICORE: hifloat8/float8_e4m3/float8_e5m2).");

    static_assert(std::is_same_v<TIdx, int32_t> || std::is_same_v<TIdx, uint32_t>,
                  "MSCATTER index type must be int32_t or uint32_t.");

    static_assert(std::is_same_v<typename GlobalTable::DType, __gm__ T>,
                  "MSCATTER destination table must be a GM GlobalTensor with element type matching the source tile.");

    static_assert(TileSrc::Loc == TileType::Vec, "MSCATTER source must be a Vec tile (UB).");
    static_assert(TileIdx::Loc == TileType::Vec, "MSCATTER indices must be a Vec tile (UB).");

    static_assert(IsValidScatterAtomic<T, Atomic>::value,
                  "MSCATTER atomic operation: Add requires int32/uint32/float/half/bfloat16; "
                  "Max/Min requires int32/uint32/float.");

    constexpr uint32_t kSrcValidR = static_cast<uint32_t>(TileSrc::ValidRow);
    constexpr uint32_t kSrcValidC = static_cast<uint32_t>(TileSrc::ValidCol);
    constexpr uint32_t kIdxValidR = static_cast<uint32_t>(TileIdx::ValidRow);
    constexpr uint32_t kIdxValidC = static_cast<uint32_t>(TileIdx::ValidCol);

    using ShapeType = typename GlobalTable::Shape;
    constexpr uint32_t kTableRows = static_cast<uint32_t>(ShapeType::staticShape[3]);
    constexpr uint32_t kTableCols = static_cast<uint32_t>(ShapeType::staticShape[4]);

    if constexpr (Mode == Coalesce::Row) {
        static_assert(kSrcValidR >= 1u && kSrcValidC >= 1u,
                      "MSCATTER Coalesce::Row requires non-empty valid source shape [R, C].");
        static_assert(
            (kIdxValidR == 1u && kIdxValidC == kSrcValidR) || (kIdxValidR == kSrcValidR && kIdxValidC == 1u),
            "MSCATTER Coalesce::Row requires index tile valid shape [1, R] or [R, 1] matching TileSrc::ValidRow.");
        static_assert(kTableCols == kSrcValidC,
                      "MSCATTER Coalesce::Row requires GlobalTensor inner dim (Shape[4]) == TileSrc::ValidCol.");
        static_assert(kTableRows >= 1u, "MSCATTER Coalesce::Row requires GlobalTensor TableRows (Shape[3]) >= 1.");
    } else {
        static_assert(kIdxValidR == kSrcValidR && kIdxValidC == kSrcValidC,
                      "MSCATTER Coalesce::Elem requires index tile valid shape == source tile valid shape.");
        static_assert(kSrcValidR >= 1u && kSrcValidC >= 1u,
                      "MSCATTER Coalesce::Elem requires non-empty valid source shape.");
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

    constexpr uint32_t kValidRows = static_cast<uint32_t>(TileSrc::ValidRow);
    constexpr uint32_t kValidCols = static_cast<uint32_t>(TileSrc::ValidCol);

    if constexpr (Mode == Coalesce::Row) {
        constexpr uint32_t TableRows = static_cast<uint32_t>(ShapeType::staticShape[3]);
        MScatterRowImpl<T, TIdx, Atomic, Oob, Conflict, TileSrc, TileIdx, kValidRows, kValidCols, TableRows>(
            tablePtr, src.data(), indices.data());
    } else {
        constexpr uint32_t TableSize =
            static_cast<uint32_t>(ShapeType::staticShape[0]) * static_cast<uint32_t>(ShapeType::staticShape[1]) *
            static_cast<uint32_t>(ShapeType::staticShape[2]) * static_cast<uint32_t>(ShapeType::staticShape[3]) *
            static_cast<uint32_t>(ShapeType::staticShape[4]);
        if constexpr (kValidRows == 1u && kValidCols == 1u) {
            MScatterScalarImpl<T, TIdx, Atomic, Oob, TileSrc, TileIdx, TableSize>(tablePtr, src.data(), indices.data());
        } else {
            MScatterElemImpl<T, TIdx, Atomic, Oob, Conflict, TileSrc, TileIdx, kValidRows, kValidCols, TableSize>(
                tablePtr, src.data(), indices.data());
        }
    }
}

} // namespace pto

#endif // MSCATTER_HPP
