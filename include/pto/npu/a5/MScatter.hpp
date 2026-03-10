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
    None = 0, // Non-atomic write
    Add = 1,  // Atomic addition
    Max = 2,  // Atomic maximum
    Min = 3   // Atomic minimum
};

enum class ScatterOOB : uint8_t
{
    Undefined = 0, // No bounds check
    Skip = 1,      // Skip OOB writes (no memory access)
    Clamp = 2,     // Clamp to valid range
    Wrap = 3       // Modulo wrap
};

template <typename T, ScatterAtomicOp Atomic>
struct IsValidScatterAtomic {
    static constexpr bool value =
        (Atomic == ScatterAtomicOp::None) ||
        ((Atomic == ScatterAtomicOp::Add) && (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
                                              std::is_same_v<T, float> || std::is_same_v<T, half>)) ||
        ((Atomic == ScatterAtomicOp::Max || Atomic == ScatterAtomicOp::Min) &&
         (std::is_same_v<T, int32_t> || std::is_same_v<T, float>));
};

namespace mscatter_cfg {
constexpr uint32_t WARP_SIZE = 32;
constexpr uint32_t NUM_WARPS = 32;
constexpr uint32_t TOTAL_THREADS = WARP_SIZE * NUM_WARPS;
} // namespace mscatter_cfg

template <ScatterOOB Mode>
AICORE PTO_INLINE uint32_t apply_scatter_oob(uint32_t idx, uint32_t tableSize, bool &skip)
{
    skip = false;
    if constexpr (Mode == ScatterOOB::Undefined) {
        return idx;
    } else if constexpr (Mode == ScatterOOB::Skip) {
        skip = (idx >= tableSize);
        return idx;
    } else if constexpr (Mode == ScatterOOB::Clamp) {
        return (idx >= tableSize) ? (tableSize - 1) : idx;
    } else {
        return idx % tableSize;
    }
}

template <ScatterAtomicOp Atomic, typename T>
AICORE PTO_INLINE void scatter_write(__gm__ T *ptr, T val)
{
    if constexpr (Atomic == ScatterAtomicOp::None) {
        *ptr = val;
    } else if constexpr (Atomic == ScatterAtomicOp::Add) {
        *ptr = *ptr + val;
    } else if constexpr (Atomic == ScatterAtomicOp::Max) {
        T old = *ptr;
        *ptr = (val > old) ? val : old;
    } else { // Min
        T old = *ptr;
        *ptr = (val < old) ? val : old;
    }
}

// SIMT Kernel: Row-Indexed Scatter
// Semantics: table[indices[i], j] = src[i, j]
template <typename T, typename TIdx, ScatterAtomicOp Atomic, ScatterOOB Mode, uint32_t NumRows, uint32_t RowWidth,
          uint32_t TableRows>
AICORE __simt_vf__ LAUNCH_BOUND(1024) PTO_INLINE
    void simt_mscatter_row_kernel(__gm__ T *__restrict__ table, __ubuf__ const T *__restrict__ src,
                                  __ubuf__ const TIdx *__restrict__ indices)
{
    const uint32_t tx = __cce_simt_get_TID_X();
    const uint32_t ty = __cce_simt_get_TID_Y();

#pragma unroll(1)
    for (uint32_t row = ty; row < NumRows; row += mscatter_cfg::NUM_WARPS) {
        uint32_t rawIdx = static_cast<uint32_t>(indices[row]);

        bool skip = false;
        uint32_t safeIdx = apply_scatter_oob<Mode>(rawIdx, TableRows, skip);

        if (!skip) {
            __ubuf__ const T *srcRow = src + row * RowWidth;
            __gm__ T *dstRow = table + safeIdx * RowWidth;

#pragma unroll(4)
            for (uint32_t col = tx; col < RowWidth; col += mscatter_cfg::WARP_SIZE) {
                scatter_write<Atomic>(&dstRow[col], srcRow[col]);
            }
        }
    }
}

// SIMT Kernel: Element-Indexed Scatter
// Semantics: table[indices[i, j]] = src[i, j]
template <typename T, typename TIdx, ScatterAtomicOp Atomic, ScatterOOB Mode, uint32_t NumRows, uint32_t NumCols,
          uint32_t TableSize>
AICORE __simt_vf__ LAUNCH_BOUND(1024) PTO_INLINE
    void simt_mscatter_elem_kernel(__gm__ T *__restrict__ table, __ubuf__ const T *__restrict__ src,
                                   __ubuf__ const TIdx *__restrict__ indices)
{
    const uint32_t tx = __cce_simt_get_TID_X();
    const uint32_t ty = __cce_simt_get_TID_Y();
    const uint32_t tid = ty * mscatter_cfg::WARP_SIZE + tx;

    constexpr uint32_t totalElements = NumRows * NumCols;

#pragma unroll(1)
    for (uint32_t i = tid; i < totalElements; i += mscatter_cfg::TOTAL_THREADS) {
        uint32_t rawIdx = static_cast<uint32_t>(indices[i]);

        bool skip = false;
        uint32_t safeIdx = apply_scatter_oob<Mode>(rawIdx, TableSize, skip);

        if (!skip) {
            scatter_write<Atomic>(&table[safeIdx], src[i]);
        }
    }
}

template <typename T, typename TIdx, ScatterAtomicOp Atomic, ScatterOOB Mode, typename SrcTileData,
          typename IdxTileData, uint32_t NumRows, uint32_t RowWidth, uint32_t TableRows>
__tf__ AICORE void MScatterRowImpl(__gm__ T *__restrict__ tablePtr, typename SrcTileData::TileDType __in__ src,
                                   typename IdxTileData::TileDType __in__ indices)
{
    __ubuf__ const T *srcPtr = (__ubuf__ const T *)__cce_get_tile_ptr(src);
    __ubuf__ const TIdx *idxPtr = (__ubuf__ const TIdx *)__cce_get_tile_ptr(indices);

    cce::async_invoke<simt_mscatter_row_kernel<T, TIdx, Atomic, Mode, NumRows, RowWidth, TableRows>>(
        cce::dim3{mscatter_cfg::WARP_SIZE, mscatter_cfg::NUM_WARPS}, tablePtr, srcPtr, idxPtr);
}

template <typename T, typename TIdx, ScatterAtomicOp Atomic, ScatterOOB Mode, typename SrcTileData,
          typename IdxTileData, uint32_t NumRows, uint32_t NumCols, uint32_t TableSize>
__tf__ AICORE void MScatterElemImpl(__gm__ T *__restrict__ tablePtr, typename SrcTileData::TileDType __in__ src,
                                    typename IdxTileData::TileDType __in__ indices)
{
    __ubuf__ const T *srcPtr = (__ubuf__ const T *)__cce_get_tile_ptr(src);
    __ubuf__ const TIdx *idxPtr = (__ubuf__ const TIdx *)__cce_get_tile_ptr(indices);

    cce::async_invoke<simt_mscatter_elem_kernel<T, TIdx, Atomic, Mode, NumRows, NumCols, TableSize>>(
        cce::dim3{mscatter_cfg::WARP_SIZE, mscatter_cfg::NUM_WARPS}, tablePtr, srcPtr, idxPtr);
}

template <ScatterAtomicOp Atomic, typename GlobalTable, typename TileSrc, typename TileIdx>
PTO_INTERNAL void MScatterCheck(const GlobalTable &table, const TileSrc &src, const TileIdx &indices)
{
    using T = typename TileSrc::DType;
    using TIdx = typename TileIdx::DType;

    static_assert(std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int16_t> ||
                      std::is_same_v<T, uint16_t> || std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
                      std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float> ||
#ifdef __CCE_AICORE__
                      std::is_same_v<T, float8_e4m3_t> || std::is_same_v<T, float8_e5m2_t> ||
#endif
                      false,
                  "MSCATTER data type must be int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/"
                  "half/bfloat16_t/float/float8_e4m3_t/float8_e5m2_t.");

    static_assert(std::is_same_v<TIdx, int32_t> || std::is_same_v<TIdx, uint32_t>,
                  "MSCATTER index type must be int32_t or uint32_t.");

    static_assert(std::is_same_v<typename GlobalTable::DType, __gm__ T>,
                  "MSCATTER destination table must be GlobalTensor with __gm__ qualifier (GM memory).");

    static_assert(TileSrc::Loc == TileType::Vec, "MSCATTER source must be Vec tile (UB/ubuf location).");

    static_assert(TileIdx::Loc == TileType::Vec, "MSCATTER indices must be Vec tile (UB/ubuf location).");

    static_assert(TileSrc::isRowMajor, "MSCATTER source tile must be row major layout.");
    static_assert(TileIdx::isRowMajor, "MSCATTER indices tile must be row major layout.");

    static_assert(GlobalTable::layout == Layout::ND, "MSCATTER destination GlobalTensor must use ND layout.");

    static_assert(IsValidScatterAtomic<T, Atomic>::value,
                  "MSCATTER atomic operation not supported for this data type. "
                  "Add requires int32/uint32/float/half; Max/Min requires int32/float.");

    static_assert(TileSrc::Rows == TileIdx::Rows, "MSCATTER src.Rows must equal indices.Rows.");
    static_assert(TileIdx::Cols == 1 || TileIdx::Cols == TileSrc::Cols,
                  "MSCATTER indices must be [N,1] for row-indexed or [N,M] for element-indexed.");

    static_assert(TileSrc::Cols * sizeof(T) % 32 == 0,
                  "MSCATTER source row width must be 32-byte aligned "
                  "(Cols * sizeof(DType) must be multiple of 32).");

    static_assert(
        GlobalTable::staticShape[0] == 1 && GlobalTable::staticShape[1] == 1 && GlobalTable::staticShape[2] == 1,
        "MSCATTER GlobalTensor shape must have dimensions [0],[1],[2] equal to 1. "
        "Use Shape<1,1,1,TableRows,RowWidth>.");
}

template <ScatterAtomicOp Atomic = ScatterAtomicOp::None, ScatterOOB Mode = ScatterOOB::Undefined, typename GlobalTable,
          typename TileSrc, typename TileIdx>
PTO_INTERNAL void MSCATTER_IMPL(GlobalTable &table, TileSrc &src, TileIdx &indices)
{
    using T = typename TileSrc::DType;
    using TIdx = typename TileIdx::DType;
    using ShapeType = typename GlobalTable::Shape;

    MScatterCheck<Atomic>(table, src, indices);

    __gm__ T *tablePtr = reinterpret_cast<__gm__ T *>(table.data());

    constexpr uint32_t NumRows = TileSrc::Rows;
    constexpr uint32_t NumCols = TileSrc::Cols;

    if constexpr (TileIdx::Cols == 1) {
        // Row-indexed scatter: indices shape is [N, 1]
        constexpr uint32_t TableRows = static_cast<uint32_t>(ShapeType::staticShape[3]);
        MScatterRowImpl<T, TIdx, Atomic, Mode, TileSrc, TileIdx, NumRows, NumCols, TableRows>(tablePtr, src.data(),
                                                                                              indices.data());
    } else {
        // Element-indexed scatter: indices shape is [N, M]
        constexpr uint32_t TableSize = static_cast<uint32_t>(ShapeType::staticShape[3] * ShapeType::staticShape[4]);
        MScatterElemImpl<T, TIdx, Atomic, Mode, TileSrc, TileIdx, NumRows, NumCols, TableSize>(tablePtr, src.data(),
                                                                                               indices.data());
    }
}

} // namespace pto

#endif // MSCATTER_HPP
