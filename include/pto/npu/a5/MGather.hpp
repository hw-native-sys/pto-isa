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
#include "common.hpp"
#include "utils.hpp"

namespace pto {

enum class GatherOOB : uint8_t
{
    Undefined = 0, // No bounds check
    Clamp = 1,     // Clamp to valid range [0, tableSize-1]
    Wrap = 2,      // Modulo wrap (idx % tableSize)
    Zero = 3       // Return zero for OOB accesses
};

namespace mgather_cfg {
constexpr uint32_t WARP_SIZE = 32;
constexpr uint32_t NUM_WARPS = 32;
constexpr uint32_t TOTAL_THREADS = WARP_SIZE * NUM_WARPS;
} // namespace mgather_cfg

template <GatherOOB Mode>
AICORE PTO_INLINE uint32_t apply_gather_oob(uint32_t idx, uint32_t tableSize)
{
    if constexpr (Mode == GatherOOB::Undefined) {
        return idx;
    } else if constexpr (Mode == GatherOOB::Clamp) {
        return (idx >= tableSize) ? (tableSize - 1) : idx;
    } else if constexpr (Mode == GatherOOB::Wrap) {
        return idx % tableSize;
    } else { // Zero - handled at value load
        return (idx >= tableSize) ? 0 : idx;
    }
}

// SIMT Kernel: Row-Indexed Gather
// Semantics: dst[i, j] = table[indices[i], j]
// Thread mapping:
//   - tx (lane): Column index within row (strided for wide rows)
//   - ty (warp): Row index (strided for many rows)
template <typename T, typename TIdx, GatherOOB Mode, uint32_t NumRows, uint32_t RowWidth, uint32_t TableRows>
AICORE __simt_vf__ LAUNCH_BOUND(1024) PTO_INLINE
    void simt_mgather_row_kernel(__ubuf__ T *__restrict__ dst, __gm__ const T *__restrict__ table,
                                 __ubuf__ const TIdx *__restrict__ indices)
{
    const uint32_t tx = __cce_simt_get_TID_X();
    const uint32_t ty = __cce_simt_get_TID_Y();

// Each warp handles rows with stride
#pragma unroll(1)
    for (uint32_t row = ty; row < NumRows; row += mgather_cfg::NUM_WARPS) {
        // Load row index (warp-uniform read)
        uint32_t rawIdx = static_cast<uint32_t>(indices[row]);

        uint32_t safeIdx = apply_gather_oob<Mode>(rawIdx, TableRows);

        __gm__ const T *srcRow = table + safeIdx * RowWidth;
        __ubuf__ T *dstRow = dst + row * RowWidth;

// Coalesced column access
#pragma unroll(4)
        for (uint32_t col = tx; col < RowWidth; col += mgather_cfg::WARP_SIZE) {
            T val;
            if constexpr (Mode == GatherOOB::Zero) {
                val = (rawIdx >= TableRows) ? static_cast<T>(0) : srcRow[col];
            } else {
                val = srcRow[col];
            }
            dstRow[col] = val;
        }
    }
}

// Semantics: dst[i, j] = table[indices[i, j]]
// Thread mapping: Linearized element index
template <typename T, typename TIdx, GatherOOB Mode, uint32_t NumRows, uint32_t NumCols, uint32_t TableSize>
AICORE __simt_vf__ LAUNCH_BOUND(1024) PTO_INLINE
    void simt_mgather_elem_kernel(__ubuf__ T *__restrict__ dst, __gm__ const T *__restrict__ table,
                                  __ubuf__ const TIdx *__restrict__ indices)
{
    const uint32_t tx = __cce_simt_get_TID_X();
    const uint32_t ty = __cce_simt_get_TID_Y();
    const uint32_t tid = ty * mgather_cfg::WARP_SIZE + tx;

    constexpr uint32_t totalElements = NumRows * NumCols;

#pragma unroll(1)
    for (uint32_t i = tid; i < totalElements; i += mgather_cfg::TOTAL_THREADS) {
        uint32_t rawIdx = static_cast<uint32_t>(indices[i]);
        uint32_t safeIdx = apply_gather_oob<Mode>(rawIdx, TableSize);

        T val;
        if constexpr (Mode == GatherOOB::Zero) {
            val = (rawIdx >= TableSize) ? static_cast<T>(0) : table[safeIdx];
        } else {
            val = table[safeIdx];
        }
        dst[i] = val;
    }
}

template <typename T, typename TIdx, GatherOOB Mode, typename DstTileData, typename IdxTileData, uint32_t NumRows,
          uint32_t RowWidth, uint32_t TableRows>
__tf__ AICORE void MGatherRowImpl(typename DstTileData::TileDType __out__ dst, __gm__ const T *__restrict__ tablePtr,
                                  typename IdxTileData::TileDType __in__ indices)
{
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ const TIdx *idxPtr = (__ubuf__ const TIdx *)__cce_get_tile_ptr(indices);

    cce::async_invoke<simt_mgather_row_kernel<T, TIdx, Mode, NumRows, RowWidth, TableRows>>(
        cce::dim3{mgather_cfg::WARP_SIZE, mgather_cfg::NUM_WARPS}, dstPtr, tablePtr, idxPtr);
}

template <typename T, typename TIdx, GatherOOB Mode, typename DstTileData, typename IdxTileData, uint32_t NumRows,
          uint32_t NumCols, uint32_t TableSize>
__tf__ AICORE void MGatherElemImpl(typename DstTileData::TileDType __out__ dst, __gm__ const T *__restrict__ tablePtr,
                                   typename IdxTileData::TileDType __in__ indices)
{
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ const TIdx *idxPtr = (__ubuf__ const TIdx *)__cce_get_tile_ptr(indices);

    cce::async_invoke<simt_mgather_elem_kernel<T, TIdx, Mode, NumRows, NumCols, TableSize>>(
        cce::dim3{mgather_cfg::WARP_SIZE, mgather_cfg::NUM_WARPS}, dstPtr, tablePtr, idxPtr);
}

template <typename TileDst, typename GlobalTable, typename TileIdx>
PTO_INTERNAL void MGatherCheck(const TileDst &dst, const GlobalTable &table, const TileIdx &indices)
{
    using T = typename TileDst::DType;
    using TIdx = typename TileIdx::DType;

    static_assert(std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> ||
                      std::is_same_v<T, uint16_t> || std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
                      std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float> ||
#ifdef __CCE_AICORE__
                      std::is_same_v<T, float8_e4m3_t> || std::is_same_v<T, float8_e5m2_t> ||
#endif
                      false,
                  "MGATHER data type must be int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/"
                  "half/bfloat16_t/float/float8_e4m3_t/float8_e5m2_t.");

    static_assert(std::is_same_v<TIdx, int32_t> || std::is_same_v<TIdx, uint32_t>,
                  "MGATHER index type must be int32_t or uint32_t.");

    static_assert(std::is_same_v<typename GlobalTable::DType, __gm__ T>,
                  "MGATHER source table must be GlobalTensor with __gm__ qualifier (GM memory).");

    static_assert(TileDst::Loc == TileType::Vec, "MGATHER destination must be Vec tile (UB/ubuf location).");

    static_assert(TileIdx::Loc == TileType::Vec, "MGATHER indices must be Vec tile (UB/ubuf location).");

    static_assert(TileDst::isRowMajor, "MGATHER destination tile must be row major layout.");
    static_assert(TileIdx::isRowMajor, "MGATHER indices tile must be row major layout.");

    static_assert(GlobalTable::layout == Layout::ND, "MGATHER source GlobalTensor must use ND layout.");

    static_assert(TileDst::Rows == TileIdx::Rows, "MGATHER dst.Rows must equal indices.Rows.");
    static_assert(TileIdx::Cols == 1 || TileIdx::Cols == TileDst::Cols,
                  "MGATHER indices must be [N,1] for row-indexed or [N,M] for element-indexed.");

    static_assert(TileDst::Cols * sizeof(T) % 32 == 0,
                  "MGATHER destination row width must be 32-byte aligned "
                  "(Cols * sizeof(DType) must be multiple of 32).");

    static_assert(
        GlobalTable::staticShape[0] == 1 && GlobalTable::staticShape[1] == 1 && GlobalTable::staticShape[2] == 1,
        "MGATHER GlobalTensor shape must have dimensions [0],[1],[2] equal to 1. "
        "Use Shape<1,1,1,TableRows,RowWidth>.");
}

template <GatherOOB Mode = GatherOOB::Undefined, typename TileDst, typename GlobalTable, typename TileIdx>
PTO_INTERNAL void MGATHER_IMPL(TileDst &dst, GlobalTable &table, TileIdx &indices)
{
    using T = typename TileDst::DType;
    using TIdx = typename TileIdx::DType;
    using ShapeType = typename GlobalTable::Shape;

    MGatherCheck(dst, table, indices);

    __gm__ const T *tablePtr = reinterpret_cast<__gm__ const T *>(table.data());

    constexpr uint32_t NumRows = TileDst::Rows;
    constexpr uint32_t NumCols = TileDst::Cols;

    if constexpr (TileIdx::Cols == 1) {
        // Row-indexed gather: indices shape is [N, 1]
        constexpr uint32_t TableRows = static_cast<uint32_t>(ShapeType::staticShape[3]);
        MGatherRowImpl<T, TIdx, Mode, TileDst, TileIdx, NumRows, NumCols, TableRows>(dst.data(), tablePtr,
                                                                                     indices.data());
    } else {
        // Element-indexed gather: indices shape is [N, M]
        constexpr uint32_t TableSize = static_cast<uint32_t>(ShapeType::staticShape[3] * ShapeType::staticShape[4]);
        MGatherElemImpl<T, TIdx, Mode, TileDst, TileIdx, NumRows, NumCols, TableSize>(dst.data(), tablePtr,
                                                                                      indices.data());
    }
}

} // namespace pto

#endif // MGATHER_HPP
