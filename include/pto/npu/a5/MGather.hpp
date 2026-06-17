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

} // namespace pto

#endif // MGATHER_HPP
