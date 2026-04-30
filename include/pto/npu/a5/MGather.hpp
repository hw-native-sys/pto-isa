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
    Undefined = 0,
    Clamp = 1,
    Wrap = 2,
    Zero = 3
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

template <typename T, typename TIdx, typename TileDst, GatherOOB Oob, uint32_t ValidRows, uint32_t ValidCols,
          uint32_t TableRows>
AICORE __simt_vf__ LAUNCH_BOUND(1024) PTO_INLINE
    void simt_mgather_row_kernel(__ubuf__ T *__restrict__ dst, __gm__ const T *__restrict__ table,
                                 __ubuf__ const TIdx *__restrict__ indices, uint32_t validRowsRT, uint32_t validColsRT,
                                 uint32_t tableRowsRT)
{
    using Launch = mgather_cfg::RowLaunch<ValidRows, ValidCols>;
    constexpr uint32_t kRowWarps = Launch::kRowWarps;
    constexpr uint32_t kWarpsPerRow = (Launch::kWarpsPerRow == 0u) ? 1u : Launch::kWarpsPerRow;
    constexpr uint32_t kColStride = kWarpsPerRow * mgather_cfg::WARP_SIZE;

    const uint32_t tx = threadIdx.x;
    const uint32_t ty = threadIdx.y;
    const uint32_t rowWarp = ty % kRowWarps;
    const uint32_t colSeg = ty / kRowWarps;

#pragma unroll(1)
    for (uint32_t row = rowWarp; row < ValidRows; row += kRowWarps) {
        const uint32_t rawIdx = static_cast<uint32_t>(indices[row]);
        uint32_t doRead;
        const uint32_t safeIdx = gather_remap<Oob>(rawIdx, TableRows, doRead);
        __gm__ const T *srcRow = table + safeIdx * ValidCols;
#pragma unroll(4)
        for (uint32_t col = colSeg * mgather_cfg::WARP_SIZE + tx; col < ValidCols; col += kColStride) {
            const T val = doRead ? srcRow[col] : static_cast<T>(0);
            dst[tile_offset_2d<TileDst>(row, col)] = val;
        }
    }
}

template <typename T, typename TIdx, typename TileDst, typename TileIdx, GatherOOB Oob, uint32_t ValidRows,
          uint32_t ValidCols, uint32_t TableSize>
AICORE __simt_vf__ LAUNCH_BOUND(1024) PTO_INLINE
    void simt_mgather_elem_kernel(__ubuf__ T *__restrict__ dst, __gm__ const T *__restrict__ table,
                                  __ubuf__ const TIdx *__restrict__ indices, uint32_t validRowsRT, uint32_t validColsRT,
                                  uint32_t tableSizeRT)
{
    constexpr uint32_t kTotalElems = ValidRows * ValidCols;
    constexpr uint32_t kLaunchThreads = mgather_cfg::ElemLaunch<kTotalElems>::kLaunchWarps * mgather_cfg::WARP_SIZE;

    const uint32_t tx = threadIdx.x;
    const uint32_t ty = threadIdx.y;
    const uint32_t tid = ty * mgather_cfg::WARP_SIZE + tx;

#pragma unroll(1)
    for (uint32_t i = tid; i < kTotalElems; i += kLaunchThreads) {
        const uint32_t r = (ValidCols == 1u) ? i : (i / ValidCols);
        const uint32_t c = (ValidCols == 1u) ? 0u : (i - r * ValidCols);
        const uint32_t dstOff = tile_offset_2d<TileDst>(r, c);
        const uint32_t idxOff = tile_offset_2d<TileIdx>(r, c);
        const uint32_t rawIdx = static_cast<uint32_t>(indices[idxOff]);
        uint32_t doRead;
        const uint32_t safeIdx = gather_remap<Oob>(rawIdx, TableSize, doRead);
        dst[dstOff] = doRead ? table[safeIdx] : static_cast<T>(0);
    }
}

template <typename T, typename TIdx, GatherOOB Oob, typename DstTileData, typename IdxTileData, uint32_t ValidRows,
          uint32_t ValidCols, uint32_t TableRows>
__tf__ AICORE void MGatherRowImpl(typename DstTileData::TileDType __out__ dst, __gm__ const T *__restrict__ tablePtr,
                                  typename IdxTileData::TileDType __in__ indices, uint32_t validRows,
                                  uint32_t validCols, uint32_t tableRows)
{
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ const TIdx *idxPtr = (__ubuf__ const TIdx *)__cce_get_tile_ptr(indices);

    constexpr uint32_t kLaunchWarps = mgather_cfg::RowLaunch<ValidRows, ValidCols>::kLaunchWarps;
    cce::async_invoke<simt_mgather_row_kernel<T, TIdx, DstTileData, Oob, ValidRows, ValidCols, TableRows>>(
        cce::dim3{mgather_cfg::WARP_SIZE, kLaunchWarps}, dstPtr, tablePtr, idxPtr);
}

template <typename T, typename TIdx, GatherOOB Oob, typename DstTileData, typename IdxTileData, uint32_t ValidRows,
          uint32_t ValidCols, uint32_t TableSize>
__tf__ AICORE void MGatherElemImpl(typename DstTileData::TileDType __out__ dst, __gm__ const T *__restrict__ tablePtr,
                                   typename IdxTileData::TileDType __in__ indices, uint32_t validRows,
                                   uint32_t validCols, uint32_t tableSize)
{
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ const TIdx *idxPtr = (__ubuf__ const TIdx *)__cce_get_tile_ptr(indices);

    constexpr uint32_t kLaunchWarps = mgather_cfg::ElemLaunch<ValidRows * ValidCols>::kLaunchWarps;
    cce::async_invoke<
        simt_mgather_elem_kernel<T, TIdx, DstTileData, IdxTileData, Oob, ValidRows, ValidCols, TableSize>>(
        cce::dim3{mgather_cfg::WARP_SIZE, kLaunchWarps}, dstPtr, tablePtr, idxPtr);
}

template <typename T, typename TIdx, GatherOOB Oob, typename DstTileData, typename IdxTileData, uint32_t TableSize>
__tf__ AICORE void MGatherScalarImpl(typename DstTileData::TileDType __out__ dst, __gm__ const T *__restrict__ tablePtr,
                                     typename IdxTileData::TileDType __in__ indices)
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
        safeIdx = (rawIdx >= TableSize) ? (TableSize - 1u) : rawIdx;
    } else if constexpr (Oob == GatherOOB::Wrap) {
        doRead = 1u;
        safeIdx = rawIdx % TableSize;
    } else {
        doRead = (rawIdx < TableSize) ? 1u : 0u;
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
                  "MGATHER data type must be int8/uint8/int16/uint16/int32/uint32/half/bfloat16/float "
                  "(and on AICORE: hifloat8/float8_e4m3/float8_e5m2).");

    static_assert(std::is_same_v<TIdx, int32_t> || std::is_same_v<TIdx, uint32_t>,
                  "MGATHER index type must be int32_t or uint32_t.");

    static_assert(std::is_same_v<typename GlobalTable::DType, __gm__ T>,
                  "MGATHER source table must be a GM GlobalTensor with element type matching the destination tile.");

    static_assert(TileDst::Loc == TileType::Vec, "MGATHER destination must be a Vec tile (UB).");
    static_assert(TileIdx::Loc == TileType::Vec, "MGATHER indices must be a Vec tile (UB).");

    constexpr uint32_t kDstValidR = static_cast<uint32_t>(TileDst::ValidRow);
    constexpr uint32_t kDstValidC = static_cast<uint32_t>(TileDst::ValidCol);
    constexpr uint32_t kIdxValidR = static_cast<uint32_t>(TileIdx::ValidRow);
    constexpr uint32_t kIdxValidC = static_cast<uint32_t>(TileIdx::ValidCol);

    using ShapeType = typename GlobalTable::Shape;
    constexpr uint32_t kTableCols = static_cast<uint32_t>(ShapeType::staticShape[4]);

    if constexpr (Mode == Coalesce::Row) {
        static_assert(kDstValidR >= 1u && kDstValidC >= 1u,
                      "MGATHER Coalesce::Row requires non-empty valid destination shape [R, C].");
        static_assert(
            (kIdxValidR == 1u && kIdxValidC == kDstValidR) || (kIdxValidR == kDstValidR && kIdxValidC == 1u),
            "MGATHER Coalesce::Row requires index tile valid shape [1, R] or [R, 1] matching TileDst::ValidRow.");
        static_assert(kTableCols == kDstValidC,
                      "MGATHER Coalesce::Row requires GlobalTensor inner dim (Shape[4]) == TileDst::ValidCol.");
    } else {
        static_assert(kDstValidR >= 1u && kDstValidC >= 1u,
                      "MGATHER Coalesce::Elem requires non-empty valid destination shape.");
        static_assert(kIdxValidR == kDstValidR && kIdxValidC == kDstValidC,
                      "MGATHER Coalesce::Elem requires index tile valid shape == destination tile valid shape.");
    }
}

template <Coalesce Mode = Coalesce::Row, GatherOOB Oob = GatherOOB::Undefined, typename TileDst, typename GlobalTable,
          typename TileIdx>
PTO_INTERNAL void MGATHER_IMPL(TileDst &dst, GlobalTable &table, TileIdx &indices)
{
    using T = typename TileDst::DType;

    MGatherCheck<Mode>(dst, table, indices);

    __gm__ const T *tablePtr = reinterpret_cast<__gm__ const T *>(table.data());

    constexpr uint32_t kValidRows = static_cast<uint32_t>(TileDst::ValidRow);
    constexpr uint32_t kValidCols = static_cast<uint32_t>(TileDst::ValidCol);

    if constexpr (Mode == Coalesce::Row) {
        constexpr uint32_t TableRows = static_cast<uint32_t>(ShapeType::staticShape[3]);
        MGatherRowImpl<T, TIdx, Oob, TileDst, TileIdx, kValidRows, kValidCols, TableRows>(dst.data(), tablePtr,
                                                                                          indices.data());
    } else {
        constexpr uint32_t TableSize =
            static_cast<uint32_t>(ShapeType::staticShape[0]) * static_cast<uint32_t>(ShapeType::staticShape[1]) *
            static_cast<uint32_t>(ShapeType::staticShape[2]) * static_cast<uint32_t>(ShapeType::staticShape[3]) *
            static_cast<uint32_t>(ShapeType::staticShape[4]);
        if constexpr (kValidRows == 1u && kValidCols == 1u) {
            MGatherScalarImpl<T, TIdx, Oob, TileDst, TileIdx, TableSize>(dst.data(), tablePtr, indices.data());
        } else {
            MGatherElemImpl<T, TIdx, Oob, TileDst, TileIdx, kValidRows, kValidCols, TableSize>(dst.data(), tablePtr,
                                                                                               indices.data());
        }
    }
}

} // namespace pto

#endif // MGATHER_HPP
