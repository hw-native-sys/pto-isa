/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCOLREDUCEIDX_HPP
#define TCOLREDUCEIDX_HPP

#include <pto/common/utils.hpp>
#include <pto/common/type.hpp>

namespace pto {
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TColReduceIdxCheck(unsigned srcValidRow, unsigned srcValidCol, unsigned dstValidRow,
                                     unsigned dstValidCol)
{
    static_assert(TileDataIn::ValidCol == 1 || TileDataIn::ValidCol == -1,
                  "Fix: TCOLARGMAX Src ValidCol must be 1 or -1");
    static_assert(
        std::is_same_v<typename TileDataIn::DType, uint32_t> || std::is_same_v<typename TileDataIn::DType, uint16_t> ||
            std::is_same_v<typename TileDataIn::DType, half> || std::is_same_v<typename TileDataIn::DType, float>,
        "Fix: TCOLARGMAX input data type must be f16/u16/f32/u32");
    static_assert(TileDataIn::Loc == pto::TileType::Vec, "Fix: TCOLARGMAX Src TileType must be Vec Tile!");
    static_assert(TileDataOut::Loc == pto::TileType::Vec, "Fix: TCOLARGMAX Dst TileType must be Vec Tile!");
    static_assert(TileDataIn::SFractal == SLayout::NoneBox, "Fix: TCOLARGMAX only support Nd or Dn fractal Tile");
    static_assert(TileDataOut::isRowMajor && TileDataOut::SFractal == SLayout::NoneBox,
                  "Fix: TCOLARGMAX only support Nd fractal Tile");
    static_assert(
        std::is_same_v<typename TileDataOut::DType, uint32_t> || std::is_same_v<typename TileDataOut::DType, int32_t>,
        "Fix: TCOLARGMAX output data type must be s32 or u32.");
    static_assert(std::is_same_v<typename TileDataIn::DType, typename TileDataTmp::DType>,
                  "Fix: TCOLARGMAX input type must be consistent with the tmp type");
    PTO_ASSERT(srcValidRow != 0 && srcValidCol != 0,
               "Fix: TCOLARGMAX input shape is invalid, validCol or validRow is 0.");
    PTO_ASSERT(dstValidRow == 1, "Fix: TCOLARGMAX output validRow must be 1");
    PTO_ASSERT(srcValidCol == dstValidCol,
               "Fix: TCOLARGMAX input validCol must be consistent with the output validCol");
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, bool IsArgMax>
__tf__ PTO_INTERNAL void TColReduceIdx16(typename TileDataOut::TileDType __out__ dst,
                                         typename TileDataIn::TileDType __in__ src,
                                         typename TileDataTmp::TileDType __in__ tmp, unsigned srcValidRow,
                                         unsigned srcValidCol)
{
    using TOUT = typename TileDataOut::DType;
    using T = typename TileDataIn::DType;
    constexpr uint32_t srcRowStride = TileDataIn::Cols;
    constexpr uint32_t elemPerRpt = REPEAT_BYTE / sizeof(T);
    constexpr uint32_t elemPerBlock = BLOCK_BYTE_SIZE / sizeof(T);
    __ubuf__ TOUT *dstPtr = (__ubuf__ TOUT *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    __ubuf__ T *tmpPtr = (__ubuf__ T *)__cce_get_tile_ptr(tmp);

    uint16_t numLoop = srcValidCol / elemPerRpt;
    uint16_t remainAfterLoop = srcValidCol % elemPerRpt;
    uint32_t tmpGapEles = numLoop > 0 ? elemPerRpt : CeilDivision(srcValidCol, elemPerBlock) * elemPerBlock;

    for (uint16_t j = 0; j < numLoop; j++) {
        pipe_barrier(PIPE_V);
        vector_dup((__ubuf__ int16_t *)tmpPtr, 0, 1, 1, 1, 0, 0);                        // cur index
        vector_dup((__ubuf__ int16_t *)tmpPtr + 2 * tmpGapEles, 0, 1, 1, 1, 0, 0);       // argmin index
        pto_copy_ubuf_to_ubuf(tmpPtr + tmpGapEles, srcPtr + j * elemPerRpt, 1, 8, 0, 0); // min elements
        pipe_barrier(PIPE_V);
        for (uint16_t i = 1; i < srcValidRow; i++) {
            vadds((__ubuf__ int16_t *)tmpPtr, (__ubuf__ int16_t *)tmpPtr, 1, 1, 1, 1, 0, 0);
            if constexpr (IsArgMax) {
                vcmp_ge((__ubuf__ half *)tmpPtr + tmpGapEles,
                        (__ubuf__ half *)srcPtr + i * srcRowStride + j * elemPerRpt, 1, 1, 1, 1, 0, 0, 0);
                pipe_barrier(PIPE_V);
                vsel((__ubuf__ half *)tmpPtr + 2 * tmpGapEles, (__ubuf__ half *)tmpPtr + 2 * tmpGapEles,
                     (__ubuf__ half *)tmpPtr, 1, 1, 1, 1, 0, 0, 0, 0);
                vmax((__ubuf__ half *)tmpPtr + tmpGapEles, (__ubuf__ half *)tmpPtr + tmpGapEles,
                     (__ubuf__ half *)srcPtr + i * srcRowStride + j * elemPerRpt, 1, 1, 1, 1, 0, 0, 0);
            } else {
                vcmp_le((__ubuf__ half *)tmpPtr + tmpGapEles,
                        (__ubuf__ half *)srcPtr + i * srcRowStride + j * elemPerRpt, 1, 1, 1, 1, 0, 0, 0);
                pipe_barrier(PIPE_V);
                vsel((__ubuf__ half *)tmpPtr + 2 * tmpGapEles, (__ubuf__ half *)tmpPtr + 2 * tmpGapEles,
                     (__ubuf__ half *)tmpPtr, 1, 1, 1, 1, 0, 0, 0, 0);
                vmin((__ubuf__ half *)tmpPtr + tmpGapEles, (__ubuf__ half *)tmpPtr + tmpGapEles,
                     (__ubuf__ half *)srcPtr + i * srcRowStride + j * elemPerRpt, 1, 1, 1, 1, 0, 0, 0);
            }
            pipe_barrier(PIPE_V);
        }

        vconv_s162f16a((__ubuf__ half *)tmpPtr + 2 * tmpGapEles, (__ubuf__ int16_t *)tmpPtr + 2 * tmpGapEles, 1, 1, 1,
                       0, 0);
        pipe_barrier(PIPE_V);
        vconv_f162s32a((__ubuf__ int32_t *)dstPtr + j * elemPerRpt, (__ubuf__ half *)tmpPtr + 2 * tmpGapEles, 2, 1, 1,
                       8, 4);
        pipe_barrier(PIPE_V);
    }
    if (remainAfterLoop > 0) {
        set_mask_count();
        set_vector_mask(0, remainAfterLoop);
        vector_dup(tmpPtr, 0, 1, 1, 1, 0, 0);
        vector_dup((__ubuf__ int16_t *)tmpPtr + 2 * tmpGapEles, 0, 1, 1, 1, 0, 0);
        vcopy((__ubuf__ int16_t *)tmpPtr + tmpGapEles, (__ubuf__ int16_t *)srcPtr + numLoop * elemPerRpt, 1, 1, 1, 0,
              0);
        pipe_barrier(PIPE_V);

        for (uint16_t i = 1; i < srcValidRow; i++) {
            vadds((__ubuf__ int16_t *)tmpPtr, (__ubuf__ int16_t *)tmpPtr, 1, 1, 1, 1, 0, 0);
            if constexpr (IsArgMax) {
                vcmp_ge((__ubuf__ half *)tmpPtr + tmpGapEles,
                        (__ubuf__ half *)srcPtr + i * srcRowStride + numLoop * elemPerRpt, 1, 1, 1, 1, 0, 0, 0);
                pipe_barrier(PIPE_V);
                vsel((__ubuf__ half *)tmpPtr + 2 * tmpGapEles, (__ubuf__ half *)tmpPtr + 2 * tmpGapEles,
                     (__ubuf__ half *)tmpPtr, 1, 1, 1, 1, 0, 0, 0, 0);
                vmax((__ubuf__ half *)tmpPtr + tmpGapEles, (__ubuf__ half *)tmpPtr + tmpGapEles,
                     (__ubuf__ half *)srcPtr + i * srcRowStride + numLoop * elemPerRpt, 1, 1, 1, 1, 0, 0, 0);
            } else {
                vcmp_le((__ubuf__ half *)tmpPtr + tmpGapEles,
                        (__ubuf__ half *)srcPtr + i * srcRowStride + numLoop * elemPerRpt, 1, 1, 1, 1, 0, 0, 0);
                pipe_barrier(PIPE_V);
                vsel((__ubuf__ half *)tmpPtr + 2 * tmpGapEles, (__ubuf__ half *)tmpPtr + 2 * tmpGapEles,
                     (__ubuf__ half *)tmpPtr, 1, 1, 1, 1, 0, 0, 0, 0);
                vmin((__ubuf__ half *)tmpPtr + tmpGapEles, (__ubuf__ half *)tmpPtr + tmpGapEles,
                     (__ubuf__ half *)srcPtr + i * srcRowStride + numLoop * elemPerRpt, 1, 1, 1, 1, 0, 0, 0);
            }
            pipe_barrier(PIPE_V);
        }

        vconv_s162f16a((__ubuf__ half *)tmpPtr + 2 * tmpGapEles, (__ubuf__ int16_t *)tmpPtr + 2 * tmpGapEles, 1, 1, 1,
                       0, 0);
        pipe_barrier(PIPE_V);
        vconv_f162s32a((__ubuf__ int32_t *)dstPtr + numLoop * elemPerRpt, (__ubuf__ half *)tmpPtr + 2 * tmpGapEles, 2,
                       1, 1, 8, 4);
        set_mask_norm();
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
    }
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, bool IsArgMax>
__tf__ PTO_INTERNAL void TColReduceIdx32(typename TileDataOut::TileDType __out__ dst,
                                         typename TileDataIn::TileDType __in__ src,
                                         typename TileDataTmp::TileDType __in__ tmp, unsigned srcValidRow,
                                         unsigned srcValidCol)
{
    using TOUT = typename TileDataOut::DType;
    using T = typename TileDataIn::DType;
    __ubuf__ TOUT *dstPtr = (__ubuf__ TOUT *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    __ubuf__ T *tmpPtr = (__ubuf__ T *)__cce_get_tile_ptr(tmp);

    constexpr uint32_t srcRowStride = TileDataIn::Cols;
    constexpr uint32_t elemPerRpt = REPEAT_BYTE / sizeof(T);
    constexpr uint32_t elemPerBlock = BLOCK_BYTE_SIZE / sizeof(T);
    uint16_t numLoop = srcValidCol / elemPerRpt;
    uint16_t remainAfterLoop = srcValidCol % elemPerRpt;
    uint32_t tmpGapEles = numLoop > 0 ? elemPerRpt : CeilDivision(srcValidCol, elemPerBlock) * elemPerBlock;

    for (uint16_t j = 0; j < numLoop; j++) {
        vector_dup(dstPtr + j * elemPerRpt, 0, 1, 1, 1, 0, 0);                           // argmin index
        vector_dup(tmpPtr, 0, 1, 1, 1, 0, 0);                                            // cur index
        pto_copy_ubuf_to_ubuf(tmpPtr + tmpGapEles, srcPtr + j * elemPerRpt, 1, 8, 0, 0); // min elements
        pipe_barrier(PIPE_V);
        for (uint16_t i = 1; i < srcValidRow; i++) {
            vadds((__ubuf__ int32_t *)tmpPtr, (__ubuf__ int32_t *)tmpPtr, 1, 1, 1, 1, 0, 0);
            if constexpr (IsArgMax) {
                vcmp_ge((__ubuf__ float *)tmpPtr + tmpGapEles,
                        (__ubuf__ float *)srcPtr + i * srcRowStride + j * elemPerRpt, 1, 1, 1, 1, 0, 0, 0);
                pipe_barrier(PIPE_V);
                vsel((__ubuf__ float *)dstPtr + j * elemPerRpt, (__ubuf__ float *)dstPtr + j * elemPerRpt,
                     (__ubuf__ float *)tmpPtr, 1, 1, 1, 1, 0, 0, 0, 0);
                vmax((__ubuf__ float *)tmpPtr + tmpGapEles, (__ubuf__ float *)tmpPtr + tmpGapEles,
                     (__ubuf__ float *)srcPtr + i * srcRowStride + j * elemPerRpt, 1, 1, 1, 1, 0, 0, 0);
            } else {
                vcmp_le((__ubuf__ float *)tmpPtr + tmpGapEles,
                        (__ubuf__ float *)srcPtr + i * srcRowStride + j * elemPerRpt, 1, 1, 1, 1, 0, 0, 0);
                pipe_barrier(PIPE_V);
                vsel((__ubuf__ float *)dstPtr + j * elemPerRpt, (__ubuf__ float *)dstPtr + j * elemPerRpt,
                     (__ubuf__ float *)tmpPtr, 1, 1, 1, 1, 0, 0, 0, 0);
                vmin((__ubuf__ float *)tmpPtr + tmpGapEles, (__ubuf__ float *)tmpPtr + tmpGapEles,
                     (__ubuf__ float *)srcPtr + i * srcRowStride + j * elemPerRpt, 1, 1, 1, 1, 0, 0, 0);
            }
            pipe_barrier(PIPE_V);
        }
    }
    if (remainAfterLoop > 0) {
        set_mask_count();
        set_vector_mask(0, remainAfterLoop);

        vector_dup(dstPtr + numLoop * elemPerRpt, 0, 1, 1, 1, 0, 0);
        vector_dup(tmpPtr, 0, 1, 1, 1, 0, 0);
        vcopy((__ubuf__ int32_t *)tmpPtr + tmpGapEles, (__ubuf__ int32_t *)srcPtr + numLoop * elemPerRpt, 1, 1, 1, 0,
              0);
        pipe_barrier(PIPE_V);

        for (uint16_t i = 1; i < srcValidRow; i++) {
            vadds((__ubuf__ int32_t *)tmpPtr, (__ubuf__ int32_t *)tmpPtr, 1, 1, 1, 1, 0, 0);
            if constexpr (IsArgMax) {
                vcmp_ge((__ubuf__ float *)tmpPtr + tmpGapEles,
                        (__ubuf__ float *)srcPtr + i * srcRowStride + numLoop * elemPerRpt, 1, 1, 1, 1, 0, 0, 0);
                pipe_barrier(PIPE_V);
                vsel((__ubuf__ float *)dstPtr + numLoop * elemPerRpt, (__ubuf__ float *)dstPtr + numLoop * elemPerRpt,
                     (__ubuf__ float *)tmpPtr, 1, 1, 1, 1, 0, 0, 0, 0);
                vmax((__ubuf__ float *)tmpPtr + tmpGapEles, (__ubuf__ float *)tmpPtr + tmpGapEles,
                     (__ubuf__ float *)srcPtr + i * srcRowStride + numLoop * elemPerRpt, 1, 1, 1, 1, 0, 0, 0);
            } else {
                vcmp_le((__ubuf__ float *)tmpPtr + tmpGapEles,
                        (__ubuf__ float *)srcPtr + i * srcRowStride + numLoop * elemPerRpt, 1, 1, 1, 1, 0, 0, 0);
                pipe_barrier(PIPE_V);
                vsel((__ubuf__ float *)dstPtr + numLoop * elemPerRpt, (__ubuf__ float *)dstPtr + numLoop * elemPerRpt,
                     (__ubuf__ float *)tmpPtr, 1, 1, 1, 1, 0, 0, 0, 0);
                vmin((__ubuf__ float *)tmpPtr + tmpGapEles, (__ubuf__ float *)tmpPtr + tmpGapEles,
                     (__ubuf__ float *)srcPtr + i * srcRowStride + numLoop * elemPerRpt, 1, 1, 1, 1, 0, 0, 0);
            }
            pipe_barrier(PIPE_V);
        }
        set_mask_norm();
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
    }
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, bool IsArgMax>
PTO_INTERNAL void TCOLARG_DISPATCH(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    unsigned srcValidRow = src.GetValidRow();
    unsigned srcValidCol = src.GetValidCol();
    TColReduceIdxCheck<TileDataOut, TileDataIn, TileDataTmp>(srcValidRow, srcValidCol, dst.GetValidRow(),
                                                             dst.GetValidCol());

    if (sizeof(typename TileDataIn::DType) == 2) {
        TColReduceIdx16<TileDataOut, TileDataIn, TileDataTmp, IsArgMax>(dst.data(), src.data(), tmp.data(), srcValidRow,
                                                                        srcValidCol);
    } else if (sizeof(typename TileDataIn::DType) == 4) {
        TColReduceIdx32<TileDataOut, TileDataIn, TileDataTmp, IsArgMax>(dst.data(), src.data(), tmp.data(), srcValidRow,
                                                                        srcValidCol);
    }
}
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TCOLARGMIN_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    TCOLARG_DISPATCH<TileDataOut, TileDataIn, TileDataTmp, false>(dst, src, tmp); // Min
}
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TCOLARGMAX_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    TCOLARG_DISPATCH<TileDataOut, TileDataIn, TileDataTmp, true>(dst, src, tmp); // Max
}
} // namespace pto
#endif
