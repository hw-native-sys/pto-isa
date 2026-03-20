/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TGATHER_HPP
#define TGATHER_HPP

#include <pto/common/constants.hpp>
#include "common.hpp"

namespace pto {
template <typename DstTileData, typename Src0TileData, typename Src1TileData>
PTO_INTERNAL void CheckValid()
{
    static_assert((sizeof(typename DstTileData::DType) == 1) || (sizeof(typename DstTileData::DType) == 2) ||
                      (sizeof(typename DstTileData::DType) == 4),
                  "Fix: TGATHER expect b8/b16/b32");
    static_assert((sizeof(typename Src1TileData::DType) == 2) || (sizeof(typename Src1TileData::DType) == 4),
                  "Fix: TGATHER expect b16/b32");
    static_assert((std::is_same<typename DstTileData::DType, typename Src0TileData::DType>::value),
                  "Fix: TGATHER expect same size for indice and dst");
}

template <typename TileDataD, typename TileDataS0, typename TileDataS1>
__tf__ AICORE void TGather_b32(typename TileDataD::TileDType __out__ dst, typename TileDataS0::TileDType __in__ src0,
                               typename TileDataS1::TileDType __in__ src1, unsigned validCol, unsigned validRow)
{
    __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ typename TileDataS0::DType *src0Ptr = (__ubuf__ typename TileDataS0::DType *)__cce_get_tile_ptr(src0);
    __ubuf__ typename TileDataS1::DType *src1Ptr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(src1);
    unsigned TShape1 = TileDataD::Cols;
    __VEC_SCOPE__
    {
        uint16_t batchSize = 256 / static_cast<uint16_t>(sizeof(typename TileDataS1::DType));
        uint16_t innerLoopNum = CeilDivision(validCol, batchSize);
        for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
            for (uint16_t j = 0; j < innerLoopNum; ++j) {
                RegTensor<typename TileDataS1::DType> index;
                vlds(index, src1Ptr, (i * TileDataS1::Cols + j * batchSize), NORM);

                uint32_t count = ((j + 1) * batchSize >= validCol ? validCol - j * batchSize : batchSize);
                vector_bool preg = CreatePredicate<typename TileDataS1::DType>(count);

                RegTensor<typename TileDataD::DType> v_output;
                vgather2(v_output, src0Ptr, (vector_u32 &)index, preg);
                vsts(v_output, dstPtr, (i * TShape1 + j * batchSize), NORM_B32, preg);
            }
        }
    }
}

template <typename TileDataD, typename TileDataS0, typename TileDataS1>
__tf__ AICORE void TGather_b16(typename TileDataD::TileDType __out__ dst, typename TileDataS0::TileDType __in__ src0,
                               typename TileDataS1::TileDType __in__ src1, unsigned validCol, unsigned validRow)
{
    __ubuf__ typename TileDataS0::DType *src0Ptr = (__ubuf__ typename TileDataS0::DType *)__cce_get_tile_ptr(src0);
    __ubuf__ typename TileDataS1::DType *src1Ptr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(src1);
    __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
    unsigned TShape1 = TileDataD::Cols;
    __VEC_SCOPE__
    {
        uint16_t batchSize = 256 / static_cast<uint16_t>(sizeof(typename TileDataS1::DType));
        uint16_t loop_num = CeilDivision(validCol, batchSize);
        for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
            for (uint16_t j = 0; j < loop_num; ++j) {
                RegTensor<typename TileDataS1::DType> index;
                vlds(index, src1Ptr, (i * TileDataS1::Cols + j * batchSize), NORM);

                uint32_t count = ((j + 1) * batchSize >= validCol ? validCol - j * batchSize : batchSize);
                vector_bool preg = CreatePredicate<typename TileDataS1::DType>(count);

                RegTensor<typename TileDataD::DType> vOutput;
                vgather2(vOutput, src0Ptr, (vector_u16 &)index, preg);
                vsts(vOutput, dstPtr, (i * TShape1 + j * batchSize), NORM_B16, preg);
            }
        }
    }
}

template <typename TileDataD, typename TileDataS0, typename TileDataS1>
__tf__ AICORE void TGather_b16_bc(typename TileDataD::TileDType __out__ dst, typename TileDataS0::TileDType __in__ src0,
                                  typename TileDataS1::TileDType __in__ src1, unsigned validCol, unsigned validRow)
{
    __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ typename TileDataS0::DType *src0Ptr = (__ubuf__ typename TileDataS0::DType *)__cce_get_tile_ptr(src0);
    __ubuf__ typename TileDataS1::DType *src1Ptr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(src1);
    unsigned TShapeDst = TileDataD::Cols;
    unsigned TShapeIdx = TileDataS1::Cols;
    __VEC_SCOPE__
    {
        uint16_t batchSize = 256 / static_cast<uint16_t>(sizeof(typename TileDataS1::DType));
        uint16_t loop_num = CeilDivision(validCol, batchSize);
        for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
            for (uint16_t j = 0; j < loop_num; ++j) {
                RegTensor<typename TileDataS1::DType> index;
                vlds(index, src1Ptr, (i * TShapeIdx + j * batchSize), NORM);

                uint32_t count = ((j + 1) * batchSize >= validCol ? validCol - j * batchSize : batchSize);
                vector_bool preg = CreatePredicate<typename TileDataS1::DType>(count);

                RegTensor<typename TileDataD::DType> v_output;
                vgather2_bc(v_output, src0Ptr, (vector_u32 &)index, preg);
                vsts(v_output, dstPtr, (i * TShapeDst + j * batchSize), PK_B32, preg);
            }
        }
    }
}

template <typename TileDataD, typename TileDataS0, typename TileDataS1>
__tf__ AICORE void TGather_fp8_e4m3(typename TileDataD::TileDType __out__ dst,
                                    typename TileDataS0::TileDType __in__ src0,
                                    typename TileDataS1::TileDType __in__ src1, unsigned validCol, unsigned validRow)
{
    __ubuf__ typename TileDataS0::DType *src0Ptr = (__ubuf__ typename TileDataS0::DType *)__cce_get_tile_ptr(src0);
    __ubuf__ typename TileDataS1::DType *src1Ptr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(src1);
    __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
    unsigned TDstShape = TileDataD::Cols;
    __VEC_SCOPE__
    {
        uint16_t batchSize = 256 / static_cast<uint16_t>(sizeof(typename TileDataS1::DType));
        uint16_t loopNum = CeilDivision(validCol, batchSize);
        for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
            for (uint16_t j = 0; j < loopNum; ++j) {
                RegTensor<typename TileDataS1::DType> index;
                vlds(index, src1Ptr, (i * TileDataS1::Cols + j * batchSize), NORM);

                uint32_t count = ((j + 1) * batchSize >= validCol ? validCol - j * batchSize : batchSize);
                vector_bool preg = plt_b16(count, POST_UPDATE);

                vector_f8e4m3 vOutput;
                vgather2(vOutput, src0Ptr, (vector_u16 &)index, preg);
                vsts((vector_u8)vOutput, (__ubuf__ uint8_t *)dstPtr, (i * TDstShape + j * batchSize), PK_B16, preg);
            }
        }
    }
}

template <typename TileDataD, typename TileDataS0, typename TileDataS1>
__tf__ AICORE void TGather_fp8_e5m2(typename TileDataD::TileDType __out__ dst,
                                    typename TileDataS0::TileDType __in__ src0,
                                    typename TileDataS1::TileDType __in__ src1, unsigned validCol, unsigned validRow)
{
    __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ typename TileDataS0::DType *src0Ptr = (__ubuf__ typename TileDataS0::DType *)__cce_get_tile_ptr(src0);
    __ubuf__ typename TileDataS1::DType *src1Ptr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(src1);
    unsigned TShape1 = TileDataD::Cols;
    __VEC_SCOPE__
    {
        constexpr uint16_t batchSize = 256 / static_cast<uint16_t>(sizeof(typename TileDataS1::DType));
        uint16_t loopNum = CeilDivision(validCol, batchSize);
        for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
            for (uint16_t j = 0; j < loopNum; ++j) {
                RegTensor<typename TileDataS1::DType> index;
                vlds(index, src1Ptr, (i * TileDataS1::Cols + j * batchSize), NORM);

                uint32_t count = ((j + 1) * batchSize >= validCol ? validCol - j * batchSize : batchSize);
                vector_bool preg = plt_b16(count, POST_UPDATE);

                vector_f8e5m2 output;
                vgather2(output, src0Ptr, (vector_u16 &)index, preg);
                vsts((vector_u8)output, (__ubuf__ uint8_t *)dstPtr, (i * TShape1 + j * batchSize), PK_B16, preg);
            }
        }
    }
}

template <typename TileDataD, typename TileDataS0, typename TileDataS1>
AICORE void TGather(__ubuf__ typename TileDataD::DType *dst, __ubuf__ typename TileDataS0::DType *src0,
                    __ubuf__ typename TileDataS1::DType *src1, unsigned validCol, unsigned validRow)
{
    if constexpr (sizeof(typename TileDataS0::DType) == 4) {
        TGather_b32<TileDataD, TileDataS0, TileDataS1>(dst, src0, src1, validCol, validRow);
    } else if constexpr (sizeof(typename TileDataS0::DType) == 2 && sizeof(typename TileDataS1::DType) == 2) {
        TGather_b16<TileDataD, TileDataS0, TileDataS1>(dst, src0, src1, validCol, validRow);
    } else if constexpr (sizeof(typename TileDataS0::DType) == 2 && sizeof(typename TileDataS1::DType) == 4) {
        TGather_b16_bc<TileDataD, TileDataS0, TileDataS1>(dst, src0, src1, validCol, validRow);
    } else if constexpr (std::is_same<typename TileDataS0::DType, float8_e4m3_t>::value) {
        TGather_fp8_e4m3<TileDataD, TileDataS0, TileDataS1>(dst, src0, src1, validCol, validRow);
    } else {
        TGather_fp8_e5m2<TileDataD, TileDataS0, TileDataS1>(dst, src0, src1, validCol, validRow);
    }
}

template <typename TileDataD, typename TileDataS0, typename TileDataS1>
PTO_INTERNAL void TGATHER_IMPL(TileDataD &dst, TileDataS0 &src0, TileDataS1 &src1)
{
    CheckValid<TileDataD, TileDataS0, TileDataS1>();

    unsigned kValidCols = dst.GetValidCol();
    unsigned kValidRows = dst.GetValidRow();

    TGather<TileDataD, TileDataS0, TileDataS1>(dst.data(), src0.data(), src1.data(), kValidCols, kValidRows);
}

template <typename T>
PTO_INTERNAL void PIntlvWithType(MaskReg &dst0, MaskReg &dst1, MaskReg src0, MaskReg src1)
{
    if constexpr (sizeof(T) == sizeof(float)) {
        pintlv_b32(dst0, dst1, src0, src1);
    } else if constexpr (sizeof(T) == sizeof(half)) {
        pintlv_b16(dst0, dst1, src0, src1);
    } else if constexpr (sizeof(T) == sizeof(uint8_t)) {
        pintlv_b8(dst0, dst1, src0, src1);
    }
}

template <typename T, MaskPattern maskPattern>
PTO_INTERNAL MaskReg GetMaskVal()
{
    MaskReg pg0;
    MaskReg pg1;
    MaskReg dstPg0;
    MaskReg dstPg1;
    if constexpr (maskPattern == MaskPattern::P0101) {
        pg0 = PSetWithType<T>(PAT_ALL);
        pg1 = PSetWithType<T>(PAT_ALLF);
        PIntlvWithType<T>(dstPg0, dstPg1, pg0, pg1);
    } else if constexpr (maskPattern == MaskPattern::P1010) {
        pg0 = PSetWithType<T>(PAT_ALL);
        pg1 = PSetWithType<T>(PAT_ALLF);
        PIntlvWithType<T>(dstPg0, dstPg1, pg1, pg0);
    } else if constexpr (maskPattern == MaskPattern::P0001) {
        pg0 = PSetWithType<T>(PAT_ALL);
        pg1 = PSetWithType<T>(PAT_ALLF);
        PIntlvWithType<T>(dstPg0, dstPg1, pg0, pg1);
        PIntlvWithType<T>(dstPg0, dstPg1, dstPg0, pg1);
    } else if constexpr (maskPattern == MaskPattern::P0010) {
        pg0 = PSetWithType<T>(PAT_ALL);
        pg1 = PSetWithType<T>(PAT_ALLF);
        PIntlvWithType<T>(dstPg0, dstPg1, pg0, pg1);
        PIntlvWithType<T>(dstPg0, dstPg1, pg1, dstPg0);
    } else if constexpr (maskPattern == MaskPattern::P0100) {
        pg0 = PSetWithType<T>(PAT_ALL);
        pg1 = PSetWithType<T>(PAT_ALLF);
        PIntlvWithType<T>(dstPg0, dstPg1, pg1, pg0);
        PIntlvWithType<T>(dstPg0, dstPg1, dstPg0, pg1);
    } else if constexpr (maskPattern == MaskPattern::P1000) {
        pg0 = PSetWithType<T>(PAT_ALL);
        pg1 = PSetWithType<T>(PAT_ALLF);
        PIntlvWithType<T>(dstPg0, dstPg1, pg1, pg0);
        PIntlvWithType<T>(dstPg0, dstPg1, pg1, dstPg0);
    } else if constexpr (maskPattern == MaskPattern::P1111) {
        dstPg0 = PSetWithType<T>(PAT_ALL);
    }
    return dstPg0;
}

template <typename DstTileData, typename SrcTileData, MaskPattern maskPattern>
__tf__ AICORE void TGather(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src,
                           uint16_t validRow, uint16_t validCol)
{
    using T = typename DstTileData::DType;
    constexpr unsigned rowStride = SrcTileData::RowStride;
    __ubuf__ typename DstTileData::DType *dstPtr = (__ubuf__ typename DstTileData::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ typename DstTileData::DType *srcPtr = (__ubuf__ typename DstTileData::DType *)__cce_get_tile_ptr(src);
    __VEC_SCOPE__
    {
        constexpr uint8_t SPR_AR_VALUE = 74;
        constexpr auto sprValue = std::integral_constant<::Spr, static_cast<::Spr>(SPR_AR_VALUE)>();
        sprclr(sprValue);

        MaskReg dstPg0 = GetMaskVal<T, maskPattern>();
        RegTensor<T> dstReg;
        RegTensor<T> srcReg;
        MaskReg loadMask;
        MaskReg executeMask;
        UnalignReg ureg;

        constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
        uint16_t innerRepeatTimes = CeilDivision(validCol, elementsPerRepeat);

        for (uint16_t i = 0; i < validRow; ++i) {
            uint32_t maskValue = validCol;
            for (uint16_t j = 0; j < innerRepeatTimes; ++j) {
                loadMask = CreatePredicate<T>(maskValue);
                vlds(srcReg, srcPtr + i * rowStride, j * elementsPerRepeat, NORM);
                pand(executeMask, dstPg0, loadMask, loadMask);
                vsqz(dstReg, srcReg, executeMask, MODE_STORED);
                vstur(ureg, dstReg, dstPtr, POST_UPDATE);
            }
        }
        vstar(ureg, dstPtr);
    }
}

template <typename DstTileData, typename SrcTileData, MaskPattern maskPattern>
PTO_INTERNAL void TGATHER_IMPL(DstTileData &dst, SrcTileData &src)
{
    using T = typename SrcTileData::DType;
    using U = typename DstTileData::DType;
    static_assert(std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int16_t> ||
                      std::is_same_v<T, uint16_t> || std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
                      std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float> ||
                      std::is_same_v<T, float8_e4m3_t> || std::is_same_v<T, float8_e5m2_t> ||
                      std::is_same_v<T, hifloat8_t>,
                  "Fix: TGATHER Src data type must be int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/"
                  "half/bfloat16_t/float/float8_e4m3_t/float8_e5m2_t/hifloat8_t.");
    static_assert(std::is_same_v<U, int8_t> || std::is_same_v<U, uint8_t> || std::is_same_v<U, int16_t> ||
                      std::is_same_v<U, uint16_t> || std::is_same_v<U, int32_t> || std::is_same_v<U, uint32_t> ||
                      std::is_same_v<U, half> || std::is_same_v<U, bfloat16_t> || std::is_same_v<U, float> ||
                      std::is_same_v<U, float8_e4m3_t> || std::is_same_v<U, float8_e5m2_t> ||
                      std::is_same_v<U, hifloat8_t>,
                  "Fix: TGATHER Dst data type must be int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/"
                  "half/bfloat16_t/float/float8_e4m3_t/float8_e5m2_t/hifloat8_t.");
    static_assert((sizeof(U) == sizeof(T)), "Fix: TGATHER expect same type size for dst and src");
    static_assert((DstTileData::Loc == TileType::Vec) && (SrcTileData::Loc == TileType::Vec),
                  "Fix: TGATHER expect vec TileType");
    static_assert((DstTileData::isRowMajor && SrcTileData::isRowMajor), "Fix: TGATHER expect row major");
    uint16_t rows = src.GetValidRow();
    uint16_t cols = src.GetValidCol();
    TGather<DstTileData, SrcTileData, maskPattern>(dst.data(), src.data(), rows, cols);
}

template <typename TileDataD, typename TileDataS, typename TileDataC, CmpMode cmpMode, uint32_t offset>
__tf__ AICORE void TGather_b32_gt(typename TileDataD::TileDType __out__ dst, typename TileDataS::TileDType __in__ src0,
                                  typename TileDataS::DType __in__ k_value, typename TileDataC::TileDType __in__ cdst,
                                  unsigned srcValidCol, unsigned srcValidRow, unsigned dstValidCol,
                                  unsigned dstValidRow)
{
    __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ typename TileDataS::DType *src0Ptr = (__ubuf__ typename TileDataS::DType *)__cce_get_tile_ptr(src0);
    __ubuf__ typename TileDataC::DType *cdstPtr = (__ubuf__ typename TileDataC::DType *)__cce_get_tile_ptr(cdst);

    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataS::DType);
    uint16_t repeatTimes = CeilDivision(srcValidCol, elementsPerRepeat);
    unsigned srcRowStride = TileDataS::RowStride;
    unsigned dstRowStride = TileDataD::RowStride;

    __VEC_SCOPE__
    {
        vector_bool preg_b32 = pset_b32(PAT_ALL);
        vector_s32 add_offset;
        vector_align align_index;
        vector_s32 index;
        vci(index, offset, INC_ORDER);
        sprclr(SPR_AR);
        vbr(add_offset, 0x00000040);

        for (uint16_t i = 0; i < (uint16_t)srcValidRow; ++i) {
            for (uint16_t j = 0; j < repeatTimes; ++j) {
                RegTensor<typename TileDataS::DType> score;
                vlds(score, src0Ptr, (i * TileDataS::Cols + j * elementsPerRepeat), NORM);

                uint32_t count = ((j + 1) * elementsPerRepeat >= srcValidCol ? srcValidCol - j * elementsPerRepeat :
                                                                               elementsPerRepeat);
                vector_bool preg = CreatePredicate<typename TileDataS::DType>(count);

                vector_bool pout_ge;
                vector_s32 sqz_index_out;
                vcmps_gt(pout_ge, score, k_value, preg);
                vsqz(sqz_index_out, index, pout_ge, MODE_STORED);
                vstur(align_index, (vector_u32)sqz_index_out, (__ubuf__ uint32_t *)dstPtr, POST_UPDATE);
                vadd(index, index, add_offset, preg_b32, MODE_ZEROING);
            }
            vstar(align_index, (__ubuf__ uint32_t *)dstPtr);
            sprsts(SPR_AR, cdstPtr, i * sizeof(typename TileDataC::DType));
            sprclr(SPR_AR);
            dstPtr = dstPtr + TileDataD::Cols;
        }
    }
}

template <typename TileDataD, typename TileDataS, typename TileDataC, CmpMode cmpMode, uint32_t offset>
__tf__ AICORE void TGather_b32_eq(typename TileDataD::TileDType __out__ dst, typename TileDataS::TileDType __in__ src0,
                                  typename TileDataS::DType __in__ k_value, typename TileDataC::TileDType __in__ cdst,
                                  unsigned srcValidCol, unsigned srcValidRow, unsigned dstValidCol,
                                  unsigned dstValidRow)
{
    __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ typename TileDataS::DType *src0Ptr = (__ubuf__ typename TileDataS::DType *)__cce_get_tile_ptr(src0);
    __ubuf__ typename TileDataC::DType *cdstPtr = (__ubuf__ typename TileDataC::DType *)__cce_get_tile_ptr(cdst);

    using T = typename TileDataD::DType;
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataS::DType);
    uint16_t repeatTimes = CeilDivision(srcValidCol, elementsPerRepeat);

    __VEC_SCOPE__
    {
        vector_bool preg_b32 = pset_b32(PAT_ALL);
        vector_align align_index;
        vector_s32 index;
        vci(index, offset, INC_ORDER);
        vector_s32 add_offset;
        vbr(add_offset, 0x00000040);
        sprclr(SPR_AR);

        for (uint16_t i = 0; i < (uint16_t)srcValidRow; ++i) {
            for (uint16_t j = 0; j < repeatTimes; ++j) {
                RegTensor<typename TileDataS::DType> score;
                vlds(score, src0Ptr, (i * TileDataS::Cols + j * elementsPerRepeat), NORM);

                uint32_t count = ((j + 1) * elementsPerRepeat >= srcValidCol ? srcValidCol - j * elementsPerRepeat :
                                                                               elementsPerRepeat);
                vector_bool preg = CreatePredicate<typename TileDataS::DType>(count);

                vector_bool pout_eq;
                vector_s32 sqz_index_out;
                vcmps_eq(pout_eq, score, k_value, preg);
                vsqz(sqz_index_out, index, pout_eq, MODE_STORED);
                vadd(index, index, add_offset, preg_b32, MODE_ZEROING);
                vstur(align_index, (vector_u32)sqz_index_out, (__ubuf__ uint32_t *)dstPtr, POST_UPDATE);
            }
            vstar(align_index, (__ubuf__ uint32_t *)dstPtr);
            sprsts(SPR_AR, cdstPtr, i * sizeof(typename TileDataC::DType));
            sprclr(SPR_AR);
            dstPtr = dstPtr + TileDataD::Cols;
        }
    }
}

template <typename TileDataD, typename TileDataS, typename TileDataC, CmpMode cmpMode, uint32_t offset>
__tf__ AICORE void TGather_b16_gt(typename TileDataD::TileDType __out__ dst, typename TileDataS::TileDType __in__ src0,
                                  typename TileDataS::DType __in__ k_value, typename TileDataC::TileDType __in__ cdst,
                                  unsigned srcValidCol, unsigned srcValidRow, unsigned dstValidCol,
                                  unsigned dstValidRow)
{
    __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ typename TileDataS::DType *src0Ptr = (__ubuf__ typename TileDataS::DType *)__cce_get_tile_ptr(src0);
    __ubuf__ typename TileDataC::DType *cdstPtr = (__ubuf__ typename TileDataC::DType *)__cce_get_tile_ptr(cdst);

    unsigned dstStride = TileDataD::Cols;
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataD::DType);
    uint16_t repeatTimes = CeilDivision(srcValidCol, elementsPerRepeat);
    float k_value_f32 = (float)k_value;

    __VEC_SCOPE__
    {
        vector_bool preg_b32 = pset_b32(PAT_ALL);
        vector_bool preg_b16 = pset_b16(PAT_ALL);
        vector_s32 add_offset;
        vector_align align_index;
        vector_u32 mask_k;
        vector_s32 index;
        vbr(mask_k, k_value_f32);

        vci(index, offset, INC_ORDER);
        vbr(add_offset, 0x00000040);
        sprclr(SPR_AR);

        for (uint16_t i = 0; i < (uint16_t)srcValidRow; ++i) {
            for (uint16_t j = 0; j < repeatTimes; ++j) {
                vector_u16 score;
                vlds(score, (__ubuf__ uint16_t *)src0Ptr, (i * TileDataS::Cols + j * elementsPerRepeat), UNPK_B16);

                vector_bool pout_ge;
                vector_s32 sqz_index_out;
                vcmp_gt(pout_ge, (vector_u32)score, mask_k, preg_b32);
                vsqz(sqz_index_out, index, pout_ge, MODE_STORED);
                vstur(align_index, (vector_u32)sqz_index_out, (__ubuf__ uint32_t *)dstPtr, POST_UPDATE);
                vadd(index, index, add_offset, preg_b32, MODE_ZEROING);
            }
            vstar(align_index, (__ubuf__ uint32_t *)dstPtr);
            sprsts(SPR_AR, cdstPtr, i * sizeof(typename TileDataC::DType));
            sprclr(SPR_AR);
            dstPtr = dstPtr + dstStride;
        }
    }
}

template <typename TileDataD, typename TileDataS, typename TileDataC, CmpMode cmpMode, uint32_t offset>
__tf__ AICORE void TGather_b16_eq(typename TileDataD::TileDType __out__ dst, typename TileDataS::TileDType __in__ src0,
                                  typename TileDataS::DType __in__ k_value, typename TileDataC::TileDType __in__ cdst,
                                  unsigned srcValidCol, unsigned srcValidRow, unsigned dstValidCol,
                                  unsigned dstValidRow)
{
    __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ typename TileDataS::DType *src0Ptr = (__ubuf__ typename TileDataS::DType *)__cce_get_tile_ptr(src0);
    __ubuf__ typename TileDataC::DType *cdstPtr = (__ubuf__ typename TileDataC::DType *)__cce_get_tile_ptr(cdst);

    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataD::DType);
    uint16_t repeatTimes = CeilDivision(srcValidCol, elementsPerRepeat);
    float k_value_f32 = (float)k_value;

    __VEC_SCOPE__
    {
        vector_bool preg_b32 = pset_b32(PAT_ALL);
        vector_bool preg_b16 = pset_b16(PAT_ALL);
        vector_align align_index;
        vector_u32 mask_k;
        vbr(mask_k, k_value_f32);

        vector_s32 index;
        vci(index, offset, INC_ORDER);
        vector_s32 idx_offset;
        vbr(idx_offset, 0x00000040);
        sprclr(SPR_AR);

        for (uint16_t i = 0; i < (uint16_t)srcValidRow; ++i) {
            for (uint16_t j = 0; j < repeatTimes; ++j) {
                vector_u16 score;
                vlds(score, (__ubuf__ uint16_t *)src0Ptr, (i * TileDataS::Cols + j * elementsPerRepeat), UNPK_B16);

                vector_bool pout_eq;
                vector_s32 sqz_index_out;
                vcmp_eq(pout_eq, (vector_u32)score, mask_k, preg_b32);
                vsqz(sqz_index_out, index, pout_eq, MODE_STORED);
                vstur(align_index, (vector_u32)sqz_index_out, (__ubuf__ uint32_t *)dstPtr, POST_UPDATE);
                vadd(index, index, idx_offset, preg_b32, MODE_ZEROING);
            }
            vstar(align_index, (__ubuf__ uint32_t *)dstPtr);
            sprsts(SPR_AR, cdstPtr, i * sizeof(typename TileDataC::DType));
            sprclr(SPR_AR);
            dstPtr = dstPtr + TileDataD::Cols;
        }
    }
}

template <typename TileDataD, typename TileDataS, typename TileDataC, CmpMode cmpMode, uint32_t offset>
__tf__ AICORE void TGather_half_gt(typename TileDataD::TileDType __out__ dst, typename TileDataS::TileDType __in__ src0,
                                   typename TileDataS::DType __in__ k_value, typename TileDataC::TileDType __in__ cdst,
                                   unsigned srcValidCol, unsigned srcValidRow, unsigned dstValidCol,
                                   unsigned dstValidRow)
{
    __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ typename TileDataS::DType *src0Ptr = (__ubuf__ typename TileDataS::DType *)__cce_get_tile_ptr(src0);
    __ubuf__ typename TileDataC::DType *cdstPtr = (__ubuf__ typename TileDataC::DType *)__cce_get_tile_ptr(cdst);

    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataD::DType);
    uint16_t repeatTimes = CeilDivision(srcValidCol, elementsPerRepeat);
    unsigned srcRowStride = TileDataS::RowStride;
    unsigned dstRowStride = TileDataD::RowStride;
    float k_value_f32 = (float)k_value;

    __VEC_SCOPE__
    {
        vector_bool preg_b32 = pset_b32(PAT_ALL);
        vector_bool preg_b16 = pset_b16(PAT_ALL);
        vector_align align_index;
        vector_f32 mask_k;
        vector_s32 index;
        vbr(mask_k, k_value_f32);

        vci(index, offset, INC_ORDER);
        vector_s32 index_offset;
        vbr(index_offset, 0x00000040);
        sprclr(SPR_AR);

        for (uint16_t i = 0; i < (uint16_t)srcValidRow; ++i) {
            for (uint16_t j = 0; j < repeatTimes; ++j) {
                vector_f16 score;
                vector_f32 score_f32;
                vlds(score, (__ubuf__ half *)src0Ptr, (i * TileDataS::Cols + j * elementsPerRepeat), UNPK_B16);
                vcvt(score_f32, score, preg_b16, PART_EVEN);

                vector_bool pout_ge;
                vector_s32 sqz_index_out;

                vcmp_gt(pout_ge, score_f32, mask_k, preg_b32);
                vsqz(sqz_index_out, index, pout_ge, MODE_STORED);

                vstur(align_index, (vector_u32)sqz_index_out, (__ubuf__ uint32_t *)dstPtr, POST_UPDATE);
                vadd(index, index, index_offset, preg_b32, MODE_ZEROING);
            }
            vstar(align_index, (__ubuf__ uint32_t *)dstPtr);
            sprsts(SPR_AR, cdstPtr, i * sizeof(typename TileDataC::DType));
            sprclr(SPR_AR);
            dstPtr = dstPtr + TileDataD::Cols;
        }
    }
}

template <typename TileDataD, typename TileDataS, typename TileDataC, CmpMode cmpMode, uint32_t offset>
__tf__ AICORE void TGather_half_eq(typename TileDataD::TileDType __out__ dst, typename TileDataS::TileDType __in__ src0,
                                   typename TileDataS::DType __in__ k_value, typename TileDataC::TileDType __in__ cdst,
                                   unsigned srcValidCol, unsigned srcValidRow, unsigned dstValidCol,
                                   unsigned dstValidRow)
{
    __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ typename TileDataS::DType *src0Ptr = (__ubuf__ typename TileDataS::DType *)__cce_get_tile_ptr(src0);
    __ubuf__ typename TileDataC::DType *cdstPtr = (__ubuf__ typename TileDataC::DType *)__cce_get_tile_ptr(cdst);

    unsigned TShape1 = TileDataD::Cols;
    using T = typename TileDataD::DType;
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataD::DType);
    uint16_t repeatTimes = CeilDivision(srcValidCol, elementsPerRepeat);
    float k_value_f32 = (float)k_value;

    __VEC_SCOPE__
    {
        vector_bool preg_b32 = pset_b32(PAT_ALL);
        vector_bool preg_b32_half = pset_b32(PAT_H);
        vector_bool preg_b16 = pset_b16(PAT_ALL);
        vector_s32 add_offset;
        vector_align align_index;
        vector_f32 mask_k;
        vbr(mask_k, k_value_f32);

        vector_s32 index;
        vci(index, offset, INC_ORDER);
        vbr(add_offset, 0x00000040);
        sprclr(SPR_AR);

        for (uint16_t i = 0; i < (uint16_t)srcValidRow; ++i) {
            for (uint16_t j = 0; j < repeatTimes; ++j) {
                vector_f16 score;
                vlds(score, (__ubuf__ half *)src0Ptr, (i * TileDataS::Cols + j * elementsPerRepeat), UNPK_B16);
                vector_f32 score_f32;
                vcvt(score_f32, score, preg_b16, PART_EVEN);

                vector_bool pout_eq;
                vector_s32 sqz_index_out;

                vcmp_eq(pout_eq, score_f32, mask_k, preg_b32);
                vsqz(sqz_index_out, index, pout_eq, MODE_STORED);

                vstur(align_index, (vector_u32)sqz_index_out, (__ubuf__ uint32_t *)dstPtr, POST_UPDATE);
                vadd(index, index, add_offset, preg_b32, MODE_ZEROING);
            }
            vstar(align_index, (__ubuf__ uint32_t *)dstPtr);
            sprsts(SPR_AR, cdstPtr, i * sizeof(typename TileDataC::DType));
            sprclr(SPR_AR);
            dstPtr = dstPtr + TShape1;
        }
    }
}

template <typename TileDataD, typename TileDataS, typename TileDataC, CmpMode cmpMode, uint32_t offset>
AICORE void TGather_cmp(typename TileDataD::TileDType dst, typename TileDataS::TileDType src0,
                        typename TileDataC::TileDType cdst, typename TileDataS::DType k_value, unsigned srcValidCol,
                        unsigned srcValidRow, unsigned dstValidCol, unsigned dstValidRow)
{
    if constexpr (sizeof(typename TileDataS::DType) == 4 && cmpMode == CmpMode::GT) {
        TGather_b32_gt<TileDataD, TileDataS, TileDataC, cmpMode, offset>(dst, src0, k_value, cdst, srcValidCol,
                                                                         srcValidRow, dstValidCol, dstValidRow);
    } else if constexpr (sizeof(typename TileDataS::DType) == 4 && cmpMode == CmpMode::EQ) {
        TGather_b32_eq<TileDataD, TileDataS, TileDataC, cmpMode, offset>(dst, src0, k_value, cdst, srcValidCol,
                                                                         srcValidRow, dstValidCol, dstValidRow);
    } else if constexpr (std::is_same_v<typename TileDataS::DType, half> && cmpMode == CmpMode::GT) {
        TGather_half_gt<TileDataD, TileDataS, TileDataC, cmpMode, offset>(dst, src0, k_value, cdst, srcValidCol,
                                                                          srcValidRow, dstValidCol, dstValidRow);
    } else if constexpr (std::is_same_v<typename TileDataS::DType, half> && cmpMode == CmpMode::EQ) {
        TGather_half_eq<TileDataD, TileDataS, TileDataC, cmpMode, offset>(dst, src0, k_value, cdst, srcValidCol,
                                                                          srcValidRow, dstValidCol, dstValidRow);
    } else if constexpr (std::is_same_v<typename TileDataS::DType, int16_t> && cmpMode == CmpMode::GT) {
        TGather_b16_gt<TileDataD, TileDataS, TileDataC, cmpMode, offset>(dst, src0, k_value, cdst, srcValidCol,
                                                                         srcValidRow, dstValidCol, dstValidRow);
    } else {
        TGather_b16_eq<TileDataD, TileDataS, TileDataC, cmpMode, offset>(dst, src0, k_value, cdst, srcValidCol,
                                                                         srcValidRow, dstValidCol, dstValidRow);
    }
}

template <typename TileDataD, typename TileDataS, typename TileDataC, typename TileDataTmp, CmpMode cmpMode,
          uint32_t offset>
PTO_INTERNAL void TGATHER_IMPL(TileDataD &dst, TileDataS &src0, typename TileDataS::DType k_value, TileDataC &cdst,
                               TileDataTmp &tmp)
{
    static_assert(
        std::is_same_v<typename TileDataD::DType, uint32_t> || std::is_same_v<typename TileDataD::DType, int32_t>,
        "Fix: TGATHER Dst data type must be int32_t/uint32_t.");
    static_assert(
        std::is_same_v<typename TileDataS::DType, float> || std::is_same_v<typename TileDataS::DType, uint32_t> ||
            std::is_same_v<typename TileDataS::DType, int32_t> || std::is_same_v<typename TileDataS::DType, uint16_t> ||
            std::is_same_v<typename TileDataS::DType, int16_t> || std::is_same_v<typename TileDataS::DType, half>,
        "Fix: TGATHER Src data type must be int16_t/uint16_t/int32_t/uint32_t/half/float.");
    static_assert((cmpMode == CmpMode::GT || cmpMode == CmpMode::EQ), "Fix: TGATHER only support GT or EQ mode");
    static_assert((TileDataD::Loc == TileType::Vec) && (TileDataS::Loc == TileType::Vec),
                  "Fix: TGATHER expect vec TileType");

    unsigned sValidCols = src0.GetValidCol();
    unsigned sValidRows = src0.GetValidRow();
    unsigned dValidCols = dst.GetValidCol();
    unsigned dValidRows = dst.GetValidRow();

    TGather_cmp<TileDataD, TileDataS, TileDataC, cmpMode, offset>(dst.data(), src0.data(), cdst.data(), k_value,
                                                                  sValidCols, sValidRows, dValidCols, dValidRows);
}

} // namespace pto
#endif
