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

namespace pto {
template <typename DstTileData, typename Src0TileData, typename Src1TileData, typename TmpTileData>
PTO_INTERNAL void CheckValid()
{
    static_assert((sizeof(typename DstTileData::DType) == 2) || (sizeof(typename DstTileData::DType) == 4),
                  "Fix: TGATHER expect b16/b32");
    static_assert((sizeof(typename Src1TileData::DType) == 4), "Fix: TGATHER expect b32");
    static_assert((std::is_same<typename DstTileData::DType, typename Src0TileData::DType>::value),
                  "Fix: TGATHER expect same datatype for src and dst");
    static_assert((std::is_same<typename TmpTileData::DType, typename Src1TileData::DType>::value),
                  "Fix: TGATHER expect same datatype for tmp buffer and indice");
}

template <typename TileDataD, typename TileDataS0, typename TileDataS1, typename TileDataTmp>
__tf__ AICORE void TGather(typename TileDataD::TileDType __out__ dst, typename TileDataS0::TileDType __in__ src0,
                           typename TileDataS1::TileDType __in__ src1, typename TileDataTmp::TileDType __in__ tmp,
                           unsigned validCol, unsigned validRow)
{
    __ubuf__ typename TileDataS0::DType *src0Ptr = (__ubuf__ typename TileDataS0::DType *)__cce_get_tile_ptr(src0);
    __ubuf__ typename TileDataS1::DType *src1Ptr = (__ubuf__ typename TileDataS1::DType *)__cce_get_tile_ptr(src1);
    __ubuf__ typename TileDataD::DType *dstPtr = (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ typename TileDataTmp::DType *tmpPtr = (__ubuf__ typename TileDataTmp::DType *)__cce_get_tile_ptr(tmp);

    unsigned TShape0 = TileDataD::Rows;
    unsigned TShape1 = TileDataD::Cols;
    unsigned idx_stride = TileDataS1::Cols;
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileDataD::DType);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataD::DType);
    constexpr unsigned stride = TileDataD::RowStride;
    unsigned numRepeatPerLine = validCol / elementsPerRepeat;
    unsigned numRemainPerLine = validCol % elementsPerRepeat;
    unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
    unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;

    if constexpr (std::is_same_v<typename TileDataS0::DType, float> ||
                  std::is_same_v<typename TileDataS0::DType, int32_t> ||
                  std::is_same_v<typename TileDataS0::DType, uint32_t>) {
        // 64 element per VL
        // counter mode
        set_mask_count();
        for (int i = 0; i < validRow; i++) {
            set_vector_mask(0, validCol);
            vmuls((__ubuf__ int32_t *)(dstPtr + i * TShape1), (__ubuf__ int32_t *)(src1Ptr + i * idx_stride),
                  sizeof(typename TileDataD::DType), 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vgather((__ubuf__ uint32_t *)(dstPtr + i * TShape1), (__ubuf__ uint32_t *)(dstPtr + i * TShape1),
                    (uintptr_t)src0Ptr, 8, 1);
        }

        set_mask_norm();
        set_vector_mask(-1, -1);
    } else {
        // 128 element per VL
        // counter mode
        set_mask_count();
        for (int i = 0; i < validRow; i++) {
            set_vector_mask(0, validCol);
            vmuls((__ubuf__ int32_t *)(tmpPtr + i * TShape1), (__ubuf__ int32_t *)(src1Ptr + i * idx_stride),
                  sizeof(typename TileDataD::DType), 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vgather((__ubuf__ uint16_t *)(dstPtr + i * TShape1), (__ubuf__ uint32_t *)(tmpPtr + i * TShape1),
                    (uintptr_t)src0Ptr, 8, 1);
        }
        set_mask_norm();
        set_vector_mask(-1, -1);
    }
}

template <typename TileDataD, typename TileDataS0, typename TileDataS1, typename TileDataTmp>
PTO_INTERNAL void TGATHER_IMPL(TileDataD &dst, TileDataS0 &src0, TileDataS1 &src1, TileDataTmp &tmp)
{
    CheckValid<TileDataD, TileDataS0, TileDataS1, TileDataTmp>();
    PTO_ASSERT(src1.GetValidCol() == tmp.GetValidCol(), "Fix: TGATHER expect same cols for indice and tmp buffer.");
    PTO_ASSERT(src1.GetValidRow() == tmp.GetValidRow(), "Fix: TGATHER expect same rows for indice and tmp buffer.");

    unsigned validCol = dst.GetValidCol();
    unsigned validRow = dst.GetValidRow();

    TGather<TileDataD, TileDataS0, TileDataS1, TileDataTmp>(dst.data(), src0.data(), src1.data(), tmp.data(), validCol,
                                                            validRow);
}

template <typename DstTileData, typename SrcTileData, MaskPattern maskPattern>
__tf__ AICORE void TGather(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src,
                           unsigned validRow, unsigned validCol)
{
    using T = typename SrcTileData::DType;
    using U = std::conditional_t<sizeof(T) == sizeof(uint32_t), uint32_t, uint16_t>;

    constexpr unsigned srcRepeatStride = SrcTileData::Cols * sizeof(T) / BLOCK_BYTE_SIZE;

    __ubuf__ typename DstTileData::DType *dstPtr = (__ubuf__ typename DstTileData::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ typename SrcTileData::DType *srcPtr = (__ubuf__ typename SrcTileData::DType *)__cce_get_tile_ptr(src);

    set_mask_count();
    set_vector_mask(0, validCol);
    vreducev2(reinterpret_cast<__ubuf__ U *>(dstPtr), reinterpret_cast<__ubuf__ U *>(srcPtr),
              reinterpret_cast<__ubuf__ U *>(srcPtr), validRow, 1, maskPattern, srcRepeatStride, 0);
    set_mask_norm();
    set_vector_mask(-1, -1);
}

template <typename DstTileData, typename SrcTileData, MaskPattern maskPattern>
PTO_INTERNAL void TGATHER_IMPL(DstTileData &dst, SrcTileData &src)
{
    using T = typename SrcTileData::DType;
    static_assert(sizeof(T) == 2 || sizeof(T) == 4, "Fix: TGATHER src element type must be 16 or 32-bit wide");
    static_assert((DstTileData::Loc == TileType::Vec) && (SrcTileData::Loc == TileType::Vec),
                  "Fix: TGATHER expect vec TileType");
    static_assert((DstTileData::isRowMajor && SrcTileData::isRowMajor), "Fix: TGATHER expect row major");
    static_assert((sizeof(typename DstTileData::DType) == sizeof(T)),
                  "Fix: TGATHER expect same type size for dst and src");
    PTO_ASSERT(dst.GetValidCol() == DstTileData::Cols, "Fix: TGATHER expect continuous memory for dst.");
    TGather<DstTileData, SrcTileData, maskPattern>(dst.data(), src.data(), src.GetValidRow(), src.GetValidCol());
}

template <typename TileDataD, typename TileDataS, typename TileDataS1>
AICORE void TGather_cmp_FillIndexAndConvertK(__ubuf__ typename TileDataD::DType *indexTmpPtr, uint32_t offset,
                                             __ubuf__ typename TileDataS::DType *cvtTmpPtr,
                                             __ubuf__ typename TileDataS1::DType *kPtr, unsigned srcValidRow)
{
    using TIN = typename TileDataS::DType;

    for (int i = 0; i < TileDataS::Rows * TileDataS::Cols; i++) {
        *(indexTmpPtr + i) = offset + i;
    }
    PtoSetWaitFlag<PIPE_S, PIPE_V>();

    unsigned numRepeat = CeilDivision(srcValidRow, (REPEAT_BYTE / sizeof(typename TileDataS1::DType)));
    if constexpr (std::is_same_v<TIN, float>) {
        vconv_s322f32(cvtTmpPtr, reinterpret_cast<__ubuf__ int32_t *>(kPtr), numRepeat, 1, 1, 8, 8);
    } else if constexpr (std::is_same_v<TIN, int32_t>) {
        for (int i = 0; i < TileDataS::Rows; i++) {
            *(cvtTmpPtr + i) = *(kPtr + i);
        }
    } else if constexpr (std::is_same_v<TIN, half>) {
        vconv_s162f16(cvtTmpPtr, reinterpret_cast<__ubuf__ int16_t *>(kPtr), numRepeat, 1, 1, 8, 8);
    }
}

template <typename TileDataS, typename TileDataTmp, CmpMode cmpMode>
AICORE void TGather_cmp_RunTcmps(__ubuf__ typename TileDataS::DType *cvtTmpPtr,
                                 __ubuf__ typename TileDataS::DType *src0Ptr, __ubuf__ uint8_t *cmpsTmpPtr,
                                 unsigned srcValidCol, unsigned srcValidRow)
{
    using TIN = typename TileDataS::DType;

    unsigned numRepeatPerLine = CeilDivision(srcValidCol, (REPEAT_BYTE / sizeof(TIN)));
    size_t nLoop = numRepeatPerLine / TCMPS_REPEAT_MAX;
    int remainPerLine = numRepeatPerLine % TCMPS_REPEAT_MAX;
    constexpr int srcAlignCols = TileDataS::Cols;
    constexpr int dstAlignCols = TileDataTmp::Cols;
    constexpr int srcOffset = TCMPS_REPEAT_MAX * REPEAT_BYTE / sizeof(TIN);
    constexpr int dstOffset = TCMPS_REPEAT_MAX * REPEAT_BYTE / sizeof(TIN) / NUM_BITS_IN_BYTE;
    set_mask_norm();
    set_vector_mask(-1, -1);
    for (size_t i = 0; i < srcValidRow; i++) {
        PtoSetWaitFlag<PIPE_V, PIPE_S>();
        TIN k_scalar = *(cvtTmpPtr + i);
        PtoSetWaitFlag<PIPE_S, PIPE_V>();
        for (size_t j = 0; j < nLoop; j++) {
            if constexpr (cmpMode == CmpMode::GT) {
                vcmpvs_gt(cmpsTmpPtr + i * dstAlignCols + j * dstOffset, src0Ptr + i * srcAlignCols + j * srcOffset,
                          k_scalar, TCMPS_REPEAT_MAX, 1, 1, 8, 8);
            } else {
                vcmpvs_eq(cmpsTmpPtr + i * dstAlignCols + j * dstOffset, src0Ptr + i * srcAlignCols + j * srcOffset,
                          k_scalar, TCMPS_REPEAT_MAX, 1, 1, 8, 8);
            }
        }
    }
    if (remainPerLine) {
        for (size_t i = 0; i < srcValidRow; i++) {
            PtoSetWaitFlag<PIPE_V, PIPE_S>();
            TIN k_scalar = *(cvtTmpPtr + i);
            PtoSetWaitFlag<PIPE_S, PIPE_V>();
            if constexpr (cmpMode == CmpMode::GT) {
                vcmpvs_gt(cmpsTmpPtr + i * dstAlignCols + nLoop * dstOffset,
                          src0Ptr + i * srcAlignCols + nLoop * srcOffset, k_scalar, remainPerLine, 1, 1, 8, 8);
            } else {
                vcmpvs_eq(cmpsTmpPtr + i * dstAlignCols + nLoop * dstOffset,
                          src0Ptr + i * srcAlignCols + nLoop * srcOffset, k_scalar, remainPerLine, 1, 1, 8, 8);
            }
        }
    }
}

template <typename TileDataD, typename TileDataS, typename TileDataS1, typename TileDataTmp, CmpMode cmpMode>
AICORE void TGather_cmp_InitIndicesKAndTcmps(__ubuf__ typename TileDataD::DType *indexTmpPtr, uint32_t offset,
                                             __ubuf__ typename TileDataS::DType *cvtTmpPtr,
                                             __ubuf__ typename TileDataS1::DType *kPtr,
                                             __ubuf__ typename TileDataS::DType *src0Ptr, __ubuf__ uint8_t *cmpsTmpPtr,
                                             unsigned srcValidCol, unsigned srcValidRow)
{
    TGather_cmp_FillIndexAndConvertK<TileDataD, TileDataS, TileDataS1>(indexTmpPtr, offset, cvtTmpPtr, kPtr,
                                                                       srcValidRow);
    TGather_cmp_RunTcmps<TileDataS, TileDataTmp, cmpMode>(cvtTmpPtr, src0Ptr, cmpsTmpPtr, srcValidCol, srcValidRow);
}

template <typename TileDataD, typename TileDataS, typename TileDataC, typename TileDataTmp>
AICORE void TGather_cmp_ReduceGatherAndCounts(__ubuf__ typename TileDataD::DType *dstPtr,
                                              __ubuf__ typename TileDataD::DType *indexTmpPtr,
                                              __ubuf__ uint8_t *cmpsTmpPtr, __ubuf__ typename TileDataC::DType *cdstPtr,
                                              unsigned srcValidRow, unsigned srcValidCol)
{
    using T = typename TileDataD::DType;

    pipe_barrier(PIPE_V);

    set_mask_count();
    set_vector_mask(0, srcValidCol);
    for (int i = 0; i < srcValidRow; i++) {
        vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(dstPtr + i * TileDataD::RowStride),
                  reinterpret_cast<__ubuf__ uint32_t *>(indexTmpPtr + i * TileDataS::RowStride),
                  reinterpret_cast<__ubuf__ uint32_t *>(cmpsTmpPtr + i * TileDataTmp::RowStride), 1, 1, 0, 8, 0);
        *(cdstPtr + i) = (T)get_rsvd_cnt();
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
}

template <typename TileDataD, typename TileDataS, typename TileDataS1, typename TileDataC, typename TileDataTmp,
          CmpMode cmpMode>
__tf__ AICORE void TGather_cmp(typename TileDataD::TileDType __out__ dst, typename TileDataS::TileDType __out__ src0,
                               typename TileDataC::TileDType __in__ cdst, typename TileDataTmp::TileDType __in__ tmp,
                               typename TileDataS1::TileDType __in__ k_value, uint32_t offset, unsigned srcValidCol,
                               unsigned srcValidRow)
{
    using T = typename TileDataD::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ typename TileDataS::DType *src0Ptr = (__ubuf__ typename TileDataS::DType *)__cce_get_tile_ptr(src0);
    __ubuf__ typename TileDataC::DType *cdstPtr = (__ubuf__ typename TileDataC::DType *)__cce_get_tile_ptr(cdst);
    __ubuf__ uint8_t *cmpsTmpPtr = (__ubuf__ uint8_t *)__cce_get_tile_ptr(tmp);
    uint32_t indexTmpAddr = static_cast<uint32_t>(reinterpret_cast<int64_t>(cmpsTmpPtr)) +
                            TileDataTmp::Rows * TileDataTmp::Cols * sizeof(uint8_t);
    __ubuf__ T *indexTmpPtr = (__ubuf__ T *)get_imm(indexTmpAddr);
    uint32_t cvtTmpAddr = static_cast<uint32_t>(reinterpret_cast<int64_t>(cmpsTmpPtr)) +
                          TileDataTmp::Rows * TileDataTmp::Cols * sizeof(uint8_t) +
                          TileDataS::Rows * TileDataS::Cols * sizeof(T);

    using TIN = typename TileDataS::DType;
    using TS = typename TileDataS1::DType;
    __ubuf__ TS *kPtr = (__ubuf__ TS *)__cce_get_tile_ptr(k_value);
    __ubuf__ TIN *cvtTmpPtr = (__ubuf__ TIN *)get_imm(cvtTmpAddr);

    TGather_cmp_InitIndicesKAndTcmps<TileDataD, TileDataS, TileDataS1, TileDataTmp, cmpMode>(
        indexTmpPtr, offset, cvtTmpPtr, kPtr, src0Ptr, cmpsTmpPtr, srcValidCol, srcValidRow);

    TGather_cmp_ReduceGatherAndCounts<TileDataD, TileDataS, TileDataC, TileDataTmp>(dstPtr, indexTmpPtr, cmpsTmpPtr,
                                                                                    cdstPtr, srcValidRow, srcValidCol);
}

template <typename TileDataD, typename TileDataS, typename TileDataS1, typename TileDataC, typename TileDataTmp,
          CmpMode cmpMode>
PTO_INTERNAL void TGATHER_IMPL(TileDataD &dst, TileDataS &src0, TileDataS1 &k_value, TileDataC &cdst, TileDataTmp &tmp,
                               uint32_t offset)
{
    static_assert(
        std::is_same_v<typename TileDataD::DType, uint32_t> || std::is_same_v<typename TileDataD::DType, int32_t>,
        "Fix: TGATHER Dst data type must be int32_t/uint32_t.");
    static_assert(std::is_same_v<typename TileDataS::DType, float> || std::is_same_v<typename TileDataS::DType, half> ||
                      (std::is_same_v<typename TileDataS::DType, int32_t> && cmpMode == CmpMode::EQ),
                  "Fix: TGATHER Src data type must be half/float or int32_t while cmpMode is EQ.");
    static_assert((cmpMode == CmpMode::GT || cmpMode == CmpMode::EQ), "Fix: TGATHER only support GT or EQ mode");
    static_assert((TileDataD::Loc == TileType::Vec) && (TileDataS::Loc == TileType::Vec),
                  "Fix: TGATHER expect vec TileType");

    unsigned sValidCols = src0.GetValidCol();
    unsigned sValidRows = src0.GetValidRow();

    TGather_cmp<TileDataD, TileDataS, TileDataS1, TileDataC, TileDataTmp, cmpMode>(
        dst.data(), src0.data(), cdst.data(), tmp.data(), k_value.data(), offset, sValidCols, sValidRows);
}
} // namespace pto
#endif
