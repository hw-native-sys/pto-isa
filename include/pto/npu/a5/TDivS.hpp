/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TDIVS_HPP
#define TDIVS_HPP

#include <pto/common/constants.hpp>
#include "common.hpp"
#include "utils.hpp"
#include "TBinSOp.hpp"
#include "custom/Div754.hpp"

namespace pto {

template <DivAlgorithm PrecisionType, typename T>
struct DivSOp {
    static constexpr bool isDynFunc = true;
    RegTensor<T> scalarReg;

    PTO_INTERNAL DivSOp(T scalar)
    {
        vbr(scalarReg, scalar);
    };

    PTO_INTERNAL void BinSInstr(RegTensor<T> &dstReg, RegTensor<T> &srcReg, T scalar, MaskReg &pReg)
    {
        if constexpr (PrecisionType == DivAlgorithm::HIGH_PRECISION && std::is_same_v<T, float>) {
            DivIEEE754FloatImpl<T, RegTensor<T>>(dstReg, srcReg, scalarReg, pReg);
        } else if constexpr (PrecisionType == DivAlgorithm::HIGH_PRECISION && std::is_same_v<T, half>) {
            DivIEEE754HalfImpl<T, RegTensor<T>>(dstReg, srcReg, scalarReg, pReg);
        } else {
            vdiv(dstReg, srcReg, scalarReg, pReg, MODE_ZEROING);
        }
    }
};

template <enum DivAlgorithm PrecisionType, typename T>
struct DivSOpS {
    static constexpr bool isDynFunc = true;
    RegTensor<T> scalarReg;

    PTO_INTERNAL DivSOpS(T scalar)
    {
        vbr(scalarReg, scalar);
    };

    PTO_INTERNAL void BinSInstr(RegTensor<T> &dstReg, RegTensor<T> &srcReg, T scalar, MaskReg &pReg)
    {
        if constexpr (PrecisionType == DivAlgorithm::HIGH_PRECISION && std::is_same_v<T, float>) {
            DivIEEE754FloatImpl<T, RegTensor<T>>(dstReg, scalarReg, srcReg, pReg);
        } else if constexpr (PrecisionType == DivAlgorithm::HIGH_PRECISION && std::is_same_v<T, half>) {
            DivIEEE754HalfImpl<T, RegTensor<T>>(dstReg, scalarReg, srcReg, pReg);
        } else {
            vdiv(dstReg, scalarReg, srcReg, pReg);
        }
    }
};
template <typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void TDivs_naive(__ubuf__ T *dst, __ubuf__ T *src0, T src1, unsigned validRow, unsigned validCol)
{
// auto mode adds in synchronization during compilation
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_V, PIPE_S>();
#endif
    for (int i = 0; i < validRow; i++) {
        for (int j = 0; j < validCol; j++) {
            int dstOffset = i * DstCols + j;
            int srcOffset = i * SrcCols + j;
            dst[dstOffset] = src0[srcOffset] / src1;
        }
    }
}

template <typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void TSDiv_naive(__ubuf__ T *dst, __ubuf__ T *src0, T src1, unsigned validRow, unsigned validCol)
{
// auto mode adds in synchronization during compilation
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_V, PIPE_S>();
#endif
    for (int i = 0; i < validRow; i++) {
        for (int j = 0; j < validCol; j++) {
            int dstOffset = i * DstCols + j;
            int srcOffset = i * SrcCols + j;
            dst[dstOffset] = src1 / src0[srcOffset];
        }
    }
}
template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc,
          unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned dstRowStride, unsigned srcRowStride>
__tf__ PTO_INTERNAL OP_NAME(TDIVS)
    OP_TYPE(element_wise) void TDivS(typename TileDataDst::TileDType __out__ dst,
                                     typename TileDataSrc::TileDType __in__ src0, typename TileDataSrc::DType src1,
                                     unsigned validRow, unsigned validCol,
                                     VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    if constexpr (std::is_integral_v<T>) {
        TDivs_naive<T, TileDataDst::Cols, TileDataSrc::Cols>(dstPtr, src0Ptr, src1, validRow, validCol);
    } else {
        BinaryInstr<DivSOp<PrecisionType, T>, TileDataDst, TileDataSrc, T, elementsPerRepeat, blockSizeElem,
                    dstRowStride, srcRowStride>(dstPtr, src0Ptr, src1, validRow, validCol, version);
    }
}

template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc,
          unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned dstRowStride, unsigned srcRowStride>
__tf__ PTO_INTERNAL OP_NAME(TDIVS)
    OP_TYPE(element_wise) void TDivS(typename TileDataDst::TileDType __out__ dst, typename TileDataSrc::DType src1,
                                     typename TileDataSrc::TileDType __in__ src0, unsigned validRow, unsigned validCol,
                                     VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    if constexpr (std::is_integral_v<T>) {
        TSDiv_naive<T, TileDataDst::Cols, TileDataSrc::Cols>(dstPtr, src0Ptr, src1, validRow, validCol);
    } else {
        BinaryInstr<DivSOpS<PrecisionType, T>, TileDataDst, TileDataSrc, T, elementsPerRepeat, blockSizeElem,
                    dstRowStride, srcRowStride>(dstPtr, src0Ptr, src1, validRow, validCol, version);
    }
}

template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TDIVS_IMPL(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar)
{
    static_assert(std::is_same<typename TileDataDst::DType, uint32_t>::value ||
                      std::is_same<typename TileDataDst::DType, int32_t>::value ||
                      std::is_same<typename TileDataDst::DType, uint16_t>::value ||
                      std::is_same<typename TileDataDst::DType, int16_t>::value ||
                      std::is_same<typename TileDataDst::DType, half>::value ||
                      std::is_same<typename TileDataDst::DType, float>::value,
                  "TDIVS: Invalid data type");

    static_assert(TileDataSrc::Loc == TileType::Vec, "TileType of src and dst tiles must be TileType::Vec.");
    static_assert(TileDataDst::Loc == TileType::Vec, "TileType of src and dst tiles must be TileType::Vec.");
    static_assert(TileDataSrc::ValidCol <= TileDataSrc::Cols,
                  "Number of valid columns must not be greater than number of tile columns.");
    static_assert(TileDataSrc::ValidRow <= TileDataSrc::Rows,
                  "Number of valid rows must not be greater than number of tile rows.");
    static_assert(TileDataDst::ValidCol <= TileDataDst::Cols,
                  "Number of valid columns must not be greater than number of tile columns.");
    static_assert(TileDataDst::ValidRow <= TileDataDst::Rows,
                  "Number of valid rows must not be greater than number of tile rows.");

    PTO_ASSERT(src0.GetValidRow() == dst.GetValidRow(), "Number of rows of src and dst must be the same.");
    PTO_ASSERT(src0.GetValidCol() == dst.GetValidCol(), "Number of columns of src and dst must be the same.");

    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileDataDst::DType);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataDst::DType);
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr unsigned srcRowStride = TileDataSrc::RowStride;
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    TDivS<PrecisionType, TileDataDst, TileDataSrc, elementsPerRepeat, blockSizeElem, dstRowStride, srcRowStride>(
        dst.data(), src0.data(), scalar, validRow, validCol);
}

template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TDIVS_IMPL(TileDataDst &dst, typename TileDataSrc::DType scalar, TileDataSrc &src0)
{
    static_assert(TileDataSrc::Loc == TileType::Vec, "TileType of src and dst tiles must be TileType::Vec.");
    static_assert(TileDataDst::Loc == TileType::Vec, "TileType of src and dst tiles must be TileType::Vec.");
    static_assert(TileDataSrc::ValidCol <= TileDataSrc::Cols,
                  "Number of valid columns must not be greater than number of tile columns.");
    static_assert(TileDataSrc::ValidRow <= TileDataSrc::Rows,
                  "Number of valid rows must not be greater than number of tile rows.");
    static_assert(TileDataDst::ValidCol <= TileDataDst::Cols,
                  "Number of valid columns must not be greater than number of tile columns.");
    static_assert(TileDataDst::ValidRow <= TileDataDst::Rows,
                  "Number of valid rows must not be greater than number of tile rows.");

    PTO_ASSERT(src0.GetValidRow() == dst.GetValidRow(), "Number of rows of src and dst must be the same.");
    PTO_ASSERT(src0.GetValidCol() == dst.GetValidCol(), "Number of columns of src and dst must be the same.");

    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileDataDst::DType);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataDst::DType);
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr unsigned srcRowStride = TileDataSrc::RowStride;
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    TDivS<PrecisionType, TileDataDst, TileDataSrc, elementsPerRepeat, blockSizeElem, dstRowStride, srcRowStride>(
        dst.data(), scalar, src0.data(), validRow, validCol);
}
} // namespace pto
#endif
