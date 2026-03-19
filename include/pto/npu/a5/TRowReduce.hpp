/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

#ifndef __ROW_REDUCE__
#define __ROW_REDUCE__

#include "common.hpp"
#include "pto/common/pto_tile.hpp"
#include <math.h>
#include <type_traits>

namespace pto {

// ==========================================
// FloatLimits: get max value for float types
// ==========================================

template <typename T>
struct FloatLimits;

template <>
struct FloatLimits<float> {
    static constexpr float max()
    {
        return __FLT_MAX__;
    }
};

template <>
struct FloatLimits<half> {
    static constexpr half max()
    {
        return static_cast<half>(65504.0f);
    }
};

// IntegerLimits: get max/min values for integer types
template <typename T>
struct IntegerLimits;

template <>
struct IntegerLimits<int32_t> {
    static constexpr int32_t max()
    {
        return 2147483647;
    }
    static constexpr int32_t min()
    {
        return -2147483648;
    }
};

template <>
struct IntegerLimits<int16_t> {
    static constexpr int16_t max()
    {
        return 32767;
    }
    static constexpr int16_t min()
    {
        return -32768;
    }
};

template <typename T>
struct FloatLimits {
    static_assert(sizeof(T) == 0,
                  "FloatLimits<T> is not specialized for this type. "
                  "Supported types: float, half.");
};

template <typename T>
struct ROWSUM {
    static constexpr T InitVal = 0;
    using RegType = typename TypeGet<T>::T;
    static PTO_INTERNAL void Accumulate(RegType &dst, RegType &src0, RegType &src1, MaskReg &pred)
    {
        vadd(dst, src0, src1, pred, MODE_ZEROING);
    }
    static PTO_INTERNAL void Reduce(RegType &dst, RegType &src, MaskReg &pred)
    {
        vcadd(dst, src, pred, MODE_ZEROING);
    }
};

// Special handling for int16 TROWSUM: vcadd outputs int32, so we need to use int32 registers and truncate
template <>
struct ROWSUM<int16_t> {
    static constexpr int16_t InitVal = 0;
    using RegType = typename TypeGet<int32_t>::T;    // Use int32 registers for accumulation
    using SrcRegType = typename TypeGet<int16_t>::T; // Input type is int16
    static PTO_INTERNAL void Accumulate(RegType &dst, RegType &src0, RegType &src1, MaskReg &pred)
    {
        vadd(dst, src0, src1, pred, MODE_ZEROING);
    }
    static PTO_INTERNAL void Reduce(RegType &dst, SrcRegType &src, MaskReg &pred)
    {
        vcadd(dst, src, pred, MODE_ZEROING);
    }
};

template <typename T>
struct ROWMAX {
    static constexpr T InitVal = -FloatLimits<T>::max();
    using RegType = typename TypeGet<T>::T;
    static PTO_INTERNAL void Accumulate(RegType &dst, RegType &src0, RegType &src1, MaskReg &pred)
    {
        vmax(dst, src0, src1, pred, MODE_ZEROING);
    }
    static PTO_INTERNAL void Reduce(RegType &dst, RegType &src, MaskReg &pred)
    {
        vcmax(dst, src, pred, MODE_ZEROING);
    }
};

// Specialization for int32
template <>
struct ROWMAX<int32_t> {
    static constexpr int32_t InitVal = -2147483648;
    using RegType = typename TypeGet<int32_t>::T;
    static PTO_INTERNAL void Accumulate(RegType &dst, RegType &src0, RegType &src1, MaskReg &pred)
    {
        vmax(dst, src0, src1, pred, MODE_ZEROING);
    }
    static PTO_INTERNAL void Reduce(RegType &dst, RegType &src, MaskReg &pred)
    {
        vcmax(dst, src, pred, MODE_ZEROING);
    }
};

// Specialization for int16
template <>
struct ROWMAX<int16_t> {
    static constexpr int16_t InitVal = -32768;
    using RegType = typename TypeGet<int16_t>::T;
    static PTO_INTERNAL void Accumulate(RegType &dst, RegType &src0, RegType &src1, MaskReg &pred)
    {
        vmax(dst, src0, src1, pred, MODE_ZEROING);
    }
    static PTO_INTERNAL void Reduce(RegType &dst, RegType &src, MaskReg &pred)
    {
        vcmax(dst, src, pred, MODE_ZEROING);
    }
};

template <typename T>
struct ROWMIN {
    static constexpr T InitVal = FloatLimits<T>::max();
    using RegType = typename TypeGet<T>::T;
    static PTO_INTERNAL void Accumulate(RegType &dst, RegType &src0, RegType &src1, MaskReg &pred)
    {
        vmin(dst, src0, src1, pred, MODE_ZEROING);
    }
    static PTO_INTERNAL void Reduce(RegType &dst, RegType &src, MaskReg &pred)
    {
        vcmin(dst, src, pred, MODE_ZEROING);
    }
};

// Specialization for int32
template <>
struct ROWMIN<int32_t> {
    static constexpr int32_t InitVal = 2147483647;
    using RegType = typename TypeGet<int32_t>::T;
    static PTO_INTERNAL void Accumulate(RegType &dst, RegType &src0, RegType &src1, MaskReg &pred)
    {
        vmin(dst, src0, src1, pred, MODE_ZEROING);
    }
    static PTO_INTERNAL void Reduce(RegType &dst, RegType &src, MaskReg &pred)
    {
        vcmin(dst, src, pred, MODE_ZEROING);
    }
};

// Specialization for int16
template <>
struct ROWMIN<int16_t> {
    static constexpr int16_t InitVal = 32767;
    using RegType = typename TypeGet<int16_t>::T;
    static PTO_INTERNAL void Accumulate(RegType &dst, RegType &src0, RegType &src1, MaskReg &pred)
    {
        vmin(dst, src0, src1, pred, MODE_ZEROING);
    }
    static PTO_INTERNAL void Reduce(RegType &dst, RegType &src, MaskReg &pred)
    {
        vcmin(dst, src, pred, MODE_ZEROING);
    }
};

template <typename TileDataOut, typename TileDataIn>
PTO_INTERNAL void TRowReduceCheck(uint32_t srcValidRows, uint32_t srcValidCols, uint32_t dstValidRow)
{
    using T = typename TileDataIn::DType;
    static_assert(
        std::is_same_v<T, half> || std::is_same_v<T, float> || std::is_same_v<T, int32_t> || std::is_same_v<T, int16_t>,
        "Row reduction only supports 'half', 'float', 'int32', or 'int16' data types. "
        "Fix: Define TileDataIn with DType = half, float, int32, or int16.");
    static_assert(std::is_same_v<T, typename TileDataOut::DType>,
                  "Input and output tile data types must match. "
                  "Fix: Ensure TileDataOut uses the same DType as TileDataIn.");
    static_assert(TileDataOut::Loc == pto::TileType::Vec && TileDataIn::Loc == pto::TileType::Vec,
                  "Row reduction only works on vector tiles (TileType::Vec). "
                  "Fix: Instantiate TileDataIn and TileDataOut with Loc_ = TileType::Vec.");
    static_assert(TileDataIn::isRowMajor && !TileDataIn::isBoxedLayout,
                  "Input tile must use standard ND layout (row-major, non-fractal). "
                  "Fix: Define TileDataIn with BFractal_ = BLayout::RowMajor and SFractal_ "
                  "= SLayout::NoneBox, e.g.,\n"
                  "     Tile<TileType::Vec, T, ROWS, COLS, BLayout::RowMajor, ..., "
                  "SLayout::NoneBox>");
    static_assert((!TileDataOut::isBoxedLayout &&
                   (TileDataOut::isRowMajor || (!TileDataOut::isRowMajor && TileDataOut::Cols == 1))),
                  "Output tile layout must be either:\n"
                  "  (a) ND layout: BLayout::RowMajor + SLayout::NoneBox, OR\n"
                  "  (b) DN layout with exactly one column: BLayout::ColMajor + "
                  "SLayout::NoneBox + Cols=1.\n"
                  "Fix: Choose one of the following for TileDataOut:\n"
                  "     - Tile<..., ROWS, COLS, BLayout::RowMajor, ValidRows, 1>   // ND\n"
                  "     - Tile<..., ROWS, 1, BLayout::ColMajor, ValidRows, 1>  // DN with Cols=1");
    // runtime checks
    PTO_ASSERT(srcValidRows != 0 && srcValidCols != 0,
               "Source valid rows or columns is zero — row reduction requires at "
               "least one element per row. "
               "Fix: Ensure srcValidRows > 0 and srcValidCols > 0.");
    PTO_ASSERT(srcValidRows == dstValidRow,
               "Input and output valid row counts must be equal in row reduction "
               "(row count is preserved). "
               "Fix: Pass dstValidRow = srcValidRows.");
}

template <typename ReduceOp, typename TileDataOut, typename TileDataIn, unsigned elementsPerRepeat>
PTO_INTERNAL void TRowReduceImpl(__ubuf__ typename TileDataOut::DType *dstPtr,
                                 __ubuf__ typename TileDataOut::DType *srcPtr, uint32_t rows, uint32_t cols,
                                 unsigned version)
{
    using TIN = typename TileDataIn::DType;
    using TAccum = typename ReduceOp::RegType; // Use the accumulator type from the reduce op
    uint16_t repeatTimes = CeilDivision(cols, elementsPerRepeat);
    __VEC_SCOPE__
    {
        RegTensor<TIN> vreg0;
        RegTensor<TAccum> vreg1;
        RegTensor<TAccum> vregdst;
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<TIN, DistVST::DIST_ONEPT>())>();
        uint32_t destItems = 1;
        MaskReg pregdst = CreatePredicate<TIN>(destItems);
        if (version == VFIMPL_2D_NO_POST_UPDATE) {
            for (uint16_t i = 0; i < (uint16_t)rows; ++i) {
                vbr(vregdst, ReduceOp::InitVal);
                uint32_t sreg = cols;
                for (uint16_t j = 0; j < (uint16_t)repeatTimes; j++) {
                    MaskReg preg = CreatePredicate<TIN>(sreg);
                    vlds(vreg0, srcPtr, i * TileDataIn::RowStride + j * elementsPerRepeat, NORM);
                    ReduceOp::Reduce(vreg1, vreg0, preg);
                    ReduceOp::Accumulate(vregdst, vregdst, vreg1, pregdst);
                }
                // For int16 TROWSUM, we need to truncate from int32 to int16
                if constexpr (std::is_same_v<TIN, int16_t> && std::is_same_v<TAccum, typename TypeGet<int32_t>::T>) {
                    RegTensor<TIN> vregdst_trunc;
                    vcvt(vregdst_trunc.reg, vregdst.reg, pregdst, RS_ENABLE, PART_EVEN);
                    vcvt(vregdst_trunc.reg, vregdst.reg, pregdst, RS_ENABLE, PART_ODD);
                    vsts(vregdst_trunc, dstPtr, i * TileDataOut::RowStride, distValue, pregdst);
                } else {
                    vsts(vregdst, dstPtr, i * TileDataOut::RowStride, distValue, pregdst);
                }
            }
        } else {
            for (uint16_t i = 0; i < (uint16_t)rows; ++i) {
                vbr(vregdst, ReduceOp::InitVal);
                __ubuf__ TIN *row_ptr = srcPtr + i * TileDataIn::RowStride;
                uint32_t sreg = cols;
                for (uint16_t j = 0; j < (uint16_t)repeatTimes; j++) {
                    MaskReg preg = CreatePredicate<TIN>(sreg);
                    vlds(vreg0, row_ptr, elementsPerRepeat, NORM, POST_UPDATE);
                    ReduceOp::Reduce(vreg1, vreg0, preg);
                    ReduceOp::Accumulate(vregdst, vregdst, vreg1, pregdst);
                }
                // For int16 TROWSUM, we need to truncate from int32 to int16
                if constexpr (std::is_same_v<TIN, int16_t> && std::is_same_v<TAccum, typename TypeGet<int32_t>::T>) {
                    RegTensor<TIN> vregdst_trunc;
                    vcvt(vregdst_trunc.reg, vregdst.reg, pregdst, RS_ENABLE, PART_EVEN);
                    vcvt(vregdst_trunc.reg, vregdst.reg, pregdst, RS_ENABLE, PART_ODD);
                    vsts(vregdst_trunc, dstPtr, TileDataOut::RowStride, distValue, pregdst, POST_UPDATE);
                } else {
                    vsts(vregdst, dstPtr, TileDataOut::RowStride, distValue, pregdst, POST_UPDATE);
                }
            }
        }
    } // end VF
}

template <typename TileDataOut, typename TileDataIn, unsigned elementsPerRepeat>
__tf__ PTO_INTERNAL OP_NAME(TROWMAX)
    OP_TYPE(reduce) void TRowMax(typename TileDataOut::TileDType __out__ dst, typename TileDataIn::TileDType __in__ src,
                                 uint32_t srcValidRows, uint32_t srcValidCols, uint32_t dstValidRow,
                                 unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    TRowReduceCheck<TileDataOut, TileDataIn>(srcValidRows, srcValidCols, dstValidRow);

    using TIN = typename TileDataIn::DType;
    __ubuf__ TIN *dstPtr = (__ubuf__ TIN *)__cce_get_tile_ptr(dst);
    __ubuf__ TIN *srcPtr = (__ubuf__ TIN *)__cce_get_tile_ptr(src);

    using rowReduceOp = ROWMAX<typename TileDataIn::DType>;
    TRowReduceImpl<rowReduceOp, TileDataOut, TileDataIn, elementsPerRepeat>(dstPtr, srcPtr, srcValidRows, srcValidCols,
                                                                            version);
}

template <typename TileDataOut, typename TileDataIn, unsigned elementsPerRepeat>
__tf__ PTO_INTERNAL OP_NAME(TROWSUM)
    OP_TYPE(reduce) void TRowSum(typename TileDataOut::TileDType __out__ dst, typename TileDataIn::TileDType __in__ src,
                                 uint32_t srcValidRows, uint32_t srcValidCols, uint32_t dstValidRow,
                                 unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    TRowReduceCheck<TileDataOut, TileDataIn>(srcValidRows, srcValidCols, dstValidRow);

    using TIN = typename TileDataIn::DType;
    __ubuf__ TIN *dstPtr = (__ubuf__ TIN *)__cce_get_tile_ptr(dst);
    __ubuf__ TIN *srcPtr = (__ubuf__ TIN *)__cce_get_tile_ptr(src);

    using rowReduceOp = ROWSUM<typename TileDataIn::DType>;
    TRowReduceImpl<rowReduceOp, TileDataOut, TileDataIn, elementsPerRepeat>(dstPtr, srcPtr, srcValidRows, srcValidCols,
                                                                            version);
}

template <typename TileDataOut, typename TileDataIn, unsigned elementsPerRepeat>
__tf__ PTO_INTERNAL OP_NAME(TROWMIN)
    OP_TYPE(reduce) void TRowMin(typename TileDataOut::TileDType __out__ dst, typename TileDataIn::TileDType __in__ src,
                                 uint32_t srcValidRows, uint32_t srcValidCols, uint32_t dstValidRow,
                                 unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    TRowReduceCheck<TileDataOut, TileDataIn>(srcValidRows, srcValidCols, dstValidRow);

    using TIN = typename TileDataIn::DType;
    __ubuf__ TIN *dstPtr = (__ubuf__ TIN *)__cce_get_tile_ptr(dst);
    __ubuf__ TIN *srcPtr = (__ubuf__ TIN *)__cce_get_tile_ptr(src);

    using rowReduceOp = ROWMIN<typename TileDataIn::DType>;
    TRowReduceImpl<rowReduceOp, TileDataOut, TileDataIn, elementsPerRepeat>(dstPtr, srcPtr, srcValidRows, srcValidCols,
                                                                            version);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWMAX_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    using T = typename TileDataIn::DType;
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    unsigned rows = src.GetValidRow();
    unsigned cols = src.GetValidCol();

    TRowMax<TileDataOut, TileDataIn, elementsPerRepeat>(dst.data(), src.data(), rows, cols, dst.GetValidRow());
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWSUM_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    using T = typename TileDataIn::DType;
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    unsigned rows = src.GetValidRow();
    unsigned cols = src.GetValidCol();

    TRowSum<TileDataOut, TileDataIn, elementsPerRepeat>(dst.data(), src.data(), rows, cols, dst.GetValidRow());
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWMIN_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    using T = typename TileDataIn::DType;
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    unsigned rows = src.GetValidRow();
    unsigned cols = src.GetValidCol();

    TRowMin<TileDataOut, TileDataIn, elementsPerRepeat>(dst.data(), src.data(), rows, cols, dst.GetValidRow());
}
} // namespace pto

#endif