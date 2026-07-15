/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef T_COL_REDUCE_IDX_OPS_HPP
#define T_COL_REDUCE_IDX_OPS_HPP

#include <pto/common/utils.hpp>
#include <pto/common/type.hpp>

namespace pto {

template <
    typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, typename TileDataTmp, bool WithVal = false>
PTO_INTERNAL void TColReduceIdxCheck(
    unsigned srcValidRow, unsigned srcValidCol, unsigned dstValidRow, unsigned dstValidCol, unsigned dstValValidRow = 0,
    unsigned dstValValidCol = 0)
{
    // 输入数据类型检查
    static_assert(
        std::is_same_v<typename TileDataIn::DType, uint32_t> || std::is_same_v<typename TileDataIn::DType, uint16_t> ||
            std::is_same_v<typename TileDataIn::DType, half> || std::is_same_v<typename TileDataIn::DType, float>,
        "Fix: TColReduceIdx input data type must be f16/u16/f32/u32");

    // 输入Tile类型检查
    static_assert(TileDataIn::Loc == pto::TileType::Vec, "Fix: TColReduceIdx Src TileType must be Vec Tile");
    static_assert(TileDataIn::SFractal == SLayout::NoneBox, "Fix: TColReduceIdx only support Nd or Dn fractal Tile");

    // 输出索引Tile类型检查
    static_assert(TileDataOutIdx::Loc == pto::TileType::Vec, "Fix: TColReduceIdx DstIdx TileType must be Vec Tile");
    static_assert(
        TileDataOutIdx::isRowMajor && TileDataOutIdx::SFractal == SLayout::NoneBox,
        "Fix: TColReduceIdx DstIdx only supports Nd fractal Tile");

    // 临时Tile类型检查
    static_assert(
        std::is_same_v<typename TileDataIn::DType, typename TileDataTmp::DType>,
        "Fix: TColReduceIdx input type must be consistent with tmp type");

    // 基础输入维度检查
    PTO_ASSERT(
        srcValidRow != 0 && srcValidCol != 0, "Fix: TColReduceIdx input shape is invalid, validCol or validRow is 0");
    PTO_ASSERT(dstValidRow == 1, "Fix: TColReduceIdx output idx validRow must be 1");
    PTO_ASSERT(srcValidCol == dstValidCol, "Fix: TColReduceIdx input validCol must equal idx output validCol");

    if constexpr (WithVal) {
        // 值输出Tile类型检查
        static_assert(
            TileDataOutVal::Loc == pto::TileType::Vec, "Fix: TColReduceOpsIdx DstVal TileType must be Vec Tile");
        static_assert(
            TileDataOutVal::isRowMajor && TileDataOutVal::SFractal == SLayout::NoneBox,
            "Fix: TColReduceOpsIdx DstVal only supports Nd fractal Tile");
        static_assert(
            std::is_same_v<typename TileDataOutVal::DType, typename TileDataIn::DType>,
            "Fix: TColReduceOpsIdx DstVal data type must match input type");

        // 值输出维度检查
        PTO_ASSERT(dstValValidRow == 1, "Fix: TColReduceOpsIdx output value validRow must be 1");
        PTO_ASSERT(dstValValidCol != 0, "Fix: TColReduceOpsIdx output value validCol must be non-zero");
        PTO_ASSERT(
            srcValidCol == dstValValidCol, "Fix: TColReduceOpsIdx input validCol must equal value output validCol");
        PTO_ASSERT(
            dstValValidRow == dstValidRow, "Fix: TColReduceOpsIdx value and idx output tiles must have same validRow");
        PTO_ASSERT(
            dstValValidCol == dstValidCol, "Fix: TColReduceOpsIdx value and idx output tiles must have same validCol");

        if constexpr (sizeof(typename TileDataIn::DType) == 2) {
            static_assert(
                std::is_same_v<typename TileDataOutIdx::DType, uint16_t> ||
                    std::is_same_v<typename TileDataOutIdx::DType, int16_t>,
                "Fix: TColReduceOpsIdx DstIdx data type must be s16 or u16 when input type size is 2 bytes");
        } else {
            static_assert(
                std::is_same_v<typename TileDataOutIdx::DType, uint32_t> ||
                    std::is_same_v<typename TileDataOutIdx::DType, int32_t>,
                "Fix: TColReduceOpsIdx DstIdx data type must be s32 or u32 when input type size is 4 bytes");
        }
    } else {
        static_assert(
            std::is_same_v<typename TileDataOutIdx::DType, uint32_t> ||
                std::is_same_v<typename TileDataOutIdx::DType, int32_t>,
            "Fix: TColReduceIdx DstIdx data type must be s32 or u32");
    }
}

template <typename TVal, bool IsArgMax>
struct TColIdxCompareOp {
    PTO_INTERNAL static void VCmp(__ubuf__ TVal* src1, __ubuf__ TVal* src2)
    {
        if constexpr (IsArgMax) {
            vcmp_ge(src1, src2, 1, 1, 1, 1, 0, 0, 0);
        } else {
            vcmp_le(src1, src2, 1, 1, 1, 1, 0, 0, 0);
        }
    }

    PTO_INTERNAL static void UpdateValue(__ubuf__ TVal* dst, __ubuf__ TVal* src1, __ubuf__ TVal* src2)
    {
        if constexpr (IsArgMax) {
            vmax(dst, src1, src2, 1, 1, 1, 1, 0, 0, 0);
        } else {
            vmin(dst, src1, src2, 1, 1, 1, 1, 0, 0, 0);
        }
    }
};

template <
    typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, typename TileDataTmp, bool IsArgMax,
    bool IsRemainingBlock, bool WithVal>
PTO_INTERNAL void ProcessSingleBlock(
    __ubuf__ typename TileDataOutVal::DType* dstVal, __ubuf__ typename TileDataOutIdx::DType* dstIdx,
    __ubuf__ typename TileDataIn::DType* src, __ubuf__ typename TileDataTmp::DType* tmp, unsigned srcValidRow,
    uint16_t blockIndex, uint32_t tmpGapElems, uint32_t elemPerRpt, uint16_t remainingElements = 0)
{
    using TIN = typename TileDataIn::DType;
    using TOUT = typename TileDataOutIdx::DType;
    constexpr uint32_t srcRowStride = TileDataIn::Cols;
    using TIdx = std::conditional_t<sizeof(TIN) == 2, int16_t, int32_t>;
    using TCmp = std::conditional_t<sizeof(TIN) == 2, half, float>;

    constexpr bool isHalfType = sizeof(TIN) == 2;

    __ubuf__ TIN* elemTmpPtr = tmp + tmpGapElems;
    __ubuf__ TOUT* currentDst;
    __ubuf__ TIdx* currentIndex;
    __ubuf__ TCmp* selectedCmpVals;

    if constexpr (isHalfType && !WithVal) {
        currentIndex = reinterpret_cast<__ubuf__ TIdx*>(tmp) + 2 * tmpGapElems;
        selectedCmpVals = reinterpret_cast<__ubuf__ TCmp*>(tmp) + 2 * tmpGapElems;
        currentDst = reinterpret_cast<__ubuf__ TOUT*>(tmp) + 2 * tmpGapElems;
    } else {
        currentDst = dstIdx + blockIndex * elemPerRpt;
        currentIndex = reinterpret_cast<__ubuf__ TIdx*>(dstIdx) + blockIndex * elemPerRpt;
        selectedCmpVals = reinterpret_cast<__ubuf__ TCmp*>(dstIdx) + blockIndex * elemPerRpt;
    }

    if constexpr (IsRemainingBlock) {
        set_mask_count();
        set_vector_mask(0, remainingElements);
    }

    vector_dup(tmp, 0, 1, 1, 1, 0, 0);          // 累积索引
    vector_dup(currentIndex, 0, 1, 1, 1, 0, 0); // 当前索引

    if constexpr (IsRemainingBlock) {
        vcopy(
            reinterpret_cast<__ubuf__ TIdx*>(elemTmpPtr),
            reinterpret_cast<__ubuf__ TIdx*>(src + blockIndex * elemPerRpt), 1, 1, 1, 0, 0);
    } else {
        pto_copy_ubuf_to_ubuf(elemTmpPtr, src + blockIndex * elemPerRpt, 1, 8, 0, 0);
    }
    pipe_barrier(PIPE_V);

    for (uint16_t rowIdx = 1; rowIdx < srcValidRow; ++rowIdx) {
        vadds(reinterpret_cast<__ubuf__ TIdx*>(tmp), reinterpret_cast<__ubuf__ TIdx*>(tmp), 1, 1, 1, 1, 0, 0);

        __ubuf__ TIN* currentSrc = src + rowIdx * srcRowStride + blockIndex * elemPerRpt;

        TColIdxCompareOp<TCmp, IsArgMax>::VCmp(
            reinterpret_cast<__ubuf__ TCmp*>(elemTmpPtr), reinterpret_cast<__ubuf__ TCmp*>(currentSrc));
        pipe_barrier(PIPE_V);

        vsel(
            reinterpret_cast<__ubuf__ TCmp*>(selectedCmpVals), reinterpret_cast<__ubuf__ TCmp*>(selectedCmpVals),
            reinterpret_cast<__ubuf__ TCmp*>(tmp), 1, 1, 1, 1, 0, 0, 0, 0);

        TColIdxCompareOp<TCmp, IsArgMax>::UpdateValue(
            reinterpret_cast<__ubuf__ TCmp*>(elemTmpPtr), reinterpret_cast<__ubuf__ TCmp*>(elemTmpPtr),
            reinterpret_cast<__ubuf__ TCmp*>(currentSrc));
        pipe_barrier(PIPE_V);
    }

    if constexpr (isHalfType && !WithVal) {
        vconv_s162f16a(
            reinterpret_cast<__ubuf__ half*>(tmp) + 2 * tmpGapElems,
            reinterpret_cast<__ubuf__ int16_t*>(tmp) + 2 * tmpGapElems, 1, 1, 1, 0, 0);
        pipe_barrier(PIPE_V);

        vconv_f162s32a(
            reinterpret_cast<__ubuf__ int32_t*>(dstIdx) + blockIndex * elemPerRpt,
            reinterpret_cast<__ubuf__ half*>(tmp) + 2 * tmpGapElems, 2, 1, 1, 8, 4);
        pipe_barrier(PIPE_V);
    }

    if constexpr (WithVal && !IsRemainingBlock) {
        pto_copy_ubuf_to_ubuf(dstVal + blockIndex * elemPerRpt, elemTmpPtr, 1, 8, 0, 0);
        pipe_barrier(PIPE_V);
    } else if constexpr (WithVal && IsRemainingBlock) {
        vcopy(
            reinterpret_cast<__ubuf__ TIdx*>(dstVal) + blockIndex * elemPerRpt,
            reinterpret_cast<__ubuf__ TIdx*>(elemTmpPtr), 1, 1, 1, 0, 0);
        pipe_barrier(PIPE_V);
    }

    if constexpr (IsRemainingBlock) {
        set_mask_norm();
        set_vector_mask(-1, -1);
        pipe_barrier(PIPE_V);
    }
}

template <
    typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, typename TileDataTmp, bool IsArgMax,
    bool WithVal>
PTO_INTERNAL void ProcessFullBlocks(
    __ubuf__ typename TileDataOutVal::DType* dstVal, __ubuf__ typename TileDataOutIdx::DType* dstIdx,
    __ubuf__ typename TileDataIn::DType* src, __ubuf__ typename TileDataTmp::DType* tmp, unsigned srcValidRow,
    unsigned srcValidCol, uint16_t numFullBlocks, uint32_t tmpGapElems, uint32_t elemPerRpt)
{
    if (numFullBlocks == 0)
        return;
    for (uint16_t blockIdx = 0; blockIdx < numFullBlocks; ++blockIdx) {
        ProcessSingleBlock<TileDataOutVal, TileDataOutIdx, TileDataIn, TileDataTmp, IsArgMax, false, WithVal>(
            dstVal, dstIdx, src, tmp, srcValidRow, blockIdx, tmpGapElems, elemPerRpt);
    }
}

template <
    typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, typename TileDataTmp, bool IsArgMax,
    bool WithVal>
PTO_INTERNAL void ProcessRemainingElements(
    __ubuf__ typename TileDataOutVal::DType* dstVal, __ubuf__ typename TileDataOutIdx::DType* dstIdx,
    __ubuf__ typename TileDataIn::DType* src, __ubuf__ typename TileDataTmp::DType* tmp, unsigned srcValidRow,
    unsigned srcValidCol, uint16_t numFullBlocks, uint32_t tmpGapElems, uint32_t elemPerRpt)
{
    uint16_t remainingAfterLoop = srcValidCol % elemPerRpt;

    if (remainingAfterLoop == 0)
        return;

    ProcessSingleBlock<TileDataOutVal, TileDataOutIdx, TileDataIn, TileDataTmp, IsArgMax, true, WithVal>(
        dstVal, dstIdx, src, tmp, srcValidRow, numFullBlocks, tmpGapElems, elemPerRpt, remainingAfterLoop);
}

template <
    typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, typename TileDataTmp, bool IsArgMax,
    bool WithVal>
PTO_INTERNAL void TColReduceIdxImpl(
    __ubuf__ typename TileDataOutVal::DType* dstVal, __ubuf__ typename TileDataOutIdx::DType* dstIdx,
    __ubuf__ typename TileDataIn::DType* src, __ubuf__ typename TileDataTmp::DType* tmp, unsigned srcValidRow,
    unsigned srcValidCol)
{
    using TIN = typename TileDataIn::DType;
    constexpr uint32_t elemPerRpt = REPEAT_BYTE / sizeof(TIN);
    constexpr uint32_t elemPerBlock = BLOCK_BYTE_SIZE / sizeof(TIN);

    uint16_t numFullBlocks = srcValidCol / elemPerRpt;
    uint32_t tmpGapElems = numFullBlocks > 0 ? elemPerRpt : CeilDivision(srcValidCol, elemPerBlock) * elemPerBlock;

    ProcessFullBlocks<TileDataOutVal, TileDataOutIdx, TileDataIn, TileDataTmp, IsArgMax, WithVal>(
        dstVal, dstIdx, src, tmp, srcValidRow, srcValidCol, numFullBlocks, tmpGapElems, elemPerRpt);

    ProcessRemainingElements<TileDataOutVal, TileDataOutIdx, TileDataIn, TileDataTmp, IsArgMax, WithVal>(
        dstVal, dstIdx, src, tmp, srcValidRow, srcValidCol, numFullBlocks, tmpGapElems, elemPerRpt);
}

template <
    bool IsArgMax, typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, typename TileDataTmp,
    bool WithVal>
__tf__ PTO_INTERNAL void TColReduceIdxInstr(
    typename TileDataOutVal::TileDType __out__ dstValData, typename TileDataOutIdx::TileDType __out__ dstIdxData,
    typename TileDataIn::TileDType __in__ srcData, typename TileDataTmp::TileDType __in__ tmpData, unsigned srcValidRow,
    unsigned srcValidCol, unsigned dstValidRow, unsigned dstValidCol)
{
    __ubuf__ typename TileDataOutVal::DType* dstVal =
        reinterpret_cast<__ubuf__ typename TileDataOutVal::DType*>(__cce_get_tile_ptr(dstValData));
    __ubuf__ typename TileDataOutIdx::DType* dstIdx =
        reinterpret_cast<__ubuf__ typename TileDataOutIdx::DType*>(__cce_get_tile_ptr(dstIdxData));
    __ubuf__ typename TileDataIn::DType* src =
        reinterpret_cast<__ubuf__ typename TileDataIn::DType*>(__cce_get_tile_ptr(srcData));
    __ubuf__ typename TileDataTmp::DType* tmp =
        reinterpret_cast<__ubuf__ typename TileDataTmp::DType*>(__cce_get_tile_ptr(tmpData));

    TColReduceIdxImpl<TileDataOutVal, TileDataOutIdx, TileDataIn, TileDataTmp, IsArgMax, WithVal>(
        dstVal, dstIdx, src, tmp, srcValidRow, srcValidCol);
}

template <
    bool IsArgMax, typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, typename TileDataTmp,
    bool WithVal = false>
PTO_INTERNAL void TColReduceIdxDispatch(
    TileDataOutVal& dstVal, TileDataOutIdx& dstIdx, TileDataIn& src, TileDataTmp& tmp)
{
    unsigned srcValidRow = src.GetValidRow();
    unsigned srcValidCol = src.GetValidCol();

    unsigned dstIdxValidRow = dstIdx.GetValidRow();
    unsigned dstIdxValidCol = dstIdx.GetValidCol();

    unsigned dstValValidRow = dstVal.GetValidRow();
    unsigned dstValValidCol = dstVal.GetValidCol();

    TColReduceIdxCheck<TileDataOutVal, TileDataOutIdx, TileDataIn, TileDataTmp, WithVal>(
        srcValidRow, srcValidCol, dstIdxValidRow, dstIdxValidCol, dstValValidRow, dstValValidCol);

    TColReduceIdxInstr<IsArgMax, TileDataOutVal, TileDataOutIdx, TileDataIn, TileDataTmp, WithVal>(
        dstVal.data(), dstIdx.data(), src.data(), tmp.data(), srcValidRow, srcValidCol, dstIdxValidRow, dstIdxValidCol);
}

template <typename TileDataOutIdx, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TCOLARGMAX_IMPL(TileDataOutIdx& dst, TileDataIn& src, TileDataTmp& tmp)
{
    TColReduceIdxDispatch<true, TileDataIn, TileDataOutIdx, TileDataIn, TileDataTmp>(src, dst, src, tmp);
}

template <typename TileDataOutIdx, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TCOLARGMIN_IMPL(TileDataOutIdx& dst, TileDataIn& src, TileDataTmp& tmp)
{
    TColReduceIdxDispatch<false, TileDataIn, TileDataOutIdx, TileDataIn, TileDataTmp>(src, dst, src, tmp);
}

template <typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TCOLARGMAX_IMPL(TileDataOutVal& dstVal, TileDataOutIdx& dstIdx, TileDataIn& src, TileDataTmp& tmp)
{
    TColReduceIdxDispatch<true, TileDataOutVal, TileDataOutIdx, TileDataIn, TileDataTmp, true>(
        dstVal, dstIdx, src, tmp);
}

template <typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TCOLARGMIN_IMPL(TileDataOutVal& dstVal, TileDataOutIdx& dstIdx, TileDataIn& src, TileDataTmp& tmp)
{
    TColReduceIdxDispatch<false, TileDataOutVal, TileDataOutIdx, TileDataIn, TileDataTmp, true>(
        dstVal, dstIdx, src, tmp);
}

} // namespace pto
#endif
