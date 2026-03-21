/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCOLSUM_HPP
#define TCOLSUM_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#ifdef PTO_NPU_ARCH_A5
#include "common.hpp"
#include "utils.hpp"
#include "TColReduceOps.hpp"
#endif

namespace pto {
template <typename T>
struct TColSumOp {
    PTO_INTERNAL static void ReduceInstr(RegTensor<T> &dst, RegTensor<T> &src0, RegTensor<T> &src1, MaskReg &pReg)
    {
        vadd(dst, src0, src1, pReg, MODE_ZEROING);
    }
};

template <typename T, unsigned TmpStride>
PTO_INTERNAL void TColSum_Binary_TmpProc(RegTensor<T> &src0VReg, RegTensor<T> &src1VReg, RegTensor<T> &dstVReg,
                                         MaskReg &pReg, __ubuf__ T *tmp, uint16_t nLoop)
{
    bool remain;
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();

    // 获取nLoop的 最高比特位-1 为循环次数, 等价于while(nLoop > 1)
    uint16_t BinaryAccLoopTimes = (nLoop > 0) ? 31 - __builtin_clz((uint32_t)nLoop) : 0;
    for (int i = 0; i < BinaryAccLoopTimes; ++i) {
        remain = nLoop % 2;
        nLoop /= 2;

        // 依赖上一次循环的数据, 设置同步vlds等vsts
        mem_bar(VST_VLD);
        for (int j = 0; j < nLoop; ++j) {
            vlds(src0VReg, tmp, (2 * j) * TmpStride, NORM);
            vlds(src1VReg, tmp, (2 * j + 1) * TmpStride, NORM);
            vadd(dstVReg, src0VReg, src1VReg, pReg, MODE_ZEROING);
            vsts(dstVReg, tmp, j * TmpStride, distValue, pReg);
        }

        if (remain) {
            // 最后剩余奇数行加入tmp最后一行
            vlds(src0VReg, tmp, (2 * nLoop) * TmpStride, NORM);
            vadd(dstVReg, src0VReg, dstVReg, pReg, MODE_ZEROING);
            vsts(dstVReg, tmp, (nLoop - 1) * TmpStride, distValue, pReg);
        }
    }
}

template <typename T, unsigned SrcStride, unsigned TmpStride, unsigned elmPerRpt>
PTO_INTERNAL void TColSum_Binary(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *tmp, unsigned validRow,
                                 unsigned validCol)
{
    uint16_t repeatTimes = CeilDivision(validCol, elmPerRpt);
    __VEC_SCOPE__
    {
        RegTensor<T> src0VReg;
        RegTensor<T> src1VReg;
        RegTensor<T> dstVReg;
        MaskReg pReg;
        uint32_t sreg = validCol;
        uint16_t i, j;
        // 相邻两行相加放入temp, nLoop为tmp有效数据行数
        uint16_t nLoop = validRow / 2;
        bool remain = validRow % 2;
        uint16_t remainAddLine = (remain && (nLoop > 0)) ? (nLoop - 1) : 0;
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();

        for (i = 0; i < repeatTimes; ++i) {
            // sreg在每次执行CreatePredicate之后会累减nElmPerRepeat，直至0
            pReg = CreatePredicate<T>(sreg);
            vbr(dstVReg, 0);

            // 将src数据进行初步运算并存入tmp
            for (j = 0; j < nLoop; ++j) {
                vlds(src0VReg, src, i * elmPerRpt + (2 * j) * SrcStride, NORM);
                vlds(src1VReg, src, i * elmPerRpt + (2 * j + 1) * SrcStride, NORM);
                vadd(dstVReg, src0VReg, src1VReg, pReg, MODE_ZEROING);
                vsts(dstVReg, tmp, j * TmpStride, distValue, pReg);
            }

            if (remain) {
                // 最后剩余奇数行加入tmp最后一行
                vlds(src0VReg, src, i * elmPerRpt + (validRow - 1) * SrcStride, NORM);
                vadd(dstVReg, src0VReg, dstVReg, pReg, MODE_ZEROING);
                vsts(dstVReg, tmp, remainAddLine * TmpStride, distValue, pReg);
            }
            TColSum_Binary_TmpProc<T, TmpStride>(src0VReg, src1VReg, dstVReg, pReg, tmp, nLoop);
            // 最后一步vsts(dstVReg, tmp)其实无作用, dstVReg已经保存最终结果
            vsts(dstVReg, dst, i * elmPerRpt, distValue, pReg);
        }
    } // end VF
}

template <typename T, typename TileDataOut, typename TileDataIn, typename TileDataTmp, bool isBinary>
__tf__ PTO_INTERNAL void TColSum(typename TileDataOut::TileDType __out__ dstData,
                                 typename TileDataIn::TileDType __in__ srcData,
                                 typename TileDataTmp::TileDType __in__ tmpData, unsigned validRow, unsigned validCol,
                                 unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    __ubuf__ T *dst = (__ubuf__ T *)__cce_get_tile_ptr(dstData);
    __ubuf__ T *src = (__ubuf__ T *)__cce_get_tile_ptr(srcData);

    if constexpr (isBinary) {
        __ubuf__ T *tmp = (__ubuf__ T *)__cce_get_tile_ptr(tmpData);
        constexpr unsigned elmPerRpt = CCE_VL / sizeof(T); // 每次repeat涉及多少个元素
        constexpr int tmpStride = TileDataTmp::RowStride * sizeof(typename TileDataTmp::DType) / sizeof(T);
        TColSum_Binary<T, TileDataIn::Cols, tmpStride, elmPerRpt>(dst, src, tmp, validRow, validCol);
    } else {
        TColReduceInstr<TColSumOp<T>, T, TileDataIn>(dst, src, validRow, validCol, version);
    }
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TCOLSUM_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, bool isBinary)
{
    int32_t validCol = src.GetValidCol();
    int32_t validRow = src.GetValidRow();
    TColReduceCheck<TileDataOut, TileDataIn>(validRow, validCol, dst.GetValidCol());
    if (validCol == 0 || validRow == 0) {
        return;
    }

    using T = typename TileDataIn::DType;
    constexpr int tmpStride = TileDataTmp::RowStride * sizeof(typename TileDataTmp::DType) / sizeof(T);
    PTO_ASSERT(validCol <= tmpStride,
               "Fix: TCOLSUM input valid columns must be less than or equal to the tmp columns.");
    if (isBinary) {
        TColSum<T, TileDataOut, TileDataIn, TileDataTmp, true>(dst.data(), src.data(), tmp.data(), validRow, validCol);
    } else {
        TColSum<T, TileDataOut, TileDataIn, TileDataTmp, false>(dst.data(), src.data(), tmp.data(), validRow, validCol);
    }
}
} // namespace pto
#endif
