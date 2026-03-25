/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

#ifndef TRANDOM_HPP
#define TRANDOM_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/common/type.hpp>
#include "common.hpp"
#include "utils.hpp"

namespace pto {
constexpr uint16_t TRANDOM_ONCE_REPEAT = 4;
constexpr uint32_t TRANDOM_CONST_0 = 0xD2511F53;
constexpr uint32_t TRANDOM_CONST_1 = 0xCD9E8D57;
constexpr uint32_t TRANDOM_CONST_KEY_ADD_0 = 0x9E3779B9;
constexpr uint32_t TRANDOM_CONST_KEY_ADD_1 = 0xBB67AE85;

template <typename T>
PTO_INTERNAL void AddWith128Bits(T &ctr0, T &ctr1, T &ctr2, T &ctr3, T &zeros, T &value, MaskReg &pd, MaskReg &pReg)
{
    vaddc(pd, ctr0, ctr0, value, pReg);
    vaddcs(pd, ctr1, ctr1, zeros, pd, pReg);
    vaddcs(pd, ctr2, ctr2, zeros, pd, pReg);
    vaddcs(pd, ctr3, ctr3, zeros, pd, pReg);
}

template <unsigned eleStrideB32OneRow, typename T>
PTO_INTERNAL void TRandomInitConst(T &incIdx, T &zeros, T &vEleStride, T &const0, T &const1)
{
    vci((RegTensor<int32_t> &)incIdx, 0);
    vbr(zeros, 0);
    vbr(vEleStride, eleStrideB32OneRow);
    vbr(const0, TRANDOM_CONST_0);
    vbr(const1, TRANDOM_CONST_1);
}

template <typename T>
PTO_INTERNAL void TRandomInitKeyCnt(T &key0, T &key1, T &ctr0, T &ctr1, T &ctr2, T &ctr3, TRandomKey key,
                                    TRandomCounter counter)
{
    vbr(key0, key[0]);
    vbr(key1, key[1]);
    vbr(ctr0, counter[0]);
    vbr(ctr1, counter[1]);
    vbr(ctr2, counter[2]);
    vbr(ctr3, counter[3]);
}

template <uint16_t Rounds, typename T>
PTO_INTERNAL void TRandomKernel(T &ctr0, T &ctr1, T &ctr2, T &ctr3, T &key0, T &key1, T &const0, T &const1, MaskReg &pg)
{
    T tmpL0, tmpH0, tmpL1, tmpH1;
    for (uint16_t i = 0; i < Rounds; ++i) {
        vmull(tmpL0, tmpH0, ctr0, const0, pg);
        vmull(tmpL1, tmpH1, ctr2, const1, pg);
        vxor(tmpH1, tmpH1, ctr1, pg);
        vxor(ctr0, tmpH1, key0, pg);
        vxor(tmpH0, tmpH0, ctr3, pg);
        vxor(ctr2, tmpH0, key1, pg);
        ctr1 = tmpL1;
        ctr3 = tmpL0;
        vadds(key0, key0, TRANDOM_CONST_KEY_ADD_0, pg);
        vadds(key1, key1, TRANDOM_CONST_KEY_ADD_1, pg);
    }
}

template <unsigned nElemPerRpt, unsigned rowStride, typename T, typename U>
PTO_INTERNAL void TRandomStore(__ubuf__ T *dst, U &ctr0, U &ctr1, U &ctr2, U &ctr3, uint16_t i, uint16_t j,
                               unsigned &sReg)
{
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
    MaskReg pReg0 = CreatePredicate<T>(sReg);
    MaskReg pReg1 = CreatePredicate<T>(sReg);
    MaskReg pReg2 = CreatePredicate<T>(sReg);
    MaskReg pReg3 = CreatePredicate<T>(sReg);
    vsts(ctr0, dst, i * rowStride + (TRANDOM_ONCE_REPEAT * j) * nElemPerRpt, distValue, pReg0);
    vsts(ctr1, dst, i * rowStride + (TRANDOM_ONCE_REPEAT * j + 1) * nElemPerRpt, distValue, pReg1);
    vsts(ctr2, dst, i * rowStride + (TRANDOM_ONCE_REPEAT * j + 2) * nElemPerRpt, distValue, pReg2);
    vsts(ctr3, dst, i * rowStride + (TRANDOM_ONCE_REPEAT * j + 3) * nElemPerRpt, distValue, pReg3);
}

template <uint16_t Rounds, typename DstTile>
__tf__ PTO_INTERNAL void TRandom(typename DstTile::TileDType __out__ dstData, TRandomKey key, TRandomCounter counter,
                                 unsigned validRow, unsigned validCol)
{
    __ubuf__ uint32_t *dst = (__ubuf__ uint32_t *)__cce_get_tile_ptr(dstData);
    using T = typename DstTile::DType;
    __VEC_SCOPE__
    {
        constexpr unsigned nElemPerRpt = CCE_VL / sizeof(T);
        constexpr unsigned rowStride = DstTile::RowStride;
        constexpr unsigned eleStrideB32OneRow = rowStride / TRANDOM_ONCE_REPEAT;

        uint16_t nLoop = CeilDivision(validCol, TRANDOM_ONCE_REPEAT * nElemPerRpt);
        unsigned sReg = validCol;
        unsigned pgTmp = nElemPerRpt;
        MaskReg pg = CreatePredicate<T>(pgTmp);
        MaskReg pReg0, pReg1, pReg2, pReg3;
        MaskReg pd;
        RegTensor<uint32_t> ctr0, ctr1, ctr2, ctr3, key0, key1, zeros, incIdx, vEleStride, const0, const1;
        RegTensor<uint32_t> tmpCtr0, tmpCtr1, tmpCtr2, tmpCtr3, tmpKey0, tmpKey1;

        TRandomInitKeyCnt(key0, key1, ctr0, ctr1, ctr2, ctr3, key, counter);
        TRandomInitConst<eleStrideB32OneRow>(incIdx, zeros, vEleStride, const0, const1);
        AddWith128Bits(ctr0, ctr1, ctr2, ctr3, zeros, incIdx, pd, pg);

        for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
            sReg = validCol;
            for (uint16_t j = 0; j < nLoop; ++j) {
                tmpCtr0 = ctr0;
                tmpCtr1 = ctr1;
                tmpCtr2 = ctr2;
                tmpCtr3 = ctr3;
                tmpKey0 = key0;
                tmpKey1 = key1;

                TRandomKernel<Rounds>(tmpCtr0, tmpCtr1, tmpCtr2, tmpCtr3, tmpKey0, tmpKey1, const0, const1, pg);

                TRandomStore<nElemPerRpt, rowStride>(dst, tmpCtr0, tmpCtr1, tmpCtr2, tmpCtr3, i, j, sReg);

                AddWith128Bits(ctr0, ctr1, ctr2, ctr3, zeros, vEleStride, pd, pg);
            }
        }
    }
}

template <uint16_t Rounds = 10, typename DstTile>
PTO_INST void TRANDOM_IMPL(DstTile &dst, TRandomKey &key, TRandomCounter &counter)
{
    using T = typename DstTile::DType;
    static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>,
                  "Fix: TRANDOM only support int32_t and uint32_t.");
    static_assert((Rounds == 10) || (Rounds == 7), "Fix: TRANDOM Rounds can only  be configured to 7 or 10.");
    static_assert(DstTile::isRowMajor, "Fix: TRANDOM only support row major layout.");
    PTO_ASSERT((key != nullptr) && (counter != nullptr), "Fix: TRANDOM key and counter must be provided.");
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();

    TRandom<Rounds, DstTile>(dst.data(), key, counter, validRow, validCol);
}
} // namespace pto
#endif // TRANDOM_HPP