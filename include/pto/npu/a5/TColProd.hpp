/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCOLPROD_HPP
#define TCOLPROD_HPP

#include "TColReduceOps.hpp"

namespace pto {
template <typename T>
struct TColProdOp {
    using PadType = T;
    static constexpr T InitVal = (T)1;
    PTO_INTERNAL static void ReduceInstr(RegTensor<T>& dst, RegTensor<T>& src0, RegTensor<T>& src1, MaskReg& pReg)
    {
        vmul(dst, src0, src1, pReg, MODE_ZEROING);
    }
};

template <typename T, typename TileDataOut, typename TileDataIn>
__tf__ PTO_INTERNAL void TColProd(
    typename TileDataOut::TileDType __out__ dstData, typename TileDataIn::TileDType __in__ srcData, unsigned validRow,
    unsigned validCol, unsigned version)
{
    __ubuf__ T* dst = (__ubuf__ T*)__cce_get_tile_ptr(dstData);
    __ubuf__ T* src = (__ubuf__ T*)__cce_get_tile_ptr(srcData);

    TColReduceInstr<TColProdOp<T>, T, TileDataIn>(dst, src, validRow, validCol, version);
}

template <typename TileDataOut, typename TileDataIn>
PTO_INTERNAL void TCOLPROD_IMPL(TileDataOut& dst, TileDataIn& src)
{
    using T = typename TileDataIn::DType;
    static_assert(
        std::is_same_v<T, float> || std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t> ||
            std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t> || std::is_same_v<T, int32_t> ||
            std::is_same_v<T, uint32_t>,
        "Fix: TCOLPROD input data type supports only int16_t/uint16_t/half/bfloat16_t/int32_t/uint32_t/float.");

    unsigned validCol = src.GetValidCol();
    unsigned validRow = src.GetValidRow();
    if (validCol == 0 || validRow == 0) {
        return;
    }

    TColReduceCheck<TileDataOut, TileDataIn>(validRow, validCol, dst.GetValidCol());

    TColProd<T, TileDataOut, TileDataIn>(dst.data(), src.data(), validRow, validCol, VFImplKind::VFIMPL_DEFAULT);
}
} // namespace pto
#endif
