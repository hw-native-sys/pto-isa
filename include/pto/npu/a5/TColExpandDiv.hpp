/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCOLEXPANDDIV_HPP
#define TCOLEXPANDDIV_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "common.hpp"
#include "utils.hpp"
#include "TColExpandBinOp.hpp"

namespace pto {

template <DivAlgorithm PrecisionType, typename T>
struct ColExpandDivOp {
    PTO_INTERNAL static void ColExpandBinaryInstr(RegTensor<T> &reg_dst, RegTensor<T> &reg_src0, RegTensor<T> &reg_src1,
                                                  MaskReg &preg)
    {
        if constexpr (PrecisionType == DivAlgorithm::HIGH_PRECISION && std::is_same_v<T, float>) {
            DivIEEE754FloatImpl<T, RegTensor<T> >(reg_dst, reg_src0, reg_src1, preg);
        } else if constexpr (PrecisionType == DivAlgorithm::HIGH_PRECISION && std::is_same_v<T, half>) {
            DivIEEE754HalfImpl<T, RegTensor<T> >(reg_dst, reg_src0, reg_src1, preg);
        } else {
            vdiv(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
        }
    }
};

template <DivAlgorithm PrecisionType, typename T>
struct ColExpandDivOp2 {
    PTO_INTERNAL static void ColExpandBinaryInstr(RegTensor<T> &reg_dst, RegTensor<T> &reg_src0, RegTensor<T> &reg_src1,
                                                  MaskReg &preg)
    {
        if constexpr (PrecisionType == DivAlgorithm::HIGH_PRECISION && std::is_same_v<T, float>) {
            DivIEEE754FloatImpl<T, RegTensor<T> >(reg_dst, reg_src1, reg_src0, preg);
        } else if constexpr (PrecisionType == DivAlgorithm::HIGH_PRECISION && std::is_same_v<T, half>) {
            DivIEEE754HalfImpl<T, RegTensor<T> >(reg_dst, reg_src1, reg_src0, preg);
        } else {
            vdiv(reg_dst, reg_src1, reg_src0, preg, MODE_ZEROING);
        }
    }
};

template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileData, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TCOLEXPANDDIV_IMPL(TileData &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    using T = typename TileData::DType;
    TCOLEXPANDOP_IMPL<ColExpandDivOp<PrecisionType, T>, ColExpandDivOp2<PrecisionType, T>, TileData, TileDataSrc0,
                      TileDataSrc1>(dst, src0, src1);
}
} // namespace pto
#endif