/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCOLEXPANDMAX_HPP
#define TCOLEXPANDMAX_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "common.hpp"
#include "utils.hpp"
#include "TColExpandBinOp.hpp"

namespace pto {

template <typename T>
struct ColExpandMaxOp {
    PTO_INTERNAL static void ColExpandBinaryInstr(
        RegTensor<T>& reg_dst, RegTensor<T>& reg_src0, RegTensor<T>& reg_src1, MaskReg& preg)
    {
        vmax(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
    }

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6)
    PTO_INTERNAL static void Int64ColExpandBinaryInstr(
        vector_s32& dstLow, vector_s32& dstHigh, vector_s32& src0Low, vector_s32& src0High, vector_s32& src1Low,
        vector_s32& src1High, MaskReg& preg)
    {
        Int64BinaryCalcRegs<Int64Op::Max, T>(dstLow, dstHigh, src0Low, src0High, src1Low, src1High, preg);
    }
#endif
};

template <typename TileData, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TCOLEXPANDMAX_IMPL(TileData& dst, TileDataSrc0& src0, TileDataSrc1& src1)
{
    using T = typename TileData::DType;
    TCOLEXPANDOP_IMPL<ColExpandMaxOp<T>, ColExpandMaxOp<T>, TileData, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
}
} // namespace pto
#endif
