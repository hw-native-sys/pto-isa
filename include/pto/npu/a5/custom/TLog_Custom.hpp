/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TLOG_CUSTOM_HPP
#define TLOG_CUSTOM_HPP

#include "common.hpp"

namespace pto {
template <typename T>
PTO_INTERNAL void LogPrecisionImpl(RegTensor<T>& dstReg, RegTensor<T>& srcReg, MaskReg& mask)
{
    using ConvUnion = std::conditional_t<sizeof(T) == sizeof(half), union HalfConvUnion, union FloatConvUnion>;
    MaskReg cmpMask;
    ConvUnion mulFactor;
    mulFactor.i = std::is_same_v<T, half> ? 0x6400 : 0x4B000000;
    ConvUnion subnormalThreshold;
    subnormalThreshold.i = std::is_same_v<T, half> ? 0x03ff : 0x007FFFFF;
    const T compensationFactor = std::is_same_v<T, half> ? -6.931471805599453094172 : -15.9423851528787421;
    RegTensor<T> tmpReg;
    RegTensor<T> srcCopy;
    RegTensor<T> dstCopy;
    vcmps_lt(cmpMask, srcReg, subnormalThreshold.f, mask);
    vmuls(tmpReg, srcReg, mulFactor.f, mask);
    vsel(srcCopy, tmpReg, srcReg, cmpMask);
    vln(dstCopy, srcCopy, mask, MODE_ZEROING);
    vadds(tmpReg, dstCopy, compensationFactor, mask, MODE_ZEROING);
    vsel(dstReg, tmpReg, dstCopy, cmpMask);
}
} // namespace pto

#endif
