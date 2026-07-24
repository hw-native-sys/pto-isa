/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TEXP_CUSTOM_HPP
#define TEXP_CUSTOM_HPP

#include "common.hpp"

namespace pto {
template <typename T>
PTO_INTERNAL void ExpPrecisionImpl(RegTensor<T>& dstReg, RegTensor<T>& srcReg, MaskReg& mask)
{
    using ConvUnion = std::conditional_t<sizeof(T) == sizeof(half), union HalfConvUnion, union FloatConvUnion>;
    RegTensor<T> regTwo;
    RegTensor<T> tmpReg;
    MaskReg subnormalMask;
    ConvUnion subnormalVal;
    subnormalVal.i = std::is_same_v<T, half> ? 0x03ff : 0x007FFFFF;
    vexp(dstReg, srcReg, mask, MODE_ZEROING);
    vcmps_le(subnormalMask, dstReg, subnormalVal.f, mask);
    vdup(regTwo, 2, subnormalMask, MODE_ZEROING);
    vdiv(tmpReg, srcReg, regTwo, subnormalMask, MODE_ZEROING);
    vexp(tmpReg, tmpReg, subnormalMask, MODE_ZEROING);
    vmul(tmpReg, tmpReg, tmpReg, subnormalMask, MODE_ZEROING);
    vsel(dstReg, tmpReg, dstReg, subnormalMask);
}
} // namespace pto

#endif
