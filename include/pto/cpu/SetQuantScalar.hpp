/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef SET_QUANT_SCALAR_HPP
#define SET_QUANT_SCALAR_HPP

namespace pto {

template <typename OutType>
PTO_INTERNAL void SET_QUANT_SCALAR_IMPL(float preQuantScalar)
{
    uint64_t quantConfig = static_cast<uint64_t>(std::bit_cast<uint32_t>(preQuantScalar));
    if constexpr (sizeof(OutType) == 1) {
        constexpr bool sign = (std::is_same_v<OutType, int8_t>) ? true : false;
        quantConfig = (quantConfig & ~(static_cast<uint64_t>(1) << 46)) | (static_cast<uint64_t>(sign) << 46);
    }
    uint8_t* quantSrc = reinterpret_cast<uint8_t*>(&quantConfig);
    uint8_t* reg_base = reinterpret_cast<uint8_t*>(NPUMemoryModel::Instance().GetREGBase());
    uint8_t* scalarReg = reg_base + QUANT_SCALAR_REG_OFFSET * sizeof(uint64_t);
    std::copy(quantSrc, quantSrc + sizeof(quantConfig), scalarReg);
}

} // namespace pto

#endif // SET_QUANT_SCALAR_HPP