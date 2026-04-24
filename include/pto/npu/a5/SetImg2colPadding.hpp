/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef SET_IMG2COL_PADDING_HPP
#define SET_IMG2COL_PADDING_HPP

namespace pto {
template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL>
PTO_INTERNAL void SET_IMG2COL_PADDING_IMPL(ConvTileData &src)
{
    if constexpr (FmatrixMode == SetFmatrixMode::FMATRIX_A_MANUAL || FmatrixMode == SetFmatrixMode::FMATRIX_B_MANUAL) {
        using DataType = typename ConvTileData::DType;
        constexpr uint16_t padValueShiftBit = 8;
        uint32_t paddingValue = 0;
        const DataType padValue = src.GetPadValue();
        if constexpr (sizeof(DataType) == 1) {
            uint8_t u8Value = *reinterpret_cast<const uint8_t *>(&padValue);
            paddingValue = (static_cast<uint16_t>(u8Value) << padValueShiftBit) | u8Value;
        } else if constexpr (sizeof(DataType) == 2) {
            paddingValue = *reinterpret_cast<const uint16_t *>(&padValue);
        } else if constexpr (sizeof(DataType) == 4) {
            paddingValue = *reinterpret_cast<const uint32_t *>(&padValue);
        }
        if constexpr (FmatrixMode == SetFmatrixMode::FMATRIX_A_MANUAL) {
            set_padding(paddingValue);
        } else if constexpr (FmatrixMode == SetFmatrixMode::FMATRIX_B_MANUAL) {
            set_padding_b(paddingValue);
        }
    }
}
} // namespace pto
#endif // SET_IMG2COL_PADDING_HPP
