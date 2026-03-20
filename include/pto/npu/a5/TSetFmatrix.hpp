/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSETFMATRIX_HPP
#define TSETFMATRIX_HPP

namespace pto {
template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL>
PTO_INTERNAL void TSETFMATRIX_IMPL(ConvTileData &src)
{
    if constexpr (FmatrixMode == SetFmatrixMode::FMATRIX_A_MANUAL || FmatrixMode == SetFmatrixMode::FMATRIX_B_MANUAL) {
        uint64_t regFmatrix = 0;
        constexpr uint32_t l1ShiftBit = 16;
        regFmatrix |= uint64_t(src.GetFmapW() & 0xFFFF);
        regFmatrix |= uint64_t(src.GetFmapH() & 0xFFFF) << l1ShiftBit;

        constexpr uint32_t padListShiftBit = 8;
        constexpr uint32_t padListShiftBase = 32;
        constexpr uint32_t padNumber = 4;

        for (uint32_t i = 0; i < padNumber; i++) {
            regFmatrix |= uint64_t(src.GetPadListArray()[i] & 0xFF) << (padListShiftBase + i * padListShiftBit);
        }
        if constexpr (FmatrixMode == SetFmatrixMode::FMATRIX_B_MANUAL) {
            set_fmatrix_b(regFmatrix);
        } else if constexpr (FmatrixMode == SetFmatrixMode::FMATRIX_A_MANUAL) {
            set_fmatrix(regFmatrix);
        }
    }
}
} // namespace pto
#endif // TSETFMATRIX_HPP
