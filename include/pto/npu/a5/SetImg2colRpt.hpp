/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef SET_IMG2COL_RPT_HPP
#define SET_IMG2COL_RPT_HPP

namespace pto {
template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL>
PTO_INTERNAL void SET_IMG2COL_RPT_IMPL(ConvTileData &src)
{
    if constexpr (FmatrixMode == SetFmatrixMode::FMATRIX_A_MANUAL || FmatrixMode == SetFmatrixMode::FMATRIX_B_MANUAL) {
        constexpr uint32_t repeatTimeShiftBit = 16;
        constexpr uint32_t repeatModeShiftBit = 24;
        constexpr uint32_t dstStrideShiftBit = 32;
        constexpr uint32_t dstMpositionShiftBit = 48;
        uint64_t rptConfig = 0;
        rptConfig |= uint64_t(src.GetRepeatStride());
        rptConfig |= uint64_t(src.GetRepeatTime()) << repeatTimeShiftBit;
        rptConfig |= uint64_t(src.GetRepeatMode()) << repeatModeShiftBit;
        rptConfig |= uint64_t(src.GetDstStride()) << dstStrideShiftBit;
        rptConfig |= uint64_t(src.GetDstMposition()) << dstMpositionShiftBit;
        if constexpr (FmatrixMode == SetFmatrixMode::FMATRIX_A_MANUAL) {
            set_l3d_rpt(rptConfig);
        } else if constexpr (FmatrixMode == SetFmatrixMode::FMATRIX_B_MANUAL) {
            set_l3d_rpt_b(rptConfig);
        }
    }
}
} // namespace pto
#endif // SET_IMG2COL_RPT_HPP
