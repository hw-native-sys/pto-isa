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
template <typename ConvTileData>
PTO_INTERNAL void SET_IMG2COL_RPT_IMPL(ConvTileData &src)
{
    uint64_t rptConfig = static_cast<uint64_t>(src.GetRepeatStride()) |
                         (static_cast<uint64_t>(src.GetRepeatTime()) << 16) |
                         (static_cast<uint64_t>(src.GetRepeatMode()) << 24);
    set_l3d_rpt(rptConfig);
}
} // namespace pto
#endif // SET_IMG2COL_RPT_HPP
