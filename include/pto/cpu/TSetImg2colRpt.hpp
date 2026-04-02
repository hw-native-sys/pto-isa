/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TSET_IMG2COL_RPT_CPU_HPP
#define TSET_IMG2COL_RPT_CPU_HPP

namespace pto {
template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL>
PTO_INTERNAL void TSET_IMG2COL_RPT_IMPL(ConvTileData &src)
{
    (void)FmatrixMode;
    PTO_CPU_ASSERT(src.GetRepeatTime() >= 0, "Fix: TSET_IMG2COL_RPT metadata must be initialized.");
}
} // namespace pto

#endif