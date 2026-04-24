/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef SETFMATRIX_CPU_HPP
#define SETFMATRIX_CPU_HPP

namespace pto {
template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL>
PTO_INTERNAL void SETFMATRIX_IMPL(ConvTileData &src)
{
    (void)FmatrixMode;
    PTO_CPU_ASSERT(src.GetFmapH() > 0 && src.GetFmapW() > 0, "Fix: SETFMATRIX requires non-zero fmap size.");
}
} // namespace pto

#endif
