/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TTRANS_COSTMODEL_HPP
#define TTRANS_COSTMODEL_HPP

#include "pto/costmodel/pto_isa_costmodel.hpp"

namespace pto {

// TTRANS: scatter_vnchwconv_b8/b16/b32 (PIPE_V) + copy_ubuf_to_ubuf (MTE1 → PIPE_V placeholder).
// See TTransOp.hpp for the full cycle model derivation.
template <typename DstTile, typename SrcTile, typename TmpTile>
PTO_INTERNAL void TTRANS_IMPL(DstTile &dst, SrcTile &src, TmpTile &tmp)
{
    using T = typename SrcTile::DType;
    auto stats = runTransOp(dst, src, tmp);
    dst.SetCycle(CostModel::GetInstance().VecInstPredictCycle<T>(stats));
}

} // namespace pto

#endif // TTRANS_COSTMODEL_HPP
