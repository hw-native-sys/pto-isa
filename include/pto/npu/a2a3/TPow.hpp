/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPOW_HPP
#define TPOW_HPP

namespace pto {
template <PowAlgorithm algo, typename DstTile, typename BaseTile, typename ExpTile, typename TmpTile>
PTO_INTERNAL void TPOW_IMPL(DstTile &dst, BaseTile &base, ExpTile &exp, TmpTile &tmp)
{}

template <PowAlgorithm algo, typename DstTile, typename BaseTile, typename TmpTile>
PTO_INTERNAL void TPOWS_IMPL(DstTile &dst, BaseTile &base, typename DstTile::DType exp, TmpTile &tmp)
{}

} // namespace pto

#endif
