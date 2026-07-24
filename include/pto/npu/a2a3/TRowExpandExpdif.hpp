/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TROWEXPANDEXPDIF_HPP
#define TROWEXPANDEXPDIF_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a2a3/TRowExpandBinOp.hpp>

namespace pto {
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TROWEXPANDEXPDIF_IMPL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1)
{
    TROWEXPANDSUB_IMPL(dst, src0, src1);
    pipe_barrier(PIPE_V);
    TEXP_IMPL(dst, dst);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
PTO_INTERNAL void TROWEXPANDEXPDIF_IMPL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, TileDataTmp& tmp)
{
    TROWEXPANDSUB_IMPL(dst, src0, src1, tmp);
    pipe_barrier(PIPE_V);
    TEXP_IMPL(dst, dst);
}
} // namespace pto
#endif
