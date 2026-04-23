/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TGET_ASYNC_COMMON_DETAIL_HPP
#define PTO_COMM_TGET_ASYNC_COMMON_DETAIL_HPP

#include "pto/common/constants.hpp"
#include "pto/common/type.hpp"
#include "pto/common/debug.h"
#include "pto/comm/comm_types.hpp"
#include "pto/comm/async_common/async_types.hpp"
#include "pto/npu/comm/async/sdma/sdma_async_intrin.hpp"

namespace pto {
namespace comm {
namespace detail {

template <typename GlobalData>
PTO_INTERNAL bool TGetAsyncIsFlatContiguous1D(GlobalData &globalData)
{
    const int shp0 = globalData.GetShape(GlobalTensorDim::DIM_0);
    const int shp1 = globalData.GetShape(GlobalTensorDim::DIM_1);
    const int shp2 = globalData.GetShape(GlobalTensorDim::DIM_2);
    const int shp3 = globalData.GetShape(GlobalTensorDim::DIM_3);
    const int shp4 = globalData.GetShape(GlobalTensorDim::DIM_4);

    const int step0 = globalData.GetStride(GlobalTensorDim::DIM_0);
    const int step1 = globalData.GetStride(GlobalTensorDim::DIM_1);
    const int step2 = globalData.GetStride(GlobalTensorDim::DIM_2);
    const int step3 = globalData.GetStride(GlobalTensorDim::DIM_3);
    const int step4 = globalData.GetStride(GlobalTensorDim::DIM_4);

    const bool packedLayout = (step4 == 1) && (step3 == shp4) && (step2 == shp3 * step3) && (step1 == shp2 * step2) &&
                              (step0 == shp1 * step1);
    const bool oneDimLogical = (shp0 == 1 && shp1 == 1 && shp2 == 1 && shp3 == 1);
    return packedLayout && oneDimLogical;
}

template <typename GlobalData>
PTO_INTERNAL uint32_t TGetAsyncGetTotalElemCount(GlobalData &globalData)
{
    const uint32_t d0 = static_cast<uint32_t>(globalData.GetShape(GlobalTensorDim::DIM_0));
    const uint32_t d1 = static_cast<uint32_t>(globalData.GetShape(GlobalTensorDim::DIM_1));
    const uint32_t d2 = static_cast<uint32_t>(globalData.GetShape(GlobalTensorDim::DIM_2));
    const uint32_t d3 = static_cast<uint32_t>(globalData.GetShape(GlobalTensorDim::DIM_3));
    const uint32_t d4 = static_cast<uint32_t>(globalData.GetShape(GlobalTensorDim::DIM_4));
    return (((d0 * d1) * d2) * d3) * d4;
}

template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL bool TGetAsyncCheckTensorCompatibility()
{
    using SrcElem = typename GlobalSrcData::RawDType;
    static_assert(std::is_same_v<SrcElem, typename GlobalDstData::RawDType>,
                  "TGET_ASYNC: src/dst element type mismatch");
    static_assert(GlobalSrcData::layout == GlobalDstData::layout, "TGET_ASYNC: src/dst layout mismatch");
    return true;
}

template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent TGET_ASYNC_SDMA_IMPL(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                                             const sdma::SdmaExecContext &execCtx)
{
    (void)TGetAsyncCheckTensorCompatibility<GlobalDstData, GlobalSrcData>();

    PTO_ASSERT(dstGlobalData.data() != nullptr && srcGlobalData.data() != nullptr,
               "TGET_ASYNC: src and dst tensor pointers must not be null.");

    PTO_ASSERT(TGetAsyncIsFlatContiguous1D(srcGlobalData),
               "TGET_ASYNC: src tensor must be flat contiguous 1D (packed layout, single logical line). "
               "Multi-dimensional or non-contiguous tensors are not supported by SDMA async path.");
    PTO_ASSERT(TGetAsyncIsFlatContiguous1D(dstGlobalData),
               "TGET_ASYNC: dst tensor must be flat contiguous 1D (packed layout, single logical line). "
               "Multi-dimensional or non-contiguous tensors are not supported by SDMA async path.");

    const uint32_t srcElems = TGetAsyncGetTotalElemCount(srcGlobalData);
    const uint32_t dstElems = TGetAsyncGetTotalElemCount(dstGlobalData);
    PTO_ASSERT(dstElems >= srcElems, "TGET_ASYNC SDMA: dst buffer too small for src data.");

    using T = typename GlobalSrcData::RawDType;
    const uint64_t eventHandle =
        sdma::__sdma_get_async(dstGlobalData.data(), srcGlobalData.data(), srcElems * sizeof(T), execCtx);
    return AsyncEvent(eventHandle, DmaEngine::SDMA);
}

} // namespace detail
} // namespace comm
} // namespace pto

#endif // PTO_COMM_TGET_ASYNC_COMMON_DETAIL_HPP
