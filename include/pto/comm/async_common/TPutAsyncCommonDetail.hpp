/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TPUT_ASYNC_COMMON_DETAIL_HPP
#define PTO_COMM_TPUT_ASYNC_COMMON_DETAIL_HPP

#include "pto/common/debug.h"
#include "pto/common/type.hpp"
#include "pto/common/constants.hpp"
#include "pto/comm/comm_types.hpp"
#include "pto/comm/async_common/async_types.hpp"
#include "pto/npu/comm/async/sdma/sdma_async_intrin.hpp"

namespace pto {
namespace comm {
namespace detail {

template <typename GlobalData>
PTO_INTERNAL bool TPutAsyncIsFlatContiguous1D(GlobalData &globalData)
{
    const int dim0 = globalData.GetShape(GlobalTensorDim::DIM_0);
    const int dim1 = globalData.GetShape(GlobalTensorDim::DIM_1);
    const int dim2 = globalData.GetShape(GlobalTensorDim::DIM_2);
    const int dim3 = globalData.GetShape(GlobalTensorDim::DIM_3);
    const int dim4 = globalData.GetShape(GlobalTensorDim::DIM_4);

    const int pitch0 = globalData.GetStride(GlobalTensorDim::DIM_0);
    const int pitch1 = globalData.GetStride(GlobalTensorDim::DIM_1);
    const int pitch2 = globalData.GetStride(GlobalTensorDim::DIM_2);
    const int pitch3 = globalData.GetStride(GlobalTensorDim::DIM_3);
    const int pitch4 = globalData.GetStride(GlobalTensorDim::DIM_4);

    const bool hasPackedLayout = (pitch4 == 1) && (pitch3 == dim4) && (pitch2 == dim3 * pitch3) &&
                                 (pitch1 == dim2 * pitch2) && (pitch0 == dim1 * pitch1);
    const bool isSingleLine = (dim0 == 1 && dim1 == 1 && dim2 == 1 && dim3 == 1);
    return hasPackedLayout && isSingleLine;
}

template <typename GlobalData>
PTO_INTERNAL uint32_t TPutAsyncGetTotalElemCount(GlobalData &globalData)
{
    const uint32_t d0 = static_cast<uint32_t>(globalData.GetShape(GlobalTensorDim::DIM_0));
    const uint32_t d1 = static_cast<uint32_t>(globalData.GetShape(GlobalTensorDim::DIM_1));
    const uint32_t d2 = static_cast<uint32_t>(globalData.GetShape(GlobalTensorDim::DIM_2));
    const uint32_t d3 = static_cast<uint32_t>(globalData.GetShape(GlobalTensorDim::DIM_3));
    const uint32_t d4 = static_cast<uint32_t>(globalData.GetShape(GlobalTensorDim::DIM_4));
    return (((d0 * d1) * d2) * d3) * d4;
}

template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL bool TPutAsyncCheckTensorCompatibility()
{
    using SrcElem = typename GlobalSrcData::RawDType;
    static_assert(std::is_same_v<SrcElem, typename GlobalDstData::RawDType>,
                  "TPUT_ASYNC: src/dst element type mismatch");
    static_assert(GlobalSrcData::layout == GlobalDstData::layout, "TPUT_ASYNC: src/dst layout mismatch");
    return true;
}

template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL AsyncEvent TPUT_ASYNC_SDMA_IMPL(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                                             const sdma::SdmaExecContext &execCtx)
{
    (void)TPutAsyncCheckTensorCompatibility<GlobalDstData, GlobalSrcData>();

    PTO_ASSERT(srcGlobalData.data() != nullptr && dstGlobalData.data() != nullptr,
               "TPUT_ASYNC: src and dst tensor pointers must not be null.");

    PTO_ASSERT(TPutAsyncIsFlatContiguous1D(srcGlobalData),
               "TPUT_ASYNC: src tensor must be flat contiguous 1D (packed layout, single logical line). "
               "Multi-dimensional or non-contiguous tensors are not supported by SDMA async path.");
    PTO_ASSERT(TPutAsyncIsFlatContiguous1D(dstGlobalData),
               "TPUT_ASYNC: dst tensor must be flat contiguous 1D (packed layout, single logical line). "
               "Multi-dimensional or non-contiguous tensors are not supported by SDMA async path.");

    const uint32_t dstElems = TPutAsyncGetTotalElemCount(dstGlobalData);
    const uint32_t srcElems = TPutAsyncGetTotalElemCount(srcGlobalData);
    PTO_ASSERT(dstElems >= srcElems, "TPUT_ASYNC SDMA: dst buffer too small for src data.");

    using T = typename GlobalSrcData::RawDType;
    const uint64_t eventHandle =
        sdma::__sdma_put_async(dstGlobalData.data(), srcGlobalData.data(), srcElems * sizeof(T), execCtx);
    return AsyncEvent(eventHandle, DmaEngine::SDMA);
}

} // namespace detail
} // namespace comm
} // namespace pto

#endif // PTO_COMM_TPUT_ASYNC_COMMON_DETAIL_HPP
