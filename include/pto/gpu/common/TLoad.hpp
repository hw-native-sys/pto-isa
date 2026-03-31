/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_GPU_COMMON_TLOAD_HPP
#define PTO_GPU_COMMON_TLOAD_HPP

#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>
#include "pto/gpu/common/tile_offsets.hpp"

namespace pto {

template <typename TileData>
PTO_INTERNAL typename TileData::DType GpuPadValue()
{
    return GetPadValue<TileData>();
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLOAD_IMPL(TileData &dst, GlobalData &src)
{
    static_assert(GlobalData::layout == pto::Layout::ND || GlobalData::layout == pto::Layout::DN,
                  "Only ND and DN GlobalTensor layouts are supported by the v1 GPU backend.");

    const int gShape0 = src.GetShape(pto::GlobalTensorDim::DIM_0);
    const int gShape1 = src.GetShape(pto::GlobalTensorDim::DIM_1);
    const int gShape2 = src.GetShape(pto::GlobalTensorDim::DIM_2);
    const int gShape3 = src.GetShape(pto::GlobalTensorDim::DIM_3);
    const int gShape4 = src.GetShape(pto::GlobalTensorDim::DIM_4);

    const int gStride0 = src.GetStride(pto::GlobalTensorDim::DIM_0);
    const int gStride1 = src.GetStride(pto::GlobalTensorDim::DIM_1);
    const int gStride2 = src.GetStride(pto::GlobalTensorDim::DIM_2);
    const int gStride3 = src.GetStride(pto::GlobalTensorDim::DIM_3);
    const int gStride4 = src.GetStride(pto::GlobalTensorDim::DIM_4);

    const int validRow = dst.GetValidRow();
    const int validCol = dst.GetValidCol();

    for (int r = 0; r < TileData::Rows; ++r) {
        for (int c = 0; c < TileData::Cols; ++c) {
            dst.data()[gpu::GetTileElementOffset<TileData>(r, c)] = GpuPadValue<TileData>();
        }
    }

    if constexpr (GlobalData::layout == pto::Layout::ND) {
        PTO_ASSERT(TileData::isRowMajor, "Fix: ND loads currently require row-major tiles in the GPU backend.");
        PTO_ASSERT(gShape0 * gShape1 * gShape2 * gShape3 == validRow && gShape4 == validCol,
                   "Fix: ND TLOAD valid shape mismatch in GPU backend.");

        int rowBase = 0;
        for (int i = 0; i < gShape0; ++i) {
            for (int j = 0; j < gShape1; ++j) {
                for (int k = 0; k < gShape2; ++k) {
                    for (int r = 0; r < gShape3; ++r) {
                        const int tileRow = rowBase + r;
                        const int srcBase = i * gStride0 + j * gStride1 + k * gStride2 + r * gStride3;
                        for (int c = 0; c < gShape4; ++c) {
                            const std::size_t dstIdx = gpu::GetTileElementOffset<TileData>(tileRow, c);
                            dst.data()[dstIdx] = src.data()[srcBase + c * gStride4];
                        }
                    }
                    rowBase += gShape3;
                }
            }
        }
    } else {
        PTO_ASSERT(!TileData::isRowMajor, "Fix: DN loads currently require col-major tiles in the GPU backend.");
        PTO_ASSERT(gShape0 * gShape1 * gShape2 * gShape4 == validCol && gShape3 == validRow,
                   "Fix: DN TLOAD valid shape mismatch in GPU backend.");

        int colBase = 0;
        for (int i = 0; i < gShape0; ++i) {
            for (int j = 0; j < gShape1; ++j) {
                for (int k = 0; k < gShape2; ++k) {
                    for (int c = 0; c < gShape4; ++c) {
                        const int tileCol = colBase + c;
                        const int srcBase = i * gStride0 + j * gStride1 + k * gStride2 + c * gStride4;
                        for (int r = 0; r < gShape3; ++r) {
                            const std::size_t dstIdx = gpu::GetTileElementOffset<TileData>(r, tileCol);
                            dst.data()[dstIdx] = src.data()[srcBase + r * gStride3];
                        }
                    }
                    colBase += gShape4;
                }
            }
        }
    }
}

} // namespace pto

#endif
