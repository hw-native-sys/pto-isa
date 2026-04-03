/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_GPU_COMMON_TSTORE_HPP
#define PTO_GPU_COMMON_TSTORE_HPP

#include <pto/common/pto_tile.hpp>
#include "pto/gpu/common/tile_offsets.hpp"

namespace pto {

template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone>
PTO_INTERNAL void TSTORE_IMPL(GlobalData &dst, TileData &src)
{
    static_assert(atomicType == AtomicType::AtomicNone,
                  "Atomic store modes are not implemented yet in the v1 GPU backend.");
    static_assert(GlobalData::layout == pto::Layout::ND || GlobalData::layout == pto::Layout::DN,
                  "Only ND and DN GlobalTensor layouts are supported by the v1 GPU backend.");

    const int gShape0 = dst.GetShape(pto::GlobalTensorDim::DIM_0);
    const int gShape1 = dst.GetShape(pto::GlobalTensorDim::DIM_1);
    const int gShape2 = dst.GetShape(pto::GlobalTensorDim::DIM_2);
    const int gShape3 = dst.GetShape(pto::GlobalTensorDim::DIM_3);
    const int gShape4 = dst.GetShape(pto::GlobalTensorDim::DIM_4);

    const int gStride0 = dst.GetStride(pto::GlobalTensorDim::DIM_0);
    const int gStride1 = dst.GetStride(pto::GlobalTensorDim::DIM_1);
    const int gStride2 = dst.GetStride(pto::GlobalTensorDim::DIM_2);
    const int gStride3 = dst.GetStride(pto::GlobalTensorDim::DIM_3);
    const int gStride4 = dst.GetStride(pto::GlobalTensorDim::DIM_4);

    const int validRow = src.GetValidRow();
    const int validCol = src.GetValidCol();

    if constexpr (GlobalData::layout == pto::Layout::ND) {
        PTO_ASSERT(TileData::isRowMajor, "Fix: ND stores currently require row-major tiles in the GPU backend.");
        PTO_ASSERT(gShape0 * gShape1 * gShape2 * gShape3 == validRow && gShape4 == validCol,
                   "Fix: ND TSTORE valid shape mismatch in GPU backend.");

        int rowBase = 0;
        for (int i = 0; i < gShape0; ++i) {
            for (int j = 0; j < gShape1; ++j) {
                for (int k = 0; k < gShape2; ++k) {
                    for (int r = 0; r < gShape3; ++r) {
                        const int tileRow = rowBase + r;
                        const int dstBase = i * gStride0 + j * gStride1 + k * gStride2 + r * gStride3;
                        for (int c = 0; c < gShape4; ++c) {
                            const std::size_t srcIdx = gpu::GetTileElementOffset<TileData>(tileRow, c);
                            dst.data()[dstBase + c * gStride4] = src.data()[srcIdx];
                        }
                    }
                    rowBase += gShape3;
                }
            }
        }
    } else {
        PTO_ASSERT(!TileData::isRowMajor, "Fix: DN stores currently require col-major tiles in the GPU backend.");
        PTO_ASSERT(gShape0 * gShape1 * gShape2 * gShape4 == validCol && gShape3 == validRow,
                   "Fix: DN TSTORE valid shape mismatch in GPU backend.");

        int colBase = 0;
        for (int i = 0; i < gShape0; ++i) {
            for (int j = 0; j < gShape1; ++j) {
                for (int k = 0; k < gShape2; ++k) {
                    for (int c = 0; c < gShape4; ++c) {
                        const int tileCol = colBase + c;
                        const int dstBase = i * gStride0 + j * gStride1 + k * gStride2 + c * gStride4;
                        for (int r = 0; r < gShape3; ++r) {
                            const std::size_t srcIdx = gpu::GetTileElementOffset<TileData>(r, tileCol);
                            dst.data()[dstBase + r * gStride3] = src.data()[srcIdx];
                        }
                    }
                    colBase += gShape4;
                }
            }
        }
    }
}

template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone>
PTO_INTERNAL void TSTORE_IMPL(GlobalData &dst, TileData &src, uint64_t preQuantScalar)
{
    (void)preQuantScalar;
    TSTORE_IMPL<TileData, GlobalData, atomicType>(dst, src);
}

template <typename TileData, typename GlobalData, typename FpTileData, AtomicType atomicType = AtomicType::AtomicNone,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TSTORE_IMPL(GlobalData &dst, TileData &src, FpTileData &fp)
{
    (void)fp;
    (void)reluPreMode;
    TSTORE_IMPL<TileData, GlobalData, atomicType>(dst, src);
}

} // namespace pto

#endif
