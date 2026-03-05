/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TBROADCAST_HPP
#define TBROADCAST_HPP

#include <pto/common/pto_tile.hpp>
#include <type_traits>

namespace pto {
namespace comm {
template <typename GlobalData>
void Copy(typename GlobalData::DType *dst, typename GlobalData::DType *src, long int shape[], long int stride[])
{
    for (size_t i = 0; i < shape[0]; i++) {
        for (size_t j = 0; j < shape[1]; j++) {
            for (size_t k = 0; k < shape[2]; k++) {
                for (size_t l = 0; l < shape[3]; l++) {
                    for (size_t m = 0; m < shape[4]; m++) {
                        int index = i * stride[0] + j * stride[1] + k * stride[2] + l * stride[3] + m * stride[4];
                        dst[index] = src[index];
                    }
                }
            }
        }
    }
}

template <typename ParallelGroupType, typename GlobalData>
PTO_INTERNAL void TBroadcast_Impl(ParallelGroupType &parallelGroup, GlobalData &src)
{
    long int shape[5] = {src.GetShape(0), src.GetShape(1), src.GetShape(2), src.GetShape(3), src.GetShape(4)};
    long int stride[5] = {src.GetStride(0), src.GetStride(1), src.GetStride(2), src.GetStride(3), src.GetStride(4)};
    int groupSize = parallelGroup.GetSize();
    for (unsigned n = 0; n < groupSize; ++n) {
        GlobalData &member = parallelGroup[n];
        Copy<GlobalData>(member.data(), src.data(), shape, stride);
    }
}

template <typename ParallelGroupType, typename GlobalSrcData, typename TileData>
PTO_INTERNAL void TBROADCAST_IMPL(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                                  TileData &stagingTileData)
{
    TBroadcast_Impl<ParallelGroupType, GlobalSrcData>(parallelGroup, srcGlobalData);
}

template <typename ParallelGroupType, typename GlobalSrcData, typename TileData>
PTO_INTERNAL void TBROADCAST_IMPL(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData, TileData &pingTile,
                                  TileData &pongTile)
{
    TBroadcast_Impl<ParallelGroupType, GlobalSrcData>(parallelGroup, srcGlobalData);
}

} // namespace comm
} // namespace pto

#endif
