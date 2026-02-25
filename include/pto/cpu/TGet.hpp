/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TGET_HPP
#define TGET_HPP

#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"

namespace pto {
namespace comm {

template <typename GlobalDstData, typename GlobalSrcData>
void TGet_Impl(typename GlobalDstData::DType *dst, typename GlobalSrcData::DType *src, long int shape[],
               long int stride[])
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

template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL void Copy_Data(GlobalDstData &dst, GlobalSrcData &src)
{
    long int shape[5] = {dst.GetShape(0), dst.GetShape(1), dst.GetShape(2), dst.GetShape(3), dst.GetShape(4)};
    long int stride[5] = {dst.GetStride(0), dst.GetStride(1), dst.GetStride(2), dst.GetStride(3), dst.GetStride(4)};
    TGet_Impl<GlobalDstData, GlobalSrcData>(dst.data(), src.data(), shape, stride);
}

template <typename GlobalDstData, typename GlobalSrcData, typename TileData>
PTO_INTERNAL void TGET_IMPL(GlobalDstData &dst, GlobalSrcData &src, TileData &src1)
{
    Copy_Data(dst, src);
}

template <typename GlobalDstData, typename GlobalSrcData, typename TileData>
PTO_INTERNAL void TGET_IMPL(GlobalDstData &dst, GlobalSrcData &src, TileData &ping, TileData &pong)
{
    Copy_Data(dst, src);
}

template <typename GlobalDstData, typename GlobalSrcData, typename TileData>
PTO_INTERNAL void TPUT_IMPL(GlobalDstData &dst, GlobalSrcData &src, TileData &src1)
{
    Copy_Data(src, dst);
}

template <typename GlobalDstData, typename GlobalSrcData, typename TileData>
PTO_INTERNAL void TPUT_IMPL(GlobalDstData &dst, GlobalSrcData &src, TileData &src1, AtomicType &atomicType)
{
    Copy_Data(src, dst);
}

template <typename GlobalDstData, typename GlobalSrcData, typename TileData>
PTO_INTERNAL void TPUT_IMPL(GlobalDstData &dst, GlobalSrcData &src, TileData &ping, TileData &pong)
{
    Copy_Data(src, dst);
}

template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL void TGET_ASYNC_IMPL(GlobalDstData &dst, GlobalSrcData &src)
{
    Copy_Data(dst, src);
}

template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL void TPUT_ASYNC_IMPL(GlobalDstData &dst, GlobalSrcData &src)
{
    Copy_Data(src, dst);
}

} // namespace comm
} // namespace pto
#endif
