/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <acl/acl.h>

using namespace std;
using namespace pto;

template <typename T, int DstRow, int DstCol, int SrcRow, int SrcCol, pto::MaskPattern maskPattern>
__global__ AICORE void runTGatherMask(__gm__ T *out, __gm__ T *src)
{
    using SrcShapeDim5 = Shape<1, 1, 1, SrcRow, SrcCol>;
    using SrcStridDim5 = pto::Stride<SrcRow * SrcCol, SrcRow * SrcCol, SrcRow * SrcCol, SrcCol, 1>;
    using GlobalSrcData = GlobalTensor<T, SrcShapeDim5, SrcStridDim5>;
    using DstShapeDim5 = Shape<1, 1, 1, DstRow, DstCol>;
    using DstStridDim5 = pto::Stride<DstRow * DstCol, DstRow * DstCol, DstRow * DstCol, DstCol, 1>;
    using GlobalDstData = GlobalTensor<T, DstShapeDim5, DstStridDim5>;

    GlobalSrcData srcGlobal(src);
    GlobalDstData dstGlobal(out);

    using DstTileData = Tile<TileType::Vec, T, DstRow, DstCol>;
    using SrcTileData = Tile<TileType::Vec, T, SrcRow, SrcCol>;

    SrcTileData srcTile;
    DstTileData dstTile;
    TASSIGN<0x0>(srcTile);
    TASSIGN<SrcTileData::Numel * sizeof(T)>(dstTile);

    TLOAD(srcTile, srcGlobal);
    TLOAD(dstTile, dstGlobal);
    PtoSetWaitFlag<PIPE_MTE2, PIPE_V>();
    TGATHER<DstTileData, SrcTileData, maskPattern, GatherAxis::GATHER_COL>(dstTile, srcTile);
    PtoSetWaitFlag<PIPE_V, PIPE_MTE3>();
    TSTORE(dstGlobal, dstTile);
}

template <typename T, int DstRow, int DstCol, int SrcRow, int SrcCol, pto::MaskPattern mask>
void launchTGatherMaskTestCase(void *out, void *src, void *stream)
{
    if constexpr (std::is_same_v<T, uint16_t>) {
        runTGatherMask<half, DstRow, DstCol, SrcRow, SrcCol, mask><<<1, nullptr, stream>>>((half *)out, (half *)src);
    } else {
        runTGatherMask<T, DstRow, DstCol, SrcRow, SrcCol, mask><<<1, nullptr, stream>>>((T *)out, (T *)src);
    }
}

template void launchTGatherMaskTestCase<uint16_t, 16, 64, 16, 64, pto::MaskPattern::P1111>(void *out, void *src,
                                                                                           void *stream);
template void launchTGatherMaskTestCase<float, 16, 64, 16, 64, pto::MaskPattern::P1111>(void *out, void *src,
                                                                                        void *stream);
template void launchTGatherMaskTestCase<int32_t, 16, 64, 16, 64, pto::MaskPattern::P1111>(void *out, void *src,
                                                                                          void *stream);

template void launchTGatherMaskTestCase<uint16_t, 16, 128, 32, 64, pto::MaskPattern::P1010>(void *out, void *src,
                                                                                            void *stream);
template void launchTGatherMaskTestCase<uint16_t, 16, 128, 32, 64, pto::MaskPattern::P0101>(void *out, void *src,
                                                                                            void *stream);
template void launchTGatherMaskTestCase<float, 16, 128, 32, 64, pto::MaskPattern::P1010>(void *out, void *src,
                                                                                         void *stream);
template void launchTGatherMaskTestCase<float, 16, 128, 32, 64, pto::MaskPattern::P0101>(void *out, void *src,
                                                                                         void *stream);
template void launchTGatherMaskTestCase<int32_t, 16, 128, 32, 64, pto::MaskPattern::P1010>(void *out, void *src,
                                                                                           void *stream);
template void launchTGatherMaskTestCase<int32_t, 16, 128, 32, 64, pto::MaskPattern::P0101>(void *out, void *src,
                                                                                           void *stream);

template void launchTGatherMaskTestCase<uint16_t, 4, 256, 16, 64, pto::MaskPattern::P1000>(void *out, void *src,
                                                                                           void *stream);
template void launchTGatherMaskTestCase<uint16_t, 4, 256, 16, 64, pto::MaskPattern::P0100>(void *out, void *src,
                                                                                           void *stream);
template void launchTGatherMaskTestCase<uint16_t, 4, 256, 16, 64, pto::MaskPattern::P0010>(void *out, void *src,
                                                                                           void *stream);
template void launchTGatherMaskTestCase<uint16_t, 4, 256, 16, 64, pto::MaskPattern::P0001>(void *out, void *src,
                                                                                           void *stream);
template void launchTGatherMaskTestCase<float, 4, 256, 16, 64, pto::MaskPattern::P1000>(void *out, void *src,
                                                                                        void *stream);
template void launchTGatherMaskTestCase<float, 4, 256, 16, 64, pto::MaskPattern::P0100>(void *out, void *src,
                                                                                        void *stream);
template void launchTGatherMaskTestCase<float, 4, 256, 16, 64, pto::MaskPattern::P0010>(void *out, void *src,
                                                                                        void *stream);
template void launchTGatherMaskTestCase<float, 4, 256, 16, 64, pto::MaskPattern::P0001>(void *out, void *src,
                                                                                        void *stream);
template void launchTGatherMaskTestCase<int32_t, 4, 256, 16, 64, pto::MaskPattern::P1000>(void *out, void *src,
                                                                                          void *stream);
template void launchTGatherMaskTestCase<int32_t, 4, 256, 16, 64, pto::MaskPattern::P0100>(void *out, void *src,
                                                                                          void *stream);
template void launchTGatherMaskTestCase<int32_t, 4, 256, 16, 64, pto::MaskPattern::P0010>(void *out, void *src,
                                                                                          void *stream);
template void launchTGatherMaskTestCase<int32_t, 4, 256, 16, 64, pto::MaskPattern::P0001>(void *out, void *src,
                                                                                          void *stream);
