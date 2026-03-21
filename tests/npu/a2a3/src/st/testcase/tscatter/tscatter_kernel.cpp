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

template <typename Tsrc0, typename Tsrc1, int kGRows0_, int kGCols0_, int kGRows1_, int kGCols1_, int kTRows_,
          int kTCols_>
PTO_INTERNAL void runTScatter(__gm__ Tsrc0 __out__ *out, __gm__ Tsrc0 __in__ *src0, __gm__ Tsrc1 __in__ *src1)
{
    using DynShapeDim5_src0 = pto::Shape<1, 1, 1, kGRows0_, kGCols0_>;
    using DynStridDim5_src0 = pto::Stride<1, 1, 1, kGCols0_, 1>;
    using GlobalData_src0 = GlobalTensor<Tsrc0, DynShapeDim5_src0, DynStridDim5_src0>;

    using DynShapeDim5_src1 = pto::Shape<1, 1, 1, kGRows1_, kGCols1_>;
    using DynStridDim5_src1 = pto::Stride<1, 1, 1, kGCols1_, 1>;
    using GlobalData_src1 = GlobalTensor<Tsrc1, DynShapeDim5_src1, DynStridDim5_src1>;

    using DynShapeDim5_dst = pto::Shape<1, 1, 1, kGRows0_, kGCols0_>;
    using DynStridDim5_dst = pto::Stride<1, 1, 1, kGCols0_, 1>;
    using GlobalData_dst = GlobalTensor<Tsrc0, DynShapeDim5_dst, DynStridDim5_dst>;

    constexpr int src0_row = kGRows0_;
    constexpr int src0_col = kGCols0_;
    constexpr int src1_row = kGRows1_;
    constexpr int src1_col = kGCols1_;
    constexpr int dst_row = kGRows0_;
    constexpr int dst_col = kGCols0_;

    using TileData_src0 = Tile<TileType::Vec, Tsrc0, kGRows0_, kGCols0_, BLayout::RowMajor, -1, -1>;
    using TileData_src1 = Tile<TileType::Vec, Tsrc1, kGRows1_, kGCols1_, BLayout::RowMajor, -1, -1>;
    using TileData_dst = Tile<TileType::Vec, Tsrc0, kGRows0_, kGCols0_, BLayout::RowMajor, -1, -1>;
    TileData_src0 src0Tile(src0_row, src0_col);
    TileData_src1 src1Tile(src1_row, src1_col); // index
    TileData_dst dstTile(dst_row, dst_col);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x20000);
    TASSIGN(dstTile, 0x28000);

    GlobalData_src0 src0Global(src0);
    GlobalData_src1 src1Global(src1);
    GlobalData_dst dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_MTE2, PIPE_V>();
    PtoSetWaitFlag<PIPE_V, PIPE_S>();
#endif
    TSCATTER(dstTile, src0Tile, src1Tile);
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_S, PIPE_V>();
    PtoSetWaitFlag<PIPE_V, PIPE_MTE2>();
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTSCATTERCase1(__gm__ int16_t *out, __gm__ int16_t *src,
                                                      __gm__ uint16_t *indexes)
{
    runTScatter<int16_t, uint16_t, 2, 32, 1, 32, 2, 32>(out, src, indexes);
}
extern "C" __global__ AICORE void launchTSCATTERCase2(__gm__ half *out, __gm__ half *src, __gm__ uint16_t *indexes)
{
    runTScatter<half, uint16_t, 63, 64, 63, 64, 63, 63>(out, src, indexes);
}
extern "C" __global__ AICORE void launchTSCATTERCase3(__gm__ int32_t *out, __gm__ int32_t *src,
                                                      __gm__ uint32_t *indexes)
{
    runTScatter<int32_t, uint32_t, 31, 128, 31, 128, 31, 128>(out, src, indexes);
}
extern "C" __global__ AICORE void launchTSCATTERCase4(__gm__ int16_t *out, __gm__ int16_t *src, __gm__ int16_t *indexes)
{
    runTScatter<int16_t, int16_t, 15, 192, 15, 192, 15, 192>(out, src, indexes);
}
extern "C" __global__ AICORE void launchTSCATTERCase5(__gm__ float *out, __gm__ float *src, __gm__ int32_t *indexes)
{
    runTScatter<float, int32_t, 7, 448, 7, 448, 7, 448>(out, src, indexes);
}
extern "C" __global__ AICORE void launchTSCATTERCase6(__gm__ int8_t *out, __gm__ int8_t *src, __gm__ uint16_t *indexes)
{
    runTScatter<int8_t, uint16_t, 256, 32, 256, 32, 256, 32>(out, src, indexes);
}
extern "C" __global__ AICORE void launchTSCATTERCase7(__gm__ float *out, __gm__ float *src, __gm__ uint32_t *indexes)
{
    runTScatter<float, uint32_t, 32, 64, 32, 64, 32, 64>(out, src, indexes);
}

template <uint32_t caseId>
void launchTScatterTestCase(void *out, void *src, void *indexes, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTSCATTERCase1<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src, (uint16_t *)indexes);
            break;
        }
        case 2: {
            launchTSCATTERCase2<<<1, nullptr, stream>>>((half *)out, (half *)src, (uint16_t *)indexes);
            break;
        }
        case 3: {
            launchTSCATTERCase3<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src, (uint32_t *)indexes);
            break;
        }
        case 4: {
            launchTSCATTERCase4<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src, (int16_t *)indexes);
            break;
        }
        case 5: {
            launchTSCATTERCase5<<<1, nullptr, stream>>>((float *)out, (float *)src, (int32_t *)indexes);
            break;
        }
        case 6: {
            launchTSCATTERCase6<<<1, nullptr, stream>>>((int8_t *)out, (int8_t *)src, (uint16_t *)indexes);
            break;
        }
        case 7: {
            launchTSCATTERCase7<<<1, nullptr, stream>>>((float *)out, (float *)src, (uint32_t *)indexes);
            break;
        }
        default: {
        }
    }
}

template void launchTScatterTestCase<1>(void *out, void *src, void *indexes, aclrtStream stream);
template void launchTScatterTestCase<2>(void *out, void *src, void *indexes, aclrtStream stream);
template void launchTScatterTestCase<3>(void *out, void *src, void *indexes, aclrtStream stream);
template void launchTScatterTestCase<4>(void *out, void *src, void *indexes, aclrtStream stream);
template void launchTScatterTestCase<5>(void *out, void *src, void *indexes, aclrtStream stream);
template void launchTScatterTestCase<6>(void *out, void *src, void *indexes, aclrtStream stream);
template void launchTScatterTestCase<7>(void *out, void *src, void *indexes, aclrtStream stream);