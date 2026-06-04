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
#include <acl/acl.h>

using namespace std;
using namespace pto;

template <typename T, int row, int validRow, int srcCol, int srcValidCol, int dstCol>
PTO_INTERNAL void runTRowMin(__gm__ T *out, __gm__ T *src)
{
    using DynDim2Shape = Shape<1, 1, 1, validRow, -1>;
    using DynDim2StrideSrc = pto::Stride<row * srcCol, row * srcCol, row * srcCol, srcCol, 1>;
    using DynDim2StrideDst = pto::Stride<row * dstCol, row * dstCol, row * dstCol, dstCol, 1>;

    using GlobalDataSrc = GlobalTensor<T, DynDim2Shape, DynDim2StrideSrc>;
    using GlobalDataDst = GlobalTensor<T, DynDim2Shape, DynDim2StrideDst>;
    GlobalDataSrc srcGlobal(src, DynDim2Shape(srcValidCol));
    GlobalDataDst dstGlobal(out, DynDim2Shape(dstCol));
    using srcTileData = Tile<TileType::Vec, T, row, srcCol, BLayout::RowMajor, validRow, srcValidCol>;
    using dstTileData = Tile<TileType::Vec, T, row, 16, BLayout::RowMajor, validRow, 1>;
    srcTileData srcTile;
    srcTileData tmpTile;
    dstTileData dstTile;
    TASSIGN<0x0>(srcTile);
    TASSIGN<row * srcCol * sizeof(T)>(tmpTile);
    TASSIGN<2 * row * srcCol * sizeof(T)>(dstTile);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TROWMIN(dstTile, srcTile, tmpTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int row, int validRow, int srcCol, int srcValidCol, int dstCol>
PTO_INTERNAL void runTRowMinDNDst(__gm__ T *out, __gm__ T *src)
{
    using ValidSrcShape = TileShape2D<T, validRow, srcValidCol>;
    using NDSrcShape = BaseShape2D<T, row, srcCol>;
    using GlobalDataSrc = GlobalTensor<T, ValidSrcShape, NDSrcShape>;
    GlobalDataSrc srcGlobal(src);

    using ValidDstShape = TileShape2D<T, dstCol, validRow>;
    using NDDstShape = BaseShape2D<T, row, dstCol>;
    using GlobalDataDst = GlobalTensor<T, ValidDstShape, NDDstShape>;
    GlobalDataDst dstGlobal(out);

    using srcTileData = Tile<TileType::Vec, T, row, srcCol, BLayout::RowMajor, row, srcCol>;
    using dstTileDataDN = Tile<TileType::Vec, T, row, 1, BLayout::ColMajor, row, 1>;
    srcTileData srcTile;
    srcTileData tmpTile;
    dstTileDataDN dstTile;
    TASSIGN<0x0>(srcTile);
    TASSIGN<row * srcCol * sizeof(T)>(tmpTile);
    TASSIGN<2 * row * srcCol * sizeof(T)>(dstTile);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TROWMIN(dstTile, srcTile, tmpTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    using dstTileDataND = Tile<TileType::Vec, T, 1, row, BLayout::RowMajor, 1, row>;
    dstTileDataND dstTileND;
    TRESHAPE(dstTileND, dstTile);
    TSTORE(dstGlobal, dstTileND);
}

extern "C" __global__ AICORE void launchTROWMINCase1(__gm__ float *out, __gm__ float *src)
{
    runTRowMin<float, 127, 127, 64, 63, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase2(__gm__ float *out, __gm__ float *src)
{
    runTRowMin<float, 63, 63, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase3(__gm__ float *out, __gm__ float *src)
{
    runTRowMin<float, 31, 31, 128, 127, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase4(__gm__ float *out, __gm__ float *src)
{
    runTRowMin<float, 15, 15, 192, 192, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase5(__gm__ float *out, __gm__ float *src)
{
    runTRowMin<float, 7, 7, 448, 447, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase6(__gm__ half *out, __gm__ half *src)
{
    runTRowMin<half, 256, 256, 16, 15, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase7(__gm__ float *out, __gm__ float *src)
{
    runTRowMinDNDst<float, 64, 64, 128, 128, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase8(__gm__ float *out, __gm__ float *src)
{
    runTRowMinDNDst<float, 32, 32, 256, 256, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase9(__gm__ float *out, __gm__ float *src)
{
    runTRowMinDNDst<float, 16, 16, 512, 512, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase10(__gm__ float *out, __gm__ float *src)
{
    runTRowMinDNDst<float, 8, 8, 1024, 1024, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase11(__gm__ int32_t *out, __gm__ int32_t *src)
{
    runTRowMin<int32_t, 63, 63, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase12(__gm__ int16_t *out, __gm__ int16_t *src)
{
    runTRowMin<int16_t, 15, 15, 192, 192, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase13(__gm__ half *out, __gm__ half *src)
{
    runTRowMinDNDst<half, 64, 64, 128, 128, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase14(__gm__ int32_t *out, __gm__ int32_t *src)
{
    runTRowMinDNDst<int32_t, 32, 32, 256, 256, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWMINCase15(__gm__ int16_t *out, __gm__ int16_t *src)
{
    runTRowMinDNDst<int16_t, 16, 16, 512, 512, 1>(out, src);
}

template <uint32_t caseId>
void launchTROWMINTestCase(void *out, void *src, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTROWMINCase1<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 2: {
            launchTROWMINCase2<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 3: {
            launchTROWMINCase3<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 4: {
            launchTROWMINCase4<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 5: {
            launchTROWMINCase5<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 6: {
            launchTROWMINCase6<<<1, nullptr, stream>>>((half *)out, (half *)src);
            break;
        }
        case 7: {
            launchTROWMINCase7<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 8: {
            launchTROWMINCase8<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 9: {
            launchTROWMINCase9<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 10: {
            launchTROWMINCase10<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 11: {
            launchTROWMINCase11<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src);
            break;
        }
        case 12: {
            launchTROWMINCase12<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src);
            break;
        }
        case 13: {
            launchTROWMINCase13<<<1, nullptr, stream>>>((half *)out, (half *)src);
            break;
        }
        case 14: {
            launchTROWMINCase14<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src);
            break;
        }
        case 15: {
            launchTROWMINCase15<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src);
            break;
        }
        default: {
        }
    }
}

template void launchTROWMINTestCase<1>(void *out, void *src, aclrtStream stream);
template void launchTROWMINTestCase<2>(void *out, void *src, aclrtStream stream);
template void launchTROWMINTestCase<3>(void *out, void *src, aclrtStream stream);
template void launchTROWMINTestCase<4>(void *out, void *src, aclrtStream stream);
template void launchTROWMINTestCase<5>(void *out, void *src, aclrtStream stream);
template void launchTROWMINTestCase<6>(void *out, void *src, aclrtStream stream);
template void launchTROWMINTestCase<7>(void *out, void *src, aclrtStream stream);
template void launchTROWMINTestCase<8>(void *out, void *src, aclrtStream stream);
template void launchTROWMINTestCase<9>(void *out, void *src, aclrtStream stream);
template void launchTROWMINTestCase<10>(void *out, void *src, aclrtStream stream);
template void launchTROWMINTestCase<11>(void *out, void *src, aclrtStream stream);
template void launchTROWMINTestCase<12>(void *out, void *src, aclrtStream stream);
template void launchTROWMINTestCase<13>(void *out, void *src, aclrtStream stream);
template void launchTROWMINTestCase<14>(void *out, void *src, aclrtStream stream);
template void launchTROWMINTestCase<15>(void *out, void *src, aclrtStream stream);
