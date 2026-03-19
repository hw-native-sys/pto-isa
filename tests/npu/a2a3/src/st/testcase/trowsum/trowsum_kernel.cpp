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

template <typename T, int row, int validRow, int srcCol, int srcValidCol, int dstCol>
PTO_INTERNAL void runTRowSum(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2StrideSrc = pto::Stride<1, 1, -1, -1, 1>;
    using DynDim2StrideDst = pto::Stride<1, 1, 1, -1, -1>;

    using GlobalDataSrc = GlobalTensor<T, DynDim2Shape, DynDim2StrideSrc>;
    using GlobalDataDst = GlobalTensor<T, DynDim2Shape, DynDim2StrideDst>;
    GlobalDataSrc srcGlobal(src, DynDim2Shape(validRow, srcValidCol), DynDim2StrideSrc(row, srcCol));
    GlobalDataDst dstGlobal(out, DynDim2Shape(validRow, dstCol), DynDim2StrideDst(dstCol, row));
    using srcTileData = Tile<TileType::Vec, T, row, srcCol, BLayout::RowMajor, -1, -1>;
    using dstTileData = Tile<TileType::Vec, T, row, 16, BLayout::RowMajor, -1, -1>;
    srcTileData srcTile(validRow, srcValidCol);
    srcTileData tmpTile(validRow, srcValidCol);
    dstTileData dstTile(validRow, dstCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(tmpTile, row * srcCol * sizeof(T));
    TASSIGN(dstTile, 2 * row * srcCol * sizeof(T));

    // 搬运数据
    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TROWSUM(dstTile, srcTile, tmpTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
}

template <typename T, int row, int validRow, int srcCol, int srcValidCol, int dstCol>
PTO_INTERNAL void runTRowSumDNDst(__gm__ T *out, __gm__ T *src)
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
    TASSIGN(srcTile, 0x0);
    TASSIGN(tmpTile, row * srcCol * sizeof(T));
    TASSIGN(dstTile, 2 * row * srcCol * sizeof(T));

    // 搬运数据
    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TROWSUM(dstTile, srcTile, tmpTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    using dstTileDataND = Tile<TileType::Vec, T, 1, row, BLayout::RowMajor, 1, row>;
    dstTileDataND dstTileND;
    TRESHAPE(dstTileND, dstTile);
    TSTORE(dstGlobal, dstTileND);
}

extern "C" __global__ AICORE void launchTROWSUMCase1(__gm__ float *out, __gm__ float *src)
{
    runTRowSum<float, 127, 127, 64, 63, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase2(__gm__ float *out, __gm__ float *src)
{
    runTRowSum<float, 63, 63, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase3(__gm__ float *out, __gm__ float *src)
{
    runTRowSum<float, 31, 31, 128, 127, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase4(__gm__ float *out, __gm__ float *src)
{
    runTRowSum<float, 15, 15, 192, 192, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase5(__gm__ float *out, __gm__ float *src)
{
    runTRowSum<float, 7, 7, 448, 447, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase6(__gm__ half *out, __gm__ half *src)
{
    runTRowSum<half, 256, 256, 16, 15, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase7(__gm__ float *out, __gm__ float *src)
{
    runTRowSumDNDst<float, 64, 64, 128, 128, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase8(__gm__ float *out, __gm__ float *src)
{
    runTRowSumDNDst<float, 32, 32, 256, 256, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase9(__gm__ float *out, __gm__ float *src)
{
    runTRowSumDNDst<float, 16, 16, 512, 512, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase10(__gm__ float *out, __gm__ float *src)
{
    runTRowSumDNDst<float, 8, 8, 1024, 1024, 1>(out, src);
}

// int32 test cases
extern "C" __global__ AICORE void launchTROWSUMCase11(__gm__ int32_t *out, __gm__ int32_t *src)
{
    runTRowSum<int32_t, 127, 127, 64, 63, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase12(__gm__ int32_t *out, __gm__ int32_t *src)
{
    runTRowSum<int32_t, 63, 63, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase13(__gm__ int32_t *out, __gm__ int32_t *src)
{
    runTRowSum<int32_t, 31, 31, 128, 127, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase14(__gm__ int32_t *out, __gm__ int32_t *src)
{
    runTRowSum<int32_t, 15, 15, 192, 192, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase15(__gm__ int32_t *out, __gm__ int32_t *src)
{
    runTRowSum<int32_t, 7, 7, 448, 447, 1>(out, src);
}

// int16 test cases - need 32-byte alignment for int16_t (2 bytes), so cols must be multiple of 16
extern "C" __global__ AICORE void launchTROWSUMCase16(__gm__ int16_t *out, __gm__ int16_t *src)
{
    runTRowSum<int16_t, 128, 128, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase17(__gm__ int16_t *out, __gm__ int16_t *src)
{
    runTRowSum<int16_t, 64, 64, 64, 64, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase18(__gm__ int16_t *out, __gm__ int16_t *src)
{
    runTRowSum<int16_t, 32, 32, 128, 128, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase19(__gm__ int16_t *out, __gm__ int16_t *src)
{
    runTRowSum<int16_t, 16, 16, 192, 192, 1>(out, src);
}
extern "C" __global__ AICORE void launchTROWSUMCase20(__gm__ int16_t *out, __gm__ int16_t *src)
{
    runTRowSum<int16_t, 8, 8, 448, 448, 1>(out, src);
}

template <uint32_t caseId>
void launchTROWSUMTestCase(void *out, void *src, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTROWSUMCase1<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 2: {
            launchTROWSUMCase2<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 3: {
            launchTROWSUMCase3<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 4: {
            launchTROWSUMCase4<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 5: {
            launchTROWSUMCase5<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 6: {
            launchTROWSUMCase6<<<1, nullptr, stream>>>((half *)out, (half *)src);
            break;
        }
        case 7: {
            launchTROWSUMCase7<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 8: {
            launchTROWSUMCase8<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 9: {
            launchTROWSUMCase9<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 10: {
            launchTROWSUMCase10<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 11: {
            launchTROWSUMCase11<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src);
            break;
        }
        case 12: {
            launchTROWSUMCase12<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src);
            break;
        }
        case 13: {
            launchTROWSUMCase13<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src);
            break;
        }
        case 14: {
            launchTROWSUMCase14<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src);
            break;
        }
        case 15: {
            launchTROWSUMCase15<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src);
            break;
        }
        case 16: {
            launchTROWSUMCase16<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src);
            break;
        }
        case 17: {
            launchTROWSUMCase17<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src);
            break;
        }
        case 18: {
            launchTROWSUMCase18<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src);
            break;
        }
        case 19: {
            launchTROWSUMCase19<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src);
            break;
        }
        case 20: {
            launchTROWSUMCase20<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src);
            break;
        }
        default: {
        }
    }
}

template void launchTROWSUMTestCase<1>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<2>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<3>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<4>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<5>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<6>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<7>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<8>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<9>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<10>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<11>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<12>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<13>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<14>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<15>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<16>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<17>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<18>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<19>(void *out, void *src, aclrtStream stream);
template void launchTROWSUMTestCase<20>(void *out, void *src, aclrtStream stream);