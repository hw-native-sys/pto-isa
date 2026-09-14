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

template <typename T, int srcRow, int srcValidRow, int dstRow, int col, int validCol>
PTO_INTERNAL void runTColSum(__gm__ T __out__* out, __gm__ T __in__* src, bool isBinary)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    GlobalData srcGlobal(src, DynDim2Shape(srcValidRow, validCol), DynDim2Stride(srcRow, col));
    GlobalData dstGlobal(out, DynDim2Shape(dstRow, validCol), DynDim2Stride(dstRow, col));

    using SrcTileData = Tile<TileType::Vec, T, srcRow, col, BLayout::RowMajor, -1, -1>;
    using TmpTileData = Tile<TileType::Vec, T, (srcRow + 1) / 2, col, BLayout::RowMajor, -1, -1>;
    using DstTileData = Tile<TileType::Vec, T, dstRow, col, BLayout::RowMajor, -1, -1>;
    SrcTileData srcTile(srcValidRow, validCol);
    TmpTileData tmpTile((srcValidRow + 1) / 2, validCol);
    DstTileData dstTile(dstRow, validCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(tmpTile, srcRow * col * sizeof(T));
    TASSIGN(dstTile, (srcRow * 3 / 2 + 1) * col * sizeof(T));

    // 搬运数据
    TLOAD(srcTile, srcGlobal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    if (!isBinary) {
        TCOLSUM(dstTile, srcTile);
    } else {
        TCOLSUM(dstTile, srcTile, tmpTile, isBinary);
    }
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
}

extern "C" __global__ AICORE void launchTCOLSUMCase01(__gm__ float* out, __gm__ float* src)
{
    runTColSum<float, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLSUMCase02(__gm__ float* out, __gm__ float* src)
{
    runTColSum<float, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLSUMCase03(__gm__ float* out, __gm__ float* src)
{
    runTColSum<float, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLSUMCase04(__gm__ float* out, __gm__ float* src)
{
    runTColSum<float, 64, 63, 1, 128, 127>(out, src, true);
}
extern "C" __global__ AICORE void launchTCOLSUMCase05(__gm__ float* out, __gm__ float* src)
{
    runTColSum<float, 64, 64, 1, 128, 128>(out, src, true);
}
extern "C" __global__ AICORE void launchTCOLSUMCase11(__gm__ half* out, __gm__ half* src)
{
    runTColSum<half, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLSUMCase12(__gm__ half* out, __gm__ half* src)
{
    runTColSum<half, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLSUMCase13(__gm__ half* out, __gm__ half* src)
{
    runTColSum<half, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLSUMCase14(__gm__ half* out, __gm__ half* src)
{
    runTColSum<half, 64, 63, 1, 128, 127>(out, src, true);
}
extern "C" __global__ AICORE void launchTCOLSUMCase15(__gm__ half* out, __gm__ half* src)
{
    runTColSum<half, 64, 64, 1, 128, 128>(out, src, true);
}
extern "C" __global__ AICORE void launchTCOLSUMCase16(__gm__ half* out, __gm__ half* src)
{
    runTColSum<half, 64, 64, 1, 128, 128>(out, src, true);
}
extern "C" __global__ AICORE void launchTCOLSUMCase21(__gm__ int8_t* out, __gm__ int8_t* src)
{
    runTColSum<int8_t, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLSUMCase22(__gm__ int8_t* out, __gm__ int8_t* src)
{
    runTColSum<int8_t, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLSUMCase23(__gm__ int8_t* out, __gm__ int8_t* src)
{
    runTColSum<int8_t, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLSUMCase24(__gm__ int8_t* out, __gm__ int8_t* src)
{
    runTColSum<int8_t, 64, 63, 1, 128, 127>(out, src, true);
}
extern "C" __global__ AICORE void launchTCOLSUMCase25(__gm__ int8_t* out, __gm__ int8_t* src)
{
    runTColSum<int8_t, 64, 64, 1, 128, 128>(out, src, true);
}
extern "C" __global__ AICORE void launchTCOLSUMCase31(__gm__ float* out, __gm__ float* src)
{
    runTColSum<float, 1, 1, 1, 512, 511>(out, src, true);
}
extern "C" __global__ AICORE void launchTCOLSUMCase41(__gm__ int64_t* out, __gm__ int64_t* src)
{
    using ShapeType = Shape<1, 1, 1, 4, 16>;
    using StrideType = pto::Stride<64, 64, 64, 16, 1>;
    using GlobalData = GlobalTensor<int64_t, ShapeType, StrideType>;
    using SrcTile = Tile<TileType::Vec, int64_t, 4, 16, BLayout::RowMajor, 4, 16>;
    using DstTile = Tile<TileType::Vec, int64_t, 1, 16, BLayout::RowMajor, 1, 16>;
    SrcTile srcTile;
    DstTile dstTile;
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x1000);
    GlobalData srcGlobal(src);
    GlobalTensor<int64_t, Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>> dstGlobal(out);
    TLOAD(srcTile, srcGlobal);
    PtoSetWaitFlag<PIPE_MTE2, PIPE_V>();
    TCOLSUM(dstTile, srcTile);
    PtoSetWaitFlag<PIPE_V, PIPE_MTE3>();
    TSTORE(dstGlobal, dstTile);
}
extern "C" __global__ AICORE void launchTCOLSUMCase42(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    using ShapeType = Shape<1, 1, 1, 4, 16>;
    using StrideType = pto::Stride<64, 64, 64, 16, 1>;
    using GlobalData = GlobalTensor<uint64_t, ShapeType, StrideType>;
    using SrcTile = Tile<TileType::Vec, uint64_t, 4, 16, BLayout::RowMajor, 4, 16>;
    using DstTile = Tile<TileType::Vec, uint64_t, 1, 16, BLayout::RowMajor, 1, 16>;
    SrcTile srcTile;
    DstTile dstTile;
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x1000);
    GlobalData srcGlobal(src);
    GlobalTensor<uint64_t, Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>> dstGlobal(out);
    TLOAD(srcTile, srcGlobal);
    PtoSetWaitFlag<PIPE_MTE2, PIPE_V>();
    TCOLSUM(dstTile, srcTile);
    PtoSetWaitFlag<PIPE_V, PIPE_MTE3>();
    TSTORE(dstGlobal, dstTile);
}
extern "C" __global__ AICORE void launchTCOLSUMCase43(__gm__ int64_t* out, __gm__ int64_t* src)
{
    using ShapeType = Shape<1, 1, 1, 4, 64>;
    using StrideType = pto::Stride<256, 256, 256, 64, 1>;
    using GlobalData = GlobalTensor<int64_t, ShapeType, StrideType>;
    using SrcTile = Tile<TileType::Vec, int64_t, 4, 64, BLayout::RowMajor, 4, 64>;
    using DstTile = Tile<TileType::Vec, int64_t, 1, 64, BLayout::RowMajor, 1, 64>;
    SrcTile srcTile;
    DstTile dstTile;
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x1000);
    GlobalData srcGlobal(src);
    GlobalTensor<int64_t, Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>> dstGlobal(out);
    TLOAD(srcTile, srcGlobal);
    PtoSetWaitFlag<PIPE_MTE2, PIPE_V>();
    TCOLSUM(dstTile, srcTile);
    PtoSetWaitFlag<PIPE_V, PIPE_MTE3>();
    TSTORE(dstGlobal, dstTile);
}
extern "C" __global__ AICORE void launchTCOLSUMCase44(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    using ShapeType = Shape<1, 1, 1, 4, 64>;
    using StrideType = pto::Stride<256, 256, 256, 64, 1>;
    using GlobalData = GlobalTensor<uint64_t, ShapeType, StrideType>;
    using SrcTile = Tile<TileType::Vec, uint64_t, 4, 64, BLayout::RowMajor, 4, 64>;
    using DstTile = Tile<TileType::Vec, uint64_t, 1, 64, BLayout::RowMajor, 1, 64>;
    SrcTile srcTile;
    DstTile dstTile;
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x1000);
    GlobalData srcGlobal(src);
    GlobalTensor<uint64_t, Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>> dstGlobal(out);
    TLOAD(srcTile, srcGlobal);
    PtoSetWaitFlag<PIPE_MTE2, PIPE_V>();
    TCOLSUM(dstTile, srcTile);
    PtoSetWaitFlag<PIPE_V, PIPE_MTE3>();
    TSTORE(dstGlobal, dstTile);
}
extern "C" __global__ AICORE void launchTCOLSUMCase45(__gm__ int64_t* out, __gm__ int64_t* src)
{
    runTColSum<int64_t, 4, 4, 1, 16, 16>(out, src, true);
}
extern "C" __global__ AICORE void launchTCOLSUMCase46(__gm__ int64_t* out, __gm__ int64_t* src)
{
    runTColSum<int64_t, 4, 4, 1, 16, 16>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLSUMCase47(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    runTColSum<uint64_t, 4, 4, 1, 16, 16>(out, src, true);
}
extern "C" __global__ AICORE void launchTCOLSUMCase48(__gm__ uint64_t* out, __gm__ uint64_t* src)
{
    runTColSum<uint64_t, 4, 4, 1, 16, 16>(out, src, false);
}

template <typename T, int rows, int validRows, int cols, int validCols, int dstCols>
__global__ AICORE void launchTCOLSUMGuard(__gm__ T* out, __gm__ T* src)
{
    constexpr int outputElements = dstCols + 64;
    constexpr BLayout dstLayout = BLayout::RowMajor;
    using SrcView = Tile<TileType::Vec, T, rows, cols, BLayout::RowMajor, rows, cols>;
    using SrcTile = Tile<TileType::Vec, T, rows, cols, BLayout::RowMajor, validRows, validCols>;
    using DstTile = Tile<TileType::Vec, T, 1, dstCols, dstLayout, 1, validCols>;
    using OutputView = Tile<TileType::Vec, T, 1, outputElements, BLayout::RowMajor, 1, outputElements>;
    using SrcGlobal = GlobalTensor<T, Shape<1, 1, 1, rows, cols>, pto::Stride<1, 1, 1, cols, 1>>;
    using OutputGlobal = GlobalTensor<T, Shape<1, 1, 1, 1, outputElements>, pto::Stride<1, 1, 1, outputElements, 1>>;
    SrcView srcView;
    SrcTile srcTile;
    DstTile dstTile;
    OutputView outputView;
    SrcGlobal srcGlobal(src);
    OutputGlobal outputGlobal(out);
    TASSIGN(srcView, 0);
    TASSIGN(srcTile, 0);
    TASSIGN(dstTile, 32768);
    // Read back padding, inactive rows, and the adjacent allocation guard.
    TASSIGN(outputView, 32768);
    TLOAD(srcView, srcGlobal);
    TLOAD(outputView, outputGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TCOLSUM(dstTile, srcTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(outputGlobal, outputView);
}

template <uint32_t caseId>
struct TColSumCaseLauncher {
    static void Launch(void* out, void* src, aclrtStream stream) {}
};

template <>
struct TColSumCaseLauncher<1> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMCase01<<<1, nullptr, stream>>>((float*)out, (float*)src);
    }
};

template <>
struct TColSumCaseLauncher<2> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMCase02<<<1, nullptr, stream>>>((float*)out, (float*)src);
    }
};

template <>
struct TColSumCaseLauncher<3> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMCase03<<<1, nullptr, stream>>>((float*)out, (float*)src);
    }
};

template <>
struct TColSumCaseLauncher<4> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMCase04<<<1, nullptr, stream>>>((float*)out, (float*)src);
    }
};

template <>
struct TColSumCaseLauncher<5> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMCase05<<<1, nullptr, stream>>>((float*)out, (float*)src);
    }
};

template <>
struct TColSumCaseLauncher<11> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMCase11<<<1, nullptr, stream>>>((half*)out, (half*)src);
    }
};

template <>
struct TColSumCaseLauncher<12> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMCase12<<<1, nullptr, stream>>>((half*)out, (half*)src);
    }
};

template <>
struct TColSumCaseLauncher<13> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMCase13<<<1, nullptr, stream>>>((half*)out, (half*)src);
    }
};

template <>
struct TColSumCaseLauncher<14> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMCase14<<<1, nullptr, stream>>>((half*)out, (half*)src);
    }
};

template <>
struct TColSumCaseLauncher<15> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMCase15<<<1, nullptr, stream>>>((half*)out, (half*)src);
    }
};

template <>
struct TColSumCaseLauncher<16> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMCase16<<<1, nullptr, stream>>>((half*)out, (half*)src);
    }
};

template <>
struct TColSumCaseLauncher<21> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMCase21<<<1, nullptr, stream>>>((int8_t*)out, (int8_t*)src);
    }
};

template <>
struct TColSumCaseLauncher<22> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMCase22<<<1, nullptr, stream>>>((int8_t*)out, (int8_t*)src);
    }
};

template <>
struct TColSumCaseLauncher<23> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMCase23<<<1, nullptr, stream>>>((int8_t*)out, (int8_t*)src);
    }
};

template <>
struct TColSumCaseLauncher<24> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMCase24<<<1, nullptr, stream>>>((int8_t*)out, (int8_t*)src);
    }
};

template <>
struct TColSumCaseLauncher<25> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMCase25<<<1, nullptr, stream>>>((int8_t*)out, (int8_t*)src);
    }
};

template <>
struct TColSumCaseLauncher<31> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMCase31<<<1, nullptr, stream>>>((float*)out, (float*)src);
    }
};

template <>
struct TColSumCaseLauncher<41> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMCase41<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src);
    }
};

template <>
struct TColSumCaseLauncher<42> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMCase42<<<1, nullptr, stream>>>((uint64_t*)out, (uint64_t*)src);
    }
};

template <>
struct TColSumCaseLauncher<43> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMCase43<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src);
    }
};

template <>
struct TColSumCaseLauncher<44> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMCase44<<<1, nullptr, stream>>>((uint64_t*)out, (uint64_t*)src);
    }
};

template <>
struct TColSumCaseLauncher<45> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMCase45<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src);
    }
};

template <>
struct TColSumCaseLauncher<46> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMCase46<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src);
    }
};

template <>
struct TColSumCaseLauncher<47> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMCase47<<<1, nullptr, stream>>>((uint64_t*)out, (uint64_t*)src);
    }
};

template <>
struct TColSumCaseLauncher<48> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMCase48<<<1, nullptr, stream>>>((uint64_t*)out, (uint64_t*)src);
    }
};

template <>
struct TColSumCaseLauncher<100> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMGuard<int64_t, 2, 2, 4, 1, 4><<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src);
    }
};

template <>
struct TColSumCaseLauncher<101> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMGuard<int64_t, 2, 2, 4, 4, 4><<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src);
    }
};

template <>
struct TColSumCaseLauncher<102> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMGuard<int64_t, 2, 2, 36, 33, 36><<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src);
    }
};

template <>
struct TColSumCaseLauncher<103> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMGuard<int64_t, 2, 2, 68, 65, 68><<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src);
    }
};

template <>
struct TColSumCaseLauncher<104> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMGuard<uint64_t, 2, 2, 4, 1, 4><<<1, nullptr, stream>>>((uint64_t*)out, (uint64_t*)src);
    }
};

template <>
struct TColSumCaseLauncher<105> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMGuard<uint64_t, 2, 2, 4, 4, 4><<<1, nullptr, stream>>>((uint64_t*)out, (uint64_t*)src);
    }
};

template <>
struct TColSumCaseLauncher<106> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMGuard<uint64_t, 2, 2, 36, 33, 36><<<1, nullptr, stream>>>((uint64_t*)out, (uint64_t*)src);
    }
};

template <>
struct TColSumCaseLauncher<107> {
    static void Launch(void* out, void* src, aclrtStream stream)
    {
        launchTCOLSUMGuard<uint64_t, 2, 2, 68, 65, 68><<<1, nullptr, stream>>>((uint64_t*)out, (uint64_t*)src);
    }
};

template <uint32_t caseId>
void launchTCOLSUMTestCase(void* out, void* src, aclrtStream stream)
{
    TColSumCaseLauncher<caseId>::Launch(out, src, stream);
}

template void launchTCOLSUMTestCase<1>(void* out, void* src, aclrtStream stream);
template void launchTCOLSUMTestCase<2>(void* out, void* src, aclrtStream stream);
template void launchTCOLSUMTestCase<3>(void* out, void* src, aclrtStream stream);
template void launchTCOLSUMTestCase<4>(void* out, void* src, aclrtStream stream);
template void launchTCOLSUMTestCase<5>(void* out, void* src, aclrtStream stream);
template void launchTCOLSUMTestCase<11>(void* out, void* src, aclrtStream stream);
template void launchTCOLSUMTestCase<12>(void* out, void* src, aclrtStream stream);
template void launchTCOLSUMTestCase<13>(void* out, void* src, aclrtStream stream);
template void launchTCOLSUMTestCase<14>(void* out, void* src, aclrtStream stream);
template void launchTCOLSUMTestCase<15>(void* out, void* src, aclrtStream stream);
template void launchTCOLSUMTestCase<16>(void* out, void* src, aclrtStream stream);
template void launchTCOLSUMTestCase<21>(void* out, void* src, aclrtStream stream);
template void launchTCOLSUMTestCase<22>(void* out, void* src, aclrtStream stream);
template void launchTCOLSUMTestCase<23>(void* out, void* src, aclrtStream stream);
template void launchTCOLSUMTestCase<24>(void* out, void* src, aclrtStream stream);
template void launchTCOLSUMTestCase<25>(void* out, void* src, aclrtStream stream);
template void launchTCOLSUMTestCase<31>(void* out, void* src, aclrtStream stream);
template void launchTCOLSUMTestCase<41>(void* out, void* src, aclrtStream stream);
template void launchTCOLSUMTestCase<42>(void* out, void* src, aclrtStream stream);
template void launchTCOLSUMTestCase<43>(void* out, void* src, aclrtStream stream);
template void launchTCOLSUMTestCase<44>(void* out, void* src, aclrtStream stream);
template void launchTCOLSUMTestCase<45>(void* out, void* src, aclrtStream stream);
template void launchTCOLSUMTestCase<46>(void* out, void* src, aclrtStream stream);
template void launchTCOLSUMTestCase<47>(void* out, void* src, aclrtStream stream);
template void launchTCOLSUMTestCase<48>(void* out, void* src, aclrtStream stream);
template void launchTCOLSUMTestCase<100>(void* out, void* src, aclrtStream stream);
template void launchTCOLSUMTestCase<101>(void* out, void* src, aclrtStream stream);
template void launchTCOLSUMTestCase<102>(void* out, void* src, aclrtStream stream);
template void launchTCOLSUMTestCase<103>(void* out, void* src, aclrtStream stream);
template void launchTCOLSUMTestCase<104>(void* out, void* src, aclrtStream stream);
template void launchTCOLSUMTestCase<105>(void* out, void* src, aclrtStream stream);
template void launchTCOLSUMTestCase<106>(void* out, void* src, aclrtStream stream);
template void launchTCOLSUMTestCase<107>(void* out, void* src, aclrtStream stream);
