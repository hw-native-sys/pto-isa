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
#include <iostream>
#include "tci_common.h"
#include "acl/acl.h"

using namespace std;
using namespace pto;

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int descending>
inline AICORE void runTCI(__gm__ T __out__ *out, T start)
{
    using DynShapeDim5_dst = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5_dst = Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData_dst = GlobalTensor<T, DynShapeDim5_dst, DynStridDim5_dst>;

    constexpr int dst_row = kGRows_;
    constexpr int dst_col = kGCols_;

    using TileData_dst = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData_dst dstTile(dst_row, dst_col);

    // A3 ub size 192kB, 0x30000
    TASSIGN(dstTile, 0x0);

    GlobalData_dst dstGlobal(out);

    TCI<TileData_dst, T, descending>(dstTile, start);

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void test_tci_b32_case1(__gm__ int32_t *out)
{
    runTCI<int32_t, FLOAT_ROW, FLOAT_T1_COL, FLOAT_ROW, FLOAT_T1_COL, ASCEND>(out, START);
}

extern "C" __global__ AICORE void test_tci_b32_case2(__gm__ int32_t *out)
{
    runTCI<int32_t, FLOAT_ROW, FLOAT_T2_COL, FLOAT_ROW, FLOAT_T2_COL, ASCEND>(out, START);
}

extern "C" __global__ AICORE void test_tci_b32_case3(__gm__ int32_t *out)
{
    runTCI<int32_t, FLOAT_ROW, FLOAT_T3_COL, FLOAT_ROW, FLOAT_T3_COL, DESCEND>(out, START);
}

extern "C" __global__ AICORE void test_tci_b32_case4(__gm__ int32_t *out)
{
    runTCI<int32_t, FLOAT_ROW, FLOAT_T4_COL, FLOAT_ROW, FLOAT_T4_COL, DESCEND>(out, START);
}

extern "C" __global__ AICORE void test_tci_b16_case1(__gm__ int16_t *out)
{
    runTCI<int16_t, HALF_ROW, HALF_T1_COL, HALF_ROW, HALF_T1_COL, ASCEND>(out, START);
}

extern "C" __global__ AICORE void test_tci_b16_case2(__gm__ int16_t *out)
{
    runTCI<int16_t, HALF_ROW, HALF_T2_COL, HALF_ROW, HALF_T2_COL, DESCEND>(out, START);
}

extern "C" __global__ AICORE void test_tci_b16_case3(__gm__ int16_t *out)
{
    runTCI<int16_t, HALF_ROW, HALF_T3_COL, HALF_ROW, HALF_T3_COL, ASCEND>(out, START);
}

extern "C" __global__ AICORE void test_tci_b16_case4(__gm__ int16_t *out)
{
    runTCI<int16_t, HALF_ROW, HALF_T4_COL, HALF_ROW, HALF_T4_COL, DESCEND>(out, START);
}

template <uint32_t descending>
void launchTCI_demo_b32_case1(int32_t *out, void *stream)
{
    cout << "launch TCI b32 start!" << endl;
    test_tci_b32_case1<<<1, nullptr, stream>>>(out);
    cout << "launch TCI b32 end!" << endl;
}

template <uint32_t descending>
void launchTCI_demo_b32_case2(int32_t *out, void *stream)
{
    cout << "launch TCI b32 start!" << endl;
    test_tci_b32_case2<<<1, nullptr, stream>>>(out);
    cout << "launch TCI b32 end!" << endl;
}

template <uint32_t descending>
void launchTCI_demo_b32_case3(int32_t *out, void *stream)
{
    cout << "launch TCI b32 start!" << endl;
    test_tci_b32_case3<<<1, nullptr, stream>>>(out);
    cout << "launch TCI b32 end!" << endl;
}

template <uint32_t descending>
void launchTCI_demo_b32_case4(int32_t *out, void *stream)
{
    cout << "launch TCI b32 start!" << endl;
    test_tci_b32_case4<<<1, nullptr, stream>>>(out);
    cout << "launch TCI b32 end!" << endl;
}

template void launchTCI_demo_b32_case1<ASCEND>(int32_t *out, void *stream);
template void launchTCI_demo_b32_case1<DESCEND>(int32_t *out, void *stream);
template void launchTCI_demo_b32_case2<ASCEND>(int32_t *out, void *stream);
template void launchTCI_demo_b32_case2<DESCEND>(int32_t *out, void *stream);
template void launchTCI_demo_b32_case3<ASCEND>(int32_t *out, void *stream);
template void launchTCI_demo_b32_case3<DESCEND>(int32_t *out, void *stream);
template void launchTCI_demo_b32_case4<ASCEND>(int32_t *out, void *stream);
template void launchTCI_demo_b32_case4<DESCEND>(int32_t *out, void *stream);

template <uint32_t descending>
void launchTCI_demo_b16_case1(int16_t *out, void *stream)
{
    cout << "launch TCI b16 start!" << endl;
    test_tci_b16_case1<<<1, nullptr, stream>>>(out);
    cout << "launch TCI b16 end!" << endl;
}

template <uint32_t descending>
void launchTCI_demo_b16_case2(int16_t *out, void *stream)
{
    cout << "launch TCI b16 start!" << endl;
    test_tci_b16_case2<<<1, nullptr, stream>>>(out);
    cout << "launch TCI b16 end!" << endl;
}

template <uint32_t descending>
void launchTCI_demo_b16_case3(int16_t *out, void *stream)
{
    cout << "launch TCI b16 start!" << endl;
    test_tci_b16_case3<<<1, nullptr, stream>>>(out);
    cout << "launch TCI b16 end!" << endl;
}

template <uint32_t descending>
void launchTCI_demo_b16_case4(int16_t *out, void *stream)
{
    cout << "launch TCI b16 start!" << endl;
    test_tci_b16_case4<<<1, nullptr, stream>>>(out);
    cout << "launch TCI b16 end!" << endl;
}

template void launchTCI_demo_b16_case1<ASCEND>(int16_t *out, void *stream);
template void launchTCI_demo_b16_case1<DESCEND>(int16_t *out, void *stream);
template void launchTCI_demo_b16_case2<ASCEND>(int16_t *out, void *stream);
template void launchTCI_demo_b16_case2<DESCEND>(int16_t *out, void *stream);
template void launchTCI_demo_b16_case3<ASCEND>(int16_t *out, void *stream);
template void launchTCI_demo_b16_case3<DESCEND>(int16_t *out, void *stream);
template void launchTCI_demo_b16_case4<ASCEND>(int16_t *out, void *stream);
template void launchTCI_demo_b16_case4<DESCEND>(int16_t *out, void *stream);