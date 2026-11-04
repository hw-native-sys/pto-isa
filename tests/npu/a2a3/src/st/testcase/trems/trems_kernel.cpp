/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <acl/acl.h>

using namespace std;
using namespace pto;

// tmp tile needs dstTileCol + packed-mask floats + 8-element address buffer
// (matches the TREMS_IMPL contract in include/pto/npu/a2a3/TRemS.hpp).
template <typename T, int dstTileRow, int dstTileCol, int row, int validRow, int col, int validCol>
PTO_INTERNAL void runTREMS(__gm__ T* out, __gm__ T* src, T scalar)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, 1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    GlobalData srcGlobal(src, DynDim2Shape(validRow, validCol), DynDim2Stride(col));
    GlobalData dstGlobal(out, DynDim2Shape(validRow, validCol), DynDim2Stride(dstTileCol));

    constexpr unsigned maskFloats = ((dstTileCol + 255) / 256) * 8;
    constexpr unsigned tmpTileCol = dstTileCol + maskFloats + 8;

    using dstTileData = Tile<TileType::Vec, T, dstTileRow, dstTileCol, BLayout::RowMajor, -1, -1>;
    using srcTileData = Tile<TileType::Vec, T, row, col, BLayout::RowMajor, -1, -1>;
    using tmpTileData = Tile<TileType::Vec, T, 1, tmpTileCol, BLayout::RowMajor, -1, -1>;
    srcTileData srcTile(validRow, validCol);
    dstTileData dstTile(validRow, validCol);
    tmpTileData tmpTile(1, tmpTileCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, row * col * sizeof(T));
    TASSIGN(tmpTile, row * col * sizeof(T) + dstTileRow * dstTileCol * sizeof(T));

    TLOAD(srcTile, srcGlobal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

    TREMS(dstTile, srcTile, scalar, tmpTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

#define DEF_TREMS_CASE(ID, T, DR, DC, R, VR, C, VC)                                               \
    extern "C" __global__ AICORE void launchTREMSCase##ID(__gm__ T* out, __gm__ T* src, T scalar) \
    {                                                                                             \
        runTREMS<T, DR, DC, R, VR, C, VC>(out, src, scalar);                                      \
    }

// High precision remainder cases: TREMS<RemSAlgorithm::HIGH_PRECISION>.
// tmp tile follows the a2a3 HP scratch contract (custom/TFmodRemHp.hpp):
// 7 row buffers + packed mask + addr buffer + sign constant.
template <typename T, int dstTileRow, int dstTileCol, int row, int validRow, int col, int validCol>
PTO_INTERNAL void runTREMSHP(__gm__ T* out, __gm__ T* src, T scalar)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, 1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    GlobalData srcGlobal(src, DynDim2Shape(validRow, validCol), DynDim2Stride(col));
    GlobalData dstGlobal(out, DynDim2Shape(validRow, validCol), DynDim2Stride(dstTileCol));

    constexpr unsigned elemsPerRepeat = 256 / sizeof(T);
    constexpr unsigned capVecs = validCol / elemsPerRepeat > 0 ? validCol / elemsPerRepeat : 1;
    constexpr unsigned maskF = ((capVecs * 8 + 31) / 32) * 8;
    constexpr unsigned tmpTileCol = 128 * capVecs + maskF + 8 + (capVecs == 1 ? 64 : 0);

    using dstTileData = Tile<TileType::Vec, T, dstTileRow, dstTileCol, BLayout::RowMajor, -1, -1>;
    using srcTileData = Tile<TileType::Vec, T, row, col, BLayout::RowMajor, -1, -1>;
    using tmpTileData = Tile<TileType::Vec, T, 1, tmpTileCol, BLayout::RowMajor, -1, -1>;
    srcTileData srcTile(validRow, validCol);
    dstTileData dstTile(validRow, validCol);
    tmpTileData tmpTile(1, tmpTileCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, row * col * sizeof(T));
    TASSIGN(tmpTile, row * col * sizeof(T) + dstTileRow * dstTileCol * sizeof(T));

    TLOAD(srcTile, srcGlobal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

    constexpr auto precisionType = RemSAlgorithm::HIGH_PRECISION;
    TREMS<precisionType>(dstTile, srcTile, scalar, tmpTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

#define DEF_TREMS_CASE_HP(ID, T, DR, DC, R, VR, C, VC)                                            \
    extern "C" __global__ AICORE void launchTREMSCase##ID(__gm__ T* out, __gm__ T* src, T scalar) \
    {                                                                                             \
        runTREMSHP<T, DR, DC, R, VR, C, VC>(out, src, scalar);                                    \
    }

// ---- original cases (ids and shapes unchanged) ----
DEF_TREMS_CASE(1, float, 32, 64, 32, 32, 64, 64)
DEF_TREMS_CASE(3, int32_t, 31, 128, 31, 31, 128, 128)
DEF_TREMS_CASE(5, float, 7, 448, 7, 7, 448, 448)
DEF_TREMS_CASE(6, float, 256, 16, 256, 256, 16, 16)
DEF_TREMS_CASE(7, float, 32, 128, 32, 32, 64, 64)     // strided dst
DEF_TREMS_CASE(9, int32_t, 31, 256, 31, 31, 128, 128) // strided dst
DEF_TREMS_CASE(11, float, 7, 512, 7, 7, 448, 448)     // strided dst
DEF_TREMS_CASE(12, float, 256, 32, 256, 256, 16, 16)  // strided dst
DEF_TREMS_CASE(15, int32_t, 1, 8192, 1, 1, 8192, 8192)
DEF_TREMS_CASE(16, float, 1, 8192, 1, 1, 8192, 8192)

// ---- size-matrix float cases: repeat boundaries (64), sub-boundary, large ----
DEF_TREMS_CASE(20, float, 1, 64, 1, 1, 64, 64)       // exactly one repeat
DEF_TREMS_CASE(21, float, 1, 72, 1, 1, 72, 65)       // one element past one repeat
DEF_TREMS_CASE(22, float, 2, 128, 2, 2, 128, 127)    // just below two repeats
DEF_TREMS_CASE(23, float, 2, 128, 2, 2, 128, 128)    // exactly two repeats
DEF_TREMS_CASE(24, float, 2, 136, 2, 2, 136, 129)    // one past two repeats
DEF_TREMS_CASE(25, float, 3, 256, 3, 3, 256, 256)    // exactly four repeats
DEF_TREMS_CASE(26, float, 3, 264, 3, 3, 264, 257)    // one past four repeats
DEF_TREMS_CASE(27, float, 8, 192, 8, 8, 192, 192)    // three repeats, multi-row
DEF_TREMS_CASE(28, float, 33, 64, 33, 33, 64, 64)    // rows > 32
DEF_TREMS_CASE(29, float, 1, 1024, 1, 1, 1024, 1024) // 16 repeats, single row
DEF_TREMS_CASE(30, float, 2, 2048, 2, 2, 2048, 2048) // 32 repeats
DEF_TREMS_CASE(31, float, 4, 64, 4, 4, 64, 63)       // just below one repeat
DEF_TREMS_CASE(32, float, 4, 8, 4, 4, 8, 8)          // tiny, below one full block
DEF_TREMS_CASE(33, float, 2, 4096, 2, 2, 4096, 4096) // 64 repeats
DEF_TREMS_CASE(34, float, 2, 512, 2, 2, 384, 384)    // strided dst, 6 repeats
DEF_TREMS_CASE(35, float, 5, 128, 5, 5, 128, 128)    // negative-scalar dataset

// ---- size-matrix int32 cases ----
DEF_TREMS_CASE(40, int32_t, 1, 64, 1, 1, 64, 64)
DEF_TREMS_CASE(41, int32_t, 2, 72, 2, 2, 72, 65)
DEF_TREMS_CASE(42, int32_t, 3, 136, 3, 3, 136, 129)
DEF_TREMS_CASE(43, int32_t, 1, 8192, 1, 1, 8192, 8192)
DEF_TREMS_CASE(44, int32_t, 2, 192, 2, 2, 192, 192) // negative-scalar dataset
DEF_TREMS_CASE(45, int32_t, 2, 256, 2, 2, 192, 192) // strided dst

// ---- high precision (issue #2): staged (>= 48 full vectors), register fallback,
// tails and strided variants ----
DEF_TREMS_CASE_HP(50, float, 8, 448, 8, 8, 448, 448)    // 56 vecs staged
DEF_TREMS_CASE_HP(51, float, 12, 256, 12, 12, 256, 256) // 48 vecs = threshold
DEF_TREMS_CASE_HP(52, float, 32, 64, 32, 32, 64, 64)    // 32 vecs register fallback
DEF_TREMS_CASE_HP(53, float, 7, 512, 7, 7, 448, 448)    // 49 vecs staged, strided dst
DEF_TREMS_CASE_HP(54, float, 2, 2048, 2, 2, 2048, 2048) // 64 vecs staged, wide rows
DEF_TREMS_CASE_HP(55, float, 5, 320, 5, 5, 320, 320)    // 25 vecs register fallback
DEF_TREMS_CASE_HP(56, float, 8, 456, 8, 8, 456, 456)    // 56 vecs staged + 8-elem tails
DEF_TREMS_CASE_HP(57, float, 4, 136, 4, 4, 136, 136)    // 8 vecs register + 8-elem tails
DEF_TREMS_CASE_HP(58, float, 1, 8192, 1, 1, 8192, 8192) // 128 vecs staged, one row
DEF_TREMS_CASE_HP(59, float, 16, 64, 16, 16, 64, 63)    // register fallback + 63-elem tails

template <uint32_t caseId>
void launchTREMSTestCase(void* out, void* src, float scalar, aclrtStream stream);

#define RUN_TREMS_CASE(ID, T)                                                                  \
    case ID:                                                                                   \
        launchTREMSCase##ID<<<1, nullptr, stream>>>((T*)out, (T*)src, static_cast<T>(scalar)); \
        break;

template <uint32_t caseId>
void launchTREMSTestCase(void* out, void* src, float scalar, aclrtStream stream)
{
    switch (caseId) {
        RUN_TREMS_CASE(1, float)
        RUN_TREMS_CASE(3, int32_t)
        RUN_TREMS_CASE(5, float)
        RUN_TREMS_CASE(6, float)
        RUN_TREMS_CASE(7, float)
        RUN_TREMS_CASE(9, int32_t)
        RUN_TREMS_CASE(11, float)
        RUN_TREMS_CASE(12, float)
        RUN_TREMS_CASE(15, int32_t)
        RUN_TREMS_CASE(16, float)
        RUN_TREMS_CASE(20, float)
        RUN_TREMS_CASE(21, float)
        RUN_TREMS_CASE(22, float)
        RUN_TREMS_CASE(23, float)
        RUN_TREMS_CASE(24, float)
        RUN_TREMS_CASE(25, float)
        RUN_TREMS_CASE(26, float)
        RUN_TREMS_CASE(27, float)
        RUN_TREMS_CASE(28, float)
        RUN_TREMS_CASE(29, float)
        RUN_TREMS_CASE(30, float)
        RUN_TREMS_CASE(31, float)
        RUN_TREMS_CASE(32, float)
        RUN_TREMS_CASE(33, float)
        RUN_TREMS_CASE(34, float)
        RUN_TREMS_CASE(35, float)
        RUN_TREMS_CASE(40, int32_t)
        RUN_TREMS_CASE(41, int32_t)
        RUN_TREMS_CASE(42, int32_t)
        RUN_TREMS_CASE(43, int32_t)
        RUN_TREMS_CASE(44, int32_t)
        RUN_TREMS_CASE(45, int32_t)
        RUN_TREMS_CASE(50, float)
        RUN_TREMS_CASE(51, float)
        RUN_TREMS_CASE(52, float)
        RUN_TREMS_CASE(53, float)
        RUN_TREMS_CASE(54, float)
        RUN_TREMS_CASE(55, float)
        RUN_TREMS_CASE(56, float)
        RUN_TREMS_CASE(57, float)
        RUN_TREMS_CASE(58, float)
        RUN_TREMS_CASE(59, float)
        default:
            break;
    }
}

#define INST_TREMS(ID) template void launchTREMSTestCase<ID>(void* out, void* src, float scalar, aclrtStream stream);
INST_TREMS(1)
INST_TREMS(3)
INST_TREMS(5)
INST_TREMS(6)
INST_TREMS(7)
INST_TREMS(9)
INST_TREMS(11)
INST_TREMS(12)
INST_TREMS(15)
INST_TREMS(16)
INST_TREMS(20)
INST_TREMS(21)
INST_TREMS(22)
INST_TREMS(23)
INST_TREMS(24)
INST_TREMS(25)
INST_TREMS(26)
INST_TREMS(27)
INST_TREMS(28)
INST_TREMS(29)
INST_TREMS(30)
INST_TREMS(31)
INST_TREMS(32)
INST_TREMS(33)
INST_TREMS(34)
INST_TREMS(35)
INST_TREMS(40)
INST_TREMS(41)
INST_TREMS(42)
INST_TREMS(43)
INST_TREMS(44)
INST_TREMS(45)
INST_TREMS(50)
INST_TREMS(51)
INST_TREMS(52)
INST_TREMS(53)
INST_TREMS(54)
INST_TREMS(55)
INST_TREMS(56)
INST_TREMS(57)
INST_TREMS(58)
INST_TREMS(59)
