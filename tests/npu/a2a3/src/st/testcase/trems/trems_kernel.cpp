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
    // int keeps its float-domain dividend + quotient chain in tmp (src is
    // read-only), so it needs one more row than float
    constexpr unsigned tmpTileCol = (std::is_same<T, int32_t>::value ? 2 * dstTileCol : dstTileCol) + maskFloats + 8;

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
// tmp tile follows the a2a3 HP scratch contract (TRem.hpp):
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
    constexpr unsigned tmpTileCol =
        128 * capVecs + maskF + 8 + (capVecs == 1 ? 64 : 0) +
        512 * (capVecs < pto::TREM_HP_FUSED_CHUNK_A2A3 ? capVecs : pto::TREM_HP_FUSED_CHUNK_A2A3);

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

// In-place aliasing probe: dst and src share ONE GM buffer and ONE UB tile
// region (both TASSIGN'd at 0x0); tmp sits behind them. highPrecision picks
// the DEFAULT or HP tmp contract. Probe shapes are compact (valid == tile),
// so TSTORE rewrites the whole GM tile and padding never participates.
template <bool highPrecision, typename T, int dstTileRow, int dstTileCol, int row, int validRow, int col, int validCol>
PTO_INTERNAL void runTREMSInplace(__gm__ T* out, T scalar)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, 1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    GlobalData srcGlobal(out, DynDim2Shape(validRow, validCol), DynDim2Stride(col));
    GlobalData dstGlobal(out, DynDim2Shape(validRow, validCol), DynDim2Stride(dstTileCol));

    constexpr unsigned maskFloatsDef = ((dstTileCol + 255) / 256) * 8;
    // int keeps its float-domain dividend + quotient chain in tmp (src is
    // read-only), so it needs one more row than float
    constexpr unsigned tmpTileColDef =
        (std::is_same<T, int32_t>::value ? 2 * dstTileCol : dstTileCol) + maskFloatsDef + 8;
    constexpr unsigned elemsPerRepeat = 256 / sizeof(T);
    constexpr unsigned capVecs = validCol / elemsPerRepeat > 0 ? validCol / elemsPerRepeat : 1;
    constexpr unsigned maskF = ((capVecs * 8 + 31) / 32) * 8;
    constexpr unsigned tmpTileColHp =
        128 * capVecs + maskF + 8 + (capVecs == 1 ? 64 : 0) +
        512 * (capVecs < pto::TREM_HP_FUSED_CHUNK_A2A3 ? capVecs : pto::TREM_HP_FUSED_CHUNK_A2A3);
    constexpr unsigned tmpTileCol = highPrecision ? tmpTileColHp : tmpTileColDef;

    using dstTileData = Tile<TileType::Vec, T, dstTileRow, dstTileCol, BLayout::RowMajor, -1, -1>;
    using srcTileData = Tile<TileType::Vec, T, row, col, BLayout::RowMajor, -1, -1>;
    using tmpTileData = Tile<TileType::Vec, T, 1, tmpTileCol, BLayout::RowMajor, -1, -1>;
    srcTileData srcTile(validRow, validCol);
    dstTileData dstTile(validRow, validCol);
    tmpTileData tmpTile(1, tmpTileCol);
    // the alias under test: dstTile and srcTile share UB offset 0
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x0);
    TASSIGN(tmpTile, row * col * sizeof(T));

    TLOAD(srcTile, srcGlobal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

    constexpr auto precisionType = highPrecision ? RemSAlgorithm::HIGH_PRECISION : RemSAlgorithm::DEFAULT;
    TREMS<precisionType>(dstTile, srcTile, scalar, tmpTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

// Disjoint control: same selectable-precision body, stock dst-behind-src layout.
template <bool highPrecision, typename T, int dstTileRow, int dstTileCol, int row, int validRow, int col, int validCol>
PTO_INTERNAL void runTREMSDisjoint(__gm__ T* out, __gm__ T* src, T scalar)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, 1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    GlobalData srcGlobal(src, DynDim2Shape(validRow, validCol), DynDim2Stride(col));
    GlobalData dstGlobal(out, DynDim2Shape(validRow, validCol), DynDim2Stride(dstTileCol));

    constexpr unsigned maskFloatsDef = ((dstTileCol + 255) / 256) * 8;
    // int keeps its float-domain dividend + quotient chain in tmp (src is
    // read-only), so it needs one more row than float
    constexpr unsigned tmpTileColDef =
        (std::is_same<T, int32_t>::value ? 2 * dstTileCol : dstTileCol) + maskFloatsDef + 8;
    constexpr unsigned elemsPerRepeat = 256 / sizeof(T);
    constexpr unsigned capVecs = validCol / elemsPerRepeat > 0 ? validCol / elemsPerRepeat : 1;
    constexpr unsigned maskF = ((capVecs * 8 + 31) / 32) * 8;
    constexpr unsigned tmpTileColHp =
        128 * capVecs + maskF + 8 + (capVecs == 1 ? 64 : 0) +
        512 * (capVecs < pto::TREM_HP_FUSED_CHUNK_A2A3 ? capVecs : pto::TREM_HP_FUSED_CHUNK_A2A3);
    constexpr unsigned tmpTileCol = highPrecision ? tmpTileColHp : tmpTileColDef;

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

    constexpr auto precisionType = highPrecision ? RemSAlgorithm::HIGH_PRECISION : RemSAlgorithm::DEFAULT;
    TREMS<precisionType>(dstTile, srcTile, scalar, tmpTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

#define DEF_TREMS_IP_CASE(ID, HP, T, DR, DC, R, VR, C, VC)                           \
    extern "C" __global__ AICORE void launchTREMSIpCase##ID(__gm__ T* out, T scalar) \
    {                                                                                \
        runTREMSInplace<HP, T, DR, DC, R, VR, C, VC>(out, scalar);                   \
    }

#define DEF_TREMS_DISJ_CASE(ID, HP, T, DR, DC, R, VR, C, VC)                                          \
    extern "C" __global__ AICORE void launchTREMSDisjCase##ID(__gm__ T* out, __gm__ T* src, T scalar) \
    {                                                                                                 \
        runTREMSDisjoint<HP, T, DR, DC, R, VR, C, VC>(out, src, scalar);                              \
    }

// ---- in-place aliasing probes: DEFAULT float/int (row loop, sub-repeat,
// single wide row; negative scalars hit the sign-correction compare) and the
// HP fallback (staged gate rejects the alias); disjoint control ----
DEF_TREMS_IP_CASE(60, false, float, 33, 64, 33, 33, 64, 64)
DEF_TREMS_IP_CASE(61, false, float, 4, 8, 4, 4, 8, 8)
DEF_TREMS_IP_CASE(62, false, float, 1, 8192, 1, 1, 8192, 8192)
DEF_TREMS_IP_CASE(63, false, int32_t, 2, 192, 2, 2, 192, 192)
DEF_TREMS_IP_CASE(64, true, float, 8, 448, 8, 8, 448, 448)
DEF_TREMS_IP_CASE(65, true, float, 32, 64, 32, 32, 64, 64)
DEF_TREMS_DISJ_CASE(66, false, float, 33, 64, 33, 33, 64, 64)
DEF_TREMS_CASE_HP(67, float, 2, 448, 2, 2, 448, 448)
DEF_TREMS_CASE_HP(68, float, 3, 448, 3, 3, 448, 448)
DEF_TREMS_CASE_HP(69, float, 5, 448, 5, 5, 448, 448)
DEF_TREMS_CASE_HP(70, float, 13, 448, 13, 13, 448, 448)
DEF_TREMS_CASE_HP(71, float, 26, 448, 26, 26, 448, 448)
DEF_TREMS_CASE_HP(72, float, 49, 64, 49, 49, 64, 64)
DEF_TREMS_CASE_HP(73, float, 47, 64, 47, 47, 64, 64)
DEF_TREMS_CASE_HP(74, float, 100, 64, 100, 100, 64, 64)
DEF_TREMS_CASE_HP(75, float, 255, 64, 255, 255, 64, 64)
DEF_TREMS_CASE_HP(76, float, 8, 464, 8, 8, 464, 457)
DEF_TREMS_CASE_HP(77, float, 8, 456, 8, 8, 456, 449)
DEF_TREMS_CASE_HP(78, float, 3, 72, 3, 3, 72, 65)
DEF_TREMS_CASE_HP(79, float, 1, 128, 1, 1, 128, 128)
DEF_TREMS_CASE_HP(80, float, 1, 192, 1, 1, 192, 192)
DEF_TREMS_CASE_HP(81, float, 1, 320, 1, 1, 320, 320)
DEF_TREMS_CASE_HP(82, float, 1, 576, 1, 1, 576, 576)
DEF_TREMS_CASE_HP(83, float, 1, 4096, 1, 1, 4096, 4096)
DEF_TREMS_CASE_HP(84, float, 1, 6400, 1, 1, 6400, 6400)
DEF_TREMS_CASE_HP(85, float, 1, 11136, 1, 1, 11136, 11136)
DEF_TREMS_CASE_HP(86, float, 2, 256, 2, 2, 192, 192) // strided dst
DEF_TREMS_CASE_HP(87, float, 4, 128, 4, 4, 72, 65)   // strided dst
DEF_TREMS_IP_CASE(88, true, float, 2, 2048, 2, 2, 2048, 2048)
DEF_TREMS_IP_CASE(89, true, float, 49, 64, 49, 49, 64, 64)
DEF_TREMS_CASE_HP(90, float, 6, 640, 6, 6, 640, 640)

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
        RUN_TREMS_CASE(67, float)
        RUN_TREMS_CASE(68, float)
        RUN_TREMS_CASE(69, float)
        RUN_TREMS_CASE(70, float)
        RUN_TREMS_CASE(71, float)
        RUN_TREMS_CASE(72, float)
        RUN_TREMS_CASE(73, float)
        RUN_TREMS_CASE(74, float)
        RUN_TREMS_CASE(75, float)
        RUN_TREMS_CASE(76, float)
        RUN_TREMS_CASE(77, float)
        RUN_TREMS_CASE(78, float)
        RUN_TREMS_CASE(79, float)
        RUN_TREMS_CASE(80, float)
        RUN_TREMS_CASE(81, float)
        RUN_TREMS_CASE(82, float)
        RUN_TREMS_CASE(83, float)
        RUN_TREMS_CASE(84, float)
        RUN_TREMS_CASE(85, float)
        RUN_TREMS_CASE(86, float)
        RUN_TREMS_CASE(87, float)
        RUN_TREMS_CASE(90, float)
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

template <uint32_t caseId>
void launchTREMSIpTestCase(void* out, float scalar, aclrtStream stream);

#define RUN_TREMS_IP_CASE(ID, T)                                                        \
    case ID:                                                                            \
        launchTREMSIpCase##ID<<<1, nullptr, stream>>>((T*)out, static_cast<T>(scalar)); \
        break;

template <uint32_t caseId>
void launchTREMSIpTestCase(void* out, float scalar, aclrtStream stream)
{
    switch (caseId) {
        RUN_TREMS_IP_CASE(60, float)
        RUN_TREMS_IP_CASE(61, float)
        RUN_TREMS_IP_CASE(62, float)
        RUN_TREMS_IP_CASE(63, int32_t)
        RUN_TREMS_IP_CASE(64, float)
        RUN_TREMS_IP_CASE(65, float)
        RUN_TREMS_IP_CASE(88, float)
        RUN_TREMS_IP_CASE(89, float)
        default:
            break;
    }
}

#define INST_TREMS_IP(ID) template void launchTREMSIpTestCase<ID>(void* out, float scalar, aclrtStream stream);
INST_TREMS_IP(60)
INST_TREMS_IP(61)
INST_TREMS_IP(62)
INST_TREMS_IP(63)
INST_TREMS_IP(64)
INST_TREMS_IP(65)
INST_TREMS_IP(88)
INST_TREMS_IP(89)
INST_TREMS(90)

template <uint32_t caseId>
void launchTREMSDisjTestCase(void* out, void* src, float scalar, aclrtStream stream);

#define RUN_TREMS_DISJ_CASE(ID, T)                                                                 \
    case ID:                                                                                       \
        launchTREMSDisjCase##ID<<<1, nullptr, stream>>>((T*)out, (T*)src, static_cast<T>(scalar)); \
        break;

template <uint32_t caseId>
void launchTREMSDisjTestCase(void* out, void* src, float scalar, aclrtStream stream)
{
    switch (caseId) {
        RUN_TREMS_DISJ_CASE(66, float)
        default:
            break;
    }
}

#define INST_TREMS_DISJ(ID) \
    template void launchTREMSDisjTestCase<ID>(void* out, void* src, float scalar, aclrtStream stream);
INST_TREMS_DISJ(66)
INST_TREMS(67)
INST_TREMS(68)
INST_TREMS(69)
INST_TREMS(70)
INST_TREMS(71)
INST_TREMS(72)
INST_TREMS(73)
INST_TREMS(74)
INST_TREMS(75)
INST_TREMS(76)
INST_TREMS(77)
INST_TREMS(78)
INST_TREMS(79)
INST_TREMS(80)
INST_TREMS(81)
INST_TREMS(82)
INST_TREMS(83)
INST_TREMS(84)
INST_TREMS(85)
INST_TREMS(86)
INST_TREMS(87)

// ===== HP exactness repro (4580025 % -7.25 returned -3.25 before the
// compensated final iteration; see HpFusedSub in TRem.hpp) =====
// One 1x64 row through TREMS and TREM in both precision modes; out rows:
// 0 TREMS DEFAULT | 1 TREMS HP | 2 TREM DEFAULT | 3 TREM HP.
extern "C" __global__ AICORE void runTREMHpExactRepro(
    __gm__ float* out, __gm__ float* src0Gm, __gm__ float* src1Gm, float scalar)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, 1, -1, 1>;
    using GlobalData = GlobalTensor<float, DynDim2Shape, DynDim2Stride>;
    GlobalData src0Global(src0Gm, DynDim2Shape(1, 64), DynDim2Stride(64));
    GlobalData src1Global(src1Gm, DynDim2Shape(1, 64), DynDim2Stride(64));

    using Row64 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1>;
    using TmpDef = Tile<TileType::Vec, float, 1, 80, BLayout::RowMajor, -1, -1>;
    using TmpHpS = Tile<TileType::Vec, float, 1, 720, BLayout::RowMajor, -1, -1>; // 128 + 8 + 8 + 64 + 512
    using TmpHpV = Tile<TileType::Vec, float, 1, 784, BLayout::RowMajor, -1, -1>; // 192 + 8 + 8 + 64 + 512

    Row64 src0Tile(1, 64), src1Tile(1, 64);
    Row64 dstSDef(1, 64), dstSHp(1, 64), dstVDef(1, 64), dstVHp(1, 64);
    TmpDef tmpDef(1, 80);
    TmpHpS tmpHpS(1, 720);
    TmpHpV tmpHpV(1, 784);

    constexpr unsigned B = 256;
    TASSIGN(src0Tile, 0x0);
    TASSIGN(dstSDef, 1 * B);
    TASSIGN(dstSHp, 2 * B);
    TASSIGN(dstVDef, 3 * B);
    TASSIGN(dstVHp, 4 * B);
    TASSIGN(tmpDef, 5 * B);
    TASSIGN(tmpHpS, 5 * B + 320);
    TASSIGN(tmpHpV, 5 * B + 320 + 2880);
    TASSIGN(src1Tile, 5 * B + 320 + 2880 + 3136);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TREMS<RemSAlgorithm::DEFAULT>(dstSDef, src0Tile, scalar, tmpDef);
    TREMS<RemSAlgorithm::HIGH_PRECISION>(dstSHp, src0Tile, scalar, tmpHpS);
    TREM<RemAlgorithm::DEFAULT, Row64, Row64, Row64, TmpDef>(dstVDef, src0Tile, src1Tile, tmpDef);
    TREM<RemAlgorithm::HIGH_PRECISION, Row64, Row64, Row64, TmpHpV>(dstVHp, src0Tile, src1Tile, tmpHpV);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    GlobalData g0(out, DynDim2Shape(1, 64), DynDim2Stride(64));
    TSTORE(g0, dstSDef);
    GlobalData g1(out + 64, DynDim2Shape(1, 64), DynDim2Stride(64));
    TSTORE(g1, dstSHp);
    GlobalData g2(out + 128, DynDim2Shape(1, 64), DynDim2Stride(64));
    TSTORE(g2, dstVDef);
    GlobalData g3(out + 192, DynDim2Shape(1, 64), DynDim2Stride(64));
    TSTORE(g3, dstVHp);
}

extern "C" void LaunchTREMHpExactRepro(float* out, float* src0, float* src1, float scalar, void* stream)
{
    runTREMHpExactRepro<<<1, nullptr, stream>>>(out, src0, src1, scalar);
}
// ===== end HP exactness repro =====
