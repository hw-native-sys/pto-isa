/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

template <typename T>
AICORE constexpr inline T CeilAlign(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

template <
    typename OutType, typename AType, typename BType, int validM, int validK, int validN, typename GlobalDataSrc0,
    typename GlobalDataSrc1>
AICORE inline void RunTMATMULImpl(__gm__ OutType* out, __gm__ AType* src0, __gm__ BType* src1)
{
    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(AType);
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, blockAlign);

    using GlobalDataOut = GlobalTensor<
        OutType, pto::Shape<1, 1, 1, validM, validN>,
        pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, AType, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, BType, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;

    using LeftTile = TileLeft<AType, M, K, validM, validK>;
    using RightTile = TileRight<BType, K, N, validK, validN>;
    using AccTile = TileAcc<OutType, M, N, validM, validN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x20000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    TEXTRACT(aTile, aMatTile, 0, 0);
    TEXTRACT(bTile, bMatTile, 0, 0);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    TMATMUL(cTile, aTile, bTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    TSTORE(dstGlobal, cTile);
}

template <typename OutType, typename AType, typename BType, int validM, int validK, int validN>
__global__ AICORE void RunTMATMUL_DN(__gm__ OutType* out, __gm__ AType* src0, __gm__ BType* src1)
{
    using GlobalDataSrc0 = GlobalTensor<
        AType, pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, 1, validM>, pto::Layout::DN>;
    using GlobalDataSrc1 = GlobalTensor<
        BType, pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, 1, validK>, pto::Layout::DN>;
    RunTMATMULImpl<OutType, AType, BType, validM, validK, validN, GlobalDataSrc0, GlobalDataSrc1>(out, src0, src1);
}

template <typename OutType, typename AType, typename BType, int validM, int validK, int validN>
__global__ AICORE void RunTMATMUL_ND(__gm__ OutType* out, __gm__ AType* src0, __gm__ BType* src1)
{
    using GlobalDataSrc0 = GlobalTensor<
        AType, pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<
        BType, pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    RunTMATMULImpl<OutType, AType, BType, validM, validK, validN, GlobalDataSrc0, GlobalDataSrc1>(out, src0, src1);
}

template <int32_t tilingKey>
void LaunchTMATMUL(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);

#define DEFINE_TMATMUL_LAUNCH_ND(KEY, OUT_T, A_T, B_T, M, K, N)                                         \
    template <>                                                                                         \
    void LaunchTMATMUL<KEY>(uint8_t * out, uint8_t * src0, uint8_t * src1, void* stream)                \
    {                                                                                                   \
        RunTMATMUL_ND<OUT_T, A_T, B_T, M, K, N><<<1, nullptr, stream>>>(                                \
            reinterpret_cast<OUT_T*>(out), reinterpret_cast<A_T*>(src0), reinterpret_cast<B_T*>(src1)); \
    }

#define DEFINE_TMATMUL_LAUNCH_DN(KEY, OUT_T, A_T, B_T, M, K, N)                                         \
    template <>                                                                                         \
    void LaunchTMATMUL<KEY>(uint8_t * out, uint8_t * src0, uint8_t * src1, void* stream)                \
    {                                                                                                   \
        RunTMATMUL_DN<OUT_T, A_T, B_T, M, K, N><<<1, nullptr, stream>>>(                                \
            reinterpret_cast<OUT_T*>(out), reinterpret_cast<A_T*>(src0), reinterpret_cast<B_T*>(src1)); \
    }

DEFINE_TMATMUL_LAUNCH_DN(1, float, half, half, 31, 96, 47)
DEFINE_TMATMUL_LAUNCH_DN(2, int32_t, int8_t, int8_t, 65, 90, 89)
DEFINE_TMATMUL_LAUNCH_DN(3, float, float, float, 16, 32, 64)
DEFINE_TMATMUL_LAUNCH_DN(4, float, half, half, 1, 256, 64)
DEFINE_TMATMUL_LAUNCH_ND(5, float, half, half, 64, 64, 64)
DEFINE_TMATMUL_LAUNCH_ND(6, int32_t, int8_t, int8_t, 96, 128, 65)
DEFINE_TMATMUL_LAUNCH_ND(7, float, float, float, 33, 63, 31)
DEFINE_TMATMUL_LAUNCH_ND(8, float, half, half, 2, 80, 48)
DEFINE_TMATMUL_LAUNCH_DN(9, float, half, half, 127, 33, 95)
DEFINE_TMATMUL_LAUNCH_DN(10, int32_t, int8_t, int8_t, 17, 33, 31)
DEFINE_TMATMUL_LAUNCH_DN(11, float, float, float, 63, 31, 15)
DEFINE_TMATMUL_LAUNCH_ND(12, float, half, half, 95, 33, 79)
DEFINE_TMATMUL_LAUNCH_ND(13, int32_t, int8_t, int8_t, 129, 95, 33)
DEFINE_TMATMUL_LAUNCH_ND(14, float, float, float, 47, 29, 25)
DEFINE_TMATMUL_LAUNCH_ND(15, int32_t, int8_t, int4b_t, 64, 64, 64)
DEFINE_TMATMUL_LAUNCH_ND(16, int32_t, int8_t, int4b_t, 96, 128, 65)
DEFINE_TMATMUL_LAUNCH_ND(17, int32_t, int8_t, int4b_t, 129, 95, 33)
DEFINE_TMATMUL_LAUNCH_ND(18, int32_t, int8_t, int4b_t, 17, 33, 31)
DEFINE_TMATMUL_LAUNCH_ND(19, int32_t, int8_t, int4b_t, 2, 80, 48)
DEFINE_TMATMUL_LAUNCH_ND(20, float, half, int8_t, 64, 64, 64)
DEFINE_TMATMUL_LAUNCH_ND(21, float, half, int8_t, 96, 128, 89)
DEFINE_TMATMUL_LAUNCH_ND(22, float, half, int8_t, 129, 95, 63)
DEFINE_TMATMUL_LAUNCH_DN(23, float, half, int8_t, 65, 90, 89)
DEFINE_TMATMUL_LAUNCH_ND(24, float, half, int8_t, 2, 90, 31)
DEFINE_TMATMUL_LAUNCH_ND(25, float, half, half, 64, 64, 64)
DEFINE_TMATMUL_LAUNCH_ND(26, float, half, half, 95, 33, 79)
DEFINE_TMATMUL_LAUNCH_DN(27, float, half, half, 127, 33, 95)
DEFINE_TMATMUL_LAUNCH_ND(28, float, half, float8_e4m3_t, 64, 64, 64)
DEFINE_TMATMUL_LAUNCH_DN(29, float, half, float8_e4m3_t, 127, 64, 95)
DEFINE_TMATMUL_LAUNCH_ND(30, float, bfloat16_t, float8_e4m3_t, 64, 64, 64)
DEFINE_TMATMUL_LAUNCH_DN(31, float, bfloat16_t, float8_e4m3_t, 127, 64, 95)
DEFINE_TMATMUL_LAUNCH_ND(32, float, bfloat16_t, int8_t, 64, 64, 64)
DEFINE_TMATMUL_LAUNCH_DN(33, float, bfloat16_t, int8_t, 65, 90, 89)
DEFINE_TMATMUL_LAUNCH_ND(34, float, half, int4b_t, 64, 64, 64)
DEFINE_TMATMUL_LAUNCH_ND(35, float, half, int4b_t, 65, 90, 89)
DEFINE_TMATMUL_LAUNCH_ND(36, float, half, int4b_t, 96, 128, 89)
DEFINE_TMATMUL_LAUNCH_ND(37, float, half, int4b_t, 129, 95, 63)
DEFINE_TMATMUL_LAUNCH_ND(38, float, half, int4b_t, 16, 64, 32)
DEFINE_TMATMUL_LAUNCH_ND(39, float, half, int4b_t, 128, 128, 128)
DEFINE_TMATMUL_LAUNCH_ND(40, float, bfloat16_t, int4b_t, 64, 64, 64)
DEFINE_TMATMUL_LAUNCH_ND(41, float, bfloat16_t, int4b_t, 65, 90, 89)
DEFINE_TMATMUL_LAUNCH_ND(42, float, bfloat16_t, int4b_t, 96, 128, 89)
DEFINE_TMATMUL_LAUNCH_ND(43, float, bfloat16_t, int4b_t, 129, 95, 63)
DEFINE_TMATMUL_LAUNCH_ND(44, float, bfloat16_t, int4b_t, 16, 64, 32)
DEFINE_TMATMUL_LAUNCH_ND(45, float, bfloat16_t, int4b_t, 128, 128, 128)
DEFINE_TMATMUL_LAUNCH_ND(46, float, bfloat16_t, int8_t, 96, 128, 89)
DEFINE_TMATMUL_LAUNCH_ND(47, float, bfloat16_t, int8_t, 129, 95, 63)
DEFINE_TMATMUL_LAUNCH_ND(48, float, bfloat16_t, int8_t, 2, 90, 31)
DEFINE_TMATMUL_LAUNCH_ND(49, float, bfloat16_t, float8_e4m3_t, 95, 64, 95)
DEFINE_TMATMUL_LAUNCH_ND(50, float, bfloat16_t, float8_e4m3_t, 2, 64, 31)
DEFINE_TMATMUL_LAUNCH_ND(51, float, half, half, 2, 80, 48)
DEFINE_TMATMUL_LAUNCH_ND(52, float, half, half, 128, 128, 128)

#undef DEFINE_TMATMUL_LAUNCH_DN
#undef DEFINE_TMATMUL_LAUNCH_ND
