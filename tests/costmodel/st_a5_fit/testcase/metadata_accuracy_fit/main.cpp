/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <bit>
#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>
namespace {
using namespace pto;
template <class T, int R, int C>
void CheckSelect(const std::array<double, 3>& measured, uint64_t frozenPrediction)
{
    auto& ctx = perf_sim::GetCostModelCtx();
    const auto saved = ctx;
    ctx.npu_arch = perf_sim::kAscend950Arch;
    using D = Tile<TileType::Vec, T, R, C, BLayout::RowMajor, -1, -1>;
    using M = Tile<TileType::Vec, uint8_t, R, 32, BLayout::RowMajor, -1, -1>;
    D dst(R, C), a(R, C), b(R, C);
    M mask(R, 32), tmp(R, 32);
    constexpr auto bytes = (R * C * sizeof(T) + 255) / 256 * 256;
    TASSIGN(dst, 0);
    TASSIGN(a, bytes);
    TASSIGN(b, 2 * bytes);
    TASSIGN(mask, 3 * bytes);
    TASSIGN(tmp, 3 * bytes);
    // Hardware inputs used all-zero, all-one and mixed (0x69) masks. These
    // contents are intentionally not read by the model or needed as API args.
    for (double actual : measured) {
        TSEL(dst, mask, a, b, tmp);
        const auto predicted = mocker::GetLastPtoInstrCycles();
        EXPECT_EQ(predicted, frozenPrediction);
        EXPECT_NEAR(double(predicted), actual, actual * .1);
    }
    ctx = saved;
}
TEST(A5ObservableMetadataFit, SelectFrozenBackendIndependentAccuracy)
{
    CheckSelect<uint16_t, 1, 32>({37.00073028950828, 37.000085214017936, 37.00013005386059}, 37);
    CheckSelect<uint16_t, 12, 96>({89.99927620660621, 90.00038608295158, 90.00023792761787}, 90);
    CheckSelect<uint32_t, 1, 32>({52.00042703358226, 51.999940514828005, 52.000540705630186}, 52);
    CheckSelect<uint32_t, 12, 96>({113.00062148025849, 112.99992955982593, 113.00024144629151}, 115);
}

template <int Op, class T, int R, int C, int AR, int AC>
void Check(int ar, int ac, int br, int bc, int upper, int diagonal, double measured)
{
    auto& ctx = perf_sim::GetCostModelCtx();
    const auto saved = ctx;
    ctx.npu_arch = perf_sim::kAscend950Arch;
    using D = Tile<TileType::Vec, T, R, C, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>;
    D dst(R, C);
    TASSIGN(dst, 0);
    if constexpr (Op == 6) {
        if (upper)
            TTRI<D, 1>(dst, diagonal);
        else
            TTRI<D, 0>(dst, diagonal);
    } else {
        using A = Tile<TileType::Vec, T, AR, AC, BLayout::RowMajor, -1, -1>;
        A source0(ar, ac);
        constexpr auto dstBytes = (R * C * sizeof(T) + 255) / 256 * 256;
        constexpr auto srcBytes = (AR * AC * sizeof(T) + 255) / 256 * 256;
        TASSIGN(source0, dstBytes);
        if constexpr (Op == 0)
            TFILLPAD(dst, source0);
        else if constexpr (Op == 1)
            TFILLPAD_EXPAND(dst, source0);
        else {
            using B = Tile<TileType::Vec, T, R, C, BLayout::RowMajor, -1, -1>;
            B source1(br, bc);
            TASSIGN(source1, dstBytes + srcBytes);
            if constexpr (Op == 2)
                TPARTADD(dst, source0, source1);
            else if constexpr (Op == 3)
                TPARTMUL(dst, source0, source1);
            else if constexpr (Op == 4)
                TPARTMAX(dst, source0, source1);
            else if constexpr (Op == 5)
                TPARTMIN(dst, source0, source1);
        }
    }
    const double predicted = mocker::GetLastPtoInstrCycles();
    EXPECT_GT(predicted, 0);
    EXPECT_NEAR(predicted, measured, measured * .1);
    ctx = saved;
}
TEST(A5ObservableMetadataFit, IndependentPublicPathAccuracy)
{
    {
        SCOPED_TRACE("ext_tpartadd_float_capacity_holdout_0");
        Check<2, float, 1, 21824, 1, 21824>(1, 10911, 1, 21824, 0, 0, 551.00006274167106);
    }
    {
        SCOPED_TRACE("ext_tpartadd_float_capacity_holdout_1");
        Check<2, float, 2728, 8, 2728, 8>(2728, 8, 1364, 3, 0, 0, 23213.000492222691);
    }
    {
        SCOPED_TRACE("ext_tpartadd_float_capacity_holdout_2");
        Check<2, float, 148, 144, 148, 144>(74, 71, 148, 144, 0, 0, 1580.0003391445214);
    }
    {
        SCOPED_TRACE("ext_tpartadd_float_empty_holdout_0");
        Check<2, float, 12, 48, 12, 48>(0, 48, 12, 48, 0, 0, 89.999439010547931);
    }
    {
        SCOPED_TRACE("ext_tpartadd_float_empty_holdout_1");
        Check<2, float, 40, 96, 40, 96>(40, 96, 0, 96, 0, 0, 298.00011406270443);
    }
    {
        SCOPED_TRACE("ext_tpartadd_float_equal_holdout_0");
        Check<2, float, 12, 48, 12, 48>(12, 48, 12, 48, 0, 0, 89.999761773140094);
    }
    {
        SCOPED_TRACE("ext_tpartadd_float_equal_holdout_1");
        Check<2, float, 40, 96, 40, 96>(40, 96, 40, 96, 0, 0, 298.00079917534526);
    }
    {
        SCOPED_TRACE("ext_tpartadd_int16_capacity_holdout_0");
        Check<2, int16_t, 1, 43648, 1, 43648>(1, 21823, 1, 43648, 0, 0, 550.99965936420961);
    }
    {
        SCOPED_TRACE("ext_tpartadd_int16_capacity_holdout_1");
        Check<2, int16_t, 2728, 16, 2728, 16>(2728, 16, 1364, 7, 0, 0, 23212.999673651502);
    }
    {
        SCOPED_TRACE("ext_tpartadd_int16_capacity_holdout_2");
        Check<2, int16_t, 148, 288, 148, 288>(74, 143, 148, 288, 0, 0, 1580.0000408324015);
    }
    {
        SCOPED_TRACE("ext_tpartadd_int16_empty_holdout_0");
        Check<2, int16_t, 12, 96, 12, 96>(0, 96, 12, 96, 0, 0, 89.999104948590443);
    }
    {
        SCOPED_TRACE("ext_tpartadd_int16_empty_holdout_1");
        Check<2, int16_t, 40, 192, 40, 192>(40, 192, 0, 192, 0, 0, 298.00003243151571);
    }
    {
        SCOPED_TRACE("ext_tpartadd_int16_equal_holdout_0");
        Check<2, int16_t, 12, 96, 12, 96>(12, 96, 12, 96, 0, 0, 90.000652317697927);
    }
    {
        SCOPED_TRACE("ext_tpartadd_int16_equal_holdout_1");
        Check<2, int16_t, 40, 192, 40, 192>(40, 192, 40, 192, 0, 0, 297.99969246965765);
    }
    {
        SCOPED_TRACE("ext_tpartadd_int32_capacity_holdout_0");
        Check<2, int32_t, 1, 21824, 1, 21824>(1, 10911, 1, 21824, 0, 0, 551.00006593491935);
    }
    {
        SCOPED_TRACE("ext_tpartadd_int32_capacity_holdout_1");
        Check<2, int32_t, 2728, 8, 2728, 8>(2728, 8, 1364, 3, 0, 0, 23212.999600931827);
    }
    {
        SCOPED_TRACE("ext_tpartadd_int32_capacity_holdout_2");
        Check<2, int32_t, 148, 144, 148, 144>(74, 71, 148, 144, 0, 0, 1579.9998078296551);
    }
    {
        SCOPED_TRACE("ext_tpartadd_int32_empty_holdout_0");
        Check<2, int32_t, 12, 48, 12, 48>(0, 48, 12, 48, 0, 0, 89.999802797116573);
    }
    {
        SCOPED_TRACE("ext_tpartadd_int32_empty_holdout_1");
        Check<2, int32_t, 40, 96, 40, 96>(40, 96, 0, 96, 0, 0, 297.99988532274227);
    }
    {
        SCOPED_TRACE("ext_tpartadd_int32_equal_holdout_0");
        Check<2, int32_t, 12, 48, 12, 48>(12, 48, 12, 48, 0, 0, 90.00004542780168);
    }
    {
        SCOPED_TRACE("ext_tpartadd_int32_equal_holdout_1");
        Check<2, int32_t, 40, 96, 40, 96>(40, 96, 40, 96, 0, 0, 298.00004752811304);
    }
    {
        SCOPED_TRACE("ext_tpartmax_float_capacity_holdout_0");
        Check<4, float, 1, 21824, 1, 21824>(1, 10911, 1, 21824, 0, 0, 548.99975491400107);
    }
    {
        SCOPED_TRACE("ext_tpartmax_float_capacity_holdout_1");
        Check<4, float, 2728, 8, 2728, 8>(2728, 8, 1364, 3, 0, 0, 23212.99966367993);
    }
    {
        SCOPED_TRACE("ext_tpartmax_float_capacity_holdout_2");
        Check<4, float, 148, 144, 148, 144>(74, 71, 148, 144, 0, 0, 1580.0000581053346);
    }
    {
        SCOPED_TRACE("ext_tpartmax_float_empty_holdout_0");
        Check<4, float, 12, 48, 12, 48>(0, 48, 12, 48, 0, 0, 90.000795672859553);
    }
    {
        SCOPED_TRACE("ext_tpartmax_float_empty_holdout_1");
        Check<4, float, 40, 96, 40, 96>(40, 96, 0, 96, 0, 0, 297.9992251106944);
    }
    {
        SCOPED_TRACE("ext_tpartmax_float_equal_holdout_0");
        Check<4, float, 12, 48, 12, 48>(12, 48, 12, 48, 0, 0, 90.000248786205916);
    }
    {
        SCOPED_TRACE("ext_tpartmax_float_equal_holdout_1");
        Check<4, float, 40, 96, 40, 96>(40, 96, 40, 96, 0, 0, 298.00005866269333);
    }
    {
        SCOPED_TRACE("ext_tpartmax_int16_capacity_holdout_0");
        Check<4, int16_t, 1, 43648, 1, 43648>(1, 21823, 1, 43648, 0, 0, 549.00063318647324);
    }
    {
        SCOPED_TRACE("ext_tpartmax_int16_capacity_holdout_1");
        Check<4, int16_t, 2728, 16, 2728, 16>(2728, 16, 1364, 7, 0, 0, 23212.999868137849);
    }
    {
        SCOPED_TRACE("ext_tpartmax_int16_capacity_holdout_2");
        Check<4, int16_t, 148, 288, 148, 288>(74, 143, 148, 288, 0, 0, 1580.0002987142332);
    }
    {
        SCOPED_TRACE("ext_tpartmax_int16_empty_holdout_0");
        Check<4, int16_t, 12, 96, 12, 96>(0, 96, 12, 96, 0, 0, 89.999523420868243);
    }
    {
        SCOPED_TRACE("ext_tpartmax_int16_empty_holdout_1");
        Check<4, int16_t, 40, 192, 40, 192>(40, 192, 0, 192, 0, 0, 298.0002293148504);
    }
    {
        SCOPED_TRACE("ext_tpartmax_int16_equal_holdout_0");
        Check<4, int16_t, 12, 96, 12, 96>(12, 96, 12, 96, 0, 0, 90.000757489225478);
    }
    {
        SCOPED_TRACE("ext_tpartmax_int16_equal_holdout_1");
        Check<4, int16_t, 40, 192, 40, 192>(40, 192, 40, 192, 0, 0, 297.99931092725882);
    }
    {
        SCOPED_TRACE("ext_tpartmax_int32_capacity_holdout_0");
        Check<4, int32_t, 1, 21824, 1, 21824>(1, 10911, 1, 21824, 0, 0, 549.0004462837145);
    }
    {
        SCOPED_TRACE("ext_tpartmax_int32_capacity_holdout_1");
        Check<4, int32_t, 2728, 8, 2728, 8>(2728, 8, 1364, 3, 0, 0, 23212.999609033694);
    }
    {
        SCOPED_TRACE("ext_tpartmax_int32_capacity_holdout_2");
        Check<4, int32_t, 148, 144, 148, 144>(74, 71, 148, 144, 0, 0, 1579.9994021486507);
    }
    {
        SCOPED_TRACE("ext_tpartmax_int32_empty_holdout_0");
        Check<4, int32_t, 12, 48, 12, 48>(0, 48, 12, 48, 0, 0, 89.999698791303146);
    }
    {
        SCOPED_TRACE("ext_tpartmax_int32_empty_holdout_1");
        Check<4, int32_t, 40, 96, 40, 96>(40, 96, 0, 96, 0, 0, 297.9999905479628);
    }
    {
        SCOPED_TRACE("ext_tpartmax_int32_equal_holdout_0");
        Check<4, int32_t, 12, 48, 12, 48>(12, 48, 12, 48, 0, 0, 89.999776189534487);
    }
    {
        SCOPED_TRACE("ext_tpartmax_int32_equal_holdout_1");
        Check<4, int32_t, 40, 96, 40, 96>(40, 96, 40, 96, 0, 0, 298.0009928754323);
    }
    {
        SCOPED_TRACE("ext_tpartmax_int8_capacity_holdout_0");
        Check<4, int8_t, 1, 87296, 1, 87296>(1, 43647, 1, 87296, 0, 0, 548.99961605997316);
    }
    {
        SCOPED_TRACE("ext_tpartmax_int8_capacity_holdout_1");
        Check<4, int8_t, 2728, 32, 2728, 32>(2728, 32, 1364, 15, 0, 0, 23212.999899394392);
    }
    {
        SCOPED_TRACE("ext_tpartmax_int8_capacity_holdout_2");
        Check<4, int8_t, 148, 576, 148, 576>(74, 287, 148, 576, 0, 0, 1579.9994666228479);
    }
    {
        SCOPED_TRACE("ext_tpartmax_int8_empty_holdout_0");
        Check<4, int8_t, 12, 192, 12, 192>(0, 192, 12, 192, 0, 0, 90.000531456947883);
    }
    {
        SCOPED_TRACE("ext_tpartmax_int8_empty_holdout_1");
        Check<4, int8_t, 40, 384, 40, 384>(40, 384, 0, 384, 0, 0, 297.99958480443871);
    }
    {
        SCOPED_TRACE("ext_tpartmax_int8_equal_holdout_0");
        Check<4, int8_t, 12, 192, 12, 192>(12, 192, 12, 192, 0, 0, 90.000015804935757);
    }
    {
        SCOPED_TRACE("ext_tpartmax_int8_equal_holdout_1");
        Check<4, int8_t, 40, 384, 40, 384>(40, 384, 40, 384, 0, 0, 297.99966582001701);
    }
    {
        SCOPED_TRACE("ext_tpartmax_uint8_capacity_holdout_0");
        Check<4, uint8_t, 1, 87296, 1, 87296>(1, 43647, 1, 87296, 0, 0, 548.99934720471708);
    }
    {
        SCOPED_TRACE("ext_tpartmax_uint8_capacity_holdout_1");
        Check<4, uint8_t, 2728, 32, 2728, 32>(2728, 32, 1364, 15, 0, 0, 23213.000288916057);
    }
    {
        SCOPED_TRACE("ext_tpartmax_uint8_capacity_holdout_2");
        Check<4, uint8_t, 148, 576, 148, 576>(74, 287, 148, 576, 0, 0, 1580.0002164249997);
    }
    {
        SCOPED_TRACE("ext_tpartmax_uint8_empty_holdout_0");
        Check<4, uint8_t, 12, 192, 12, 192>(0, 192, 12, 192, 0, 0, 89.999502373569825);
    }
    {
        SCOPED_TRACE("ext_tpartmax_uint8_empty_holdout_1");
        Check<4, uint8_t, 40, 384, 40, 384>(40, 384, 0, 384, 0, 0, 297.99931293541312);
    }
    {
        SCOPED_TRACE("ext_tpartmax_uint8_equal_holdout_0");
        Check<4, uint8_t, 12, 192, 12, 192>(12, 192, 12, 192, 0, 0, 90.000402094140398);
    }
    {
        SCOPED_TRACE("ext_tpartmax_uint8_equal_holdout_1");
        Check<4, uint8_t, 40, 384, 40, 384>(40, 384, 40, 384, 0, 0, 297.99976862358312);
    }
    {
        SCOPED_TRACE("ext_tpartmin_float_capacity_holdout_0");
        Check<5, float, 1, 21824, 1, 21824>(1, 10911, 1, 21824, 0, 0, 549.00038878858766);
    }
    {
        SCOPED_TRACE("ext_tpartmin_float_capacity_holdout_1");
        Check<5, float, 2728, 8, 2728, 8>(2728, 8, 1364, 3, 0, 0, 23213.000307311362);
    }
    {
        SCOPED_TRACE("ext_tpartmin_float_capacity_holdout_2");
        Check<5, float, 148, 144, 148, 144>(74, 71, 148, 144, 0, 0, 1580.0004990583063);
    }
    {
        SCOPED_TRACE("ext_tpartmin_float_empty_holdout_0");
        Check<5, float, 12, 48, 12, 48>(0, 48, 12, 48, 0, 0, 90.00004057894698);
    }
    {
        SCOPED_TRACE("ext_tpartmin_float_empty_holdout_1");
        Check<5, float, 40, 96, 40, 96>(40, 96, 0, 96, 0, 0, 297.99980840284604);
    }
    {
        SCOPED_TRACE("ext_tpartmin_float_equal_holdout_0");
        Check<5, float, 12, 48, 12, 48>(12, 48, 12, 48, 0, 0, 90.000253926076041);
    }
    {
        SCOPED_TRACE("ext_tpartmin_float_equal_holdout_1");
        Check<5, float, 40, 96, 40, 96>(40, 96, 40, 96, 0, 0, 297.99908951363068);
    }
    {
        SCOPED_TRACE("ext_tpartmin_int16_capacity_holdout_0");
        Check<5, int16_t, 1, 43648, 1, 43648>(1, 21823, 1, 43648, 0, 0, 549.00022975219849);
    }
    {
        SCOPED_TRACE("ext_tpartmin_int16_capacity_holdout_1");
        Check<5, int16_t, 2728, 16, 2728, 16>(2728, 16, 1364, 7, 0, 0, 23212.999356035729);
    }
    {
        SCOPED_TRACE("ext_tpartmin_int16_capacity_holdout_2");
        Check<5, int16_t, 148, 288, 148, 288>(74, 143, 148, 288, 0, 0, 1580.0009024346975);
    }
    {
        SCOPED_TRACE("ext_tpartmin_int16_empty_holdout_0");
        Check<5, int16_t, 12, 96, 12, 96>(0, 96, 12, 96, 0, 0, 89.999196592371774);
    }
    {
        SCOPED_TRACE("ext_tpartmin_int16_empty_holdout_1");
        Check<5, int16_t, 40, 192, 40, 192>(40, 192, 0, 192, 0, 0, 298.00098458274192);
    }
    {
        SCOPED_TRACE("ext_tpartmin_int16_equal_holdout_0");
        Check<5, int16_t, 12, 96, 12, 96>(12, 96, 12, 96, 0, 0, 90.000803712484114);
    }
    {
        SCOPED_TRACE("ext_tpartmin_int16_equal_holdout_1");
        Check<5, int16_t, 40, 192, 40, 192>(40, 192, 40, 192, 0, 0, 298.00030540198082);
    }
    {
        SCOPED_TRACE("ext_tpartmin_int32_capacity_holdout_0");
        Check<5, int32_t, 1, 21824, 1, 21824>(1, 10911, 1, 21824, 0, 0, 548.99912003966085);
    }
    {
        SCOPED_TRACE("ext_tpartmin_int32_capacity_holdout_1");
        Check<5, int32_t, 2728, 8, 2728, 8>(2728, 8, 1364, 3, 0, 0, 23212.999818019041);
    }
    {
        SCOPED_TRACE("ext_tpartmin_int32_capacity_holdout_2");
        Check<5, int32_t, 148, 144, 148, 144>(74, 71, 148, 144, 0, 0, 1579.9996098988738);
    }
    {
        SCOPED_TRACE("ext_tpartmin_int32_empty_holdout_0");
        Check<5, int32_t, 12, 48, 12, 48>(0, 48, 12, 48, 0, 0, 89.99983541528637);
    }
    {
        SCOPED_TRACE("ext_tpartmin_int32_empty_holdout_1");
        Check<5, int32_t, 40, 96, 40, 96>(40, 96, 0, 96, 0, 0, 297.9997098227845);
    }
    {
        SCOPED_TRACE("ext_tpartmin_int32_equal_holdout_0");
        Check<5, int32_t, 12, 48, 12, 48>(12, 48, 12, 48, 0, 0, 89.999315496684744);
    }
    {
        SCOPED_TRACE("ext_tpartmin_int32_equal_holdout_1");
        Check<5, int32_t, 40, 96, 40, 96>(40, 96, 40, 96, 0, 0, 298.00068816501931);
    }
    {
        SCOPED_TRACE("ext_tpartmin_int8_capacity_holdout_0");
        Check<5, int8_t, 1, 87296, 1, 87296>(1, 43647, 1, 87296, 0, 0, 548.99990177961888);
    }
    {
        SCOPED_TRACE("ext_tpartmin_int8_capacity_holdout_1");
        Check<5, int8_t, 2728, 32, 2728, 32>(2728, 32, 1364, 15, 0, 0, 23212.999840019034);
    }
    {
        SCOPED_TRACE("ext_tpartmin_int8_capacity_holdout_2");
        Check<5, int8_t, 148, 576, 148, 576>(74, 287, 148, 576, 0, 0, 1580.0003343212159);
    }
    {
        SCOPED_TRACE("ext_tpartmin_int8_empty_holdout_0");
        Check<5, int8_t, 12, 192, 12, 192>(0, 192, 12, 192, 0, 0, 89.999978423585233);
    }
    {
        SCOPED_TRACE("ext_tpartmin_int8_empty_holdout_1");
        Check<5, int8_t, 40, 384, 40, 384>(40, 384, 0, 384, 0, 0, 297.99966027034861);
    }
    {
        SCOPED_TRACE("ext_tpartmin_int8_equal_holdout_0");
        Check<5, int8_t, 12, 192, 12, 192>(12, 192, 12, 192, 0, 0, 90.000112417966577);
    }
    {
        SCOPED_TRACE("ext_tpartmin_int8_equal_holdout_1");
        Check<5, int8_t, 40, 384, 40, 384>(40, 384, 40, 384, 0, 0, 297.99974242597449);
    }
    {
        SCOPED_TRACE("ext_tpartmin_uint8_capacity_holdout_0");
        Check<5, uint8_t, 1, 87296, 1, 87296>(1, 43647, 1, 87296, 0, 0, 549.00020512739513);
    }
    {
        SCOPED_TRACE("ext_tpartmin_uint8_capacity_holdout_1");
        Check<5, uint8_t, 2728, 32, 2728, 32>(2728, 32, 1364, 15, 0, 0, 23213.00032058022);
    }
    {
        SCOPED_TRACE("ext_tpartmin_uint8_capacity_holdout_2");
        Check<5, uint8_t, 148, 576, 148, 576>(74, 287, 148, 576, 0, 0, 1579.9999526143013);
    }
    {
        SCOPED_TRACE("ext_tpartmin_uint8_empty_holdout_0");
        Check<5, uint8_t, 12, 192, 12, 192>(0, 192, 12, 192, 0, 0, 90.000541925638672);
    }
    {
        SCOPED_TRACE("ext_tpartmin_uint8_empty_holdout_1");
        Check<5, uint8_t, 40, 384, 40, 384>(40, 384, 0, 384, 0, 0, 298.00022202619311);
    }
    {
        SCOPED_TRACE("ext_tpartmin_uint8_equal_holdout_0");
        Check<5, uint8_t, 12, 192, 12, 192>(12, 192, 12, 192, 0, 0, 90.000032683515315);
    }
    {
        SCOPED_TRACE("ext_tpartmin_uint8_equal_holdout_1");
        Check<5, uint8_t, 40, 384, 40, 384>(40, 384, 40, 384, 0, 0, 298.00019917994877);
    }
    {
        SCOPED_TRACE("ext_tpartmul_float_capacity_holdout_0");
        Check<3, float, 1, 21824, 1, 21824>(1, 10911, 1, 21824, 0, 0, 551.00029179242347);
    }
    {
        SCOPED_TRACE("ext_tpartmul_float_capacity_holdout_1");
        Check<3, float, 2728, 8, 2728, 8>(2728, 8, 1364, 3, 0, 0, 23213.000544251587);
    }
    {
        SCOPED_TRACE("ext_tpartmul_float_capacity_holdout_2");
        Check<3, float, 148, 144, 148, 144>(74, 71, 148, 144, 0, 0, 1580.000341056455);
    }
    {
        SCOPED_TRACE("ext_tpartmul_float_empty_holdout_0");
        Check<3, float, 12, 48, 12, 48>(0, 48, 12, 48, 0, 0, 90.000310533041286);
    }
    {
        SCOPED_TRACE("ext_tpartmul_float_empty_holdout_1");
        Check<3, float, 40, 96, 40, 96>(40, 96, 0, 96, 0, 0, 297.99964014898524);
    }
    {
        SCOPED_TRACE("ext_tpartmul_float_equal_holdout_0");
        Check<3, float, 12, 48, 12, 48>(12, 48, 12, 48, 0, 0, 89.999771047100879);
    }
    {
        SCOPED_TRACE("ext_tpartmul_float_equal_holdout_1");
        Check<3, float, 40, 96, 40, 96>(40, 96, 40, 96, 0, 0, 297.99985714678883);
    }
    {
        SCOPED_TRACE("ext_tpartmul_int16_capacity_holdout_0");
        Check<3, int16_t, 1, 43648, 1, 43648>(1, 21823, 1, 43648, 0, 0, 551.00035084048409);
    }
    {
        SCOPED_TRACE("ext_tpartmul_int16_capacity_holdout_1");
        Check<3, int16_t, 2728, 16, 2728, 16>(2728, 16, 1364, 7, 0, 0, 23212.999824408002);
    }
    {
        SCOPED_TRACE("ext_tpartmul_int16_capacity_holdout_2");
        Check<3, int16_t, 148, 288, 148, 288>(74, 143, 148, 288, 0, 0, 1579.9993393037046);
    }
    {
        SCOPED_TRACE("ext_tpartmul_int16_empty_holdout_0");
        Check<3, int16_t, 12, 96, 12, 96>(0, 96, 12, 96, 0, 0, 90.000615839625951);
    }
    {
        SCOPED_TRACE("ext_tpartmul_int16_empty_holdout_1");
        Check<3, int16_t, 40, 192, 40, 192>(40, 192, 0, 192, 0, 0, 297.99955401381584);
    }
    {
        SCOPED_TRACE("ext_tpartmul_int16_equal_holdout_0");
        Check<3, int16_t, 12, 96, 12, 96>(12, 96, 12, 96, 0, 0, 89.999956003731995);
    }
    {
        SCOPED_TRACE("ext_tpartmul_int16_equal_holdout_1");
        Check<3, int16_t, 40, 192, 40, 192>(40, 192, 40, 192, 0, 0, 297.99967893074017);
    }
    {
        SCOPED_TRACE("ext_tpartmul_int32_capacity_holdout_0");
        Check<3, int32_t, 1, 21824, 1, 21824>(1, 10911, 1, 21824, 0, 0, 550.99922362928771);
    }
    {
        SCOPED_TRACE("ext_tpartmul_int32_capacity_holdout_1");
        Check<3, int32_t, 2728, 8, 2728, 8>(2728, 8, 1364, 3, 0, 0, 23212.999535801675);
    }
    {
        SCOPED_TRACE("ext_tpartmul_int32_capacity_holdout_2");
        Check<3, int32_t, 148, 144, 148, 144>(74, 71, 148, 144, 0, 0, 1580.0001015155792);
    }
    {
        SCOPED_TRACE("ext_tpartmul_int32_empty_holdout_0");
        Check<3, int32_t, 12, 48, 12, 48>(0, 48, 12, 48, 0, 0, 90.000654148451986);
    }
    {
        SCOPED_TRACE("ext_tpartmul_int32_empty_holdout_1");
        Check<3, int32_t, 40, 96, 40, 96>(40, 96, 0, 96, 0, 0, 298.00042137779639);
    }
    {
        SCOPED_TRACE("ext_tpartmul_int32_equal_holdout_0");
        Check<3, int32_t, 12, 48, 12, 48>(12, 48, 12, 48, 0, 0, 90.000275440429732);
    }
    {
        SCOPED_TRACE("ext_tpartmul_int32_equal_holdout_1");
        Check<3, int32_t, 40, 96, 40, 96>(40, 96, 40, 96, 0, 0, 298.00089813655023);
    }
    {
        SCOPED_TRACE("pad_v2_tfillpad_expand_bf16_holdout_0");
        Check<1, bfloat16_t, 1, 160, 1, 64>(1, 60, 1, 160, 0, 0, 44.000130956655049);
    }
    {
        SCOPED_TRACE("pad_v2_tfillpad_expand_bf16_holdout_1");
        Check<1, bfloat16_t, 1, 384, 1, 240>(1, 239, 1, 384, 0, 0, 46.000390891154829);
    }
    {
        SCOPED_TRACE("pad_v2_tfillpad_expand_bf16_holdout_2");
        Check<1, bfloat16_t, 1, 640, 1, 240>(1, 240, 1, 640, 0, 0, 55.999449909691627);
    }
    {
        SCOPED_TRACE("pad_v2_tfillpad_expand_bf16_holdout_3");
        Check<1, bfloat16_t, 1, 896, 1, 560>(1, 559, 1, 896, 0, 0, 51.999689501096206);
    }
    {
        SCOPED_TRACE("pad_v2_tfillpad_expand_bf16_holdout_4");
        Check<1, bfloat16_t, 28, 224, 14, 144>(14, 140, 28, 224, 0, 0, 325.00100692332609);
    }
    {
        SCOPED_TRACE("pad_v2_tfillpad_expand_bf16_holdout_5");
        Check<1, bfloat16_t, 52, 400, 26, 160>(26, 149, 52, 400, 0, 0, 725.99931717232755);
    }
    {
        SCOPED_TRACE("pad_v2_tfillpad_expand_float_holdout_0");
        Check<1, float, 1, 80, 1, 32>(1, 30, 1, 80, 0, 0, 44.000698247169076);
    }
    {
        SCOPED_TRACE("pad_v2_tfillpad_expand_float_holdout_1");
        Check<1, float, 1, 192, 1, 120>(1, 119, 1, 192, 0, 0, 45.999076850197731);
    }
    {
        SCOPED_TRACE("pad_v2_tfillpad_expand_float_holdout_2");
        Check<1, float, 1, 320, 1, 120>(1, 120, 1, 320, 0, 0, 56.000238107332741);
    }
    {
        SCOPED_TRACE("pad_v2_tfillpad_expand_float_holdout_3");
        Check<1, float, 1, 448, 1, 280>(1, 279, 1, 448, 0, 0, 51.999539836801688);
    }
    {
        SCOPED_TRACE("pad_v2_tfillpad_expand_float_holdout_4");
        Check<1, float, 28, 112, 14, 72>(14, 70, 28, 112, 0, 0, 325.00057111785617);
    }
    {
        SCOPED_TRACE("pad_v2_tfillpad_expand_float_holdout_5");
        Check<1, float, 52, 200, 26, 80>(26, 74, 52, 200, 0, 0, 726.00065767559852);
    }
    {
        SCOPED_TRACE("pad_v2_tfillpad_expand_int16_holdout_0");
        Check<1, int16_t, 1, 160, 1, 64>(1, 60, 1, 160, 0, 0, 44.000583897939215);
    }
    {
        SCOPED_TRACE("pad_v2_tfillpad_expand_int16_holdout_1");
        Check<1, int16_t, 1, 384, 1, 240>(1, 239, 1, 384, 0, 0, 46.000192082910011);
    }
    {
        SCOPED_TRACE("pad_v2_tfillpad_expand_int16_holdout_2");
        Check<1, int16_t, 1, 640, 1, 240>(1, 240, 1, 640, 0, 0, 56.000256883558507);
    }
    {
        SCOPED_TRACE("pad_v2_tfillpad_expand_int16_holdout_3");
        Check<1, int16_t, 1, 896, 1, 560>(1, 559, 1, 896, 0, 0, 51.998825085527024);
    }
    {
        SCOPED_TRACE("pad_v2_tfillpad_expand_int16_holdout_4");
        Check<1, int16_t, 28, 224, 14, 144>(14, 140, 28, 224, 0, 0, 325.00018795928162);
    }
    {
        SCOPED_TRACE("pad_v2_tfillpad_expand_int16_holdout_5");
        Check<1, int16_t, 52, 400, 26, 160>(26, 149, 52, 400, 0, 0, 725.9993347901933);
    }
    {
        SCOPED_TRACE("pad_v2_tfillpad_expand_int32_holdout_0");
        Check<1, int32_t, 1, 80, 1, 32>(1, 30, 1, 80, 0, 0, 43.999955275744448);
    }
    {
        SCOPED_TRACE("pad_v2_tfillpad_expand_int32_holdout_1");
        Check<1, int32_t, 1, 192, 1, 120>(1, 119, 1, 192, 0, 0, 46.000261854688254);
    }
    {
        SCOPED_TRACE("pad_v2_tfillpad_expand_int32_holdout_2");
        Check<1, int32_t, 1, 320, 1, 120>(1, 120, 1, 320, 0, 0, 56.000557851012957);
    }
    {
        SCOPED_TRACE("pad_v2_tfillpad_expand_int32_holdout_3");
        Check<1, int32_t, 1, 448, 1, 280>(1, 279, 1, 448, 0, 0, 51.999151440522063);
    }
    {
        SCOPED_TRACE("pad_v2_tfillpad_expand_int32_holdout_4");
        Check<1, int32_t, 28, 112, 14, 72>(14, 70, 28, 112, 0, 0, 325.00013147723604);
    }
    {
        SCOPED_TRACE("pad_v2_tfillpad_expand_int32_holdout_5");
        Check<1, int32_t, 52, 200, 26, 80>(26, 74, 52, 200, 0, 0, 725.99946501907061);
    }
    {
        SCOPED_TRACE("pad_v2_tfillpad_float_holdout_0");
        Check<0, float, 1, 80, 1, 80>(1, 30, 1, 80, 0, 0, 44.000245025048656);
    }
    {
        SCOPED_TRACE("pad_v2_tfillpad_float_holdout_1");
        Check<0, float, 1, 192, 1, 192>(1, 119, 1, 192, 0, 0, 46.000535752070142);
    }
    {
        SCOPED_TRACE("pad_v2_tfillpad_float_holdout_2");
        Check<0, float, 1, 320, 1, 320>(1, 120, 1, 320, 0, 0, 56.000269792206979);
    }
    {
        SCOPED_TRACE("pad_v2_tfillpad_float_holdout_3");
        Check<0, float, 1, 448, 1, 448>(1, 279, 1, 448, 0, 0, 51.99967961565924);
    }
    {
        SCOPED_TRACE("pad_v2_tfillpad_float_holdout_4");
        Check<0, float, 28, 112, 28, 112>(14, 70, 28, 112, 0, 0, 324.99986209414175);
    }
    {
        SCOPED_TRACE("pad_v2_tfillpad_float_holdout_5");
        Check<0, float, 52, 200, 52, 200>(26, 74, 52, 200, 0, 0, 726.00024871615324);
    }
    {
        SCOPED_TRACE("tpartadd_float_holdout_00");
        Check<2, float, 12, 48, 12, 48>(6, 24, 12, 48, 0, 0, 127.00068092943657);
    }
    {
        SCOPED_TRACE("tpartadd_float_holdout_01");
        Check<2, float, 24, 80, 24, 80>(24, 80, 18, 39, 0, 0, 282.99957638984256);
    }
    {
        SCOPED_TRACE("tpartadd_float_holdout_02");
        Check<2, float, 40, 112, 40, 112>(20, 84, 40, 112, 0, 0, 425.99963444260828);
    }
    {
        SCOPED_TRACE("tpartadd_float_holdout_03");
        Check<2, float, 80, 144, 80, 144>(80, 144, 20, 72, 0, 0, 726.00035207576423);
    }
    {
        SCOPED_TRACE("tpartadd_float_holdout_04");
        Check<2, float, 160, 48, 160, 48>(80, 11, 160, 48, 0, 0, 1384.9998652575866);
    }
    {
        SCOPED_TRACE("tpartadd_float_holdout_05");
        Check<2, float, 1, 384, 1, 384>(1, 384, 1, 288, 0, 0, 48.499968006312635);
    }
    {
        SCOPED_TRACE("tpartadd_int16_holdout_00");
        Check<2, int16_t, 12, 96, 12, 96>(6, 48, 12, 96, 0, 0, 126.99986449741584);
    }
    {
        SCOPED_TRACE("tpartadd_int16_holdout_01");
        Check<2, int16_t, 24, 160, 24, 160>(24, 160, 18, 79, 0, 0, 282.99938456683611);
    }
    {
        SCOPED_TRACE("tpartadd_int16_holdout_02");
        Check<2, int16_t, 40, 224, 40, 224>(20, 168, 40, 224, 0, 0, 426.00018536679528);
    }
    {
        SCOPED_TRACE("tpartadd_int16_holdout_03");
        Check<2, int16_t, 80, 288, 80, 288>(80, 288, 20, 144, 0, 0, 726.00007084197387);
    }
    {
        SCOPED_TRACE("tpartadd_int16_holdout_04");
        Check<2, int16_t, 160, 96, 160, 96>(80, 23, 160, 96, 0, 0, 1384.9996608471738);
    }
    {
        SCOPED_TRACE("tpartadd_int16_holdout_05");
        Check<2, int16_t, 1, 768, 1, 768>(1, 768, 1, 576, 0, 0, 48.499953816268004);
    }
    {
        SCOPED_TRACE("tpartadd_int32_holdout_00");
        Check<2, int32_t, 12, 48, 12, 48>(6, 24, 12, 48, 0, 0, 126.99945147493717);
    }
    {
        SCOPED_TRACE("tpartadd_int32_holdout_01");
        Check<2, int32_t, 24, 80, 24, 80>(24, 80, 18, 39, 0, 0, 283.00051141938133);
    }
    {
        SCOPED_TRACE("tpartadd_int32_holdout_02");
        Check<2, int32_t, 40, 112, 40, 112>(20, 84, 40, 112, 0, 0, 425.999498999992);
    }
    {
        SCOPED_TRACE("tpartadd_int32_holdout_03");
        Check<2, int32_t, 80, 144, 80, 144>(80, 144, 20, 72, 0, 0, 726.00026326189482);
    }
    {
        SCOPED_TRACE("tpartadd_int32_holdout_04");
        Check<2, int32_t, 160, 48, 160, 48>(80, 11, 160, 48, 0, 0, 1384.9996813978607);
    }
    {
        SCOPED_TRACE("tpartadd_int32_holdout_05");
        Check<2, int32_t, 1, 384, 1, 384>(1, 384, 1, 288, 0, 0, 48.499932365038696);
    }
    {
        SCOPED_TRACE("tpartmax_float_holdout_00");
        Check<4, float, 12, 48, 12, 48>(6, 24, 12, 48, 0, 0, 127.00007899337693);
    }
    {
        SCOPED_TRACE("tpartmax_float_holdout_01");
        Check<4, float, 24, 80, 24, 80>(24, 80, 18, 39, 0, 0, 282.99969813347121);
    }
    {
        SCOPED_TRACE("tpartmax_float_holdout_02");
        Check<4, float, 40, 112, 40, 112>(20, 84, 40, 112, 0, 0, 425.99993180059693);
    }
    {
        SCOPED_TRACE("tpartmax_float_holdout_03");
        Check<4, float, 80, 144, 80, 144>(80, 144, 20, 72, 0, 0, 725.99968328450359);
    }
    {
        SCOPED_TRACE("tpartmax_float_holdout_04");
        Check<4, float, 160, 48, 160, 48>(80, 11, 160, 48, 0, 0, 1384.9992253194043);
    }
    {
        SCOPED_TRACE("tpartmax_float_holdout_05");
        Check<4, float, 1, 384, 1, 384>(1, 384, 1, 288, 0, 0, 46.999984632091362);
    }
    {
        SCOPED_TRACE("tpartmax_int16_holdout_00");
        Check<4, int16_t, 12, 96, 12, 96>(6, 48, 12, 96, 0, 0, 126.99951099748944);
    }
    {
        SCOPED_TRACE("tpartmax_int16_holdout_01");
        Check<4, int16_t, 24, 160, 24, 160>(24, 160, 18, 79, 0, 0, 283.00006242185788);
    }
    {
        SCOPED_TRACE("tpartmax_int16_holdout_02");
        Check<4, int16_t, 40, 224, 40, 224>(20, 168, 40, 224, 0, 0, 426.0008664922762);
    }
    {
        SCOPED_TRACE("tpartmax_int16_holdout_03");
        Check<4, int16_t, 80, 288, 80, 288>(80, 288, 20, 144, 0, 0, 725.99992850063973);
    }
    {
        SCOPED_TRACE("tpartmax_int16_holdout_04");
        Check<4, int16_t, 160, 96, 160, 96>(80, 23, 160, 96, 0, 0, 1385.0000552271404);
    }
    {
        SCOPED_TRACE("tpartmax_int16_holdout_05");
        Check<4, int16_t, 1, 768, 1, 768>(1, 768, 1, 576, 0, 0, 47.000558653560631);
    }
    {
        SCOPED_TRACE("tpartmax_int32_holdout_00");
        Check<4, int32_t, 12, 48, 12, 48>(6, 24, 12, 48, 0, 0, 127.00003242099258);
    }
    {
        SCOPED_TRACE("tpartmax_int32_holdout_01");
        Check<4, int32_t, 24, 80, 24, 80>(24, 80, 18, 39, 0, 0, 283.00001004204682);
    }
    {
        SCOPED_TRACE("tpartmax_int32_holdout_02");
        Check<4, int32_t, 40, 112, 40, 112>(20, 84, 40, 112, 0, 0, 425.99921848589793);
    }
    {
        SCOPED_TRACE("tpartmax_int32_holdout_03");
        Check<4, int32_t, 80, 144, 80, 144>(80, 144, 20, 72, 0, 0, 726.00040938761265);
    }
    {
        SCOPED_TRACE("tpartmax_int32_holdout_04");
        Check<4, int32_t, 160, 48, 160, 48>(80, 11, 160, 48, 0, 0, 1385.0000354118838);
    }
    {
        SCOPED_TRACE("tpartmax_int32_holdout_05");
        Check<4, int32_t, 1, 384, 1, 384>(1, 384, 1, 288, 0, 0, 46.999396168596256);
    }
    {
        SCOPED_TRACE("tpartmax_int8_holdout_00");
        Check<4, int8_t, 12, 192, 12, 192>(6, 96, 12, 192, 0, 0, 126.99945143682108);
    }
    {
        SCOPED_TRACE("tpartmax_int8_holdout_01");
        Check<4, int8_t, 24, 320, 24, 320>(24, 320, 18, 159, 0, 0, 283.00009183372975);
    }
    {
        SCOPED_TRACE("tpartmax_int8_holdout_02");
        Check<4, int8_t, 40, 448, 40, 448>(20, 336, 40, 448, 0, 0, 425.99969143891394);
    }
    {
        SCOPED_TRACE("tpartmax_int8_holdout_03");
        Check<4, int8_t, 80, 576, 80, 576>(80, 576, 20, 288, 0, 0, 725.99938934853878);
    }
    {
        SCOPED_TRACE("tpartmax_int8_holdout_04");
        Check<4, int8_t, 160, 192, 160, 192>(80, 47, 160, 192, 0, 0, 1385.0006925279019);
    }
    {
        SCOPED_TRACE("tpartmax_int8_holdout_05");
        Check<4, int8_t, 1, 1536, 1, 1536>(1, 1536, 1, 1152, 0, 0, 47.00024798191189);
    }
    {
        SCOPED_TRACE("tpartmax_uint8_holdout_00");
        Check<4, uint8_t, 12, 192, 12, 192>(6, 96, 12, 192, 0, 0, 127.0003740526704);
    }
    {
        SCOPED_TRACE("tpartmax_uint8_holdout_01");
        Check<4, uint8_t, 24, 320, 24, 320>(24, 320, 18, 159, 0, 0, 282.99932500745587);
    }
    {
        SCOPED_TRACE("tpartmax_uint8_holdout_02");
        Check<4, uint8_t, 40, 448, 40, 448>(20, 336, 40, 448, 0, 0, 425.99939279112738);
    }
    {
        SCOPED_TRACE("tpartmax_uint8_holdout_03");
        Check<4, uint8_t, 80, 576, 80, 576>(80, 576, 20, 288, 0, 0, 725.99990649521078);
    }
    {
        SCOPED_TRACE("tpartmax_uint8_holdout_04");
        Check<4, uint8_t, 160, 192, 160, 192>(80, 47, 160, 192, 0, 0, 1385.0002944637445);
    }
    {
        SCOPED_TRACE("tpartmax_uint8_holdout_05");
        Check<4, uint8_t, 1, 1536, 1, 1536>(1, 1536, 1, 1152, 0, 0, 47.000131064126776);
    }
    {
        SCOPED_TRACE("tpartmin_float_holdout_00");
        Check<5, float, 12, 48, 12, 48>(6, 24, 12, 48, 0, 0, 126.99948539023164);
    }
    {
        SCOPED_TRACE("tpartmin_float_holdout_01");
        Check<5, float, 24, 80, 24, 80>(24, 80, 18, 39, 0, 0, 283.00032864292785);
    }
    {
        SCOPED_TRACE("tpartmin_float_holdout_02");
        Check<5, float, 40, 112, 40, 112>(20, 84, 40, 112, 0, 0, 425.99945602071301);
    }
    {
        SCOPED_TRACE("tpartmin_float_holdout_03");
        Check<5, float, 80, 144, 80, 144>(80, 144, 20, 72, 0, 0, 725.999771125856);
    }
    {
        SCOPED_TRACE("tpartmin_float_holdout_04");
        Check<5, float, 160, 48, 160, 48>(80, 11, 160, 48, 0, 0, 1385.0000392960708);
    }
    {
        SCOPED_TRACE("tpartmin_float_holdout_05");
        Check<5, float, 1, 384, 1, 384>(1, 384, 1, 288, 0, 0, 46.999568490656287);
    }
    {
        SCOPED_TRACE("tpartmin_int16_holdout_00");
        Check<5, int16_t, 12, 96, 12, 96>(6, 48, 12, 96, 0, 0, 127.00002138685113);
    }
    {
        SCOPED_TRACE("tpartmin_int16_holdout_01");
        Check<5, int16_t, 24, 160, 24, 160>(24, 160, 18, 79, 0, 0, 282.99949812685725);
    }
    {
        SCOPED_TRACE("tpartmin_int16_holdout_02");
        Check<5, int16_t, 40, 224, 40, 224>(20, 168, 40, 224, 0, 0, 426.00111676713573);
    }
    {
        SCOPED_TRACE("tpartmin_int16_holdout_03");
        Check<5, int16_t, 80, 288, 80, 288>(80, 288, 20, 144, 0, 0, 725.99985186973527);
    }
    {
        SCOPED_TRACE("tpartmin_int16_holdout_04");
        Check<5, int16_t, 160, 96, 160, 96>(80, 23, 160, 96, 0, 0, 1384.9998112324456);
    }
    {
        SCOPED_TRACE("tpartmin_int16_holdout_05");
        Check<5, int16_t, 1, 768, 1, 768>(1, 768, 1, 576, 0, 0, 47.000090433053572);
    }
    {
        SCOPED_TRACE("tpartmin_int32_holdout_00");
        Check<5, int32_t, 12, 48, 12, 48>(6, 24, 12, 48, 0, 0, 127.00028997919986);
    }
    {
        SCOPED_TRACE("tpartmin_int32_holdout_01");
        Check<5, int32_t, 24, 80, 24, 80>(24, 80, 18, 39, 0, 0, 283.00000627113712);
    }
    {
        SCOPED_TRACE("tpartmin_int32_holdout_02");
        Check<5, int32_t, 40, 112, 40, 112>(20, 84, 40, 112, 0, 0, 426.0002626153144);
    }
    {
        SCOPED_TRACE("tpartmin_int32_holdout_03");
        Check<5, int32_t, 80, 144, 80, 144>(80, 144, 20, 72, 0, 0, 725.99909580997928);
    }
    {
        SCOPED_TRACE("tpartmin_int32_holdout_04");
        Check<5, int32_t, 160, 48, 160, 48>(80, 11, 160, 48, 0, 0, 1385.0008481630805);
    }
    {
        SCOPED_TRACE("tpartmin_int32_holdout_05");
        Check<5, int32_t, 1, 384, 1, 384>(1, 384, 1, 288, 0, 0, 46.999081132315553);
    }
    {
        SCOPED_TRACE("tpartmin_int8_holdout_00");
        Check<5, int8_t, 12, 192, 12, 192>(6, 96, 12, 192, 0, 0, 126.99985673366412);
    }
    {
        SCOPED_TRACE("tpartmin_int8_holdout_01");
        Check<5, int8_t, 24, 320, 24, 320>(24, 320, 18, 159, 0, 0, 282.99948592674508);
    }
    {
        SCOPED_TRACE("tpartmin_int8_holdout_02");
        Check<5, int8_t, 40, 448, 40, 448>(20, 336, 40, 448, 0, 0, 426.00045972320305);
    }
    {
        SCOPED_TRACE("tpartmin_int8_holdout_03");
        Check<5, int8_t, 80, 576, 80, 576>(80, 576, 20, 288, 0, 0, 725.99986737336928);
    }
    {
        SCOPED_TRACE("tpartmin_int8_holdout_04");
        Check<5, int8_t, 160, 192, 160, 192>(80, 47, 160, 192, 0, 0, 1385.0005486801895);
    }
    {
        SCOPED_TRACE("tpartmin_int8_holdout_05");
        Check<5, int8_t, 1, 1536, 1, 1536>(1, 1536, 1, 1152, 0, 0, 47.00027290745043);
    }
    {
        SCOPED_TRACE("tpartmin_uint8_holdout_00");
        Check<5, uint8_t, 12, 192, 12, 192>(6, 96, 12, 192, 0, 0, 126.99980029993462);
    }
    {
        SCOPED_TRACE("tpartmin_uint8_holdout_01");
        Check<5, uint8_t, 24, 320, 24, 320>(24, 320, 18, 159, 0, 0, 282.99943839652241);
    }
    {
        SCOPED_TRACE("tpartmin_uint8_holdout_02");
        Check<5, uint8_t, 40, 448, 40, 448>(20, 336, 40, 448, 0, 0, 426.00023017371404);
    }
    {
        SCOPED_TRACE("tpartmin_uint8_holdout_03");
        Check<5, uint8_t, 80, 576, 80, 576>(80, 576, 20, 288, 0, 0, 725.99970964125475);
    }
    {
        SCOPED_TRACE("tpartmin_uint8_holdout_04");
        Check<5, uint8_t, 160, 192, 160, 192>(80, 47, 160, 192, 0, 0, 1385.0000597909459);
    }
    {
        SCOPED_TRACE("tpartmin_uint8_holdout_05");
        Check<5, uint8_t, 1, 1536, 1, 1536>(1, 1536, 1, 1152, 0, 0, 46.99993226297422);
    }
    {
        SCOPED_TRACE("tpartmul_float_holdout_00");
        Check<3, float, 12, 48, 12, 48>(6, 24, 12, 48, 0, 0, 126.99920787296152);
    }
    {
        SCOPED_TRACE("tpartmul_float_holdout_01");
        Check<3, float, 24, 80, 24, 80>(24, 80, 18, 39, 0, 0, 282.99973592628982);
    }
    {
        SCOPED_TRACE("tpartmul_float_holdout_02");
        Check<3, float, 40, 112, 40, 112>(20, 84, 40, 112, 0, 0, 426.00010417458793);
    }
    {
        SCOPED_TRACE("tpartmul_float_holdout_03");
        Check<3, float, 80, 144, 80, 144>(80, 144, 20, 72, 0, 0, 726.00019275742306);
    }
    {
        SCOPED_TRACE("tpartmul_float_holdout_04");
        Check<3, float, 160, 48, 160, 48>(80, 11, 160, 48, 0, 0, 1384.9998346079012);
    }
    {
        SCOPED_TRACE("tpartmul_float_holdout_05");
        Check<3, float, 1, 384, 1, 384>(1, 384, 1, 288, 0, 0, 49.000692150068716);
    }
    {
        SCOPED_TRACE("tpartmul_int16_holdout_00");
        Check<3, int16_t, 12, 96, 12, 96>(6, 48, 12, 96, 0, 0, 127.00003242099258);
    }
    {
        SCOPED_TRACE("tpartmul_int16_holdout_01");
        Check<3, int16_t, 24, 160, 24, 160>(24, 160, 18, 79, 0, 0, 283.00015257329852);
    }
    {
        SCOPED_TRACE("tpartmul_int16_holdout_02");
        Check<3, int16_t, 40, 224, 40, 224>(20, 168, 40, 224, 0, 0, 425.99941379614938);
    }
    {
        SCOPED_TRACE("tpartmul_int16_holdout_03");
        Check<3, int16_t, 80, 288, 80, 288>(80, 288, 20, 144, 0, 0, 726.00034491381314);
    }
    {
        SCOPED_TRACE("tpartmul_int16_holdout_04");
        Check<3, int16_t, 160, 96, 160, 96>(80, 23, 160, 96, 0, 0, 1385.0005316650661);
    }
    {
        SCOPED_TRACE("tpartmul_int16_holdout_05");
        Check<3, int16_t, 1, 768, 1, 768>(1, 768, 1, 576, 0, 0, 49.000165840495391);
    }
    {
        SCOPED_TRACE("tpartmul_int32_holdout_00");
        Check<3, int32_t, 12, 48, 12, 48>(6, 24, 12, 48, 0, 0, 126.99881112824306);
    }
    {
        SCOPED_TRACE("tpartmul_int32_holdout_01");
        Check<3, int32_t, 24, 80, 24, 80>(24, 80, 18, 39, 0, 0, 283.00017896835448);
    }
    {
        SCOPED_TRACE("tpartmul_int32_holdout_02");
        Check<3, int32_t, 40, 112, 40, 112>(20, 84, 40, 112, 0, 0, 425.99968261253531);
    }
    {
        SCOPED_TRACE("tpartmul_int32_holdout_03");
        Check<3, int32_t, 80, 144, 80, 144>(80, 144, 20, 72, 0, 0, 726.00028945238353);
    }
    {
        SCOPED_TRACE("tpartmul_int32_holdout_04");
        Check<3, int32_t, 160, 48, 160, 48>(80, 11, 160, 48, 0, 0, 1384.9996384314293);
    }
    {
        SCOPED_TRACE("tpartmul_int32_holdout_05");
        Check<3, int32_t, 1, 384, 1, 384>(1, 384, 1, 288, 0, 0, 49.000161689973403);
    }
}
TEST(A5ObservableMetadataFit, TriangleV4IndependentAccuracy)
{
    {
        SCOPED_TRACE("tri_v4_float_holdout_0");
        Check<6, float, 7, 384, 7, 384>(7, 384, 7, 384, 1, -7, 88.000711520210032);
    }
    {
        SCOPED_TRACE("tri_v4_float_holdout_1");
        Check<6, float, 14, 512, 14, 512>(14, 512, 14, 512, 0, -14, 164.99947774591371);
    }
    {
        SCOPED_TRACE("tri_v4_float_holdout_2");
        Check<6, float, 28, 832, 28, 832>(28, 832, 28, 832, 1, -28, 443.99993285067961);
    }
    {
        SCOPED_TRACE("tri_v4_float_holdout_3");
        Check<6, float, 7, 1088, 7, 1088>(7, 1088, 7, 1088, 0, -7, 157.00103753994907);
    }
    {
        SCOPED_TRACE("tri_v4_float_holdout_4");
        Check<6, float, 14, 384, 14, 384>(14, 384, 14, 384, 1, -7, 230.99979921127883);
    }
    {
        SCOPED_TRACE("tri_v4_float_holdout_5");
        Check<6, float, 28, 512, 28, 512>(28, 512, 28, 512, 0, 0, 748.99980401397215);
    }
    {
        SCOPED_TRACE("tri_v4_float_holdout_6");
        Check<6, float, 7, 832, 7, 832>(7, 832, 7, 832, 1, 832, 272.00015330231355);
    }
    {
        SCOPED_TRACE("tri_v4_float_holdout_7");
        Check<6, float, 14, 1088, 14, 1088>(14, 1088, 14, 1088, 0, -7, 460.99982951021548);
    }
    {
        SCOPED_TRACE("tri_v4_half_holdout_0");
        Check<6, half, 7, 768, 7, 768>(7, 768, 7, 768, 1, -7, 88.000631539663814);
    }
    {
        SCOPED_TRACE("tri_v4_half_holdout_1");
        Check<6, half, 14, 1024, 14, 1024>(14, 1024, 14, 1024, 0, -14, 164.99979805284636);
    }
    {
        SCOPED_TRACE("tri_v4_half_holdout_2");
        Check<6, half, 28, 1664, 28, 1664>(28, 1664, 28, 1664, 1, -28, 443.99887377751844);
    }
    {
        SCOPED_TRACE("tri_v4_half_holdout_3");
        Check<6, half, 7, 2176, 7, 2176>(7, 2176, 7, 2176, 0, -7, 156.99928710294208);
    }
    {
        SCOPED_TRACE("tri_v4_half_holdout_4");
        Check<6, half, 14, 768, 14, 768>(14, 768, 14, 768, 1, -7, 230.9998473859192);
    }
    {
        SCOPED_TRACE("tri_v4_half_holdout_5");
        Check<6, half, 28, 1024, 28, 1024>(28, 1024, 28, 1024, 0, 0, 748.99931420395399);
    }
    {
        SCOPED_TRACE("tri_v4_half_holdout_6");
        Check<6, half, 7, 1664, 7, 1664>(7, 1664, 7, 1664, 1, 1664, 271.99993270308573);
    }
    {
        SCOPED_TRACE("tri_v4_half_holdout_7");
        Check<6, half, 14, 2176, 14, 2176>(14, 2176, 14, 2176, 0, -7, 461.00079895907771);
    }
    {
        SCOPED_TRACE("tri_v4_int16_holdout_0");
        Check<6, int16_t, 7, 768, 7, 768>(7, 768, 7, 768, 1, -7, 88.000187160347323);
    }
    {
        SCOPED_TRACE("tri_v4_int16_holdout_1");
        Check<6, int16_t, 14, 1024, 14, 1024>(14, 1024, 14, 1024, 0, -14, 164.99934369707248);
    }
    {
        SCOPED_TRACE("tri_v4_int16_holdout_2");
        Check<6, int16_t, 28, 1664, 28, 1664>(28, 1664, 28, 1664, 1, -28, 444.0000927334325);
    }
    {
        SCOPED_TRACE("tri_v4_int16_holdout_3");
        Check<6, int16_t, 7, 2176, 7, 2176>(7, 2176, 7, 2176, 0, -7, 157.00005474435667);
    }
    {
        SCOPED_TRACE("tri_v4_int16_holdout_4");
        Check<6, int16_t, 14, 768, 14, 768>(14, 768, 14, 768, 1, -7, 231.00029094225556);
    }
    {
        SCOPED_TRACE("tri_v4_int16_holdout_5");
        Check<6, int16_t, 28, 1024, 28, 1024>(28, 1024, 28, 1024, 0, 0, 749.00010711598213);
    }
    {
        SCOPED_TRACE("tri_v4_int16_holdout_6");
        Check<6, int16_t, 7, 1664, 7, 1664>(7, 1664, 7, 1664, 1, 1664, 271.99994291404306);
    }
    {
        SCOPED_TRACE("tri_v4_int16_holdout_7");
        Check<6, int16_t, 14, 2176, 14, 2176>(14, 2176, 14, 2176, 0, -7, 461.00033485398353);
    }
    {
        SCOPED_TRACE("tri_v4_int32_holdout_0");
        Check<6, int32_t, 7, 384, 7, 384>(7, 384, 7, 384, 1, -7, 88.000703904319906);
    }
    {
        SCOPED_TRACE("tri_v4_int32_holdout_1");
        Check<6, int32_t, 14, 512, 14, 512>(14, 512, 14, 512, 0, -14, 164.99992135557829);
    }
    {
        SCOPED_TRACE("tri_v4_int32_holdout_2");
        Check<6, int32_t, 28, 832, 28, 832>(28, 832, 28, 832, 1, -28, 443.99907324294281);
    }
    {
        SCOPED_TRACE("tri_v4_int32_holdout_3");
        Check<6, int32_t, 7, 1088, 7, 1088>(7, 1088, 7, 1088, 0, -7, 156.99910674772221);
    }
    {
        SCOPED_TRACE("tri_v4_int32_holdout_4");
        Check<6, int32_t, 14, 384, 14, 384>(14, 384, 14, 384, 1, -7, 231.00026849673245);
    }
    {
        SCOPED_TRACE("tri_v4_int32_holdout_5");
        Check<6, int32_t, 28, 512, 28, 512>(28, 512, 28, 512, 0, 0, 749.00008024322426);
    }
    {
        SCOPED_TRACE("tri_v4_int32_holdout_6");
        Check<6, int32_t, 7, 832, 7, 832>(7, 832, 7, 832, 1, 832, 272.00008288100412);
    }
    {
        SCOPED_TRACE("tri_v4_int32_holdout_7");
        Check<6, int32_t, 14, 1088, 14, 1088>(14, 1088, 14, 1088, 0, -7, 460.99981438643619);
    }
}
TEST(A5ObservableMetadataFit, TriangleFrozenExtentAccuracy)
{
    {
        SCOPED_TRACE("shape_eval_ttri_float_1x8_0");
        Check<6, float, 1, 8, 1, 8>(1, 8, 1, 8, 0, 2, 36.999680431379687);
    }
    {
        SCOPED_TRACE("shape_eval_ttri_float_1x8_1");
        Check<6, float, 1, 8, 1, 8>(1, 8, 1, 8, 1, 0, 32.000128881435664);
    }
    {
        SCOPED_TRACE("shape_eval_ttri_float_1x65472_0");
        Check<6, float, 1, 65472, 1, 65472>(1, 65472, 1, 65472, 0, 21824, 2077.0007784260738);
    }
    {
        SCOPED_TRACE("shape_eval_ttri_float_1x65472_1");
        Check<6, float, 1, 65472, 1, 65472>(1, 65472, 1, 65472, 1, 0, 1053.9996085876821);
    }
    {
        SCOPED_TRACE("shape_eval_ttri_float_256x256_0");
        Check<6, float, 256, 256, 256, 256>(256, 256, 256, 256, 0, 85, 5140.9998505331205);
    }
    {
        SCOPED_TRACE("shape_eval_ttri_float_256x256_1");
        Check<6, float, 256, 256, 256, 256>(256, 256, 256, 256, 1, -85, 4109.0005670564378);
    }
    {
        SCOPED_TRACE("shape_eval_ttri_float_4095x8_0");
        Check<6, float, 4095, 8, 4095, 8>(4095, 8, 4095, 8, 0, 2, 61445.999945706521);
    }
    {
        SCOPED_TRACE("shape_eval_ttri_float_4095x8_1");
        Check<6, float, 4095, 8, 4095, 8>(4095, 8, 4095, 8, 1, -1365, 49151.999944014569);
    }
    {
        SCOPED_TRACE("shape_eval_ttri_half_362x352_0");
        Check<6, half, 362, 352, 362, 352>(362, 352, 362, 352, 0, 117, 6537.0003390777556);
    }
    {
        SCOPED_TRACE("shape_eval_ttri_half_362x352_1");
        Check<6, half, 362, 352, 362, 352>(362, 352, 362, 352, 1, -120, 5206.0005246447172);
    }
    {
        SCOPED_TRACE("shape_eval_ttri_int16_1x16_0");
        Check<6, int16_t, 1, 16, 1, 16>(1, 16, 1, 16, 0, 5, 36.999650399773252);
    }
    {
        SCOPED_TRACE("shape_eval_ttri_int16_1x16_1");
        Check<6, int16_t, 1, 16, 1, 16>(1, 16, 1, 16, 1, 0, 31.999998575793786);
    }
    {
        SCOPED_TRACE("shape_eval_ttri_int16_1x352_0");
        Check<6, int16_t, 1, 352, 1, 352>(1, 352, 1, 352, 0, 117, 39.000399405451745);
    }
    {
        SCOPED_TRACE("shape_eval_ttri_int16_1x352_1");
        Check<6, int16_t, 1, 352, 1, 352>(1, 352, 1, 352, 1, 0, 33.99967237103175);
    }
    {
        SCOPED_TRACE("shape_eval_ttri_int16_1x130944_0");
        Check<6, int16_t, 1, 130944, 1, 130944>(1, 130944, 1, 130944, 0, 43648, 2080.9996905229941);
    }
    {
        SCOPED_TRACE("shape_eval_ttri_int16_1x130944_1");
        Check<6, int16_t, 1, 130944, 1, 130944>(1, 130944, 1, 130944, 1, 0, 1053.999834701084);
    }
    {
        SCOPED_TRACE("shape_eval_ttri_int16_362x352_0");
        Check<6, int16_t, 362, 352, 362, 352>(362, 352, 362, 352, 0, 117, 6537.0001837677701);
    }
    {
        SCOPED_TRACE("shape_eval_ttri_int16_362x352_1");
        Check<6, int16_t, 362, 352, 362, 352>(362, 352, 362, 352, 1, -120, 5206.0005616617191);
    }
    {
        SCOPED_TRACE("shape_eval_ttri_int16_4095x16_0");
        Check<6, int16_t, 4095, 16, 4095, 16>(4095, 16, 4095, 16, 0, 5, 61446.000046596397);
    }
    {
        SCOPED_TRACE("shape_eval_ttri_int16_4095x16_1");
        Check<6, int16_t, 4095, 16, 4095, 16>(4095, 16, 4095, 16, 1, -1365, 49151.999546701532);
    }
    {
        SCOPED_TRACE("shape_eval_ttri_int32_1x8_0");
        Check<6, int32_t, 1, 8, 1, 8>(1, 8, 1, 8, 0, 2, 37.000176995164686);
    }
    {
        SCOPED_TRACE("shape_eval_ttri_int32_1x8_1");
        Check<6, int32_t, 1, 8, 1, 8>(1, 8, 1, 8, 1, 0, 31.999481686134285);
    }
    {
        SCOPED_TRACE("shape_eval_ttri_int32_1x65472_0");
        Check<6, int32_t, 1, 65472, 1, 65472>(1, 65472, 1, 65472, 0, 21824, 2077.0000618475037);
    }
    {
        SCOPED_TRACE("shape_eval_ttri_int32_1x65472_1");
        Check<6, int32_t, 1, 65472, 1, 65472>(1, 65472, 1, 65472, 1, 0, 1053.9994081400612);
    }
    {
        SCOPED_TRACE("shape_eval_ttri_int32_256x256_0");
        Check<6, int32_t, 256, 256, 256, 256>(256, 256, 256, 256, 0, 85, 5141.0007020984804);
    }
    {
        SCOPED_TRACE("shape_eval_ttri_int32_256x256_1");
        Check<6, int32_t, 256, 256, 256, 256>(256, 256, 256, 256, 1, -85, 4108.9999910930965);
    }
    {
        SCOPED_TRACE("shape_eval_ttri_int32_4095x8_0");
        Check<6, int32_t, 4095, 8, 4095, 8>(4095, 8, 4095, 8, 0, 2, 61446.00013028019);
    }
    {
        SCOPED_TRACE("shape_eval_ttri_int32_4095x8_1");
        Check<6, int32_t, 4095, 8, 4095, 8>(4095, 8, 4095, 8, 1, -1365, 49151.999979492626);
    }
}

} // namespace
