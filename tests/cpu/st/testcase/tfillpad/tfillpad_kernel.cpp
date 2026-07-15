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
#include <limits>
#include <bit>
#include <algorithm>

using namespace std;
using namespace pto;

// Custom pad value constant for -1.0f (bit pattern 0xBF800000)
constexpr PadValue PadCustomNeg1 = PadValueCustom(-1.0f);
constexpr PadValue PadCustomNeg1_Half = PadValueCustom16(0xBC00); // fp16 -1.0

#define LOGSIZE 128
#define PRINTLOG 4
#define LOG(x)

template <int shape0, int shape1, int shape2, int shape3, int shape4>
__inline__ auto getOptDynShape(int gShape0, int gShape1, int gShape2, int gShape3, int gShape4)
{
    if constexpr (shape0 == 1) {
        using DynShapeDim5 = pto::Shape<1, -1, -1, -1, -1>;
        DynShapeDim5 dynShape(gShape1, gShape2, gShape3, gShape4);
        return dynShape;
    } else if constexpr (shape0 == 1 && shape1 == 1) {
        using DynShapeDim5 = pto::Shape<1, 1, -1, -1, -1>;
        DynShapeDim5 dynShape(gShape2, gShape3, gShape4);
        return dynShape;
    } else if constexpr (shape0 == 1 && shape1 == 1 && shape2 == 1) {
        using DynShapeDim5 = pto::Shape<1, 1, 1, -1, -1>;
        DynShapeDim5 dynShape(gShape3, gShape4);
        return dynShape;
    } else if constexpr (shape0 == 1 && shape1 == 1 && shape2 == 1 && shape3 == 1) {
        using DynShapeDim5 = pto::Shape<1, 1, 1, 1, -1>;
        DynShapeDim5 dynShape(gShape4);
        return dynShape;
    } else {
        using DynShapeDim5 = pto::Shape<-1, -1, -1, -1, -1>;
        DynShapeDim5 dynShape(gShape0, gShape1, gShape2, gShape3, gShape4);
        return dynShape;
    }
}

// case shape is static, but testing would do dynamic or static test
template <
    typename T, int shape0, int shape1, int shape2, int shape3, int shape4, int tRows, int tCols, BLayout major,
    int dyn>
__inline__ auto getGloablTensor(T* addr, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4)
{
    if constexpr (dyn) {
        int stride0 = gShape1 * gShape2 * shape3 * shape4;
        int stride1 = gShape2 * shape3 * shape4;
        int stride2 = shape3 * shape4;

        using DynStrideDim5 = pto::Stride<-1, -1, -1, -1, -1>;
        auto dynShape =
            getOptDynShape<shape0, shape1, shape2, shape3, shape4>(gShape0, gShape1, gShape2, gShape3, gShape4);
        using GlobalData = GlobalTensor<T, decltype(dynShape), DynStrideDim5>;

        if constexpr (major == BLayout::RowMajor) {
            GlobalData srcGlobal(addr, dynShape, DynStrideDim5(stride0, stride1, stride2, shape4, 1));
            return srcGlobal;
        } else {
            GlobalData srcGlobal(addr, dynShape, DynStrideDim5(stride0, stride1, stride2, 1, shape4));
            return srcGlobal;
        }
    } else // static
    {
        constexpr int stride0 = shape1 * shape2 * shape3 * shape4;
        constexpr int stride1 = shape2 * shape3 * shape4;
        constexpr int stride2 = shape3 * shape4;
        using StaticShapeDim5 = Shape<shape0, shape1, shape2, tRows, tCols>;

        if constexpr (major == BLayout::RowMajor) {
            using StaticStrideDim5 = pto::Stride<stride0, stride1, stride2, shape4, 1>;
            using GlobalData = GlobalTensor<T, StaticShapeDim5, StaticStrideDim5>;
            GlobalData srcGlobal(addr);
            return srcGlobal;
        } else {
            using StaticStrideDim5 = pto::Stride<stride0, stride1, stride2, 1, shape4>;
            using GlobalData = GlobalTensor<T, StaticShapeDim5, StaticStrideDim5>;
            GlobalData srcGlobal(addr);
            return srcGlobal;
        }
    }
}

#define type_32_aligned(T) (32 / sizeof(T))
#define align_to_32B(x, T) ((((x) + type_32_aligned(T) - 1) / type_32_aligned(T)) * (type_32_aligned(T)))

template <
    typename T, int shape0, int shape1, int shape2, int shape3, int shape4, int kTRows_, int kTCols_, int dyn_,
    PadValue LoadPadVal_ = PadValue::Null, PadValue FillPadVal_ = PadValue::Null, bool inplace = false,
    bool expand = false>
void runTFILLPAD(T* out, T* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols)
{
#ifdef DEBUGLOG
    gLog += block_idx * LOGSIZE;
#endif
    T* ubaddr0 = 0x0;
    T* ubaddr1 = (T*)0x18000;
    if (inplace)
        ubaddr1 = ubaddr0;

    constexpr int shape4_aligned = align_to_32B(shape4, T);
    constexpr int kGTRows = kTRows_ / shape0 / shape1 / shape2; // Dst Tile Rows, merged all shape0*shape1*shape2 row
    // int srcOffset = (block_idx) * (shape3/block_num) * shape4;
    auto srcGlobal =
        getGloablTensor<T, shape0, shape1, shape2, shape3, shape4, kGTRows, shape4, BLayout::RowMajor, dyn_>(
            src, gShape0, gShape1, gShape2, shape3, shape4);
    // int dstOffset = (block_idx) * (shape3/block_num) * kTCols_;
    auto dstGlobal =
        getGloablTensor<T, shape0, shape1, shape2, shape3, kTCols_, kGTRows, kTCols_, BLayout::RowMajor, 0>(
            out, gShape0, gShape1, gShape2, shape3, kTCols_); // dst TStore GlobalTensor just use static

    volatile uint64_t t0, t1, t2;

    using TileDataP =
        Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, kTCols_, SLayout::NoneBox, 512, FillPadVal_>;
    TileDataP vecTileP(kTRows_);
    TASSIGN(vecTileP, (uint64_t)ubaddr1);

    if constexpr (expand) {
        using TileData = Tile<
            TileType::Vec, T, kTRows_, shape4_aligned, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, LoadPadVal_>;

        TileData vecTile(shape3, shape4);
        TASSIGN(vecTile, (uint64_t)ubaddr0);

        TLOAD(vecTile, srcGlobal);
        TFILLPAD_EXPAND(vecTileP, vecTile);
    } else {
        using TileData =
            Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, LoadPadVal_>;

        TileData vecTile(shape3, shape4);
        TASSIGN(vecTile, (uint64_t)ubaddr0);

        TLOAD(vecTile, srcGlobal);
        if constexpr (inplace)
            TFILLPAD_INPLACE(vecTileP, vecTile);
        else
            TFILLPAD(vecTileP, vecTile);
    }
    TSTORE(dstGlobal, vecTileP);
}

void launchTFILLPAD_1(uint8_t* out, uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols)
{
    runTFILLPAD<float, 1, 1, 1, 128, 127, 128, 128, 1, PadValue::Max, PadValue::Max>(
        (float*)out, (float*)src, gShape0, gShape1, gShape2, gRows, gCols);
}

void launchTFILLPAD_2(uint8_t* out, uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols)
{
    runTFILLPAD<float, 1, 1, 1, 128, 127, 128, 160, 1, PadValue::Max, PadValue::Max>(
        (float*)out, (float*)src, gShape0, gShape1, gShape2, gRows, gCols);
}

void launchTFILLPAD_3(uint8_t* out, uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols)
{
    runTFILLPAD<float, 1, 1, 1, 128, 127, 128, 160, 1, PadValue::Min, PadValue::Max>(
        (float*)out, (float*)src, gShape0, gShape1, gShape2, gRows, gCols);
}

void launchTFILLPAD_4(uint8_t* out, uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols)
{
    runTFILLPAD<float, 1, 1, 1, 260, 7, 260, 16, 1, PadValue::Min, PadValue::Max>(
        (float*)out, (float*)src, gShape0, gShape1, gShape2, gRows, gCols);
}

void launchTFILLPAD_5(uint8_t* out, uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols)
{
    runTFILLPAD<float, 1, 1, 1, 260, 7, 260, 16, 1, PadValue::Min, PadValue::Max, true>(
        (float*)out, (float*)src, gShape0, gShape1, gShape2, gRows, gCols);
}

void launchTFILLPAD_6(uint8_t* out, uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols)
{
    runTFILLPAD<uint16_t, 1, 1, 1, 260, 7, 260, 32, 1, PadValue::Min, PadValue::Max>(
        (uint16_t*)out, (uint16_t*)src, gShape0, gShape1, gShape2, gRows, gCols);
}

void launchTFILLPAD_7(uint8_t* out, uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols)
{
    runTFILLPAD<uint8_t, 1, 1, 1, 260, 7, 260, 64, 1, PadValue::Min, PadValue::Max>(
        (uint8_t*)out, (uint8_t*)src, gShape0, gShape1, gShape2, gRows, gCols);
}

void launchTFILLPAD_8(uint8_t* out, uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols)
{
    runTFILLPAD<uint16_t, 1, 1, 1, 259, 7, 260, 32, 1, PadValue::Min, PadValue::Max, false, true>(
        (uint16_t*)out, (uint16_t*)src, gShape0, gShape1, gShape2, gRows, gCols);
}

void launchTFILLPAD_9(uint8_t* out, uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols)
{
    runTFILLPAD<int8_t, 1, 1, 1, 259, 7, 260, 64, 1, PadValue::Min, PadValue::Max, false, true>(
        (int8_t*)out, (int8_t*)src, gShape0, gShape1, gShape2, gRows, gCols);
}

// Test case 10: PadCustomNeg1 - custom pad value (float)
void launchTFILLPAD_10(uint8_t* out, uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols)
{
    runTFILLPAD<float, 1, 1, 1, 128, 64, 128, 128, 1, PadValue::Null, PadCustomNeg1>(
        (float*)out, (float*)src, gShape0, gShape1, gShape2, gRows, gCols);
}

#ifdef CPU_SIM_BFLOAT_ENABLED
// Test case 11: PadCustomNeg1 - custom pad value (bfloat16)
void launchTFILLPAD_11(uint8_t* out, uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols)
{
    runTFILLPAD<bfloat16_t, 1, 1, 1, 128, 64, 128, 128, 1, PadValue::Null, PadCustomNeg1>(
        (bfloat16_t*)out, (bfloat16_t*)src, gShape0, gShape1, gShape2, gRows, gCols);
}
#endif

// Test case 12: PadCustomNeg1_Half - custom pad value (half/fp16)
void launchTFILLPAD_12(uint8_t* out, uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols)
{
    runTFILLPAD<half, 1, 1, 1, 128, 64, 128, 128, 1, PadValue::Null, PadCustomNeg1_Half>(
        (half*)out, (half*)src, gShape0, gShape1, gShape2, gRows, gCols);
}

// Test case 13: s8, 1x32 tile, 15 valid cols, pad zero
void launchTFILLPAD_13(uint8_t* out, uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols)
{
    runTFILLPAD<int8_t, 1, 1, 1, 1, 15, 1, 32, 0, PadValue::Null, PadValue::Zero>(
        (int8_t*)out, (int8_t*)src, gShape0, gShape1, gShape2, gRows, gCols);
}

// Test case 14: s8, 1x32 tile, 16 valid cols, pad zero
void launchTFILLPAD_14(uint8_t* out, uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols)
{
    runTFILLPAD<int8_t, 1, 1, 1, 1, 16, 1, 32, 0, PadValue::Null, PadValue::Zero>(
        (int8_t*)out, (int8_t*)src, gShape0, gShape1, gShape2, gRows, gCols);
}

// Test case 15: u8, 1x32 tile, 15 valid cols, pad zero
void launchTFILLPAD_15(uint8_t* out, uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols)
{
    runTFILLPAD<uint8_t, 1, 1, 1, 1, 15, 1, 32, 0, PadValue::Null, PadValue::Zero>(
        (uint8_t*)out, (uint8_t*)src, gShape0, gShape1, gShape2, gRows, gCols);
}

// Test case 16: u8, 1x32 tile, 16 valid cols, pad zero
void launchTFILLPAD_16(uint8_t* out, uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols)
{
    runTFILLPAD<uint8_t, 1, 1, 1, 1, 16, 1, 32, 0, PadValue::Null, PadValue::Zero>(
        (uint8_t*)out, (uint8_t*)src, gShape0, gShape1, gShape2, gRows, gCols);
}

template <int32_t testKey>
void launchTFILLPAD(uint8_t* out, uint8_t* src, void* stream)
{
    if constexpr (testKey == 1) {
        launchTFILLPAD_1(out, src, 1, 1, 1, 128, 127);
    } else if constexpr (testKey == 2) {
        launchTFILLPAD_2(out, src, 1, 1, 1, 128, 160);
    } else if constexpr (testKey == 3) {
        launchTFILLPAD_3(out, src, 1, 1, 1, 128, 160);
    } else if constexpr (testKey == 4) {
        launchTFILLPAD_4(out, src, 1, 1, 1, 260, 7);
    } else if constexpr (testKey == 5) {
        launchTFILLPAD_5(out, src, 1, 1, 1, 260, 7);
    } else if constexpr (testKey == 6) {
        launchTFILLPAD_6(out, src, 1, 1, 1, 260, 7);
    } else if constexpr (testKey == 7) {
        launchTFILLPAD_7(out, src, 1, 1, 1, 260, 7);
    } else if constexpr (testKey == 8) {
        launchTFILLPAD_8(out, src, 1, 1, 1, 260, 7);
    } else if constexpr (testKey == 9) {
        launchTFILLPAD_9(out, src, 1, 1, 1, 260, 7);
    } else if constexpr (testKey == 10) {
        launchTFILLPAD_10(out, src, 1, 1, 1, 128, 64);
#ifdef CPU_SIM_BFLOAT_ENABLED
    } else if constexpr (testKey == 11) {
        launchTFILLPAD_11(out, src, 1, 1, 1, 128, 64);
#endif
    } else if constexpr (testKey == 12) {
        launchTFILLPAD_12(out, src, 1, 1, 1, 128, 64);
    } else if constexpr (testKey == 13) {
        launchTFILLPAD_13(out, src, 1, 1, 1, 1, 15);
    } else if constexpr (testKey == 14) {
        launchTFILLPAD_14(out, src, 1, 1, 1, 1, 16);
    } else if constexpr (testKey == 15) {
        launchTFILLPAD_15(out, src, 1, 1, 1, 1, 15);
    } else if constexpr (testKey == 16) {
        launchTFILLPAD_16(out, src, 1, 1, 1, 1, 16);
    }
}

template <
    typename U, int Shape0, int Shape1, int Shape2, int Shape3, int Shape4, int kTRows_, int kTCols_,
    PadValue PadVal_ = PadValue::Null>
int get_input_golden_case(uint8_t* input, uint8_t* golden)
{
    constexpr int shape4_aligned = align_to_32B(Shape4, U); // 128
    int in_shape[5] = {Shape0, Shape1, Shape2, Shape3, Shape4};
    int out_shape[5] = {Shape0, Shape1, Shape2, kTRows_, kTCols_};
    int in_capacity = in_shape[0] * in_shape[1] * in_shape[2] * in_shape[3] * in_shape[4];
    int out_capacity = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3] * out_shape[4];
    int in_byteSize = in_capacity * sizeof(U);
    int out_byteSize = out_capacity * sizeof(U);

    U u_padVal[1] = {0};
    if constexpr (isCustomPadValue(PadVal_)) {
        // Custom pad value - extract bits and convert
        constexpr uint32_t bits = getCustomPadBits(PadVal_);
        if constexpr (std::is_same_v<U, float>) {
            u_padVal[0] = std::bit_cast<U>(bits);
        } else if constexpr (sizeof(U) == 2) {
            // fp16 and bf16 both use lower 16 bits
            // (PadValueCustom stores native type bits directly)
            uint16_t u16_bits = static_cast<uint16_t>(bits & 0xFFFF);
            u_padVal[0] = std::bit_cast<U>(u16_bits);
        } else if constexpr (sizeof(U) == 1) {
            uint8_t u8_bits = static_cast<uint8_t>(bits & 0xFF);
            u_padVal[0] = std::bit_cast<U>(u8_bits);
        } else {
            u_padVal[0] = std::bit_cast<U>(bits);
        }
    } else if (std::numeric_limits<U>::has_infinity) {
        if (PadVal_ == PadValue::Max)
            u_padVal[0] = std::numeric_limits<U>::infinity();
        else if (PadVal_ == PadValue::Min)
            u_padVal[0] = -std::numeric_limits<U>::infinity();
    } else {
        if (PadVal_ == PadValue::Max)
            u_padVal[0] = std::numeric_limits<U>::max();
        else if (PadVal_ == PadValue::Min)
            u_padVal[0] = std::numeric_limits<U>::min();
    }

    U in_arr[Shape0][Shape1][Shape2][Shape3][Shape4] = {};
    U gold_arr[Shape0][Shape1][Shape2][kTRows_][kTCols_] = {};
    for (int x0 = 0; x0 < Shape0; x0++)
        for (int x1 = 0; x1 < Shape1; x1++)
            for (int x2 = 0; x2 < Shape2; x2++)
                for (int i = 0; i < kTRows_; i++) {
                    for (int j = 0; j < kTCols_; j++) {
                        if (i < Shape3 && j < Shape4) {
                            in_arr[x0][x1][x2][i][j] = x0 * Shape1 * Shape2 * Shape3 * Shape4 +
                                                       x1 * Shape2 * Shape3 * Shape4 + x2 * Shape3 * Shape4 +
                                                       i * Shape4 + j;
                            gold_arr[x0][x1][x2][i][j] = in_arr[x0][x1][x2][i][j];
                        } else {
                            gold_arr[x0][x1][x2][i][j] = *(U*)(u_padVal);
                        }
                    }
                }
    std::copy((uint8_t*)in_arr, ((uint8_t*)(in_arr)) + in_byteSize, input);
    std::copy((uint8_t*)gold_arr, ((uint8_t*)(gold_arr)) + out_byteSize, golden);
    return sizeof(gold_arr);
}

template <int32_t testKey>
int get_input_golden(uint8_t* input, uint8_t* golden)
{
    if constexpr (testKey == 1) {
        return get_input_golden_case<float, 1, 1, 1, 128, 127, 128, 128, PadValue::Max>(input, golden);
    } else if constexpr (testKey == 2 || testKey == 3) {
        return get_input_golden_case<float, 1, 1, 1, 128, 127, 128, 160, PadValue::Max>(input, golden);
    } else if constexpr (testKey == 4 || testKey == 5) {
        return get_input_golden_case<float, 1, 1, 1, 260, 7, 260, 16, PadValue::Max>(input, golden);
    } else if constexpr (testKey == 6) {
        return get_input_golden_case<uint16_t, 1, 1, 1, 260, 7, 260, 32, PadValue::Max>(input, golden);
    } else if constexpr (testKey == 7) {
        return get_input_golden_case<uint8_t, 1, 1, 1, 260, 7, 260, 64, PadValue::Max>(input, golden);
    } else if constexpr (testKey == 8) {
        return get_input_golden_case<uint16_t, 1, 1, 1, 259, 7, 260, 32, PadValue::Max>(input, golden);
    } else if constexpr (testKey == 9) {
        return get_input_golden_case<int8_t, 1, 1, 1, 259, 7, 260, 64, PadValue::Max>(input, golden);
    } else if constexpr (testKey == 10) {
        return get_input_golden_case<float, 1, 1, 1, 128, 64, 128, 128, PadCustomNeg1>(input, golden);
#ifdef CPU_SIM_BFLOAT_ENABLED
    } else if constexpr (testKey == 11) {
        return get_input_golden_case<bfloat16_t, 1, 1, 1, 128, 64, 128, 128, PadCustomNeg1>(input, golden);
#endif
    } else if constexpr (testKey == 12) {
        return get_input_golden_case<half, 1, 1, 1, 128, 64, 128, 128, PadCustomNeg1_Half>(input, golden);
    } else if constexpr (testKey == 13) {
        return get_input_golden_case<int8_t, 1, 1, 1, 1, 15, 1, 32, PadValue::Zero>(input, golden);
    } else if constexpr (testKey == 14) {
        return get_input_golden_case<int8_t, 1, 1, 1, 1, 16, 1, 32, PadValue::Zero>(input, golden);
    } else if constexpr (testKey == 15) {
        return get_input_golden_case<uint8_t, 1, 1, 1, 1, 15, 1, 32, PadValue::Zero>(input, golden);
    } else if constexpr (testKey == 16) {
        return get_input_golden_case<uint8_t, 1, 1, 1, 1, 16, 1, 32, PadValue::Zero>(input, golden);
    }
    return 0;
}

template void launchTFILLPAD<1>(uint8_t* out, uint8_t* src, void* stream);
template void launchTFILLPAD<2>(uint8_t* out, uint8_t* src, void* stream);
template void launchTFILLPAD<3>(uint8_t* out, uint8_t* src, void* stream);
template void launchTFILLPAD<4>(uint8_t* out, uint8_t* src, void* stream);
template void launchTFILLPAD<5>(uint8_t* out, uint8_t* src, void* stream);
template void launchTFILLPAD<6>(uint8_t* out, uint8_t* src, void* stream);
template void launchTFILLPAD<7>(uint8_t* out, uint8_t* src, void* stream);
template void launchTFILLPAD<8>(uint8_t* out, uint8_t* src, void* stream);
template void launchTFILLPAD<9>(uint8_t* out, uint8_t* src, void* stream);
template void launchTFILLPAD<10>(uint8_t* out, uint8_t* src, void* stream);
#ifdef CPU_SIM_BFLOAT_ENABLED
template void launchTFILLPAD<11>(uint8_t* out, uint8_t* src, void* stream);
#endif
template void launchTFILLPAD<12>(uint8_t* out, uint8_t* src, void* stream);
template void launchTFILLPAD<13>(uint8_t* out, uint8_t* src, void* stream);
template void launchTFILLPAD<14>(uint8_t* out, uint8_t* src, void* stream);
template void launchTFILLPAD<15>(uint8_t* out, uint8_t* src, void* stream);
template void launchTFILLPAD<16>(uint8_t* out, uint8_t* src, void* stream);

template int get_input_golden<1>(uint8_t* input, uint8_t* golden);
template int get_input_golden<2>(uint8_t* input, uint8_t* golden);
template int get_input_golden<3>(uint8_t* input, uint8_t* golden);
template int get_input_golden<4>(uint8_t* input, uint8_t* golden);
template int get_input_golden<5>(uint8_t* input, uint8_t* golden);
template int get_input_golden<6>(uint8_t* input, uint8_t* golden);
template int get_input_golden<7>(uint8_t* input, uint8_t* golden);
template int get_input_golden<8>(uint8_t* input, uint8_t* golden);
template int get_input_golden<9>(uint8_t* input, uint8_t* golden);
template int get_input_golden<10>(uint8_t* input, uint8_t* golden);
#ifdef CPU_SIM_BFLOAT_ENABLED
template int get_input_golden<11>(uint8_t* input, uint8_t* golden);
#endif
template int get_input_golden<12>(uint8_t* input, uint8_t* golden);
template int get_input_golden<13>(uint8_t* input, uint8_t* golden);
template int get_input_golden<14>(uint8_t* input, uint8_t* golden);
template int get_input_golden<15>(uint8_t* input, uint8_t* golden);
template int get_input_golden<16>(uint8_t* input, uint8_t* golden);
