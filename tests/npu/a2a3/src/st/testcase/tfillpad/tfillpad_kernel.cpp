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
#include <algorithm>

using namespace std;
using namespace pto;

// Custom pad value for test case 12
// -1.0f has bit pattern 0xBF800000
constexpr PadValue PadCustomNeg1 = PadValueCustom(-1.0f);

#define LOGSIZE 128
#define PRINTLOG 4
#define DEBUGLOG
#ifdef DEBUGLOG
#define LOG(x) *(gLog++) = x;
#else
#define LOG(x)
#endif

// case shape is static, but testing would do dynamic or static test
template <int shape0, int shape1, int shape2, int shape3, int shape4>
AICORE __inline__ auto getOptDynShape(int gShape0, int gShape1, int gShape2, int gShape3, int gShape4)
{
    if constexpr (shape0 == 1) {
        using DynShapeDim5 = Shape<1, -1, -1, -1, -1>;
        DynShapeDim5 dynShape(gShape1, gShape2, gShape3, gShape4);
        return dynShape;
    } else if constexpr (shape0 == 1 && shape1 == 1) {
        using DynShapeDim5 = Shape<1, 1, -1, -1, -1>;
        DynShapeDim5 dynShape(gShape2, gShape3, gShape4);
        return dynShape;
    } else if constexpr (shape0 == 1 && shape1 == 1 && shape2 == 1) {
        using DynShapeDim5 = Shape<1, 1, 1, -1, -1>;
        DynShapeDim5 dynShape(gShape3, gShape4);
        return dynShape;
    } else if constexpr (shape0 == 1 && shape1 == 1 && shape2 == 1 && shape3 == 1) {
        using DynShapeDim5 = Shape<1, 1, 1, 1, -1>;
        DynShapeDim5 dynShape(gShape4);
        return dynShape;
    } else {
        using DynShapeDim5 = Shape<-1, -1, -1, -1, -1>;
        DynShapeDim5 dynShape(gShape0, gShape1, gShape2, gShape3, gShape4);
        return dynShape;
    }
}

// case shape is static, but testing would do dynamic or static test
template <
    typename T, int shape0, int shape1, int shape2, int shape3, int shape4, int tRows, int tCols, BLayout major,
    int dyn>
AICORE __inline__ auto getGlobalTensor(__gm__ T* addr, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4)
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

inline AICORE uint64_t get_syscnt() // dont use get_sys_cnt(), need volatile for profiling
{
    uint64_t syscnt;
    asm volatile("MOV %0, SYS_CNT\n" : "+l"(syscnt));
    return syscnt;
}

#define type_32_aligned(T) (32 / sizeof(T))
#define align_to_32B(x, T) ((((x) + type_32_aligned(T) - 1) / type_32_aligned(T)) * (type_32_aligned(T)));

template <
    typename T, int shape0, int shape1, int shape2, int shape3, int shape4, int kTRows_, int kTCols_, int dyn_,
    PadValue LoadPadVal_ = PadValue::Null, PadValue FillPadVal_ = PadValue::Null, bool inplace = false,
    bool expand = false>
AICORE void runTFILLPAD(
    __gm__ T* out, __gm__ T* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols, __gm__ uint64_t* gLog)
{
#ifndef __PTO_AUTO__
    // Avoid stack dcache miss
    {
#define INIT_STACK 8192
        uint64_t stack[INIT_STACK / sizeof(uint64_t)]; // 8KB
        volatile uint64_t* pStack = stack;
        for (int i = 0; i < INIT_STACK; i += 64 / sizeof(uint64_t)) // cacheline is 64B
        {
            *(pStack++) = 0;
        }
        dsb(DSB_ALL);
    }

    // Avoid icache miss in profiling: preload 4KB icache and wait
    uint64_t pc;
    asm volatile("MOV %0, PC\n" : "+l"(pc));
    preload((void*)pc, 2);
    while (get_icache_prl_st()) {
#if defined(__DAV_C220_CUBE__) || defined(__DAV_C220_VEC__)
        // seems to compile for a2a3; will crash in HiIPUJumpOpt pass for A5
        asm("nop");
#endif
    }
#endif

#ifdef DEBUGLOG
    gLog += block_idx * LOGSIZE;
#endif
    __ubuf__ T* ubaddr0 = 0x0;
    __ubuf__ T* ubaddr1 = (__ubuf__ T*)0x18000;
    if (inplace)
        ubaddr1 = ubaddr0;

    constexpr int shape4_aligned = align_to_32B(shape4, T);
    constexpr int kGTRows = kTRows_ / shape0 / shape1 / shape2; // Dst Tile Rows, merged all shape0*shape1*shape2 row
    int srcOffset = (block_idx) * (shape3 / block_num) * shape4;
    auto srcGlobal =
        getGlobalTensor<T, shape0, shape1, shape2, shape3, shape4, kGTRows, shape4, BLayout::RowMajor, dyn_>(
            src + srcOffset, gShape0, gShape1, gShape2, shape3, shape4);
    int dstOffset = (block_idx) * (shape3 / block_num) * kTCols_;
    auto dstGlobal =
        getGlobalTensor<T, shape0, shape1, shape2, shape3, kTCols_, kGTRows, kTCols_, BLayout::RowMajor, 0>(
            out + dstOffset, gShape0, gShape1, gShape2, kGTRows, kTCols_); // dst TStore GlobalTensor just use static

    volatile uint64_t t0, t1, t2;
    constexpr PadValue PadCustomNeg1_Test = PadValueCustom(-1.0f); // Test device usage
    static_assert(
        PadCustomNeg1_Test == static_cast<PadValue>(0x00000001BF800000ULL), "PadValueCustom float device test");
    constexpr PadValue PadCustomNeg1_Half_Test = PadValueCustom((half)-1.0); // fp16 using half type
    static_assert(
        PadCustomNeg1_Half_Test == static_cast<PadValue>(0x000000010000BC00ULL), "PadValueCustom16 fp16 device test");
    constexpr PadValue PadCustomNeg1_Bf16_Test = PadValueCustom((bfloat16_t)-1.0); // bf16 using bfloat16_t type
    static_assert(
        PadCustomNeg1_Bf16_Test == static_cast<PadValue>(0x000000010000BF80ULL), "PadValueCustom bf16 encoding test");
    // Verify decoding: getCustomPadBits should return 0xBF80 (bf16 -1.0), NOT 0 from bits >> 16
    static_assert(getCustomPadBits(PadCustomNeg1_Bf16_Test) == 0xBF80U, "PadValueCustom bf16 decoding test");

    // Test custom pad bits extraction for each type (catches decode bugs!)
    // For 16-bit types, bits & 0xFFFF must return the correct fp16/bf16 bits
    constexpr uint32_t float_bits = getCustomPadBits(PadCustomNeg1_Test);
    constexpr uint32_t half_bits = getCustomPadBits(PadCustomNeg1_Half_Test) & 0xFFFF;
    constexpr uint32_t bf16_bits = getCustomPadBits(PadCustomNeg1_Bf16_Test) & 0xFFFF;
    static_assert(float_bits == 0xBF800000U, "Custom pad float: expected -1.0f bits");
    static_assert(half_bits == 0xBC00U, "Custom pad half: expected fp16 -1.0 bits (0xBC00)");
    static_assert(bf16_bits == 0xBF80U, "Custom pad bf16: expected bf16 -1.0 bits (0xBF80)");

    using TileDataP =
        Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, FillPadVal_>;
    TileDataP vecTileP(kTRows_, kTCols_);
    TASSIGN(vecTileP, (uint64_t)ubaddr1);

    if constexpr (expand) {
        using TileData = Tile<
            TileType::Vec, T, kTRows_, shape4_aligned, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, LoadPadVal_>;
        // using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

        TileData vecTile(shape3, shape4);
        TASSIGN(vecTile, (uint64_t)ubaddr0);

        // TLOAD(vecTile, srcGlobal); //warm up...
        TLOAD(vecTile, srcGlobal);
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
        t0 = get_syscnt();
        TFILLPAD<TFillPadMode::Expand>(vecTileP, vecTile);
    } else {
        using TileData =
            Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, LoadPadVal_>;
        // using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

        TileData vecTile(shape3, shape4);
        TASSIGN(vecTile, (uint64_t)ubaddr0);

        // TLOAD(vecTile, srcGlobal); //warm up...
        TLOAD(vecTile, srcGlobal);
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
        t0 = get_syscnt();
        if constexpr (inplace) {
#ifdef __PTO_AUTO__
            TRESHAPE(vecTileP, vecTile);
#endif
            TFILLPAD<TFillPadMode::InPlace>(vecTileP, vecTile);
        } else
            TFILLPAD(vecTileP, vecTile);
    }
    t1 = get_syscnt();
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, vecTileP);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
#endif
    t2 = get_syscnt(); /*FIXME: compile would insert a dcci at above set/wait t2 timing may not be very correct*/
    LOG(t0);
    LOG(t1 - t0);
    LOG(t2 - t1);
}

extern "C" __global__ AICORE void launchTFILLPAD_1(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    runTFILLPAD<float, 1, 1, 1, 128, 127, 128, 128, 1, PadValue::Max, PadValue::Max>(
        (__gm__ float*)out, (__gm__ float*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

extern "C" __global__ AICORE void launchTFILLPAD_2(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    runTFILLPAD<float, 1, 1, 1, 128, 127, 128, 160, 1, PadValue::Max, PadValue::Max>(
        (__gm__ float*)out, (__gm__ float*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

extern "C" __global__ AICORE void launchTFILLPAD_3(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    runTFILLPAD<float, 1, 1, 1, 128, 127, 128, 160, 1, PadValue::Min, PadValue::Max>(
        (__gm__ float*)out, (__gm__ float*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

extern "C" __global__ AICORE void launchTFILLPAD_4(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    runTFILLPAD<float, 1, 1, 1, 260, 7, 260, 16, 1, PadValue::Min, PadValue::Max>(
        (__gm__ float*)out, (__gm__ float*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

extern "C" __global__ AICORE void launchTFILLPAD_5(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    runTFILLPAD<float, 1, 1, 1, 260, 7, 260, 16, 1, PadValue::Min, PadValue::Max, true>(
        (__gm__ float*)out, (__gm__ float*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

extern "C" __global__ AICORE void launchTFILLPAD_6(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    runTFILLPAD<uint16_t, 1, 1, 1, 260, 7, 260, 32, 1, PadValue::Min, PadValue::Max>(
        (__gm__ uint16_t*)out, (__gm__ uint16_t*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

extern "C" __global__ AICORE void launchTFILLPAD_7(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    runTFILLPAD<int8_t, 1, 1, 1, 260, 7, 260, 64, 1, PadValue::Min, PadValue::Max>(
        (__gm__ int8_t*)out, (__gm__ int8_t*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

extern "C" __global__ AICORE void launchTFILLPAD_8(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    runTFILLPAD<uint16_t, 1, 1, 1, 259, 7, 260, 32, 1, PadValue::Min, PadValue::Max, false, true>(
        (__gm__ uint16_t*)out, (__gm__ uint16_t*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

extern "C" __global__ AICORE void launchTFILLPAD_9(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    runTFILLPAD<int8_t, 1, 1, 1, 259, 7, 260, 64, 1, PadValue::Min, PadValue::Max, false, true>(
        (__gm__ int8_t*)out, (__gm__ int8_t*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

extern "C" __global__ AICORE void launchTFILLPAD_10(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    runTFILLPAD<int16_t, 1, 1, 1, 260, 7, 260, 32, 1, PadValue::Min, PadValue::Min>(
        (__gm__ int16_t*)out, (__gm__ int16_t*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

extern "C" __global__ AICORE void launchTFILLPAD_11(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    runTFILLPAD<int32_t, 1, 1, 1, 260, 7, 260, 32, 1, PadValue::Min, PadValue::Min>(
        (__gm__ int32_t*)out, (__gm__ int32_t*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

// Case 12: Custom pad value (-1.0f)
extern "C" __global__ AICORE void launchTFILLPAD_12(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    runTFILLPAD<float, 1, 1, 1, 128, 64, 128, 128, 1, PadValue::Null, PadCustomNeg1>(
        (__gm__ float*)out, (__gm__ float*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

// Case 13: Custom pad value for both TLOAD and TFILLPAD (32B unaligned: 127 cols)
extern "C" __global__ AICORE void launchTFILLPAD_13(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    runTFILLPAD<float, 1, 1, 1, 128, 127, 128, 160, 1, PadCustomNeg1, PadCustomNeg1>(
        (__gm__ float*)out, (__gm__ float*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

// Case 14: s8, 1x32 tile, 15 valid cols, pad zero
extern "C" __global__ AICORE void launchTFILLPAD_14(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    runTFILLPAD<int8_t, 1, 1, 1, 1, 15, 1, 32, 0, PadValue::Null, PadValue::Zero>(
        (__gm__ int8_t*)out, (__gm__ int8_t*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

// Case 15: s8, 1x32 tile, 16 valid cols, pad zero
extern "C" __global__ AICORE void launchTFILLPAD_15(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    runTFILLPAD<int8_t, 1, 1, 1, 1, 16, 1, 32, 0, PadValue::Null, PadValue::Zero>(
        (__gm__ int8_t*)out, (__gm__ int8_t*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

// Case 16: u8, 1x32 tile, 15 valid cols, pad zero
extern "C" __global__ AICORE void launchTFILLPAD_16(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    runTFILLPAD<uint8_t, 1, 1, 1, 1, 15, 1, 32, 0, PadValue::Null, PadValue::Zero>(
        (__gm__ uint8_t*)out, (__gm__ uint8_t*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

// Case 17: u8, 1x32 tile, 16 valid cols, pad zero
extern "C" __global__ AICORE void launchTFILLPAD_17(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    runTFILLPAD<uint8_t, 1, 1, 1, 1, 16, 1, 32, 0, PadValue::Null, PadValue::Zero>(
        (__gm__ uint8_t*)out, (__gm__ uint8_t*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

// Case 18: s8, 1x32 tile, 15 valid cols, pad min
extern "C" __global__ AICORE void launchTFILLPAD_18(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    runTFILLPAD<int8_t, 1, 1, 1, 1, 15, 1, 32, 0, PadValue::Null, PadValue::Min>(
        (__gm__ int8_t*)out, (__gm__ int8_t*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

// Case 19: s8, 1x32 tile, 15 valid cols, pad max
extern "C" __global__ AICORE void launchTFILLPAD_19(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    runTFILLPAD<int8_t, 1, 1, 1, 1, 15, 1, 32, 0, PadValue::Null, PadValue::Max>(
        (__gm__ int8_t*)out, (__gm__ int8_t*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

// Case 20: u8, 1x32 tile, 15 valid cols, pad min
extern "C" __global__ AICORE void launchTFILLPAD_20(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    runTFILLPAD<uint8_t, 1, 1, 1, 1, 15, 1, 32, 0, PadValue::Null, PadValue::Min>(
        (__gm__ uint8_t*)out, (__gm__ uint8_t*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

// Case 21: u8, 1x32 tile, 15 valid cols, pad max
extern "C" __global__ AICORE void launchTFILLPAD_21(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    runTFILLPAD<uint8_t, 1, 1, 1, 1, 15, 1, 32, 0, PadValue::Null, PadValue::Max>(
        (__gm__ uint8_t*)out, (__gm__ uint8_t*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

// Case 22: s8, 1x64 tile, 40 valid cols (>32B), pad min->max — catches Handle32BAlignedPad_Byte underflow
extern "C" __global__ AICORE void launchTFILLPAD_22(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    runTFILLPAD<int8_t, 1, 1, 1, 1, 40, 1, 64, 0, PadValue::Min, PadValue::Max>(
        (__gm__ int8_t*)out, (__gm__ int8_t*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

#if !defined(PTO_NPU_ARCH_A5)
// ---------------------------------------------------------------------------
// Case 23 (UB-OOB repro, GitHub #291): a FULL-width 16384-fp32 (64 KiB) tile
// TASSIGN'd at 0x20000 ends at 0x30000 == the top of the 192 KiB AIV UB.
// Before fix: TFILLPAD issued tail-pad vector_dup at dst + srcValidCol32B
// = 0x30000 (one past UB) even though padCols == 0 -> AIV fault (-100).
// After PadRightSingleRow padCols<=0 early-return: must PASS on a2a3 NPU.
// ---------------------------------------------------------------------------
AICORE void runTFILLPAD_UB_OOB(__gm__ float* out, __gm__ float* src, int, int, int, int, int gCols, __gm__ uint64_t*)
{
    constexpr int kCols = 16384; // fp32: 16384 * 4 B = 64 KiB = one full UB slot
    using TileData =
        Tile<TileType::Vec, float, 1, kCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>;
    using ShapeDyn = Shape<1, 1, 1, 1, -1>;
    using StrideDyn = Stride<1, 1, 1, 1, -1>;

    __ubuf__ float* ub = (__ubuf__ float*)0x20000; // top slot; tile ends at 0x30000
    TileData tile(1, kCols);
    TASSIGN(tile, (uint64_t)ub);
    tile.ColMaskInternal = kCols; // full width: srcValidCol == copyDstCols -> padCols == 0

    ShapeDyn shape(1, 1, 1, 1, gCols);
    StrideDyn stride(gCols, gCols, gCols, gCols, 1);
    GlobalTensor<float, ShapeDyn, StrideDyn, Layout::ND> srcG(src, shape, stride);
    GlobalTensor<float, ShapeDyn, StrideDyn, Layout::ND> dstG(out, shape, stride);

    TLOAD(tile, srcG);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // InPlace PadValue::Zero — matches surrounding master cases (TFillPadMode::InPlace).
    TFILLPAD<TFillPadMode::InPlace>(tile, tile); // padCols==0 early-return; must PASS on NPU

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstG, tile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
}

extern "C" __global__ AICORE void launchTFILLPAD_23(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    runTFILLPAD_UB_OOB((__gm__ float*)out, (__gm__ float*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}
#endif // !PTO_NPU_ARCH_A5

#if defined(PTO_NPU_ARCH_A5)
// Low-precision PadValue::Zero/Min/Max sugar (fp8 / hif8 / fp4x2)
extern "C" __global__ AICORE void launchTFILLPAD_23(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    static_assert(PadValueMap<float8_e4m3_t, PadValue::Zero>::value == uint8_t(0));
    runTFILLPAD<float8_e4m3_t, 1, 1, 1, 1, 15, 1, 32, 0, PadValue::Null, PadValue::Zero>(
        (__gm__ float8_e4m3_t*)out, (__gm__ float8_e4m3_t*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}
extern "C" __global__ AICORE void launchTFILLPAD_24(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    static_assert(PadValueMap<float8_e4m3_t, PadValue::Min>::value == uint8_t(0xFE));
    runTFILLPAD<float8_e4m3_t, 1, 1, 1, 1, 15, 1, 32, 0, PadValue::Null, PadValue::Min>(
        (__gm__ float8_e4m3_t*)out, (__gm__ float8_e4m3_t*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}
extern "C" __global__ AICORE void launchTFILLPAD_25(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    static_assert(PadValueMap<float8_e4m3_t, PadValue::Max>::value == uint8_t(0x7E));
    runTFILLPAD<float8_e4m3_t, 1, 1, 1, 1, 15, 1, 32, 0, PadValue::Null, PadValue::Max>(
        (__gm__ float8_e4m3_t*)out, (__gm__ float8_e4m3_t*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}
extern "C" __global__ AICORE void launchTFILLPAD_26(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    runTFILLPAD<float8_e5m2_t, 1, 1, 1, 1, 15, 1, 32, 0, PadValue::Null, PadValue::Zero>(
        (__gm__ float8_e5m2_t*)out, (__gm__ float8_e5m2_t*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}
extern "C" __global__ AICORE void launchTFILLPAD_27(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    static_assert(PadValueMap<float8_e5m2_t, PadValue::Min>::value == uint8_t(0xFC));
    runTFILLPAD<float8_e5m2_t, 1, 1, 1, 1, 15, 1, 32, 0, PadValue::Null, PadValue::Min>(
        (__gm__ float8_e5m2_t*)out, (__gm__ float8_e5m2_t*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}
extern "C" __global__ AICORE void launchTFILLPAD_28(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    static_assert(PadValueMap<float8_e5m2_t, PadValue::Max>::value == uint8_t(0x7C));
    runTFILLPAD<float8_e5m2_t, 1, 1, 1, 1, 15, 1, 32, 0, PadValue::Null, PadValue::Max>(
        (__gm__ float8_e5m2_t*)out, (__gm__ float8_e5m2_t*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}
extern "C" __global__ AICORE void launchTFILLPAD_29(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    runTFILLPAD<hifloat8_t, 1, 1, 1, 1, 15, 1, 32, 0, PadValue::Null, PadValue::Zero>(
        (__gm__ hifloat8_t*)out, (__gm__ hifloat8_t*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}
extern "C" __global__ AICORE void launchTFILLPAD_30(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    static_assert(PadValueMap<hifloat8_t, PadValue::Min>::value == uint8_t(0xEF));
    runTFILLPAD<hifloat8_t, 1, 1, 1, 1, 15, 1, 32, 0, PadValue::Null, PadValue::Min>(
        (__gm__ hifloat8_t*)out, (__gm__ hifloat8_t*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}
extern "C" __global__ AICORE void launchTFILLPAD_31(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    static_assert(PadValueMap<hifloat8_t, PadValue::Max>::value == uint8_t(0x6F));
    runTFILLPAD<hifloat8_t, 1, 1, 1, 1, 15, 1, 32, 0, PadValue::Null, PadValue::Max>(
        (__gm__ hifloat8_t*)out, (__gm__ hifloat8_t*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}
extern "C" __global__ AICORE void launchTFILLPAD_32(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    runTFILLPAD<float4_e2m1x2_t, 1, 1, 1, 1, 15, 1, 64, 0, PadValue::Null, PadValue::Zero>(
        (__gm__ float4_e2m1x2_t*)out, (__gm__ float4_e2m1x2_t*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}
extern "C" __global__ AICORE void launchTFILLPAD_33(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    static_assert(PadValueMap<float4_e2m1x2_t, PadValue::Min>::value == uint8_t(0xFF));
    runTFILLPAD<float4_e2m1x2_t, 1, 1, 1, 1, 15, 1, 64, 0, PadValue::Null, PadValue::Min>(
        (__gm__ float4_e2m1x2_t*)out, (__gm__ float4_e2m1x2_t*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}
extern "C" __global__ AICORE void launchTFILLPAD_34(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    static_assert(PadValueMap<float4_e2m1x2_t, PadValue::Max>::value == uint8_t(0x77));
    runTFILLPAD<float4_e2m1x2_t, 1, 1, 1, 1, 15, 1, 64, 0, PadValue::Null, PadValue::Max>(
        (__gm__ float4_e2m1x2_t*)out, (__gm__ float4_e2m1x2_t*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}
extern "C" __global__ AICORE void launchTFILLPAD_35(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    runTFILLPAD<float4_e1m2x2_t, 1, 1, 1, 1, 15, 1, 64, 0, PadValue::Null, PadValue::Zero>(
        (__gm__ float4_e1m2x2_t*)out, (__gm__ float4_e1m2x2_t*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}
extern "C" __global__ AICORE void launchTFILLPAD_36(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    runTFILLPAD<float4_e1m2x2_t, 1, 1, 1, 1, 15, 1, 64, 0, PadValue::Null, PadValue::Min>(
        (__gm__ float4_e1m2x2_t*)out, (__gm__ float4_e1m2x2_t*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}
extern "C" __global__ AICORE void launchTFILLPAD_37(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    static_assert(PadValueMap<float4_e1m2x2_t, PadValue::Max>::value == uint8_t(0x77));
    runTFILLPAD<float4_e1m2x2_t, 1, 1, 1, 1, 15, 1, 64, 0, PadValue::Null, PadValue::Max>(
        (__gm__ float4_e1m2x2_t*)out, (__gm__ float4_e1m2x2_t*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}
extern "C" __global__ AICORE void launchTFILLPAD_38(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    constexpr PadValue kPad = PadValueCustom(uint8_t(0x42));
    runTFILLPAD<float8_e4m3_t, 1, 1, 1, 1, 15, 1, 32, 0, PadValue::Null, kPad>(
        (__gm__ float8_e4m3_t*)out, (__gm__ float8_e4m3_t*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}
extern "C" __global__ AICORE void launchTFILLPAD_39(
    __gm__ uint8_t* out, __gm__ uint8_t* src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t* gLog)
{
    constexpr PadValue kPad = PadValueCustom(uint8_t(0x33));
    runTFILLPAD<float4_e2m1x2_t, 1, 1, 1, 1, 15, 1, 64, 0, PadValue::Null, kPad>(
        (__gm__ float4_e2m1x2_t*)out, (__gm__ float4_e2m1x2_t*)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}
#endif // PTO_NPU_ARCH_A5

template <int32_t testKey>
void launchTFILLPAD(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream)
{
    if constexpr (testKey == 1) {
        launchTFILLPAD_1<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 128, 127, gLog);
    } else if constexpr (testKey == 2) {
        launchTFILLPAD_2<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 128, 160, gLog);
    } else if constexpr (testKey == 3) {
        launchTFILLPAD_3<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 128, 160, gLog);
    } else if constexpr (testKey == 4) {
        launchTFILLPAD_4<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 260, 7, gLog);
    } else if constexpr (testKey == 5) {
        launchTFILLPAD_5<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 260, 7, gLog);
    } else if constexpr (testKey == 6) {
        launchTFILLPAD_6<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 260, 7, gLog);
    } else if constexpr (testKey == 7) {
        launchTFILLPAD_7<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 260, 7, gLog);
    } else if constexpr (testKey == 8) {
        launchTFILLPAD_8<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 260, 7, gLog);
    } else if constexpr (testKey == 9) {
        launchTFILLPAD_9<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 260, 7, gLog);
    } else if constexpr (testKey == 10) {
        launchTFILLPAD_10<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 260, 7, gLog);
    } else if constexpr (testKey == 11) {
        launchTFILLPAD_11<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 260, 7, gLog);
    } else if constexpr (testKey == 12) {
        launchTFILLPAD_12<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 128, 64, gLog);
    } else if constexpr (testKey == 13) {
        launchTFILLPAD_13<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 128, 127, gLog);
    } else if constexpr (testKey == 14) {
        launchTFILLPAD_14<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 1, 15, gLog);
    } else if constexpr (testKey == 15) {
        launchTFILLPAD_15<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 1, 16, gLog);
    } else if constexpr (testKey == 16) {
        launchTFILLPAD_16<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 1, 15, gLog);
    } else if constexpr (testKey == 17) {
        launchTFILLPAD_17<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 1, 16, gLog);
    } else if constexpr (testKey == 18) {
        launchTFILLPAD_18<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 1, 15, gLog);
    } else if constexpr (testKey == 19) {
        launchTFILLPAD_19<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 1, 15, gLog);
    } else if constexpr (testKey == 20) {
        launchTFILLPAD_20<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 1, 15, gLog);
    } else if constexpr (testKey == 21) {
        launchTFILLPAD_21<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 1, 15, gLog);
    } else if constexpr (testKey == 22) {
        launchTFILLPAD_22<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 1, 40, gLog);
    }
#if !defined(PTO_NPU_ARCH_A5)
    else if constexpr (testKey == 23) {
        launchTFILLPAD_23<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 1, 16384, gLog);
    }
#endif // !PTO_NPU_ARCH_A5
#if defined(PTO_NPU_ARCH_A5)
    else if constexpr (testKey == 23) {
        launchTFILLPAD_23<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 1, 15, gLog);
    } else if constexpr (testKey == 24) {
        launchTFILLPAD_24<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 1, 15, gLog);
    } else if constexpr (testKey == 25) {
        launchTFILLPAD_25<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 1, 15, gLog);
    } else if constexpr (testKey == 26) {
        launchTFILLPAD_26<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 1, 15, gLog);
    } else if constexpr (testKey == 27) {
        launchTFILLPAD_27<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 1, 15, gLog);
    } else if constexpr (testKey == 28) {
        launchTFILLPAD_28<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 1, 15, gLog);
    } else if constexpr (testKey == 29) {
        launchTFILLPAD_29<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 1, 15, gLog);
    } else if constexpr (testKey == 30) {
        launchTFILLPAD_30<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 1, 15, gLog);
    } else if constexpr (testKey == 31) {
        launchTFILLPAD_31<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 1, 15, gLog);
    } else if constexpr (testKey == 32) {
        launchTFILLPAD_32<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 1, 15, gLog);
    } else if constexpr (testKey == 33) {
        launchTFILLPAD_33<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 1, 15, gLog);
    } else if constexpr (testKey == 34) {
        launchTFILLPAD_34<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 1, 15, gLog);
    } else if constexpr (testKey == 35) {
        launchTFILLPAD_35<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 1, 15, gLog);
    } else if constexpr (testKey == 36) {
        launchTFILLPAD_36<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 1, 15, gLog);
    } else if constexpr (testKey == 37) {
        launchTFILLPAD_37<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 1, 15, gLog);
    } else if constexpr (testKey == 38) {
        launchTFILLPAD_38<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 1, 15, gLog);
    } else if constexpr (testKey == 39) {
        launchTFILLPAD_39<<<1, nullptr, stream>>>(out, src, 1, 1, 1, 1, 15, gLog);
    }
#endif // PTO_NPU_ARCH_A5
}

template <typename T>
constexpr T getGoldenZero()
{
    return T{0};
}

template <
    typename U, int Shape0, int Shape1, int Shape2, int Shape3, int Shape4, int kTRows_, int kTCols_,
    auto PadVal_ = PadValue::Null>
int get_input_golden_case(uint8_t* input, uint8_t* golden)
{
    auto arr = getGoldenZero<U>();
    using T = decltype(arr);

    constexpr int shape4_aligned = align_to_32B(Shape4, T);
    int in_shape[5] = {Shape0, Shape1, Shape2, Shape3, Shape4};
    int out_shape[5] = {Shape0, Shape1, Shape2, kTRows_, kTCols_};
    int in_capacity = in_shape[0] * in_shape[1] * in_shape[2] * in_shape[3] * in_shape[4];
    int out_capacity = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3] * out_shape[4];
    int in_byteSize = in_capacity * sizeof(T);
    int out_byteSize = out_capacity * sizeof(T);

    U u_padVal[1] = {0};
    if constexpr (static_cast<uint64_t>(PadVal_) >= static_cast<uint64_t>(PadValue::CustomBase)) {
        // Custom pad value - extract float bits
        uint32_t bits = static_cast<uint32_t>(static_cast<uint64_t>(PadVal_) & 0xFFFFFFFFULL);
        u_padVal[0] = *reinterpret_cast<const U*>(&bits);
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
    T t_padVal = *(T*)(u_padVal);

    T in_arr[Shape0][Shape1][Shape2][Shape3][Shape4] = {};
    T gold_arr[Shape0][Shape1][Shape2][kTRows_][kTCols_] = {};
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
                            gold_arr[x0][x1][x2][i][j] = t_padVal;
                        }
                    } // j
                } // i

    std::copy((uint8_t*)in_arr, ((uint8_t*)(in_arr)) + in_byteSize, input);
    std::copy((uint8_t*)gold_arr, ((uint8_t*)(gold_arr)) + out_byteSize, golden);
    return sizeof(gold_arr);
}

template <
    typename U, int Shape0, int Shape1, int Shape2, int Shape3, int Shape4, int kTRows_, int kTCols_, uint8_t PadBits>
int get_input_golden_bits(uint8_t* input, uint8_t* golden)
{
    auto arr = getGoldenZero<U>();
    using T = decltype(arr);
    int in_capacity = Shape0 * Shape1 * Shape2 * Shape3 * Shape4;
    int out_capacity = Shape0 * Shape1 * Shape2 * kTRows_ * kTCols_;
    int in_byteSize = in_capacity * sizeof(T);
    int out_byteSize = out_capacity * sizeof(T);
    T t_padVal = static_cast<T>(PadBits);
    T in_arr[Shape0][Shape1][Shape2][Shape3][Shape4] = {};
    T gold_arr[Shape0][Shape1][Shape2][kTRows_][kTCols_] = {};
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
                            gold_arr[x0][x1][x2][i][j] = t_padVal;
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
        return get_input_golden_case<int8_t, 1, 1, 1, 260, 7, 260, 64, PadValue::Max>(input, golden);
    } else if constexpr (testKey == 8) {
        return get_input_golden_case<uint16_t, 1, 1, 1, 259, 7, 260, 32, PadValue::Max>(input, golden);
    } else if constexpr (testKey == 9) {
        return get_input_golden_case<int8_t, 1, 1, 1, 259, 7, 260, 64, PadValue::Max>(input, golden);
    } else if constexpr (testKey == 10) {
        return get_input_golden_case<int16_t, 1, 1, 1, 260, 7, 260, 32, PadValue::Min>(input, golden);
    } else if constexpr (testKey == 11) {
        return get_input_golden_case<int32_t, 1, 1, 1, 260, 7, 260, 32, PadValue::Min>(input, golden);
    } else if constexpr (testKey == 12) {
        return get_input_golden_case<float, 1, 1, 1, 128, 64, 128, 128, PadCustomNeg1>(input, golden);
    } else if constexpr (testKey == 13) {
        return get_input_golden_case<float, 1, 1, 1, 128, 127, 128, 160, PadCustomNeg1>(input, golden);
    } else if constexpr (testKey == 14) {
        return get_input_golden_case<int8_t, 1, 1, 1, 1, 15, 1, 32, PadValue::Zero>(input, golden);
    } else if constexpr (testKey == 15) {
        return get_input_golden_case<int8_t, 1, 1, 1, 1, 16, 1, 32, PadValue::Zero>(input, golden);
    } else if constexpr (testKey == 16) {
        return get_input_golden_case<uint8_t, 1, 1, 1, 1, 15, 1, 32, PadValue::Zero>(input, golden);
    } else if constexpr (testKey == 17) {
        return get_input_golden_case<uint8_t, 1, 1, 1, 1, 16, 1, 32, PadValue::Zero>(input, golden);
    } else if constexpr (testKey == 18) {
        return get_input_golden_case<int8_t, 1, 1, 1, 1, 15, 1, 32, PadValue::Min>(input, golden);
    } else if constexpr (testKey == 19) {
        return get_input_golden_case<int8_t, 1, 1, 1, 1, 15, 1, 32, PadValue::Max>(input, golden);
    } else if constexpr (testKey == 20) {
        return get_input_golden_case<uint8_t, 1, 1, 1, 1, 15, 1, 32, PadValue::Min>(input, golden);
    } else if constexpr (testKey == 21) {
        return get_input_golden_case<uint8_t, 1, 1, 1, 1, 15, 1, 32, PadValue::Max>(input, golden);
    } else if constexpr (testKey == 22) {
        return get_input_golden_case<int8_t, 1, 1, 1, 1, 40, 1, 64, PadValue::Max>(input, golden);
    }
#if !defined(PTO_NPU_ARCH_A5)
    else if constexpr (testKey == 23) {
        // Pass-through (GT 1x16384 -> VT 1x16384, PadValue::Zero). On buggy pto-isa
        // this case never reaches the compare: the kernel faults the AIV first (-100).
        return get_input_golden_case<float, 1, 1, 1, 1, 16384, 1, 16384, PadValue::Zero>(input, golden);
    }
#endif // !PTO_NPU_ARCH_A5
#if defined(PTO_NPU_ARCH_A5)
    else if constexpr (testKey == 23) {
        return get_input_golden_bits<uint8_t, 1, 1, 1, 1, 15, 1, 32, 0x00>(input, golden);
    } else if constexpr (testKey == 24) {
        return get_input_golden_bits<uint8_t, 1, 1, 1, 1, 15, 1, 32, 0xFE>(input, golden);
    } else if constexpr (testKey == 25) {
        return get_input_golden_bits<uint8_t, 1, 1, 1, 1, 15, 1, 32, 0x7E>(input, golden);
    } else if constexpr (testKey == 26) {
        return get_input_golden_bits<uint8_t, 1, 1, 1, 1, 15, 1, 32, 0x00>(input, golden);
    } else if constexpr (testKey == 27) {
        return get_input_golden_bits<uint8_t, 1, 1, 1, 1, 15, 1, 32, 0xFC>(input, golden);
    } else if constexpr (testKey == 28) {
        return get_input_golden_bits<uint8_t, 1, 1, 1, 1, 15, 1, 32, 0x7C>(input, golden);
    } else if constexpr (testKey == 29) {
        return get_input_golden_bits<uint8_t, 1, 1, 1, 1, 15, 1, 32, 0x00>(input, golden);
    } else if constexpr (testKey == 30) {
        return get_input_golden_bits<uint8_t, 1, 1, 1, 1, 15, 1, 32, 0xEF>(input, golden);
    } else if constexpr (testKey == 31) {
        return get_input_golden_bits<uint8_t, 1, 1, 1, 1, 15, 1, 32, 0x6F>(input, golden);
    } else if constexpr (testKey == 32) {
        return get_input_golden_bits<uint8_t, 1, 1, 1, 1, 8, 1, 32, 0x00>(input, golden);
    } else if constexpr (testKey == 33) {
        return get_input_golden_bits<uint8_t, 1, 1, 1, 1, 8, 1, 32, 0xFF>(input, golden);
    } else if constexpr (testKey == 34) {
        return get_input_golden_bits<uint8_t, 1, 1, 1, 1, 8, 1, 32, 0x77>(input, golden);
    } else if constexpr (testKey == 35) {
        return get_input_golden_bits<uint8_t, 1, 1, 1, 1, 8, 1, 32, 0x00>(input, golden);
    } else if constexpr (testKey == 36) {
        return get_input_golden_bits<uint8_t, 1, 1, 1, 1, 8, 1, 32, 0xFF>(input, golden);
    } else if constexpr (testKey == 37) {
        return get_input_golden_bits<uint8_t, 1, 1, 1, 1, 8, 1, 32, 0x77>(input, golden);
    } else if constexpr (testKey == 38) {
        return get_input_golden_bits<uint8_t, 1, 1, 1, 1, 15, 1, 32, 0x42>(input, golden);
    } else if constexpr (testKey == 39) {
        return get_input_golden_bits<uint8_t, 1, 1, 1, 1, 8, 1, 32, 0x33>(input, golden);
    }
#endif // PTO_NPU_ARCH_A5

    return 0;
}

template void launchTFILLPAD<1>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);  // 实例化 Key=0 的版本
template void launchTFILLPAD<2>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);  // 实例化 Key=0 的版本
template void launchTFILLPAD<3>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);  // 实例化 Key=0 的版本
template void launchTFILLPAD<4>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);  // 实例化 Key=0 的版本
template void launchTFILLPAD<5>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);  // 实例化 Key=0 的版本
template void launchTFILLPAD<6>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);  // 实例化 Key=0 的版本
template void launchTFILLPAD<7>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);  // 实例化 Key=0 的版本
template void launchTFILLPAD<8>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);  // 实例化 Key=0 的版本
template void launchTFILLPAD<9>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);  // 实例化 Key=0 的版本
template void launchTFILLPAD<10>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream); // 实例化 Key=0 的版本
template void launchTFILLPAD<11>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);
template void launchTFILLPAD<12>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream); // 实例化 Key=0 的版本
template void launchTFILLPAD<13>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);
template void launchTFILLPAD<14>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);
template void launchTFILLPAD<15>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);
template void launchTFILLPAD<16>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);
template void launchTFILLPAD<17>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);
template void launchTFILLPAD<18>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);
template void launchTFILLPAD<19>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);
template void launchTFILLPAD<20>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);
template void launchTFILLPAD<21>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);
template void launchTFILLPAD<22>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);

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
template int get_input_golden<11>(uint8_t* input, uint8_t* golden);
template int get_input_golden<12>(uint8_t* input, uint8_t* golden);
template int get_input_golden<13>(uint8_t* input, uint8_t* golden);
template int get_input_golden<14>(uint8_t* input, uint8_t* golden);
template int get_input_golden<15>(uint8_t* input, uint8_t* golden);
template int get_input_golden<16>(uint8_t* input, uint8_t* golden);
template int get_input_golden<17>(uint8_t* input, uint8_t* golden);
template int get_input_golden<18>(uint8_t* input, uint8_t* golden);
template int get_input_golden<19>(uint8_t* input, uint8_t* golden);
template int get_input_golden<20>(uint8_t* input, uint8_t* golden);
template int get_input_golden<21>(uint8_t* input, uint8_t* golden);
template int get_input_golden<22>(uint8_t* input, uint8_t* golden);

#if !defined(PTO_NPU_ARCH_A5)
template void launchTFILLPAD<23>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);
template int get_input_golden<23>(uint8_t* input, uint8_t* golden);
#endif // !PTO_NPU_ARCH_A5

#if defined(PTO_NPU_ARCH_A5)

template void launchTFILLPAD<23>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);
template void launchTFILLPAD<24>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);
template void launchTFILLPAD<25>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);
template void launchTFILLPAD<26>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);
template void launchTFILLPAD<27>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);
template void launchTFILLPAD<28>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);
template void launchTFILLPAD<29>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);
template void launchTFILLPAD<30>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);
template void launchTFILLPAD<31>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);
template void launchTFILLPAD<32>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);
template void launchTFILLPAD<33>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);
template void launchTFILLPAD<34>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);
template void launchTFILLPAD<35>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);
template void launchTFILLPAD<36>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);
template void launchTFILLPAD<37>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);
template void launchTFILLPAD<38>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);
template void launchTFILLPAD<39>(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);
template int get_input_golden<23>(uint8_t* input, uint8_t* golden);
template int get_input_golden<24>(uint8_t* input, uint8_t* golden);
template int get_input_golden<25>(uint8_t* input, uint8_t* golden);
template int get_input_golden<26>(uint8_t* input, uint8_t* golden);
template int get_input_golden<27>(uint8_t* input, uint8_t* golden);
template int get_input_golden<28>(uint8_t* input, uint8_t* golden);
template int get_input_golden<29>(uint8_t* input, uint8_t* golden);
template int get_input_golden<30>(uint8_t* input, uint8_t* golden);
template int get_input_golden<31>(uint8_t* input, uint8_t* golden);
template int get_input_golden<32>(uint8_t* input, uint8_t* golden);
template int get_input_golden<33>(uint8_t* input, uint8_t* golden);
template int get_input_golden<34>(uint8_t* input, uint8_t* golden);
template int get_input_golden<35>(uint8_t* input, uint8_t* golden);
template int get_input_golden<36>(uint8_t* input, uint8_t* golden);
template int get_input_golden<37>(uint8_t* input, uint8_t* golden);
template int get_input_golden<38>(uint8_t* input, uint8_t* golden);
template int get_input_golden<39>(uint8_t* input, uint8_t* golden);
#endif // PTO_NPU_ARCH_A5
