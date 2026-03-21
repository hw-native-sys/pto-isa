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
constexpr PadValue PadCustomNeg1 = static_cast<PadValue>(0xBF80000000000001ULL);

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
template <typename T, int shape0, int shape1, int shape2, int shape3, int shape4, int tRows, int tCols, BLayout major,
          int dyn>
AICORE __inline__ auto getGlobalTensor(__gm__ T *addr, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4)
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

template <typename T, int shape0, int shape1, int shape2, int shape3, int shape4, int kTRows_, int kTCols_, int dyn_,
          PadValue LoadPadVal_ = PadValue::Null, PadValue FillPadVal_ = PadValue::Null, bool inplace = false,
          bool expand = false>
AICORE void runTFILLPAD(__gm__ T *out, __gm__ T *src, int gShape0, int gShape1, int gShape2, int gRows, int gCols,
                        __gm__ uint64_t *gLog)
{
#ifndef __PTO_AUTO__
    // Avoid stack dcache miss
    {
#define INIT_STACK 8192
        uint64_t stack[INIT_STACK / sizeof(uint64_t)]; // 8KB
        volatile uint64_t *pStack = stack;
        for (int i = 0; i < INIT_STACK; i += 64 / sizeof(uint64_t)) // cacheline is 64B
        {
            *(pStack++) = 0;
        }
        dsb(DSB_ALL);
    }

    // Avoid icache miss in profiling: preload 4KB icache and wait
    uint64_t pc;
    asm volatile("MOV %0, PC\n" : "+l"(pc));
    preload((void *)pc, 2);
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
    __ubuf__ T *ubaddr0 = 0x0;
    __ubuf__ T *ubaddr1 = (__ubuf__ T *)0x18000;
    if (inplace)
        ubaddr1 = ubaddr0;

    constexpr int shape4_aligned = align_to_32B(shape4, T);
    constexpr int kGTRows = kTRows_ / shape0 / shape1 / shape2; // Dst Tile Rows, merged all shape0*shape1*shape2 row
    int srcOffset = (block_idx) * (shape3 / block_num) * shape4;
    auto srcGlobal =
        getGlobalTensor<T, shape0, shape1, shape2, shape3, shape4, kGTRows, shape4, BLayout::RowMajor, dyn_>(
            src + srcOffset, gShape0, gShape1, gShape2, kGTRows, shape4);
    int dstOffset = (block_idx) * (shape3 / block_num) * kTCols_;
    auto dstGlobal =
        getGlobalTensor<T, shape0, shape1, shape2, shape3, kTCols_, kGTRows, kTCols_, BLayout::RowMajor, 0>(
            out + dstOffset, gShape0, gShape1, gShape2, kGTRows, kTCols_); // dst TStore GlobalTensor just use static

    volatile uint64_t t0, t1, t2;

    using TileDataP =
        Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, FillPadVal_>;
    TileDataP vecTileP(kTRows_, kTCols_);
    TASSIGN(vecTileP, (uint64_t)ubaddr1);

    if constexpr (expand) {
        using TileData = Tile<TileType::Vec, T, kTRows_, shape4_aligned, BLayout::RowMajor, -1, -1, SLayout::NoneBox,
                              512, LoadPadVal_>;
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
        TFILLPAD_EXPAND(vecTileP, vecTile);
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
        if constexpr (inplace)
            TFILLPAD_INPLACE(vecTileP, vecTile);
        else
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

extern "C" __global__ AICORE void launchTFILLPAD_1(__gm__ uint8_t *out, __gm__ uint8_t *src, int gShape0, int gShape1,
                                                   int gShape2, int gRows, int gCols, __gm__ uint64_t *gLog)
{
    runTFILLPAD<float, 1, 1, 1, 128, 127, 128, 128, 1, PadValue::Max, PadValue::Max>(
        (__gm__ float *)out, (__gm__ float *)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

extern "C" __global__ AICORE void launchTFILLPAD_2(__gm__ uint8_t *out, __gm__ uint8_t *src, int gShape0, int gShape1,
                                                   int gShape2, int gRows, int gCols, __gm__ uint64_t *gLog)
{
    runTFILLPAD<float, 1, 1, 1, 128, 127, 128, 160, 1, PadValue::Max, PadValue::Max>(
        (__gm__ float *)out, (__gm__ float *)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

extern "C" __global__ AICORE void launchTFILLPAD_3(__gm__ uint8_t *out, __gm__ uint8_t *src, int gShape0, int gShape1,
                                                   int gShape2, int gRows, int gCols, __gm__ uint64_t *gLog)
{
    runTFILLPAD<float, 1, 1, 1, 128, 127, 128, 160, 1, PadValue::Min, PadValue::Max>(
        (__gm__ float *)out, (__gm__ float *)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

extern "C" __global__ AICORE void launchTFILLPAD_4(__gm__ uint8_t *out, __gm__ uint8_t *src, int gShape0, int gShape1,
                                                   int gShape2, int gRows, int gCols, __gm__ uint64_t *gLog)
{
    runTFILLPAD<float, 1, 1, 1, 260, 7, 260, 16, 1, PadValue::Min, PadValue::Max>(
        (__gm__ float *)out, (__gm__ float *)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

extern "C" __global__ AICORE void launchTFILLPAD_5(__gm__ uint8_t *out, __gm__ uint8_t *src, int gShape0, int gShape1,
                                                   int gShape2, int gRows, int gCols, __gm__ uint64_t *gLog)
{
    runTFILLPAD<float, 1, 1, 1, 260, 7, 260, 16, 1, PadValue::Min, PadValue::Max, true>(
        (__gm__ float *)out, (__gm__ float *)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

extern "C" __global__ AICORE void launchTFILLPAD_6(__gm__ uint8_t *out, __gm__ uint8_t *src, int gShape0, int gShape1,
                                                   int gShape2, int gRows, int gCols, __gm__ uint64_t *gLog)
{
    runTFILLPAD<uint16_t, 1, 1, 1, 260, 7, 260, 32, 1, PadValue::Min, PadValue::Max>(
        (__gm__ uint16_t *)out, (__gm__ uint16_t *)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

extern "C" __global__ AICORE void launchTFILLPAD_7(__gm__ uint8_t *out, __gm__ uint8_t *src, int gShape0, int gShape1,
                                                   int gShape2, int gRows, int gCols, __gm__ uint64_t *gLog)
{
    runTFILLPAD<int8_t, 1, 1, 1, 260, 7, 260, 64, 1, PadValue::Min, PadValue::Max>(
        (__gm__ int8_t *)out, (__gm__ int8_t *)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

extern "C" __global__ AICORE void launchTFILLPAD_8(__gm__ uint8_t *out, __gm__ uint8_t *src, int gShape0, int gShape1,
                                                   int gShape2, int gRows, int gCols, __gm__ uint64_t *gLog)
{
    runTFILLPAD<uint16_t, 1, 1, 1, 259, 7, 260, 32, 1, PadValue::Min, PadValue::Max, false, true>(
        (__gm__ uint16_t *)out, (__gm__ uint16_t *)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

extern "C" __global__ AICORE void launchTFILLPAD_9(__gm__ uint8_t *out, __gm__ uint8_t *src, int gShape0, int gShape1,
                                                   int gShape2, int gRows, int gCols, __gm__ uint64_t *gLog)
{
    runTFILLPAD<int8_t, 1, 1, 1, 259, 7, 260, 64, 1, PadValue::Min, PadValue::Max, false, true>(
        (__gm__ int8_t *)out, (__gm__ int8_t *)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

extern "C" __global__ AICORE void launchTFILLPAD_10(__gm__ uint8_t *out, __gm__ uint8_t *src, int gShape0, int gShape1,
                                                    int gShape2, int gRows, int gCols, __gm__ uint64_t *gLog)
{
    runTFILLPAD<int16_t, 1, 1, 1, 260, 7, 260, 32, 1, PadValue::Min, PadValue::Min>(
        (__gm__ int16_t *)out, (__gm__ int16_t *)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

extern "C" __global__ AICORE void launchTFILLPAD_11(__gm__ uint8_t *out, __gm__ uint8_t *src, int gShape0, int gShape1,
                                                    int gShape2, int gRows, int gCols, __gm__ uint64_t *gLog)
{
    runTFILLPAD<int32_t, 1, 1, 1, 260, 7, 260, 32, 1, PadValue::Min, PadValue::Min>(
        (__gm__ int32_t *)out, (__gm__ int32_t *)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

// Case 12: Custom pad value (-1.0f)
extern "C" __global__ AICORE void launchTFILLPAD_12(__gm__ uint8_t *out, __gm__ uint8_t *src, int gShape0, int gShape1,
                                                    int gShape2, int gRows, int gCols, __gm__ uint64_t *gLog)
{
    runTFILLPAD<float, 1, 1, 1, 128, 64, 128, 128, 1, PadValue::Null, PadCustomNeg1>(
        (__gm__ float *)out, (__gm__ float *)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

// Case 13: Custom pad value for both TLOAD and TFILLPAD (32B unaligned: 127 cols)
extern "C" __global__ AICORE void launchTFILLPAD_13(__gm__ uint8_t *out, __gm__ uint8_t *src, int gShape0, int gShape1,
                                                    int gShape2, int gRows, int gCols, __gm__ uint64_t *gLog)
{
    runTFILLPAD<float, 1, 1, 1, 128, 127, 128, 160, 1, PadCustomNeg1, PadCustomNeg1>(
        (__gm__ float *)out, (__gm__ float *)src, gShape0, gShape1, gShape2, gRows, gCols, gLog);
}

template <int32_t testKey>
void launchTFILLPAD(uint8_t *out, uint8_t *src, uint64_t *gLog, void *stream)
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
    }
}

template <typename T>
constexpr auto getGoldenZero()
{
    if constexpr (sizeof(T) == 4) {
        return (uint32_t)0;
    } else if constexpr (sizeof(T) == 2) {
        return (uint16_t)0;
    } else if constexpr (sizeof(T) == 1) {
        return (uint8_t)0;
    }
}

template <typename U, int Shape0, int Shape1, int Shape2, int Shape3, int Shape4, int kTRows_, int kTCols_,
          auto PadVal_ = PadValue::Null>
int get_input_golden_case(uint8_t *input, uint8_t *golden)
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
    if constexpr ((static_cast<uint64_t>(PadVal_) & 0xFFFFFFFFULL) == 0x00000001ULL) {
        // Custom pad value - extract float bits
        uint32_t bits = static_cast<uint32_t>(static_cast<uint64_t>(PadVal_) >> 32);
        u_padVal[0] = *reinterpret_cast<const U *>(&bits);
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
    T t_padVal = *(T *)(u_padVal);

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
                }     // i

    std::copy((uint8_t *)in_arr, ((uint8_t *)(in_arr)) + in_byteSize, input);
    std::copy((uint8_t *)gold_arr, ((uint8_t *)(gold_arr)) + out_byteSize, golden);
    return sizeof(gold_arr);
}

template <int32_t testKey>
int get_input_golden(uint8_t *input, uint8_t *golden)
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
    }

    return 0;
}

template void launchTFILLPAD<1>(uint8_t *out, uint8_t *src, uint64_t *gLog, void *stream);  // 实例化 Key=0 的版本
template void launchTFILLPAD<2>(uint8_t *out, uint8_t *src, uint64_t *gLog, void *stream);  // 实例化 Key=0 的版本
template void launchTFILLPAD<3>(uint8_t *out, uint8_t *src, uint64_t *gLog, void *stream);  // 实例化 Key=0 的版本
template void launchTFILLPAD<4>(uint8_t *out, uint8_t *src, uint64_t *gLog, void *stream);  // 实例化 Key=0 的版本
template void launchTFILLPAD<5>(uint8_t *out, uint8_t *src, uint64_t *gLog, void *stream);  // 实例化 Key=0 的版本
template void launchTFILLPAD<6>(uint8_t *out, uint8_t *src, uint64_t *gLog, void *stream);  // 实例化 Key=0 的版本
template void launchTFILLPAD<7>(uint8_t *out, uint8_t *src, uint64_t *gLog, void *stream);  // 实例化 Key=0 的版本
template void launchTFILLPAD<8>(uint8_t *out, uint8_t *src, uint64_t *gLog, void *stream);  // 实例化 Key=0 的版本
template void launchTFILLPAD<9>(uint8_t *out, uint8_t *src, uint64_t *gLog, void *stream);  // 实例化 Key=0 的版本
template void launchTFILLPAD<10>(uint8_t *out, uint8_t *src, uint64_t *gLog, void *stream); // 实例化 Key=0 的版本
template void launchTFILLPAD<11>(uint8_t *out, uint8_t *src, uint64_t *gLog, void *stream);
template void launchTFILLPAD<12>(uint8_t *out, uint8_t *src, uint64_t *gLog, void *stream); // 实例化 Key=0 的版本
template void launchTFILLPAD<13>(uint8_t *out, uint8_t *src, uint64_t *gLog, void *stream);

template int get_input_golden<1>(uint8_t *input, uint8_t *golden);
template int get_input_golden<2>(uint8_t *input, uint8_t *golden);
template int get_input_golden<3>(uint8_t *input, uint8_t *golden);
template int get_input_golden<4>(uint8_t *input, uint8_t *golden);
template int get_input_golden<5>(uint8_t *input, uint8_t *golden);
template int get_input_golden<6>(uint8_t *input, uint8_t *golden);
template int get_input_golden<7>(uint8_t *input, uint8_t *golden);
template int get_input_golden<8>(uint8_t *input, uint8_t *golden);
template int get_input_golden<9>(uint8_t *input, uint8_t *golden);
template int get_input_golden<10>(uint8_t *input, uint8_t *golden);
template int get_input_golden<11>(uint8_t *input, uint8_t *golden);
template int get_input_golden<12>(uint8_t *input, uint8_t *golden);
template int get_input_golden<13>(uint8_t *input, uint8_t *golden);
