/*
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// TopK interface test kernel: one AIV block per case, single row [1, N].
// The kernel loads the row into UB, calls the pto::TOPK instruction (TopK.hpp) and stores
// the (value, index) results back to GM.
//
// The arch macro guard only wraps device-side includes/helpers: the host compilation pass of
// -xcce does not define the arch macros and only needs the __global__ entry signatures and the
// launchers (see topk_simdvf_radix_kernel.cpp for the same pattern).

#include <pto/pto-inst.hpp>
#include <iostream>

using namespace std;
using namespace pto;

#if defined(__CCE_AICORE__)

// Standalone TopK implementation lived under include/pto/npu/a5/; now local to this exp.
#include <TopK.hpp>

template <typename T, int N, int K>
AICORE inline constexpr uint32_t TopKCaseTmpWords()
{
    constexpr int KP = ((K + 32 / sizeof(T) - 1) / (32 / sizeof(T))) * (32 / sizeof(T));
    constexpr uint32_t tmpBytes = pto::TopKTmpBytesFor(static_cast<uint32_t>(KP), static_cast<uint32_t>(N), sizeof(T));
    return (tmpBytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);
}

#else

// Host -xcce pass: device TopK.hpp / arch macros are not available.
namespace pto {
enum class TopKAlgo { ITER_SELECT, RADIX_SELECT };
}
template <typename T, int N, int K>
inline constexpr uint32_t TopKCaseTmpWords()
{
    return 1u;
}

#endif

template <typename T, int N, int K>
__global__[aicore] void runTopKCase(
    __gm__ T __out__* outVal, __gm__ int32_t __out__* outIdx, __gm__ T __in__* scores, __gm__ int32_t __in__* srcIdx,
    int32_t largestIn, int32_t algoIn)
{
    constexpr int KP = ((K + 32 / sizeof(T) - 1) / (32 / sizeof(T))) * (32 / sizeof(T)); // 32B-aligned out footprint
    constexpr uint32_t TMP_WORDS = TopKCaseTmpWords<T, N, K>();

    using SrcData = GlobalTensor<T, Shape<-1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>>;
    using ValData = GlobalTensor<T, Shape<-1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>>;
    using IdxData = GlobalTensor<int32_t, Shape<-1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>>;
    using SrcIdxData = GlobalTensor<int32_t, Shape<-1, -1, -1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>>;

    SrcData sGlobal(scores, pto::Shape(1, 1, 1, 1, N), pto::Stride(N, N, N, N, 1));
    ValData vGlobal(outVal, pto::Shape(1, 1, 1, 1, KP), pto::Stride(KP, KP, KP, KP, 1));
    IdxData iGlobal(outIdx, pto::Shape(1, 1, 1, 1, KP), pto::Stride(KP, KP, KP, KP, 1));
    SrcIdxData siGlobal(srcIdx, pto::Shape(1, 1, 1, 1, N), pto::Stride(N, N, N, N, 1));

    using SrcTileType = Tile<TileType::Vec, T, 1, N, BLayout::RowMajor, -1, -1>;
    using ValTileType = Tile<TileType::Vec, T, 1, KP, BLayout::RowMajor, -1, -1>;
    using IdxTileType = Tile<TileType::Vec, int32_t, 1, KP, BLayout::RowMajor, -1, -1>;
    using SrcIdxTileType = Tile<TileType::Vec, int32_t, 1, N, BLayout::RowMajor, -1, -1>;
    using TmpTileType = Tile<TileType::Vec, uint32_t, 1, TMP_WORDS, BLayout::RowMajor, -1, -1>;

    SrcTileType srcTile(1, N);
    ValTileType valTile(1, KP);
    IdxTileType idxTile(1, KP);
    SrcIdxTileType srcIdxTile(1, N);
    TmpTileType tmpTile(1, TMP_WORDS);

    constexpr std::size_t srcBytes = static_cast<std::size_t>(N) * sizeof(T);
    constexpr std::size_t valBytes = static_cast<std::size_t>(KP) * sizeof(T);
    constexpr std::size_t idxBytes = static_cast<std::size_t>(KP) * sizeof(int32_t);
    constexpr std::size_t srcIdxBytes = static_cast<std::size_t>(N) * sizeof(int32_t);
    constexpr std::size_t tmpBytes = static_cast<std::size_t>(TMP_WORDS) * sizeof(uint32_t);

    constexpr std::size_t offSrc = 0;
    constexpr std::size_t offVal = offSrc + srcBytes;
    constexpr std::size_t offIdx = offVal + valBytes;
    constexpr std::size_t offSrcIdx = offIdx + idxBytes;
    constexpr std::size_t offTmp = offSrcIdx + srcIdxBytes;
    static_assert(offTmp + tmpBytes <= 0x40000ULL, "TopK case exceeds UB");

    TASSIGN(srcTile, static_cast<uint32_t>(offSrc));
    TASSIGN(valTile, static_cast<uint32_t>(offVal));
    TASSIGN(idxTile, static_cast<uint32_t>(offIdx));
    TASSIGN(srcIdxTile, static_cast<uint32_t>(offSrcIdx));
    TASSIGN(tmpTile, static_cast<uint32_t>(offTmp));

    const bool largest = largestIn != 0;

    TLOAD(srcTile, sGlobal);
    TLOAD(srcIdxTile, siGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    if constexpr (std::is_same<T, float>::value) {
        if (algoIn == static_cast<int32_t>(TopKAlgo::ITER_SELECT)) {
            TOPK<TopKAlgo::ITER_SELECT, static_cast<uint32_t>(K)>(
                valTile, idxTile, srcTile, srcIdxTile, tmpTile, largest);
        } else {
            TOPK<TopKAlgo::RADIX_SELECT, static_cast<uint32_t>(K)>(
                valTile, idxTile, srcTile, srcIdxTile, tmpTile, largest);
        }
    } else {
        TOPK<TopKAlgo::RADIX_SELECT, static_cast<uint32_t>(K)>(valTile, idxTile, srcTile, srcIdxTile, tmpTile, largest);
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(vGlobal, valTile);
    TSTORE(iGlobal, idxTile);
}

template <typename T, int N, int K>
void launchTopK(T* outVal, int32_t* outIdx, T* scores, int32_t* srcIdx, bool largest, TopKAlgo algo, void* stream)
{
    cout << "launch topk (n=" << N << " k=" << K << " largest=" << largest << " algo=" << static_cast<int>(algo)
         << ") start!" << endl;
    runTopKCase<T, N, K>
        <<<1, nullptr, stream>>>(outVal, outIdx, scores, srcIdx, largest ? 1 : 0, static_cast<int32_t>(algo));
    cout << "launch topk end!" << endl;
}

// fp32 golden-driven cases (iter_sort small N + radix_sort large N)
template void launchTopK<float, 128, 3>(
    float* outVal, int32_t* outIdx, float* scores, int32_t* srcIdx, bool largest, TopKAlgo algo, void* stream);
template void launchTopK<float, 256, 8>(
    float* outVal, int32_t* outIdx, float* scores, int32_t* srcIdx, bool largest, TopKAlgo algo, void* stream);
template void launchTopK<float, 256, 12>(
    float* outVal, int32_t* outIdx, float* scores, int32_t* srcIdx, bool largest, TopKAlgo algo, void* stream);
template void launchTopK<float, 288, 8>(
    float* outVal, int32_t* outIdx, float* scores, int32_t* srcIdx, bool largest, TopKAlgo algo, void* stream);
template void launchTopK<float, 384, 6>(
    float* outVal, int32_t* outIdx, float* scores, int32_t* srcIdx, bool largest, TopKAlgo algo, void* stream);
template void launchTopK<float, 512, 12>(
    float* outVal, int32_t* outIdx, float* scores, int32_t* srcIdx, bool largest, TopKAlgo algo, void* stream);
template void launchTopK<float, 896, 16>(
    float* outVal, int32_t* outIdx, float* scores, int32_t* srcIdx, bool largest, TopKAlgo algo, void* stream);
template void launchTopK<float, 4096, 512>(
    float* outVal, int32_t* outIdx, float* scores, int32_t* srcIdx, bool largest, TopKAlgo algo, void* stream);
template void launchTopK<float, 16384, 256>(
    float* outVal, int32_t* outIdx, float* scores, int32_t* srcIdx, bool largest, TopKAlgo algo, void* stream);
template void launchTopK<float, 16384, 2048>(
    float* outVal, int32_t* outIdx, float* scores, int32_t* srcIdx, bool largest, TopKAlgo algo, void* stream);
