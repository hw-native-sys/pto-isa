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
#include "acl/acl.h"
#include "engram_common.h"

using namespace std;
using namespace pto;

constexpr float kGateBiasF = 0.125f;
constexpr int kHashHeads = 8;

__global__ AICORE __attribute__((aiv)) void warmup_kernel()
{}

template <int kEmbDim, int kBlockSize>
inline AICORE void runEngramBaseline(__gm__ float __out__ *output, __gm__ float __in__ *table,
                                     __gm__ int32_t __in__ *indices, __gm__ float __in__ *hidden,
                                     __gm__ float __in__ *gateWeight, int tableRows)
{
    static_assert(kEmbDim >= 128 && kEmbDim <= 1024 && (kEmbDim & (kEmbDim - 1)) == 0,
                  "EmbDim must be power-of-2 in [128, 1024]");
    static_assert(kBlockSize >= 1 && kBlockSize <= 64 && (kBlockSize & (kBlockSize - 1)) == 0,
                  "BlockSize must be power-of-2 in [1, 64]");

    constexpr int H = kHashHeads;
    constexpr int D = kEmbDim;
    constexpr int rowBytesF = D * (int)sizeof(float);
    constexpr int rowAlF = ((rowBytesF + 31) / 32) * 32;
    constexpr int idxPad = ((H * (int)sizeof(int32_t) + 31) / 32) * 32 / (int)sizeof(int32_t);
    constexpr int idxAl = ((idxPad * (int)sizeof(int32_t) + 31) / 32) * 32;

    constexpr int hidOff = 0;
    constexpr int gwOff = hidOff + rowAlF;
    constexpr int idxOff = gwOff + rowAlF;
    constexpr int lookOff = idxOff + idxAl;
    constexpr int aggOff = lookOff + H * rowAlF;
    constexpr int tmpOff = aggOff + rowAlF;
    constexpr int gsOff = tmpOff + rowAlF;
    constexpr int colSumTmpOff = gsOff + 32;

    using TileRowF = Tile<TileType::Vec, float, 1, D, BLayout::RowMajor, -1, -1>;
    using TileIdxLoad = Tile<TileType::Vec, int32_t, 1, idxPad, BLayout::RowMajor, -1, -1>;
    using TileLookF2D = Tile<TileType::Vec, float, H, D, BLayout::RowMajor, H, D>;
    using TileGSF = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1>;
    using TileGSF_CM = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1>;
    using EmbRowGMShape = pto::Shape<1, 1, 1, 1, D>;
    using EmbRowGMStride = pto::Stride<1, 1, 1, D, 1>;
    using GlobalEmbRow = GlobalTensor<float, EmbRowGMShape, EmbRowGMStride>;
    using IdxGMShape = pto::Shape<1, 1, 1, 1, idxPad>;
    using IdxGMStride = pto::Stride<1, 1, 1, idxPad, 1>;
    using GlobalIdx = GlobalTensor<int32_t, IdxGMShape, IdxGMStride>;
    using OutGMShape = pto::Shape<1, 1, 1, 1, D>;
    using OutGMStride = pto::Stride<1, 1, 1, D, 1>;
    using GlobalOut = GlobalTensor<float, OutGMShape, OutGMStride>;
    using HidGMShape = pto::Shape<1, 1, 1, 1, D>;
    using HidGMStride = pto::Stride<1, 1, 1, D, 1>;
    using GlobalHid = GlobalTensor<float, HidGMShape, HidGMStride>;

    TileRowF hiddenF(1, D);
    TileRowF gateWF(1, D);
    TileIdxLoad idxTile(1, idxPad);
    TileRowF aggF(1, D);
    TileRowF tmpF(1, D);
    TileGSF gsF(1, 8);

    TASSIGN(hiddenF, hidOff);
    TASSIGN(gateWF, gwOff);
    TASSIGN(idxTile, idxOff);
    TASSIGN(aggF, aggOff);
    TASSIGN(tmpF, tmpOff);
    TASSIGN(gsF, gsOff);

    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    for (int pos = 0; pos < kBlockSize; ++pos) {
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
        GlobalHid hidGM(hidden + pos * D);
        GlobalHid gwGM(gateWeight + pos * D);
        GlobalIdx idxGM(indices + pos * H);

        TLOAD(hiddenF, hidGM);
        TLOAD(gateWF, gwGM);
        TLOAD(idxTile, idxGM);

        __ubuf__ const int32_t *idxPtr = (__ubuf__ const int32_t *)((__ubuf__ uint8_t *)idxTile.data());
        PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();
        for (int h = 0; h < H; ++h) {
            int32_t rowIdx = idxPtr[h];
            GlobalEmbRow embGM(table + (int)rowIdx * D);
            TileRowF headTile(1, D);
            TASSIGN(headTile, lookOff + h * rowAlF);
            TLOAD(headTile, embGM);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);

        {
            TileLookF2D lookupF2D;
            TASSIGN(lookupF2D, lookOff);
            TileLookF2D colSumTmp;
            TASSIGN(colSumTmp, colSumTmpOff);
            TCOLSUM(aggF, lookupF2D, colSumTmp, false);
            constexpr float invH = 1.0f / (float)H;
            TMULS(aggF, aggF, invH);
        }

        TMUL(tmpF, hiddenF, gateWF);
        TROWSUM(gsF, tmpF, tmpF);
        TADDS(gsF, gsF, kGateBiasF);
        TMULS(gsF, gsF, -1.0f);
        TEXP(gsF, gsF);
        TADDS(gsF, gsF, 1.0f);
        TDIVS(gsF, 1.0f, gsF);

        {
            TileGSF_CM gsCM(8, 1);
            TASSIGN(gsCM, gsOff);
            TROWEXPANDMUL(tmpF, aggF, gsCM);
        }
        TADD(tmpF, hiddenF, tmpF);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);

        GlobalOut outGM(output + pos * D);
        TSTORE(outGM, tmpF);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    }
}

template <uint32_t CWB, uint32_t B, uint32_t CC = 1>
struct ColChunksImpl {
    static constexpr bool kDivides = (CWB % CC == 0);
    static constexpr bool kFits = kDivides && ((CWB / CC) * B <= 32u);
    static constexpr uint32_t value = kFits ? CC : ColChunksImpl<CWB, B, CC + 1>::value;
};
template <uint32_t CWB, uint32_t B>
struct ColChunksImpl<CWB, B, CWB> {
    static constexpr uint32_t value = CWB;
};

template <uint32_t kEmbDim, uint32_t kBlockSize>
__simt_vf__ AICORE LAUNCH_BOUND(1024) PTO_INLINE
    void simt_engram_v2(__gm__ float *__restrict__ gmOutput, __gm__ const float *__restrict__ gmTable,
                        __gm__ const int32_t *__restrict__ gmIndices, __gm__ const float *__restrict__ gmHidden,
                        __gm__ const float *__restrict__ gmGateW, uint32_t tableRows)
{
    constexpr uint32_t H = (uint32_t)kHashHeads;
    constexpr uint32_t D = kEmbDim;
    constexpr uint32_t B = kBlockSize;
    constexpr uint32_t kLanes = 32u;
    constexpr float kInvH = 1.0f / (float)H;

    const uint32_t tx = __cce_simt_get_TID_X();
    const uint32_t ty = __cce_simt_get_TID_Y();

    __ubuf__ float *scrBuf = (__ubuf__ float *)((__ubuf__ uint8_t *)0);

    if constexpr (B == 1u) {
        constexpr uint32_t kWarps = D / kLanes;
        if (ty >= kWarps)
            return;
        const uint32_t col = ty * kLanes + tx;

        float h_val = gmHidden[col];
        float g_val = gmGateW[col];

        int32_t idx[H];
#pragma unroll
        for (uint32_t h = 0; h < H; ++h)
            idx[h] = gmIndices[h];

        float warp_dot = __builtin_cce_redux_add_f32(h_val * g_val);

        scrBuf[ty] = warp_dot;
        __sync_workitems();

        float dot;
        if constexpr (kWarps <= 16u) {
            dot = kGateBiasF;
#pragma unroll
            for (uint32_t w = 0; w < kWarps; ++w)
                dot += scrBuf[w];
        } else {
            if (ty == 0u) {
                float partial = scrBuf[tx];
                float total = __builtin_cce_redux_add_f32(partial);
                scrBuf[0] = total + kGateBiasF;
            }
            __sync_workitems();
            dot = scrBuf[0];
        }

        float gate = 1.0f / (1.0f + __builtin_cce_expf(-dot));

        float agg = gmTable[(uint32_t)idx[0] * D + col];
#pragma unroll
        for (uint32_t h = 1; h < H; ++h)
            agg += gmTable[(uint32_t)idx[h] * D + col];

        gmOutput[col] = h_val + (gate * kInvH) * agg;
    } else {
        constexpr uint32_t kColWarpsBase = D / kLanes;
        constexpr uint32_t kColChunksRaw = ColChunksImpl<kColWarpsBase, B>::value;
        constexpr uint32_t kColChunksMax = (D <= 256u) ? kColWarpsBase : 8u;
        constexpr uint32_t kColChunks = (kColChunksRaw < kColChunksMax) ? kColChunksRaw : kColChunksMax;
        constexpr uint32_t kColWarps = kColWarpsBase / kColChunks;
        constexpr uint32_t kTotalWarps = kColWarps * B;
        constexpr uint32_t kLaunchWarps = (kTotalWarps <= 32u) ? kTotalWarps : 32u;

        for (uint32_t warpId = ty; warpId < kTotalWarps; warpId += kLaunchWarps) {
            const uint32_t posId = warpId / kColWarps;
            const uint32_t colWarp = (kColWarps > 1u) ? (warpId % kColWarps) : 0u;

            float dot_partial = 0.0f;
            float h_reg[kColChunks];
#pragma unroll
            for (uint32_t c = 0; c < kColChunks; ++c) {
                uint32_t col = (colWarp * kColChunks + c) * kLanes + tx;
                h_reg[c] = gmHidden[posId * D + col];
                dot_partial += h_reg[c] * gmGateW[posId * D + col];
            }
            float warp_dot = __builtin_cce_redux_add_f32(dot_partial);

            float dot;
            if constexpr (kColWarps == 1u) {
                dot = warp_dot + kGateBiasF;
            } else if constexpr (kColWarps <= 16u) {
                scrBuf[ty] = warp_dot;
                __sync_workitems();
                dot = kGateBiasF;
                uint32_t scrBase;
                if constexpr (kTotalWarps <= kLaunchWarps)
                    scrBase = posId * kColWarps;
                else
                    scrBase = (ty / kColWarps) * kColWarps;
#pragma unroll
                for (uint32_t w = 0; w < kColWarps; ++w)
                    dot += scrBuf[scrBase + w];
            } else {
                scrBuf[ty] = warp_dot;
                __sync_workitems();
                float partial = scrBuf[tx];
                dot = __builtin_cce_redux_add_f32(partial) + kGateBiasF;
            }

            float gate = 1.0f / (1.0f + __builtin_cce_expf(-dot));

            int32_t idx[H];
#pragma unroll
            for (uint32_t h = 0; h < H; ++h)
                idx[h] = gmIndices[posId * H + h];

#pragma unroll
            for (uint32_t c = 0; c < kColChunks; ++c) {
                uint32_t col = (colWarp * kColChunks + c) * kLanes + tx;
                float agg = gmTable[(uint32_t)idx[0] * D + col];
#pragma unroll
                for (uint32_t h = 1; h < H; ++h)
                    agg += gmTable[(uint32_t)idx[h] * D + col];
                gmOutput[posId * D + col] = h_reg[c] + (gate * kInvH) * agg;
            }

            if constexpr (kTotalWarps > kLaunchWarps && kColWarps > 1u) {
                __sync_workitems();
            }
        }
    }
}

template <uint32_t kEmbDim, uint32_t kBlockSize>
__simt_vf__ AICORE LAUNCH_BOUND(512) PTO_INLINE
    void simt_engram_v2_lb512(__gm__ float *__restrict__ gmOutput, __gm__ const float *__restrict__ gmTable,
                              __gm__ const int32_t *__restrict__ gmIndices, __gm__ const float *__restrict__ gmHidden,
                              __gm__ const float *__restrict__ gmGateW, uint32_t tableRows)
{
    constexpr uint32_t H = (uint32_t)kHashHeads;
    constexpr uint32_t D = kEmbDim;
    constexpr uint32_t B = kBlockSize;
    constexpr uint32_t kLanes = 32u;
    constexpr float kInvH = 1.0f / (float)H;
    constexpr uint32_t kColWarpsBase = D / kLanes;
    constexpr uint32_t kCC = 8u;
    constexpr uint32_t kColWarps = kColWarpsBase / kCC;
    constexpr uint32_t kTotalWarps = kColWarps * B;
    constexpr uint32_t kMaxWarps = 16u;
    constexpr uint32_t kLaunchWarps = (kTotalWarps <= kMaxWarps) ? kTotalWarps : kMaxWarps;

    const uint32_t tx = __cce_simt_get_TID_X();
    const uint32_t ty = __cce_simt_get_TID_Y();

    __ubuf__ float *scrBuf = (__ubuf__ float *)((__ubuf__ uint8_t *)0);

    for (uint32_t warpId = ty; warpId < kTotalWarps; warpId += kLaunchWarps) {
        const uint32_t posId = warpId / kColWarps;
        const uint32_t colWarp = (kColWarps > 1u) ? (warpId % kColWarps) : 0u;

        int32_t idx[H];
#pragma unroll
        for (uint32_t h = 0; h < H; ++h)
            idx[h] = gmIndices[posId * H + h];

        float dot_partial = 0.0f;
        float h_reg[kCC];
#pragma unroll
        for (uint32_t c = 0; c < kCC; ++c) {
            uint32_t col = (colWarp * kCC + c) * kLanes + tx;
            h_reg[c] = gmHidden[posId * D + col];
            dot_partial += h_reg[c] * gmGateW[posId * D + col];
        }
        float warp_dot = __builtin_cce_redux_add_f32(dot_partial);

        float dot;
        if constexpr (kColWarps == 1u) {
            dot = warp_dot + kGateBiasF;
        } else if constexpr (kColWarps <= 16u) {
            scrBuf[ty] = warp_dot;
            __sync_workitems();
            dot = kGateBiasF;
            uint32_t scrBase;
            if constexpr (kTotalWarps <= kLaunchWarps)
                scrBase = posId * kColWarps;
            else
                scrBase = (ty / kColWarps) * kColWarps;
#pragma unroll
            for (uint32_t w = 0; w < kColWarps; ++w)
                dot += scrBuf[scrBase + w];
        }

        float gate = 1.0f / (1.0f + __builtin_cce_expf(-dot));

#pragma unroll
        for (uint32_t c = 0; c < kCC; ++c) {
            uint32_t col = (colWarp * kCC + c) * kLanes + tx;
            float t0 = gmTable[(uint32_t)idx[0] * D + col];
            float t1 = gmTable[(uint32_t)idx[1] * D + col];
            float t2 = gmTable[(uint32_t)idx[2] * D + col];
            float t3 = gmTable[(uint32_t)idx[3] * D + col];
            float t4 = gmTable[(uint32_t)idx[4] * D + col];
            float t5 = gmTable[(uint32_t)idx[5] * D + col];
            float t6 = gmTable[(uint32_t)idx[6] * D + col];
            float t7 = gmTable[(uint32_t)idx[7] * D + col];
            float agg = (t0 + t1) + (t2 + t3) + (t4 + t5) + (t6 + t7);
            gmOutput[posId * D + col] = h_reg[c] + (gate * kInvH) * agg;
        }

        if constexpr (kTotalWarps > kLaunchWarps && kColWarps > 1u) {
            __sync_workitems();
        }
    }
}

template <uint32_t kEmbDim, uint32_t kBlockSize>
__tf__ AICORE void FusedEngramImpl(__gm__ float *__restrict__ gmOutput, __gm__ const float *__restrict__ gmTable,
                                   __gm__ const int32_t *__restrict__ gmIndices,
                                   __gm__ const float *__restrict__ gmHidden, __gm__ const float *__restrict__ gmGateW,
                                   uint32_t tableRows)
{
    constexpr uint32_t kLanes = 32u;

    if constexpr (kEmbDim >= 512u && kBlockSize >= 16u) {
        constexpr uint32_t kCC = 8u;
        constexpr uint32_t kColWarps = kEmbDim / (32u * kCC);
        constexpr uint32_t kTotalWarps = kColWarps * kBlockSize;
        constexpr uint32_t kLaunchWarps = (kTotalWarps <= 16u) ? kTotalWarps : 16u;
        cce::async_invoke<simt_engram_v2_lb512<kEmbDim, kBlockSize>>(cce::dim3{32, kLaunchWarps}, gmOutput, gmTable,
                                                                     gmIndices, gmHidden, gmGateW, tableRows);
    } else if constexpr (kBlockSize == 1u) {
        constexpr uint32_t kWarps = kEmbDim / kLanes;
        cce::async_invoke<simt_engram_v2<kEmbDim, 1>>(cce::dim3{32, kWarps}, gmOutput, gmTable, gmIndices, gmHidden,
                                                      gmGateW, tableRows);
    } else {
        constexpr uint32_t kColWarpsBase = kEmbDim / kLanes;
        constexpr uint32_t kColChunksRaw = ColChunksImpl<kColWarpsBase, kBlockSize>::value;
        constexpr uint32_t kColChunksMax = (kEmbDim <= 256u) ? kColWarpsBase : 8u;
        constexpr uint32_t kColChunks = (kColChunksRaw < kColChunksMax) ? kColChunksRaw : kColChunksMax;
        constexpr uint32_t kColWarps = kColWarpsBase / kColChunks;
        constexpr uint32_t kTotalWarps = kColWarps * kBlockSize;
        constexpr uint32_t kLaunchWarps = (kTotalWarps <= 32u) ? kTotalWarps : 32u;

        cce::async_invoke<simt_engram_v2<kEmbDim, kBlockSize>>(cce::dim3{32, kLaunchWarps}, gmOutput, gmTable,
                                                               gmIndices, gmHidden, gmGateW, tableRows);
    }
}

template <int kEmbDim, int kBlockSize>
inline AICORE void runEngramFused(__gm__ float __out__ *output, __gm__ float __in__ *table,
                                  __gm__ int32_t __in__ *indices, __gm__ float __in__ *hidden,
                                  __gm__ float __in__ *gateWeight, int tableRows)
{
    FusedEngramImpl<(uint32_t)kEmbDim, (uint32_t)kBlockSize>(
        output, table, reinterpret_cast<__gm__ const int32_t *>(indices),
        reinterpret_cast<__gm__ const float *>(hidden), reinterpret_cast<__gm__ const float *>(gateWeight),
        (uint32_t)tableRows);
    pipe_barrier(PIPE_ALL);
}

#define ENGRAM_BASELINE(D, B)                                                                                \
    extern "C" __global__ AICORE void runEngram_baseline_E##D##_B##B(__gm__ float *out, __gm__ float *table, \
                                                                     __gm__ int32_t *idx, __gm__ float *hid, \
                                                                     __gm__ float *gw, int tableRows)        \
    {                                                                                                        \
        runEngramBaseline<D, B>(out, table, idx, hid, gw, tableRows);                                        \
    }

#define ENGRAM_FUSED(D, B)                                                                                \
    extern "C" __global__ AICORE void runEngram_fused_E##D##_B##B(__gm__ float *out, __gm__ float *table, \
                                                                  __gm__ int32_t *idx, __gm__ float *hid, \
                                                                  __gm__ float *gw, int tableRows)        \
    {                                                                                                     \
        runEngramFused<D, B>(out, table, idx, hid, gw, tableRows);                                        \
    }

#define ENGRAM_LAUNCH_BASELINE(D, B)                                                                              \
    template <>                                                                                                   \
    void LaunchEngramBaseline<D, B>(float *out, float *table, int32_t *idx, float *hid, float *gw, int tableRows, \
                                    void *stream)                                                                 \
    {                                                                                                             \
        warmup_kernel<<<64, nullptr, stream>>>();                                                                 \
        runEngram_baseline_E##D##_B##B<<<1, nullptr, stream>>>(out, table, idx, hid, gw, tableRows);              \
    }

#define ENGRAM_LAUNCH_FUSED(D, B)                                                                              \
    template <>                                                                                                \
    void LaunchEngramFused<D, B>(float *out, float *table, int32_t *idx, float *hid, float *gw, int tableRows, \
                                 void *stream)                                                                 \
    {                                                                                                          \
        warmup_kernel<<<64, nullptr, stream>>>();                                                              \
        runEngram_fused_E##D##_B##B<<<1, nullptr, stream>>>(out, table, idx, hid, gw, tableRows);              \
    }

#define ENGRAM_INST(D, B)        \
    ENGRAM_BASELINE(D, B)        \
    ENGRAM_FUSED(D, B)           \
    ENGRAM_LAUNCH_BASELINE(D, B) \
    ENGRAM_LAUNCH_FUSED(D, B)

template <int kEmbDim, int kBlockSize>
void LaunchEngramBaseline(float *out, float *table, int32_t *idx, float *hid, float *gw, int tableRows, void *stream);

template <int kEmbDim, int kBlockSize>
void LaunchEngramFused(float *out, float *table, int32_t *idx, float *hid, float *gw, int tableRows, void *stream);

#ifdef PERF_ANALYSIS
ENGRAM_INST(128, 1);
ENGRAM_INST(128, 4);
ENGRAM_INST(128, 16);
ENGRAM_INST(128, 64);
ENGRAM_INST(256, 1);
ENGRAM_INST(256, 4);
ENGRAM_INST(256, 16);
ENGRAM_INST(256, 64);
ENGRAM_INST(512, 1);
ENGRAM_INST(512, 4);
ENGRAM_INST(512, 16);
ENGRAM_INST(512, 64);
ENGRAM_INST(1024, 1);
ENGRAM_INST(1024, 4);
ENGRAM_INST(1024, 16);
ENGRAM_INST(1024, 64);
#else
ENGRAM_INST(128, 1);
ENGRAM_INST(256, 4);
ENGRAM_INST(512, 1);
ENGRAM_INST(1024, 1);
#endif
