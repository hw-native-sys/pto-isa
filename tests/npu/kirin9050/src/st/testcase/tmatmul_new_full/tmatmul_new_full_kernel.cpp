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

using namespace pto;

template <typename T>
AICORE constexpr inline T CeilAlign(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}

template <typename PtrT>
__tf__ AICORE inline PtrT GetTilePtr(PtrT ptr)
{
    return (PtrT)__cce_get_tile_ptr(ptr);
}

template <
    typename OutType, typename AType, typename BType, typename BiasType, int validM, int validK, int validN,
    bool isBias>
__global__ AICORE void RunTMATMUL(
    __gm__ OutType* out, __gm__ AType* src0, __gm__ BType* src1, __gm__ BiasType* src2, __gm__ uint64_t* quantVec)
{
    constexpr int blockAlign = 32;
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, blockAlign);
    constexpr int c0 = blockAlign / sizeof(OutType);
    constexpr int c0A = blockAlign / sizeof(AType);
    constexpr int c0B = blockAlign / sizeof(BType);
    constexpr int M_outer = M / 16;
    constexpr int N_outer = N / c0;
    constexpr int K_outerA = K / c0A;
    constexpr int K_outerB = K / 16;
    constexpr int N_outerB = N / c0B;
    constexpr bool isFp16 = std::is_same_v<OutType, half>;
    constexpr bool isQuant = !std::is_same_v<OutType, int32_t>;

    using GlobalDataSrc0 = GlobalTensor<
        AType, pto::Shape<K_outerA, 1, M_outer, 16, c0A>, pto::Stride<M * c0A, M * c0A, 16 * c0A, c0A, 1>,
        pto::Layout::NZ>;
    using GlobalDataSrc1 = GlobalTensor<
        BType, pto::Shape<N_outerB, 1, K_outerB, 16, c0B>, pto::Stride<K * c0B, K * c0B, 16 * c0B, c0B, 1>,
        pto::Layout::NZ>;
    using GlobalDataSrc2 = GlobalTensor<BiasType, pto::Shape<1, 1, 1, 1, N>, pto::Stride<1 * N, 1 * N, 1 * N, N, 1>>;
    using GlobalDataOut = GlobalTensor<
        OutType, pto::Shape<N_outer, 1, M_outer, 16, c0>, pto::Stride<M * c0, M * c0, 16 * c0, c0, 1>, pto::Layout::NZ>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, AType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, BType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    using TileBiasData = Tile<TileType::Mat, BiasType, 1, N, BLayout::RowMajor, 1, N>;
    using BiasTileData = Tile<TileType::Bias, OutType, 1, N, BLayout::RowMajor, 1, N>;

    using RightTile = TileRight<BType, K, N, K, N>;
    using AccTile = Tile<TileType::Mat, OutType, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 1024>;
    using VecTile = Tile<TileType::Vec, OutType, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 1024>;

    using QuantMatTile = Tile<TileType::Mat, uint64_t, 1, N, BLayout::RowMajor, 1, N>;
    using QuantFbTile = Tile<TileType::Scaling, uint64_t, 1, N, BLayout::RowMajor, 1, N>;

    constexpr uint64_t l1Base =
        M * K * sizeof(AType) + K * N * sizeof(BType) + N * sizeof(BiasType) + M * N * sizeof(OutType);

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileBiasData biasDataTile;
    TASSIGN<0x0>(aMatTile);
    TASSIGN<M * K * sizeof(AType)>(bMatTile);
    TASSIGN<M * K * sizeof(AType) + K * N * sizeof(BType)>(biasDataTile);

    AccTile cMatTile;
    TASSIGN<M * K * sizeof(AType) + K * N * sizeof(BType) + N * sizeof(BiasType)>(cMatTile);

    QuantMatTile quantMatTile;
    TASSIGN<l1Base>(quantMatTile);

    VecTile cVecTile;
    TASSIGN<0x0>(cVecTile);

    RightTile bTile;
    TASSIGN<0x0>(bTile);

    BiasTileData biasTile;
    TASSIGN<0x0>(biasTile);

    QuantFbTile quantFbTile;
    TASSIGN<0x0>(quantFbTile);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    if constexpr (isBias) {
        TLOAD(biasDataTile, src2Global);
    }

    if constexpr (isQuant) {
        using QuantGlobal = GlobalTensor<uint64_t, pto::Shape<1, 1, 1, 1, N>, pto::Stride<N, N, N, N, 1>>;
        QuantGlobal quantGlobal(quantVec);
        TLOAD(quantMatTile, quantGlobal);
    }

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    if constexpr (isQuant && N > 256) {
        // FBUF on Kirin9050 can only hold 2KB
        // For N > 256 with DeqF16Vector, split the matmul along N into chunks
        // of 256, reloading FBUF and L0B each chunk
        static_assert(!isBias, "N > 256 with bias and quant is not yet supported.");
        constexpr int QUANT_MAX = 256;
        constexpr int nChunks = N / QUANT_MAX;
        constexpr int chunkN = QUANT_MAX;

        using RightTileChunk = TileRight<BType, K, chunkN, K, chunkN>;
        using AccTileChunk =
            Tile<TileType::Mat, OutType, M, chunkN, BLayout::ColMajor, M, chunkN, SLayout::RowMajor, 1024>;
        using QuantMatTileChunk = Tile<TileType::Mat, uint64_t, 1, chunkN, BLayout::RowMajor, 1, chunkN>;
        using QuantFbTileChunk = Tile<TileType::Scaling, uint64_t, 1, chunkN, BLayout::RowMajor, 1, chunkN>;

        RightTileChunk bTileChunk;
        TASSIGN<0x0>(bTileChunk);

        QuantFbTileChunk quantFbTileChunk;
        TASSIGN<0x0>(quantFbTileChunk);

        AccTileChunk cMatTileChunk;
        QuantMatTileChunk quantMatTileChunk;

        // Each NZ Block in CBUF: 16 rows * c0 cols * sizeof(OutType) bytes
        constexpr uint32_t cBlockBytes = 16 * c0 * sizeof(OutType);
        constexpr uint32_t cChunkOffset = (chunkN / c0) * cBlockBytes;

        for (int chunk = 0; chunk < nChunks; chunk++) {
            // Point quantMatTileChunk to the correct offset in CBUF
            TASSIGN(quantMatTileChunk, (uint64_t)GetTilePtr(quantMatTile.data()) + chunk * chunkN * sizeof(uint64_t));
            TMOV(quantFbTileChunk, quantMatTileChunk);

            // Extract B columns [chunk*chunkN, (chunk+1)*chunkN] from cbuf to l0b
            TEXTRACT(bTileChunk, bMatTile, 0, chunk * chunkN);

            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

            // Point cMatTileChunk to the correct CBUF offset within the full C Tile
            TASSIGN(cMatTileChunk, (uint64_t)GetTilePtr(cMatTile.data()) + chunk * cChunkOffset);

            MatmulMacroConfig chunkCfg;
            chunkCfg.preQuantTileAddr = (uint64_t)GetTilePtr(quantFbTileChunk.data());
            TMATMUL(cMatTileChunk, aMatTile, bTileChunk, chunkCfg);

            if (chunk < nChunks - 1) {
                set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
                wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
            }
        }

        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    } else {
        if constexpr (isQuant) {
            TMOV(quantFbTile, quantMatTile);
        }

        TMOV(bTile, bMatTile);
        if constexpr (isBias) {
            TMOV(biasTile, biasDataTile);
        }

        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

        MatmulMacroConfig cfg;
        if constexpr (isQuant) {
            cfg.preQuantTileAddr = (uint64_t)GetTilePtr(quantFbTile.data());
        }

        if constexpr (isBias) {
            TMATMUL(cMatTile, aMatTile, bTile, biasTile, cfg);
        } else {
            TMATMUL(cMatTile, aMatTile, bTile, cfg);
        }

        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    }

    TMOV(cVecTile, cMatTile);

    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, cVecTile);

    out = dstGlobal.data();
}

template <
    typename OutType, typename AType, typename BType, typename BiasType, int validM, int validK, int validN,
    bool isBias>
__global__ AICORE void RunTMATMULRelu(
    __gm__ OutType* out, __gm__ AType* src0, __gm__ BType* src1, __gm__ BiasType* src2, __gm__ uint64_t* quantVec,
    float reluScalar, float clipReluVal)
{
    constexpr int blockAlign = 32;
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, blockAlign);
    constexpr int c0 = blockAlign / sizeof(OutType);
    constexpr int c0A = blockAlign / sizeof(AType);
    constexpr int c0B = blockAlign / sizeof(BType);
    constexpr int M_outer = M / 16;
    constexpr int N_outer = N / c0;
    constexpr int K_outerA = K / c0A;
    constexpr int K_outerB = K / 16;
    constexpr int N_outerB = N / c0B;
    constexpr bool isFp16 = std::is_same_v<OutType, half>;
    constexpr bool isQuant = !std::is_same_v<OutType, int32_t>;

    using GlobalDataSrc0 = GlobalTensor<
        AType, pto::Shape<K_outerA, 1, M_outer, 16, c0A>, pto::Stride<M * c0A, M * c0A, 16 * c0A, c0A, 1>,
        pto::Layout::NZ>;
    using GlobalDataSrc1 = GlobalTensor<
        BType, pto::Shape<N_outerB, 1, K_outerB, 16, c0B>, pto::Stride<K * c0B, K * c0B, 16 * c0B, c0B, 1>,
        pto::Layout::NZ>;
    using GlobalDataSrc2 = GlobalTensor<BiasType, pto::Shape<1, 1, 1, 1, N>, pto::Stride<1 * N, 1 * N, 1 * N, N, 1>>;
    using GlobalDataOut = GlobalTensor<
        OutType, pto::Shape<N_outer, 1, M_outer, 16, c0>, pto::Stride<M * c0, M * c0, 16 * c0, c0, 1>, pto::Layout::NZ>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, AType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, BType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    using TileBiasData = Tile<TileType::Mat, BiasType, 1, N, BLayout::RowMajor, 1, N>;
    using BiasTileData = Tile<TileType::Bias, OutType, 1, N, BLayout::RowMajor, 1, N>;

    using RightTile = TileRight<BType, K, N, K, N>;
    using AccTile = Tile<TileType::Mat, OutType, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 1024>;
    using VecTile = Tile<TileType::Vec, OutType, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 1024>;

    using QuantMatTile = Tile<TileType::Mat, uint64_t, 1, N, BLayout::RowMajor, 1, N>;
    using QuantFbTile = Tile<TileType::Scaling, uint64_t, 1, N, BLayout::RowMajor, 1, N>;

    constexpr uint64_t l1Base =
        M * K * sizeof(AType) + K * N * sizeof(BType) + N * sizeof(BiasType) + M * N * sizeof(OutType);

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileBiasData biasDataTile;
    TASSIGN<0x0>(aMatTile);
    TASSIGN<M * K * sizeof(AType)>(bMatTile);
    TASSIGN<M * K * sizeof(AType) + K * N * sizeof(BType)>(biasDataTile);

    AccTile cMatTile;
    TASSIGN<M * K * sizeof(AType) + K * N * sizeof(BType) + N * sizeof(BiasType)>(cMatTile);

    QuantMatTile quantMatTile;
    TASSIGN<l1Base>(quantMatTile);

    VecTile cVecTile;
    TASSIGN<0x0>(cVecTile);

    RightTile bTile;
    TASSIGN<0x0>(bTile);

    BiasTileData biasTile;
    TASSIGN<0x0>(biasTile);

    QuantFbTile quantFbTile;
    TASSIGN<0x0>(quantFbTile);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    if constexpr (isBias) {
        TLOAD(biasDataTile, src2Global);
    }

    if constexpr (isQuant) {
        using QuantGlobal = GlobalTensor<uint64_t, pto::Shape<1, 1, 1, 1, N>, pto::Stride<N, N, N, N, 1>>;
        QuantGlobal quantGlobal(quantVec);
        TLOAD(quantMatTile, quantGlobal);
    }

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    if constexpr (isQuant) {
        TMOV(quantFbTile, quantMatTile);
    }

    TMOV(bTile, bMatTile);
    if constexpr (isBias) {
        TMOV(biasTile, biasDataTile);
    }

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    MatmulMacroConfig cfg;
    cfg.reluMode = MatmulReluMode::ScalarRelu;
    cfg.reluScalar = reluScalar;
    cfg.clipReluVal = clipReluVal;
    if constexpr (isQuant) {
        cfg.preQuantTileAddr = (uint64_t)GetTilePtr(quantFbTile.data());
    }

    if constexpr (isBias) {
        TMATMUL(cMatTile, aMatTile, bTile, biasTile, cfg);
    } else {
        TMATMUL(cMatTile, aMatTile, bTile, cfg);
    }

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    TMOV(cVecTile, cMatTile);

    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, cVecTile);

    out = dstGlobal.data();
}

template <
    typename OutType, typename AType, typename BType, typename BiasType, int validM, int validK, int validN,
    bool isBias>
__global__ AICORE void RunTMATMULNormalRelu(
    __gm__ OutType* out, __gm__ AType* src0, __gm__ BType* src1, __gm__ BiasType* src2, __gm__ uint64_t* quantVec)
{
    constexpr int blockAlign = 32;
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, blockAlign);
    constexpr int c0 = blockAlign / sizeof(OutType);
    constexpr int c0A = blockAlign / sizeof(AType);
    constexpr int c0B = blockAlign / sizeof(BType);
    constexpr int M_outer = M / 16;
    constexpr int N_outer = N / c0;
    constexpr int K_outerA = K / c0A;
    constexpr int K_outerB = K / 16;
    constexpr int N_outerB = N / c0B;
    constexpr bool isQuant = !std::is_same_v<OutType, int32_t>;

    using GlobalDataSrc0 = GlobalTensor<
        AType, pto::Shape<K_outerA, 1, M_outer, 16, c0A>, pto::Stride<M * c0A, M * c0A, 16 * c0A, c0A, 1>,
        pto::Layout::NZ>;
    using GlobalDataSrc1 = GlobalTensor<
        BType, pto::Shape<N_outerB, 1, K_outerB, 16, c0B>, pto::Stride<K * c0B, K * c0B, 16 * c0B, c0B, 1>,
        pto::Layout::NZ>;
    using GlobalDataSrc2 = GlobalTensor<BiasType, pto::Shape<1, 1, 1, 1, N>, pto::Stride<1 * N, 1 * N, 1 * N, N, 1>>;
    using GlobalDataOut = GlobalTensor<
        OutType, pto::Shape<N_outer, 1, M_outer, 16, c0>, pto::Stride<M * c0, M * c0, 16 * c0, c0, 1>, pto::Layout::NZ>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, AType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, BType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    using TileBiasData = Tile<TileType::Mat, BiasType, 1, N, BLayout::RowMajor, 1, N>;
    using BiasTileData = Tile<TileType::Bias, OutType, 1, N, BLayout::RowMajor, 1, N>;

    using RightTile = TileRight<BType, K, N, K, N>;
    using AccTile = Tile<TileType::Mat, OutType, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 1024>;
    using VecTile = Tile<TileType::Vec, OutType, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 1024>;

    using QuantMatTile = Tile<TileType::Mat, uint64_t, 1, N, BLayout::RowMajor, 1, N>;
    using QuantFbTile = Tile<TileType::Scaling, uint64_t, 1, N, BLayout::RowMajor, 1, N>;

    constexpr uint64_t l1Base =
        M * K * sizeof(AType) + K * N * sizeof(BType) + N * sizeof(BiasType) + M * N * sizeof(OutType);

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileBiasData biasDataTile;
    TASSIGN<0x0>(aMatTile);
    TASSIGN<M * K * sizeof(AType)>(bMatTile);
    TASSIGN<M * K * sizeof(AType) + K * N * sizeof(BType)>(biasDataTile);

    AccTile cMatTile;
    TASSIGN<M * K * sizeof(AType) + K * N * sizeof(BType) + N * sizeof(BiasType)>(cMatTile);

    QuantMatTile quantMatTile;
    TASSIGN<l1Base>(quantMatTile);

    VecTile cVecTile;
    TASSIGN<0x0>(cVecTile);

    RightTile bTile;
    TASSIGN<0x0>(bTile);

    BiasTileData biasTile;
    TASSIGN<0x0>(biasTile);

    QuantFbTile quantFbTile;
    TASSIGN<0x0>(quantFbTile);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    if constexpr (isBias) {
        TLOAD(biasDataTile, src2Global);
    }

    if constexpr (isQuant) {
        using QuantGlobal = GlobalTensor<uint64_t, pto::Shape<1, 1, 1, 1, N>, pto::Stride<N, N, N, N, 1>>;
        QuantGlobal quantGlobal(quantVec);
        TLOAD(quantMatTile, quantGlobal);
    }

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    if constexpr (isQuant) {
        TMOV(quantFbTile, quantMatTile);
    }

    TMOV(bTile, bMatTile);
    if constexpr (isBias) {
        TMOV(biasTile, biasDataTile);
    }

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    MatmulMacroConfig cfg;
    cfg.reluMode = MatmulReluMode::NormalRelu;
    if constexpr (isQuant) {
        cfg.preQuantTileAddr = (uint64_t)GetTilePtr(quantFbTile.data());
    }

    if constexpr (isBias) {
        TMATMUL(cMatTile, aMatTile, bTile, biasTile, cfg);
    } else {
        TMATMUL(cMatTile, aMatTile, bTile, cfg);
    }

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    TMOV(cVecTile, cMatTile);

    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, cVecTile);

    out = dstGlobal.data();
}

template <
    typename OutType, typename AType, typename BType, typename BiasType, int validM, int validK, int validN,
    bool isBias>
__global__ AICORE void RunTMATMULVectorRelu(
    __gm__ OutType* out, __gm__ AType* src0, __gm__ BType* src1, __gm__ BiasType* src2, __gm__ uint64_t* quantVec,
    __gm__ uint32_t* vectorReluVec)
{
    constexpr int blockAlign = 32;
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, blockAlign);
    constexpr int c0 = blockAlign / sizeof(OutType);
    constexpr int c0A = blockAlign / sizeof(AType);
    constexpr int c0B = blockAlign / sizeof(BType);
    constexpr int M_outer = M / 16;
    constexpr int N_outer = N / c0;
    constexpr int K_outerA = K / c0A;
    constexpr int K_outerB = K / 16;
    constexpr int N_outerB = N / c0B;
    constexpr bool isQuant = !std::is_same_v<OutType, int32_t>;

    using GlobalDataSrc0 = GlobalTensor<
        AType, pto::Shape<K_outerA, 1, M_outer, 16, c0A>, pto::Stride<M * c0A, M * c0A, 16 * c0A, c0A, 1>,
        pto::Layout::NZ>;
    using GlobalDataSrc1 = GlobalTensor<
        BType, pto::Shape<N_outerB, 1, K_outerB, 16, c0B>, pto::Stride<K * c0B, K * c0B, 16 * c0B, c0B, 1>,
        pto::Layout::NZ>;
    using GlobalDataSrc2 = GlobalTensor<BiasType, pto::Shape<1, 1, 1, 1, N>, pto::Stride<1 * N, 1 * N, 1 * N, N, 1>>;
    using GlobalDataOut = GlobalTensor<
        OutType, pto::Shape<N_outer, 1, M_outer, 16, c0>, pto::Stride<M * c0, M * c0, 16 * c0, c0, 1>, pto::Layout::NZ>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, AType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, BType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    using TileBiasData = Tile<TileType::Mat, BiasType, 1, N, BLayout::RowMajor, 1, N>;
    using BiasTileData = Tile<TileType::Bias, OutType, 1, N, BLayout::RowMajor, 1, N>;

    using RightTile = TileRight<BType, K, N, K, N>;
    using AccTile = Tile<TileType::Mat, OutType, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 1024>;
    using VecTile = Tile<TileType::Vec, OutType, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 1024>;

    using QuantMatTile = Tile<TileType::Mat, uint64_t, 1, N, BLayout::RowMajor, 1, N>;
    using QuantFbTile = Tile<TileType::Scaling, uint64_t, 1, N, BLayout::RowMajor, 1, N>;
    using ReluMatTile = Tile<TileType::Mat, uint32_t, 1, N, BLayout::RowMajor, 1, N>;
    using ReluFbTile = Tile<TileType::Scaling, uint32_t, 1, N, BLayout::RowMajor, 1, N>;

    constexpr uint64_t l1Base = M * K * sizeof(AType) + K * N * sizeof(BType) + N * sizeof(BiasType) +
                                M * N * sizeof(OutType) + (isQuant ? N * sizeof(uint64_t) : 0) + N * sizeof(uint32_t);

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileBiasData biasDataTile;
    TASSIGN<0x0>(aMatTile);
    TASSIGN<M * K * sizeof(AType)>(bMatTile);
    TASSIGN<M * K * sizeof(AType) + K * N * sizeof(BType)>(biasDataTile);

    AccTile cMatTile;
    TASSIGN<M * K * sizeof(AType) + K * N * sizeof(BType) + N * sizeof(BiasType)>(cMatTile);

    QuantMatTile quantMatTile;
    TASSIGN<l1Base>(quantMatTile);

    ReluMatTile reluMatTile;
    TASSIGN<l1Base + (isQuant ? N * sizeof(uint64_t) : 0)>(reluMatTile);

    VecTile cVecTile;
    TASSIGN<0x0>(cVecTile);

    RightTile bTile;
    TASSIGN<0x0>(bTile);

    BiasTileData biasTile;
    TASSIGN<0x0>(biasTile);

    QuantFbTile quantFbTile;
    TASSIGN<0x0>(quantFbTile);

    ReluFbTile reluFbTile;
    TASSIGN<0x800>(reluFbTile);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    if constexpr (isBias) {
        TLOAD(biasDataTile, src2Global);
    }

    if constexpr (isQuant) {
        using QuantGlobal = GlobalTensor<uint64_t, pto::Shape<1, 1, 1, 1, N>, pto::Stride<N, N, N, N, 1>>;
        QuantGlobal quantGlobal(quantVec);
        TLOAD(quantMatTile, quantGlobal);
    }

    using ReluGlobal = GlobalTensor<uint32_t, pto::Shape<1, 1, 1, 1, N>, pto::Stride<N, N, N, N, 1>>;
    ReluGlobal reluGlobal(vectorReluVec);
    TLOAD(reluMatTile, reluGlobal);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    if constexpr (isQuant) {
        TMOV(quantFbTile, quantMatTile);
    }
    TMOV(reluFbTile, reluMatTile);

    TMOV(bTile, bMatTile);
    if constexpr (isBias) {
        TMOV(biasTile, biasDataTile);
    }

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    MatmulMacroConfig cfg;
    cfg.reluMode = MatmulReluMode::VectorRelu;
    cfg.vectorReluTileAddr = (uint64_t)GetTilePtr(reluFbTile.data());
    if constexpr (isQuant) {
        cfg.preQuantTileAddr = (uint64_t)GetTilePtr(quantFbTile.data());
    }

    if constexpr (isBias) {
        TMATMUL(cMatTile, aMatTile, bTile, biasTile, cfg);
    } else {
        TMATMUL(cMatTile, aMatTile, bTile, cfg);
    }

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    TMOV(cVecTile, cMatTile);

    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, cVecTile);

    out = dstGlobal.data();
}

template <
    typename OutType, typename AType, typename BType, typename BiasType, int validK, int validN, bool isBias,
    bool useTgemv = false>
__global__ AICORE void RunGEMV(
    __gm__ OutType* out, __gm__ AType* src0, __gm__ BType* src1, __gm__ BiasType* src2, __gm__ uint64_t* quantVec)
{
    constexpr int blockAlign = 32;
    constexpr int M = 16;
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, blockAlign);
    constexpr int c0 = blockAlign / sizeof(OutType);
    constexpr int c0A = blockAlign / sizeof(AType);
    constexpr int c0B = blockAlign / sizeof(BType);
    constexpr int M_outer = M / 16;
    constexpr int N_outer = N / c0;
    constexpr int K_outerA = K / c0A;
    constexpr int K_outerB = K / 16;
    constexpr int N_outerB = N / c0B;
    constexpr bool isFp16 = std::is_same_v<OutType, half>;
    constexpr bool isQuant = !std::is_same_v<OutType, int32_t>;

    using GlobalDataSrc0 = GlobalTensor<
        AType, pto::Shape<K_outerA, 1, M_outer, 16, c0A>, pto::Stride<M * c0A, M * c0A, 16 * c0A, c0A, 1>,
        pto::Layout::NZ>;
    using GlobalDataSrc1 = GlobalTensor<
        BType, pto::Shape<N_outerB, 1, K_outerB, 16, c0B>, pto::Stride<K * c0B, K * c0B, 16 * c0B, c0B, 1>,
        pto::Layout::NZ>;
    using GlobalDataSrc2 = GlobalTensor<BiasType, pto::Shape<1, 1, 1, 1, N>, pto::Stride<1 * N, 1 * N, 1 * N, N, 1>>;
    using GlobalDataOut = GlobalTensor<
        OutType, pto::Shape<N_outer, 1, M_outer, 16, c0>, pto::Stride<M * c0, M * c0, 16 * c0, c0, 1>, pto::Layout::NZ>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(src2);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, AType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, BType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    using TileBiasData = Tile<TileType::Mat, BiasType, 1, N, BLayout::RowMajor, 1, N>;
    using BiasTileData = Tile<TileType::Bias, OutType, 1, N, BLayout::RowMajor, 1, N>;

    using RightTile = TileRight<BType, K, N, K, N>;
    using AccTile = Tile<TileType::Mat, OutType, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 1024>;
    using VecTile = Tile<TileType::Vec, OutType, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 1024>;

    using QuantMatTile = Tile<TileType::Mat, uint64_t, 1, N, BLayout::RowMajor, 1, N>;
    using QuantFbTile = Tile<TileType::Scaling, uint64_t, 1, N, BLayout::RowMajor, 1, N>;

    constexpr uint64_t l1Base =
        M * K * sizeof(AType) + K * N * sizeof(BType) + N * sizeof(BiasType) + M * N * sizeof(OutType);

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileBiasData biasDataTile;
    TASSIGN<0x0>(aMatTile);
    TASSIGN<M * K * sizeof(AType)>(bMatTile);
    TASSIGN<M * K * sizeof(AType) + K * N * sizeof(BType)>(biasDataTile);

    AccTile cMatTile;
    TASSIGN<M * K * sizeof(AType) + K * N * sizeof(BType) + N * sizeof(BiasType)>(cMatTile);

    QuantMatTile quantMatTile;
    TASSIGN<l1Base>(quantMatTile);

    VecTile cVecTile;
    TASSIGN<0x0>(cVecTile);

    RightTile bTile;
    TASSIGN<0x0>(bTile);

    BiasTileData biasTile;
    TASSIGN<0x0>(biasTile);

    QuantFbTile quantFbTile;
    TASSIGN<0x0>(quantFbTile);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    if constexpr (isBias) {
        TLOAD(biasDataTile, src2Global);
    }

    if constexpr (isQuant) {
        using QuantGlobal = GlobalTensor<uint64_t, pto::Shape<1, 1, 1, 1, N>, pto::Stride<N, N, N, N, 1>>;
        QuantGlobal quantGlobal(quantVec);
        TLOAD(quantMatTile, quantGlobal);
    }

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    if constexpr (isQuant) {
        TMOV(quantFbTile, quantMatTile);
    }
    TMOV(bTile, bMatTile);
    if constexpr (isBias) {
        TMOV(biasTile, biasDataTile);
    }

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    MatmulMacroConfig cfg;
    cfg.gemvCtrl = true;
    if constexpr (isQuant) {
        cfg.preQuantTileAddr = (uint64_t)GetTilePtr(quantFbTile.data());
    }

    if constexpr (isBias) {
        TMATMUL(cMatTile, aMatTile, bTile, biasTile, cfg);
    } else if constexpr (useTgemv) {
        TGEMV(cMatTile, aMatTile, bTile, cfg);
    } else {
        TMATMUL(cMatTile, aMatTile, bTile, cfg);
    }

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    TMOV(cVecTile, cMatTile);

    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, cVecTile);

    out = dstGlobal.data();
}

template <
    typename OutType, typename AType, typename BType, typename BiasType, int validM, int validK, int validN,
    bool isBias, bool isGemv = false>
__global__ AICORE void RunTMATMUL_SPLIT_K(
    __gm__ OutType* out, __gm__ AType* src0, __gm__ BType* src1, __gm__ BiasType* src2, __gm__ uint64_t* quantVec)
{
    constexpr int blockAlign = 32;
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, blockAlign);
    constexpr int BASEK = 64;
    constexpr int iter = K / BASEK;
    constexpr int c0 = blockAlign / sizeof(OutType);
    constexpr int c0A = blockAlign / sizeof(AType);
    constexpr int c0B = blockAlign / sizeof(BType);
    constexpr int M_outer = M / 16;
    constexpr int N_outer = N / c0;
    constexpr int N_outerB = N / c0B;
    constexpr int BK_outerA = BASEK / c0A;
    constexpr int BK_outerB = BASEK / 16;
    constexpr bool isFp16 = std::is_same_v<OutType, half>;
    constexpr bool isQuant = !std::is_same_v<OutType, int32_t>;

    using GlobalDataSrc0Tile = GlobalTensor<
        AType, pto::Shape<BK_outerA, 1, M_outer, 16, c0A>, pto::Stride<M * c0A, M * c0A, 16 * c0A, c0A, 1>,
        pto::Layout::NZ>;
    using GlobalDataSrc1Tile = GlobalTensor<
        BType, pto::Shape<N_outerB, 1, BK_outerB, 16, c0B>, pto::Stride<K * c0B, K * c0B, 16 * c0B, c0B, 1>,
        pto::Layout::NZ>;
    using GlobalDataSrc2 = GlobalTensor<BiasType, pto::Shape<1, 1, 1, 1, N>, pto::Stride<1 * N, 1 * N, 1 * N, N, 1>>;
    using GlobalDataOut = GlobalTensor<
        OutType, pto::Shape<N_outer, 1, M_outer, 16, c0>, pto::Stride<M * c0, M * c0, 16 * c0, c0, 1>, pto::Layout::NZ>;
    GlobalDataSrc2 src2Global(src2);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, AType, M, BASEK, BLayout::ColMajor, M, BASEK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, BType, BASEK, N, BLayout::ColMajor, BASEK, N, SLayout::RowMajor, 512>;
    using TileBiasData = Tile<TileType::Mat, BiasType, 1, N, BLayout::RowMajor, 1, N>;
    using BiasTileData = Tile<TileType::Bias, OutType, 1, N, BLayout::RowMajor, 1, N>;

    using RightTile = TileRight<BType, BASEK, N, BASEK, N>;
    using AccTile = Tile<TileType::Mat, int32_t, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 1024>;
    using OutTile = Tile<TileType::Mat, OutType, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 1024>;
    using VecTile = Tile<TileType::Vec, OutType, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 1024>;

    using QuantMatTile = Tile<TileType::Mat, uint64_t, 1, N, BLayout::RowMajor, 1, N>;
    using QuantFbTile = Tile<TileType::Scaling, uint64_t, 1, N, BLayout::RowMajor, 1, N>;

    constexpr uint64_t l1Base = M * K * sizeof(AType) + K * N * sizeof(BType) + N * sizeof(BiasType) +
                                M * N * sizeof(int32_t) + M * N * sizeof(OutType);

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileBiasData biasDataTile;
    TASSIGN<0x0>(aMatTile);
    TASSIGN<M * K * sizeof(AType)>(bMatTile);
    TASSIGN<M * K * sizeof(AType) + K * N * sizeof(BType)>(biasDataTile);

    AccTile cMatTile;
    TASSIGN<M * K * sizeof(AType) + K * N * sizeof(BType) + N * sizeof(BiasType)>(cMatTile);

    OutTile outMatTile;
    TASSIGN<M * K * sizeof(AType) + K * N * sizeof(BType) + N * sizeof(BiasType) + M * N * sizeof(int32_t)>(outMatTile);

    QuantMatTile quantMatTile;
    TASSIGN<l1Base>(quantMatTile);

    VecTile cVecTile;
    TASSIGN<0x0>(cVecTile);

    RightTile bTile;
    TASSIGN<0x0>(bTile);

    BiasTileData biasTile;
    TASSIGN<0x0>(biasTile);

    QuantFbTile quantFbTile;
    TASSIGN<0x0>(quantFbTile);

    if constexpr (isQuant) {
        using QuantGlobal = GlobalTensor<uint64_t, pto::Shape<1, 1, 1, 1, N>, pto::Stride<N, N, N, N, 1>>;
        QuantGlobal quantGlobal(quantVec);
        TLOAD(quantMatTile, quantGlobal);
    }

    for (int i = 0; i < iter; i++) {
        GlobalDataSrc0Tile src0Global(src0 + i * M * BASEK);
        GlobalDataSrc1Tile src1Global(src1 + i * BK_outerB * 16 * c0B);
        TLOAD(aMatTile, src0Global);
        TLOAD(bMatTile, src1Global);
        if constexpr (isBias) {
            TLOAD(biasDataTile, src2Global);
        }

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        if constexpr (isQuant) {
            TMOV(quantFbTile, quantMatTile);
        }
        TMOV(bTile, bMatTile);
        if constexpr (isBias) {
            TMOV(biasTile, biasDataTile);
        }

        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

        MatmulMacroConfig cfg;
        cfg.gemvCtrl = isGemv;
        if constexpr (isQuant) {
            cfg.preQuantTileAddr = (uint64_t)GetTilePtr(quantFbTile.data());
        }

        if (i == 0) {
            if constexpr (isBias) {
                TMATMUL_MACRO_ACC_IMPL<AccPhase::Unspecified, AccTile, TileMatAData, RightTile, BiasTileData>(
                    cMatTile, aMatTile, bTile, &biasTile, cfg);
            } else {
                TMATMUL_MACRO_ACC_IMPL<AccPhase::Unspecified, AccTile, TileMatAData, RightTile, void, true>(
                    cMatTile, aMatTile, bTile, nullptr, cfg);
            }
        } else if (i < iter - 1) {
            TMATMUL_MACRO_ACC_IMPL<AccPhase::Unspecified, AccTile, TileMatAData, RightTile, void, false>(
                cMatTile, aMatTile, bTile, nullptr, cfg);
        } else {
            __cbuf__ int32_t* accBiasPtr = (__cbuf__ int32_t*)GetTilePtr(cMatTile.data());
            TMatmulMacro<OutTile, TileMatAData, RightTile, false, false, true>(
                outMatTile.data(), aMatTile.data(), bTile.data(), accBiasPtr, 0, aMatTile.GetValidRow(),
                aMatTile.GetValidCol(), bTile.GetValidCol(), cfg);
        }

        set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
    }

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    TMOV(cVecTile, outMatTile);

    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, cVecTile);

    out = dstGlobal.data();
}

template <int32_t tilingKey>
void LaunchTMATMUL(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* quantVec, void* stream)
{
    if constexpr (tilingKey == 1) {
        RunTMATMUL<half, half, half, half, 40, 50, 60, false><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1), nullptr,
            reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 2) {
        RunTMATMUL<int32_t, int8_t, int8_t, int32_t, 6, 7, 8, false><<<1, nullptr, stream>>>(
            reinterpret_cast<int32_t*>(out), reinterpret_cast<int8_t*>(src0), reinterpret_cast<int8_t*>(src1), nullptr,
            reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 3) {
        RunTMATMUL<half, half, half, half, 1, 16, 512, false><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1), nullptr,
            reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 4) {
        RunTMATMUL<int32_t, int8_t, int8_t, int32_t, 26, 15, 27, false><<<1, nullptr, stream>>>(
            reinterpret_cast<int32_t*>(out), reinterpret_cast<int8_t*>(src0), reinterpret_cast<int8_t*>(src1), nullptr,
            reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 5) {
        RunTMATMUL<int32_t, int8_t, int8_t, int32_t, 101, 1, 99, false><<<1, nullptr, stream>>>(
            reinterpret_cast<int32_t*>(out), reinterpret_cast<int8_t*>(src0), reinterpret_cast<int8_t*>(src1), nullptr,
            reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 6) {
        RunTMATMUL<half, half, half, half, 33, 16, 2, false><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1), nullptr,
            reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 7) {
        RunTMATMUL<half, half, half, half, 17, 16, 2, false><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1), nullptr,
            reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 8) {
        RunTMATMUL<int32_t, int8_t, int8_t, int32_t, 33, 15, 2, false><<<1, nullptr, stream>>>(
            reinterpret_cast<int32_t*>(out), reinterpret_cast<int8_t*>(src0), reinterpret_cast<int8_t*>(src1), nullptr,
            reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 100) {
        RunTMATMUL<half, half, half, half, 16, 16, 16, false><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1), nullptr,
            reinterpret_cast<uint64_t*>(quantVec));
    }
}

template void LaunchTMATMUL<1>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* quantVec, void* stream);
template void LaunchTMATMUL<2>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* quantVec, void* stream);
template void LaunchTMATMUL<3>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* quantVec, void* stream);
template void LaunchTMATMUL<4>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* quantVec, void* stream);
template void LaunchTMATMUL<5>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* quantVec, void* stream);
template void LaunchTMATMUL<6>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* quantVec, void* stream);
template void LaunchTMATMUL<7>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* quantVec, void* stream);
template void LaunchTMATMUL<8>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* quantVec, void* stream);
template void LaunchTMATMUL<100>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* quantVec, void* stream);

template <int32_t tilingKey>
void LaunchTMATMULS16(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* quantVec, void* stream)
{
    if constexpr (tilingKey == 1) {
        RunTMATMUL<int16_t, half, half, int16_t, 16, 64, 64, false><<<1, nullptr, stream>>>(
            reinterpret_cast<int16_t*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1), nullptr,
            reinterpret_cast<uint64_t*>(quantVec));
    }
}

template void LaunchTMATMULS16<1>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* quantVec, void* stream);

template <int32_t tilingKey>
void LaunchTMATMULBIAS(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream)
{
    if constexpr (tilingKey == 1) {
        RunTMATMUL<int32_t, int8_t, int8_t, int32_t, 8, 7, 6, true><<<1, nullptr, stream>>>(
            reinterpret_cast<int32_t*>(out), reinterpret_cast<int8_t*>(src0), reinterpret_cast<int8_t*>(src1),
            reinterpret_cast<int32_t*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 2) {
        RunTMATMUL<half, half, half, half, 16, 15, 16, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 3) {
        RunTMATMUL<int32_t, int8_t, int8_t, int32_t, 66, 11, 1, true><<<1, nullptr, stream>>>(
            reinterpret_cast<int32_t*>(out), reinterpret_cast<int8_t*>(src0), reinterpret_cast<int8_t*>(src1),
            reinterpret_cast<int32_t*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 4) {
        RunTMATMUL<half, half, half, half, 1, 16, 1, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 5) {
        RunTMATMUL<half, half, half, half, 29, 11, 41, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 6) {
        RunTMATMUL<half, half, half, half, 2, 16, 1, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 7) {
        RunTMATMUL<half, half, half, half, 4, 16, 1, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 8) {
        RunTMATMUL<half, half, half, half, 8, 16, 1, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 9) {
        RunTMATMUL<half, half, half, half, 4, 16, 2, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 10) {
        RunTMATMUL<half, half, half, half, 4, 16, 4, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 11) {
        RunTMATMUL<half, half, half, half, 4, 16, 8, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 12) {
        RunTMATMUL<half, half, half, half, 4, 1, 1, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 13) {
        RunTMATMUL<half, half, half, half, 4, 2, 1, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 14) {
        RunTMATMUL<half, half, half, half, 4, 4, 1, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 15) {
        RunTMATMUL<half, half, half, half, 4, 8, 1, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 16) {
        RunTMATMUL<half, half, half, half, 16, 16, 16, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 17) {
        RunTMATMUL<half, half, half, half, 2, 16, 3, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 18) {
        RunTMATMUL<half, half, half, half, 2, 16, 5, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 19) {
        RunTMATMUL<half, half, half, half, 2, 16, 12, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 20) {
        RunTMATMUL<half, half, half, half, 2, 16, 32, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 21) {
        RunTMATMUL<int32_t, int8_t, int8_t, int32_t, 4, 16, 2, true><<<1, nullptr, stream>>>(
            reinterpret_cast<int32_t*>(out), reinterpret_cast<int8_t*>(src0), reinterpret_cast<int8_t*>(src1),
            reinterpret_cast<int32_t*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 22) {
        RunTMATMUL<int32_t, int8_t, int8_t, int32_t, 4, 16, 16, true><<<1, nullptr, stream>>>(
            reinterpret_cast<int32_t*>(out), reinterpret_cast<int8_t*>(src0), reinterpret_cast<int8_t*>(src1),
            reinterpret_cast<int32_t*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 23) {
        RunTMATMUL<int32_t, int8_t, int8_t, int32_t, 4, 16, 32, true><<<1, nullptr, stream>>>(
            reinterpret_cast<int32_t*>(out), reinterpret_cast<int8_t*>(src0), reinterpret_cast<int8_t*>(src1),
            reinterpret_cast<int32_t*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 24) {
        RunTMATMUL<int32_t, int8_t, int8_t, int32_t, 4, 16, 63, true><<<1, nullptr, stream>>>(
            reinterpret_cast<int32_t*>(out), reinterpret_cast<int8_t*>(src0), reinterpret_cast<int8_t*>(src1),
            reinterpret_cast<int32_t*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 25) {
        RunTMATMUL<half, half, half, half, 2, 16, 33, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 26) {
        RunTMATMUL<half, half, half, half, 2, 16, 48, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 27) {
        RunTMATMUL<half, half, half, half, 2, 16, 63, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 28) {
        RunTMATMUL<half, half, half, half, 2, 16, 64, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 29) {
        RunTMATMUL<half, half, half, half, 29, 11, 2, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 30) {
        RunTMATMUL<half, half, half, half, 2, 16, 41, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 31) {
        RunTMATMUL<half, half, half, half, 17, 16, 2, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 32) {
        RunTMATMUL<half, half, half, half, 20, 16, 2, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 33) {
        RunTMATMUL<half, half, half, half, 32, 16, 2, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 34) {
        RunTMATMUL<half, half, half, half, 33, 16, 2, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 35) {
        RunTMATMUL<int32_t, int8_t, int8_t, int32_t, 33, 15, 2, true><<<1, nullptr, stream>>>(
            reinterpret_cast<int32_t*>(out), reinterpret_cast<int8_t*>(src0), reinterpret_cast<int8_t*>(src1),
            reinterpret_cast<int32_t*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    }
}

template void LaunchTMATMULBIAS<1>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<2>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<3>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<4>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<5>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<6>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<7>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<8>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<9>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<10>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<11>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<12>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<13>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<14>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<15>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<16>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<17>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<18>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<19>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<20>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<21>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<22>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<23>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<24>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<25>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<26>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<27>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<28>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<29>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<30>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<31>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<32>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<33>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<34>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS<35>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);

template <int32_t tilingKey>
void LaunchTMATMUL_SPLIT_K(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* quantVec, void* stream)
{
    if constexpr (tilingKey == 1) {
        RunTMATMUL_SPLIT_K<half, half, half, half, 16, 128, 64, false><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1), nullptr,
            reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 2) {
        RunTMATMUL_SPLIT_K<half, half, half, half, 16, 256, 64, false><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1), nullptr,
            reinterpret_cast<uint64_t*>(quantVec));
    }
}

template void LaunchTMATMUL_SPLIT_K<1>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* quantVec, void* stream);
template void LaunchTMATMUL_SPLIT_K<2>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* quantVec, void* stream);

template <int32_t tilingKey>
void LaunchTMATMULBIAS_SPLIT_K(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream)
{
    if constexpr (tilingKey == 1) {
        RunTMATMUL_SPLIT_K<half, half, half, half, 16, 128, 64, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 2) {
        RunTMATMUL_SPLIT_K<half, half, half, half, 16, 256, 64, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    }
}

template void LaunchTMATMULBIAS_SPLIT_K<1>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchTMATMULBIAS_SPLIT_K<2>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);

template <int32_t tilingKey>
void LaunchGEMV(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* quantVec, void* stream)
{
    if constexpr (tilingKey == 1) {
        RunGEMV<half, half, half, half, 64, 64, false><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1), nullptr,
            reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 2) {
        RunGEMV<int32_t, int8_t, int8_t, int32_t, 64, 64, false><<<1, nullptr, stream>>>(
            reinterpret_cast<int32_t*>(out), reinterpret_cast<int8_t*>(src0), reinterpret_cast<int8_t*>(src1), nullptr,
            reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 3) {
        RunGEMV<half, half, half, half, 64, 64, false, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1), nullptr,
            reinterpret_cast<uint64_t*>(quantVec));
    }
}

template void LaunchGEMV<1>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* quantVec, void* stream);
template void LaunchGEMV<2>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* quantVec, void* stream);
template void LaunchGEMV<3>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* quantVec, void* stream);

template <int32_t tilingKey>
void LaunchGEMVBIAS(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream)
{
    if constexpr (tilingKey == 1) {
        RunGEMV<half, half, half, half, 64, 64, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1),
            reinterpret_cast<half*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    } else if constexpr (tilingKey == 2) {
        RunGEMV<int32_t, int8_t, int8_t, int32_t, 64, 64, true><<<1, nullptr, stream>>>(
            reinterpret_cast<int32_t*>(out), reinterpret_cast<int8_t*>(src0), reinterpret_cast<int8_t*>(src1),
            reinterpret_cast<int32_t*>(src2), reinterpret_cast<uint64_t*>(quantVec));
    }
}

template void LaunchGEMVBIAS<1>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);
template void LaunchGEMVBIAS<2>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* quantVec, void* stream);

template <int32_t tilingKey>
void LaunchGEMV_SPLIT_K(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* quantVec, void* stream)
{
    if constexpr (tilingKey == 1) {
        RunTMATMUL_SPLIT_K<half, half, half, half, 1, 128, 64, false, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1), nullptr,
            reinterpret_cast<uint64_t*>(quantVec));
    }
}

template void LaunchGEMV_SPLIT_K<1>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* quantVec, void* stream);

template <int32_t tilingKey>
void LaunchTMATMULRelu(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* quantVec, void* stream, float reluScalar, float clipReluVal)
{
    if constexpr (tilingKey == 1) {
        RunTMATMULRelu<half, half, half, half, 16, 64, 64, false><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1), nullptr,
            reinterpret_cast<uint64_t*>(quantVec), reluScalar, clipReluVal);
    }
}

template void LaunchTMATMULRelu<1>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* quantVec, void* stream, float reluScalar, float clipReluVal);

template <int32_t tilingKey>
void LaunchTMATMULReluS8(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* quantVec, void* stream, float reluScalar, float clipReluVal)
{
    if constexpr (tilingKey == 1) {
        RunTMATMULRelu<int8_t, half, half, int8_t, 16, 64, 64, false><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1), nullptr,
            reinterpret_cast<uint64_t*>(quantVec), reluScalar, clipReluVal);
    }
}

template void LaunchTMATMULReluS8<1>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* quantVec, void* stream, float reluScalar, float clipReluVal);

template <int32_t tilingKey>
void LaunchTMATMULNormalRelu(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* quantVec, void* stream)
{
    if constexpr (tilingKey == 1) {
        RunTMATMULNormalRelu<half, half, half, half, 16, 64, 64, false><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1), nullptr,
            reinterpret_cast<uint64_t*>(quantVec));
    }
}

template void LaunchTMATMULNormalRelu<1>(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* quantVec, void* stream);

template <int32_t tilingKey>
void LaunchTMATMULVectorRelu(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* quantVec, void* stream, uint8_t* vectorReluVec)
{
    if constexpr (tilingKey == 1) {
        RunTMATMULVectorRelu<half, half, half, half, 16, 64, 64, false><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1), nullptr,
            reinterpret_cast<uint64_t*>(quantVec), reinterpret_cast<uint32_t*>(vectorReluVec));
    }
}

template void LaunchTMATMULVectorRelu<1>(
    uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* quantVec, void* stream, uint8_t* vectorReluVec);
