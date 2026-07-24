/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_PTO_GMM_PRELOAD_ASYNC_FIXPIPE_QUANT_HPP
#define DISPATCH_MEGA_COMBINE_PTO_GMM_PRELOAD_ASYNC_FIXPIPE_QUANT_HPP

#include <type_traits>

#include "const_args.hpp"
#include "common_helpers.hpp"
#include "pto_sync_substrate.hpp"
#include "pto/common/pto_tile.hpp"
#include "pto/pto-inst.hpp"

constexpr uint32_t BYTE_PER_C0 = 32;
constexpr uint32_t BYTE_PER_FRACTAL = BYTE_PER_C0 * 16;
constexpr uint32_t C0_NUM_PER_FRACTAL = 16;

template <typename Element, int Rows, int Cols>
AICORE inline void PtoLoadNdGmToNzL1(
    uint64_t dstL1Offset, __gm__ Element* src, uint32_t rows, uint32_t cols, uint32_t leadingDim)
{
    using L1Tile = pto::Tile<
        pto::TileType::Mat, Element, Rows, Cols, pto::BLayout::ColMajor, pto::DYNAMIC, pto::DYNAMIC,
        pto::SLayout::RowMajor>;
    using SrcShape = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
    using SrcStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using SrcGlobal = pto::GlobalTensor<Element, SrcShape, SrcStride, pto::Layout::ND>;

    SrcShape srcShape(rows, cols);
    SrcStride srcStride(
        static_cast<int64_t>(rows) * leadingDim, static_cast<int64_t>(rows) * leadingDim,
        static_cast<int64_t>(rows) * leadingDim, leadingDim);
    SrcGlobal srcGlobal(src, srcShape, srcStride);
    L1Tile dstTile(rows, cols);
    pto::TASSIGN(dstTile, dstL1Offset);
    pto::TLOAD(dstTile, srcGlobal);
}

template <typename Element, int Rows, int Cols>
AICORE inline void PtoLoadPackedWeightGmToL1(
    uint64_t dstL1Offset, __gm__ Element* src, uint32_t validRows, uint32_t validCols, uint32_t fullRows)
{
    constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    using L1Tile = pto::Tile<
        pto::TileType::Mat, Element, Rows, Cols, pto::BLayout::ColMajor, pto::DYNAMIC, pto::DYNAMIC,
        pto::SLayout::RowMajor>;
    using SrcShape = pto::Shape<1, pto::DYNAMIC, pto::DYNAMIC, C0_NUM_PER_FRACTAL, ELE_NUM_PER_C0>;
    using SrcStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, ELE_NUM_PER_C0, 1>;
    using SrcGlobal = pto::GlobalTensor<Element, SrcShape, SrcStride, pto::Layout::NZ>;

    const uint32_t rowBlocks = ceilDiv<C0_NUM_PER_FRACTAL>(validRows);
    const uint32_t colBlocks = ceilDiv<ELE_NUM_PER_C0>(validCols);
    const uint32_t validRowsAligned = rowBlocks * C0_NUM_PER_FRACTAL;
    const uint32_t validColsAligned = colBlocks * ELE_NUM_PER_C0;
    const uint32_t fullRowsAligned = roundUp<C0_NUM_PER_FRACTAL>(fullRows);
    const uint32_t srcColBlockStride = fullRowsAligned * ELE_NUM_PER_C0;
    const uint32_t rowBlockStride = BYTE_PER_FRACTAL / sizeof(Element);

    SrcShape srcShape(colBlocks, rowBlocks);
    SrcStride srcStride(static_cast<int64_t>(srcColBlockStride) * colBlocks, srcColBlockStride, rowBlockStride);
    SrcGlobal srcGlobal(src, srcShape, srcStride);
    L1Tile dstTile(validRowsAligned, validColsAligned);
    pto::TASSIGN(dstTile, dstL1Offset);
    pto::TLOAD(dstTile, srcGlobal);
}

template <typename ElementDst, int Rows, int Cols>
AICORE inline void PtoStoreQuantAccToGm(
    __gm__ ElementDst* dst, uint64_t accOffset, uint64_t scaleOffset, uint32_t validRow, uint32_t validCol,
    uint32_t leadingDim)
{
    using DstShape = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
    using DstStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using GlobalDataOut = pto::GlobalTensor<ElementDst, DstShape, DstStride, pto::Layout::ND>;
    using AccTile = pto::TileAccCompact<int32_t, Rows, Cols, pto::DYNAMIC, pto::DYNAMIC>;
    using ScalingTile = pto::Tile<
        pto::TileType::Scaling, uint64_t, 1, Cols, pto::BLayout::RowMajor, 1, pto::DYNAMIC, pto::SLayout::NoneBox>;

    DstShape dstShape(validRow, validCol);
    DstStride dstStride(
        static_cast<int64_t>(validRow) * leadingDim, static_cast<int64_t>(validRow) * leadingDim,
        static_cast<int64_t>(validRow) * leadingDim, leadingDim);
    GlobalDataOut dstGlobal(dst, dstShape, dstStride);
    AccTile accTile(validRow, validCol);
    ScalingTile scalingTile(validCol);

    pto::TASSIGN(accTile, accOffset);
    pto::TASSIGN(scalingTile, scaleOffset);
    pto::TSTORE_FP<AccTile, GlobalDataOut, ScalingTile>(dstGlobal, accTile, scalingTile);
}

template <int Cols>
AICORE inline void StagePerChannelScale(
    uint64_t l1SOffset, uint64_t fixpipeOffset, __gm__ uint64_t* gmBlockS, uint32_t cols)
{
    using ScaleMatTile = pto::Tile<
        pto::TileType::Mat, uint64_t, 1, Cols, pto::BLayout::RowMajor, 1, pto::DYNAMIC, pto::SLayout::NoneBox>;
    using ScalingTile = pto::Tile<
        pto::TileType::Scaling, uint64_t, 1, Cols, pto::BLayout::RowMajor, 1, pto::DYNAMIC, pto::SLayout::NoneBox>;

    using ScaleShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using ScaleStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using ScaleGlobal = pto::GlobalTensor<uint64_t, ScaleShape, ScaleStride, pto::Layout::ND>;
    ScaleShape scaleShape(cols);
    ScaleStride scaleStride(cols, cols, cols, cols);
    ScaleGlobal gmBlockSGlobal(gmBlockS, scaleShape, scaleStride);
    ScaleMatTile scaleMatTile(cols);
    ScalingTile scalingTile(cols);

    pto::TASSIGN(scaleMatTile, l1SOffset);
    pto::TASSIGN(scalingTile, fixpipeOffset);
    pto::TLOAD(scaleMatTile, gmBlockSGlobal);
    set_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID0);
    pto::TMOV(scalingTile, scaleMatTile);
}

template <
    uint32_t PRELOAD_STAGES_, uint32_t L1_STAGES_, uint32_t L0A_STAGES_, uint32_t L0B_STAGES_, uint32_t L0C_STAGES_,
    bool ENABLE_SHUFFLE_K_, uint32_t L1_M_, uint32_t L1_N_, uint32_t L1_K_, uint32_t L0_M_, uint32_t L0_N_,
    uint32_t L0_K_, typename ElementA_, typename ElementB_, typename ElementC_>
struct PtoGmmPreloadAsyncFixpipe {
public:
    using ArchTag = AtlasA2;
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;
    using ElementAccumulator = int32_t;

    static_assert(PRELOAD_STAGES_ == 1, "mega_moe GMM uses single-stage L1 preload");
    static_assert(
        std::is_same_v<ElementA, int8_t> && std::is_same_v<ElementB, int8_t>,
        "mega_moe GMM supports int8 activations and weights only");
    static_assert(std::is_same_v<ElementC, half>, "mega_moe GMM output must be half");

    static constexpr uint32_t PRELOAD_STAGES = PRELOAD_STAGES_;
    static constexpr uint32_t L1_STAGES = L1_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;

    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
    static constexpr uint32_t L1_M = L1_M_;
    static constexpr uint32_t L1_N = L1_N_;
    static constexpr uint32_t L1_K = L1_K_;
    static constexpr uint32_t L0_M = L0_M_;
    static constexpr uint32_t L0_N = L0_N_;
    static constexpr uint32_t L0_K = L0_K_;

    static constexpr uint32_t L1A_TILE_SIZE = L1_M * L1_K * sizeof(ElementA);
    static constexpr uint32_t L1B_TILE_SIZE = L1_N * L1_K * sizeof(ElementB);
    static constexpr uint32_t L1S_TILE_SIZE = L1_N * sizeof(uint64_t);
    static constexpr uint32_t L0A_TILE_SIZE = L0_M * L0_K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0_K * L0_N * sizeof(ElementB);
    static constexpr uint32_t L0C_TILE_SIZE = L1_M * L1_N * sizeof(ElementAccumulator);
    static constexpr uint32_t L1_M_ALIGN = C0_NUM_PER_FRACTAL;
    static constexpr uint32_t L1_N_ALIGN = BYTE_PER_C0 / sizeof(ElementB);

    static_assert(
        (L1A_TILE_SIZE + L1B_TILE_SIZE + L1S_TILE_SIZE) * L1_STAGES <= ArchTag::L1_SIZE,
        "L1 tile exceeding the L1 space");
    static_assert(L0A_TILE_SIZE * L0A_STAGES <= ArchTag::L0A_SIZE, "L0 tile exceeding the L0A space");
    static_assert(L0B_TILE_SIZE * L0B_STAGES <= ArchTag::L0B_SIZE, "L0 tile exceeding the L0B space");
    static_assert(L0C_TILE_SIZE * L0C_STAGES <= ArchTag::L0C_SIZE, "L0 tile exceeding the L0C space");
    static_assert(L1_M == L0_M && L1_N == L0_N, "L1/L0 M/N tile sizes must match");

    struct GmmBlockParams {
        __gm__ ElementA* gmBlockA = nullptr;
        __gm__ ElementB* gmBlockB = nullptr;
        __gm__ ElementC* gmBlockC = nullptr;
        __gm__ uint64_t* gmBlockS = nullptr;
        uint32_t actualM = 0;
        uint32_t actualN = 0;
        uint32_t actualK = 0;
        uint32_t aLeadingDim = 0;
        uint32_t bFullRows = 0;
        uint32_t bColStart = 0;
        uint32_t cLeadingDim = 0;
    };

    AICORE inline PtoGmmPreloadAsyncFixpipe(uint32_t l1BufAddrStart = 0, uint32_t fpAddrStart = 0)
    {
        syncGroupIdx = 0;
        fixpipeBaseOffset = fpAddrStart;
        InitL1(l1BufAddrStart);
        InitL0A();
        InitL0B();
        InitL0C();
    }

    AICORE inline ~PtoGmmPreloadAsyncFixpipe()
    {
        SynchronizeBlock();
        for (uint32_t i = 0; i < L1_STAGES; ++i) {
            wait_flag(PIPE_MTE1, PIPE_MTE2, l1AEventList[i]);
            wait_flag(PIPE_MTE1, PIPE_MTE2, l1BEventList[i]);
        }
        for (uint32_t i = 0; i < L0A_STAGES; ++i) {
            wait_flag(PIPE_M, PIPE_MTE1, l0AEventList[i]);
        }
        for (uint32_t i = 0; i < L0B_STAGES; ++i) {
            wait_flag(PIPE_M, PIPE_MTE1, l0BEventList[i]);
        }
        for (uint32_t i = 0; i < L0C_STAGES; ++i) {
            wait_flag(PIPE_FIX, PIPE_M, l0CEventList[i]);
        }
        wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);
    }

    AICORE inline void RunTile(const GmmBlockParams& params)
    {
        const uint32_t kTileCount = ceilDiv<L1_K>(params.actualK);
        if (kTileCount == 0U) {
            return;
        }
        const uint32_t mRound = roundUp<L1_M_ALIGN>(params.actualM);
        const uint32_t nRound = roundUp<L1_N_ALIGN>(params.actualN);

        uint32_t startTileIdx = 0;
        if constexpr (ENABLE_SHUFFLE_K) {
            startTileIdx = get_block_idx() % kTileCount;
        }

        for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; ++kLoopIdx) {
            if (hasPendingKTile_) {
                L1TileGmm(pendingParams_);
            }

            const uint32_t kTileIdx = (startTileIdx + kLoopIdx < kTileCount) ? (startTileIdx + kLoopIdx) :
                                                                               (startTileIdx + kLoopIdx - kTileCount);
            const uint32_t kActual = (kTileIdx < kTileCount - 1) ? L1_K : (params.actualK - kTileIdx * L1_K);
            const uint32_t kStart = kTileIdx * L1_K;
            __gm__ ElementA* gmTileA = params.gmBlockA + kStart;

            constexpr uint32_t bElemsPerC0 = BYTE_PER_C0 / sizeof(ElementB);
            const uint32_t bRowsAligned = roundUp<C0_NUM_PER_FRACTAL>(params.bFullRows);
            const uint64_t gmTileBOffset =
                static_cast<uint64_t>(kStart / C0_NUM_PER_FRACTAL) * (BYTE_PER_FRACTAL / sizeof(ElementB)) +
                static_cast<uint64_t>(params.bColStart / bElemsPerC0) * bRowsAligned * bElemsPerC0 +
                static_cast<uint64_t>(kStart % C0_NUM_PER_FRACTAL) * bElemsPerC0 + params.bColStart % bElemsPerC0;
            __gm__ ElementB* gmTileB = params.gmBlockB + gmTileBOffset;

            wait_flag(PIPE_MTE1, PIPE_MTE2, l1AEventList[l1ListId]);
            PtoLoadNdGmToNzL1<ElementA, L1_M, L1_K>(
                l1AOffsetList[l1ListId], gmTileA, params.actualM, kActual, params.aLeadingDim);
            set_flag(PIPE_MTE2, PIPE_MTE1, l1AEventList[l1ListId]);

            wait_flag(PIPE_MTE1, PIPE_MTE2, l1BEventList[l1ListId]);
            PtoLoadPackedWeightGmToL1<ElementB, L1_K, L1_N>(
                l1BOffsetList[l1ListId], gmTileB, kActual, params.actualN, params.bFullRows);
            set_flag(PIPE_MTE2, PIPE_MTE1, l1BEventList[l1ListId]);

            pendingParams_.l1ListId = l1ListId;
            pendingParams_.mRound = mRound;
            pendingParams_.nRound = nRound;
            pendingParams_.kActual = kActual;
            pendingParams_.actualM = params.actualM;
            pendingParams_.actualN = params.actualN;
            pendingParams_.cLeadingDim = params.cLeadingDim;
            pendingParams_.isKLoopFirst = (kLoopIdx == 0);
            pendingParams_.isKLoopLast = (kLoopIdx == kTileCount - 1);
            if (kLoopIdx == kTileCount - 1) {
                pendingParams_.gmBlockC = params.gmBlockC;
                pendingParams_.gmBlockS = params.gmBlockS;
            }
            hasPendingKTile_ = true;

            l1ListId = (l1ListId + 1 < L1_STAGES) ? (l1ListId + 1) : 0;
        }
    }

    AICORE inline void SynchronizeBlock()
    {
        if (hasPendingKTile_) {
            L1TileGmm(pendingParams_);
            hasPendingKTile_ = false;
        }
    }

    AICORE inline void Finalize(int32_t target, int32_t flag = 0)
    {
        for (; syncGroupIdx <= target; ++syncGroupIdx) {
            const int32_t flagId = syncGroupIdx / 15 + flag;
            CrossCoreSetFlag<0x2, PIPE_FIX>(flagId);
        }
    }

private:
    struct L1TileGmmParams {
        uint32_t l1ListId;
        uint32_t mRound;
        uint32_t nRound;
        uint32_t kActual;
        uint32_t actualM;
        uint32_t actualN;
        uint32_t cLeadingDim;
        bool isKLoopFirst;
        bool isKLoopLast;
        __gm__ ElementC* gmBlockC;
        __gm__ uint64_t* gmBlockS;
    };

    AICORE inline void InitL1(uint32_t l1BufAddrStart)
    {
        const uint32_t l1AOffset = l1BufAddrStart;
        const uint32_t l1BOffset = l1BufAddrStart + L1A_TILE_SIZE * L1_STAGES;

        for (uint32_t i = 0; i < L1_STAGES; ++i) {
            l1AOffsetList[i] = l1AOffset + L1A_TILE_SIZE * i;
            l1BOffsetList[i] = l1BOffset + L1B_TILE_SIZE * i;
            l1AEventList[i] = static_cast<int32_t>(i);
            l1BEventList[i] = static_cast<int32_t>(i + L1_STAGES);
            set_flag(PIPE_MTE1, PIPE_MTE2, l1AEventList[i]);
            set_flag(PIPE_MTE1, PIPE_MTE2, l1BEventList[i]);
        }
        l1SBaseOffset = l1BOffset + L1B_TILE_SIZE * L1_STAGES;
        set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);
    }

    AICORE inline void InitL0A()
    {
        for (uint32_t i = 0; i < L0A_STAGES; ++i) {
            l0AOffsetList[i] = L0A_TILE_SIZE * i;
            l0AEventList[i] = static_cast<int32_t>(i);
            set_flag(PIPE_M, PIPE_MTE1, l0AEventList[i]);
        }
    }

    AICORE inline void InitL0B()
    {
        for (uint32_t i = 0; i < L0B_STAGES; ++i) {
            l0BOffsetList[i] = L0B_TILE_SIZE * i;
            l0BEventList[i] = static_cast<int32_t>(i + L0A_STAGES);
            set_flag(PIPE_M, PIPE_MTE1, l0BEventList[i]);
        }
    }

    AICORE inline void InitL0C()
    {
        for (uint32_t i = 0; i < L0C_STAGES; ++i) {
            l0COffsetList[i] = L0C_TILE_SIZE * i;
            l0CEventList[i] = static_cast<int32_t>(i);
            set_flag(PIPE_FIX, PIPE_M, l0CEventList[i]);
        }
    }

    using APanelTile = pto::Tile<
        pto::TileType::Mat, ElementA, L1_M, L1_K, pto::BLayout::ColMajor, pto::DYNAMIC, pto::DYNAMIC,
        pto::SLayout::RowMajor>;
    using BPanelTile = pto::Tile<
        pto::TileType::Mat, ElementB, L1_K, L1_N, pto::BLayout::ColMajor, pto::DYNAMIC, pto::DYNAMIC,
        pto::SLayout::RowMajor>;
    using LeftTile = pto::TileLeftCompact<ElementA, L0_M, L0_K, pto::DYNAMIC, pto::DYNAMIC>;
    using RightTile = pto::TileRightCompact<ElementB, L0_K, L0_N, pto::DYNAMIC, pto::DYNAMIC>;
    using AccPanelTile = pto::TileAccCompact<ElementAccumulator, L1_M, L1_N, pto::DYNAMIC, pto::DYNAMIC>;

    AICORE inline void AssignL1Panels(
        const L1TileGmmParams& params, APanelTile& aPanel, BPanelTile& bPanel, AccPanelTile& accPanel)
    {
        pto::TASSIGN(aPanel, l1AOffsetList[params.l1ListId]);
        pto::TASSIGN(bPanel, l1BOffsetList[params.l1ListId]);
        pto::TASSIGN(accPanel, l0COffsetList[l0CListId]);
        if (params.isKLoopFirst) {
            wait_flag(PIPE_FIX, PIPE_M, l0CEventList[l0CListId]);
        }
    }

    AICORE inline void ExtractL0A(LeftTile& aTile, const APanelTile& aPanel, uint32_t l1ListId, uint32_t kPartIdx)
    {
        pto::TASSIGN(aTile, l0AOffsetList[l0AListId]);
        wait_flag(PIPE_M, PIPE_MTE1, l0AEventList[l0AListId]);
        if (kPartIdx == 0) {
            wait_flag(PIPE_MTE2, PIPE_MTE1, l1AEventList[l1ListId]);
        }
        pto::TEXTRACT(aTile, aPanel, 0, kPartIdx * L0_K);
    }

    AICORE inline void ExtractL0B(RightTile& bTile, const BPanelTile& bPanel, uint32_t l1ListId, uint32_t kPartIdx)
    {
        pto::TASSIGN(bTile, l0BOffsetList[l0BListId]);
        wait_flag(PIPE_M, PIPE_MTE1, l0BEventList[l0BListId]);
        if (kPartIdx == 0) {
            wait_flag(PIPE_MTE2, PIPE_MTE1, l1BEventList[l1ListId]);
        }
        pto::TEXTRACT(bTile, bPanel, kPartIdx * L0_K, 0);
    }

    AICORE inline void ReleaseL1PanelsIfLastPart(uint32_t l1ListId, uint32_t kPartIdx, uint32_t kPartLoop)
    {
        if (kPartIdx == kPartLoop - 1) {
            set_flag(PIPE_MTE1, PIPE_MTE2, l1AEventList[l1ListId]);
            set_flag(PIPE_MTE1, PIPE_MTE2, l1BEventList[l1ListId]);
        }
    }

    AICORE inline void RunL0MatmulPart(AccPanelTile& accPanel, LeftTile& aTile, RightTile& bTile, bool firstAccumPart)
    {
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        if (firstAccumPart) {
            pto::TMATMUL(accPanel, aTile, bTile);
        } else {
            pto::TMATMUL_ACC(accPanel, aTile, bTile);
        }
        set_flag(PIPE_M, PIPE_MTE1, l0BEventList[l0BListId]);
        l0BListId = (l0BListId + 1 < L0B_STAGES) ? (l0BListId + 1) : 0;
        set_flag(PIPE_M, PIPE_MTE1, l0AEventList[l0AListId]);
        l0AListId = (l0AListId + 1 < L0A_STAGES) ? (l0AListId + 1) : 0;
    }

    AICORE inline void StoreL1TileResult(const L1TileGmmParams& params)
    {
        wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);
        StagePerChannelScale<L1_N>(l1SBaseOffset, fixpipeBaseOffset, params.gmBlockS, params.actualN);
        set_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID0);
        pipe_barrier(PIPE_FIX);
        set_flag(PIPE_M, PIPE_FIX, l0CEventList[l0CListId]);
        wait_flag(PIPE_M, PIPE_FIX, l0CEventList[l0CListId]);
        PtoStoreQuantAccToGm<ElementC, L1_M, L1_N>(
            params.gmBlockC, l0COffsetList[l0CListId], fixpipeBaseOffset, params.actualM, params.actualN,
            params.cLeadingDim);
        set_flag(PIPE_FIX, PIPE_M, l0CEventList[l0CListId]);
        l0CListId = (l0CListId + 1 < L0C_STAGES) ? (l0CListId + 1) : 0;
        set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);
    }

    AICORE inline void L1TileGmm(L1TileGmmParams const& params)
    {
        const uint32_t kPartLoop = ceilDiv<L0_K>(params.kActual);
        APanelTile aPanel(params.mRound, params.kActual);
        BPanelTile bPanel(params.kActual, params.nRound);
        AccPanelTile accPanel(params.mRound, params.nRound);
        AssignL1Panels(params, aPanel, bPanel, accPanel);

        for (uint32_t kPartIdx = 0; kPartIdx < kPartLoop; ++kPartIdx) {
            const uint32_t kPartActual = (kPartIdx < kPartLoop - 1) ? L0_K : (params.kActual - kPartIdx * L0_K);

            LeftTile aTile(params.mRound, kPartActual);
            ExtractL0A(aTile, aPanel, params.l1ListId, kPartIdx);
            RightTile bTile(kPartActual, params.nRound);
            ExtractL0B(bTile, bPanel, params.l1ListId, kPartIdx);
            ReleaseL1PanelsIfLastPart(params.l1ListId, kPartIdx, kPartLoop);
            RunL0MatmulPart(accPanel, aTile, bTile, params.isKLoopFirst && kPartIdx == 0);
        }

        if (!params.isKLoopLast) {
            return;
        }
        StoreL1TileResult(params);
    }

    uint64_t fixpipeBaseOffset{0};
    uint64_t l1AOffsetList[L1_STAGES];
    uint64_t l1BOffsetList[L1_STAGES];
    uint64_t l1SBaseOffset{0};
    int32_t syncGroupIdx{0};
    int32_t l1AEventList[L1_STAGES];
    int32_t l1BEventList[L1_STAGES];
    uint32_t l1ListId{0};

    uint64_t l0AOffsetList[L0A_STAGES];
    int32_t l0AEventList[L0A_STAGES];
    uint32_t l0AListId{0};

    uint64_t l0BOffsetList[L0B_STAGES];
    int32_t l0BEventList[L0B_STAGES];
    uint32_t l0BListId{0};

    uint64_t l0COffsetList[L0C_STAGES_];
    int32_t l0CEventList[L0C_STAGES_];
    uint32_t l0CListId{0};

    L1TileGmmParams pendingParams_{};
    bool hasPendingKTile_{false};
};

#endif // DISPATCH_MEGA_COMBINE_PTO_GMM_PRELOAD_ASYNC_FIXPIPE_QUANT_HPP
