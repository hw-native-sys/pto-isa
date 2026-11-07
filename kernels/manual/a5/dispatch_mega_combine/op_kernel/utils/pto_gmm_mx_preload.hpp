/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_PTO_GMM_MX_PRELOAD_HPP
#define DISPATCH_MEGA_COMBINE_PTO_GMM_MX_PRELOAD_HPP

#include "const_args.hpp"
#include "../gmm1_swiglu_cv_pipe.h"
#include "../gmm2_combine_cv_pipe.h"
#include "../gmm_task_queue_device.h"
#include "pto/common/pto_tile.hpp"
#include "pto/pto-inst.hpp"

template <
    uint32_t L1_M_, uint32_t L1_N_, uint32_t L1_K_, uint32_t L0_M_, uint32_t L0_N_, uint32_t L0_K_,
    uint32_t SCALE_PREFETCH_K_, uint32_t BASE_L1_K_ = L1_K_, uint32_t SMALL_M_MAX_ = 0U,
    typename ElementA_ = float8_e4m3_t, typename ElementB_ = float8_e4m3_t, typename ScaleElement_ = float8_e8m0_t,
    typename ElementC_ = bfloat16_t>
class PtoGmmMxPreloadPipeline {
public:
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ScaleElement = ScaleElement_;
    using ElementC = ElementC_;

    static constexpr uint32_t L1_M = L1_M_;
    static constexpr uint32_t L1_N = L1_N_;
    static constexpr uint32_t L1_K = L1_K_;
    static constexpr uint32_t BASE_L1_K = BASE_L1_K_;
    static constexpr uint32_t L0_M = L0_M_;
    static constexpr uint32_t L0_N = L0_N_;
    static constexpr uint32_t L0_K = L0_K_;
    static constexpr uint32_t SCALE_GROUP_SIZE = kMegaMoeMxGroupSize;
    static constexpr uint32_t L0_SCALE_K = L0_K / SCALE_GROUP_SIZE;
    static constexpr uint32_t BUFFER_NUM = 2U;
    static constexpr uint32_t BASE_SCALE_K = SCALE_PREFETCH_K_;
    static constexpr uint32_t L1_SLOT_BYTES = AtlasA5::L1_SIZE / BUFFER_NUM;
    static constexpr uint32_t SCALE_TRANSFER_BYTES = 64U * 1024U;
    static constexpr uint8_t L2_CACHE_NORMAL = 0U;
    static constexpr uint8_t L2_CACHE_DISABLE = 4U; // A5 CACHE_MODE_DISABLE encoding.
    static constexpr uint32_t L0_PINGPONG_BYTES = 32U * 1024U;
    static constexpr uint32_t L0_C_TILE_BYTES = L0_M * L0_N * sizeof(float);

    struct L1Params {
        uint32_t blockM = L1_M;
        uint32_t dataK = BASE_L1_K;
        uint32_t scaleK = BASE_SCALE_K;

        AICORE inline bool operator!=(const L1Params& other) const
        {
            return blockM != other.blockM || dataK != other.dataK || scaleK != other.scaleK;
        }
    };

    static_assert(L1_M == L0_M && L1_N == L0_N, "MX GMM requires identical scheduler/L0 M/N tiles");
    static_assert(BASE_L1_K == 256U && L1_K == 512U, "L1 data panel K must be 256/512");
    static_assert(L1_K % L0_K == 0U && BASE_L1_K % L0_K == 0U, "L1 K panels must be divisible by L0 K");
    static_assert(BASE_SCALE_K % SCALE_GROUP_SIZE == 0U, "invalid baseline MX scale K");
    static_assert(
        static_cast<uint64_t>(L1_M) * BASE_L1_K * sizeof(ElementA) +
                static_cast<uint64_t>(BASE_L1_K) * L1_N * sizeof(ElementB) +
                static_cast<uint64_t>(L1_M) * (BASE_SCALE_K / SCALE_GROUP_SIZE) * sizeof(ScaleElement) +
                static_cast<uint64_t>(BASE_SCALE_K / SCALE_GROUP_SIZE) * L1_N * sizeof(ScaleElement) <=
            L1_SLOT_BYTES,
        "baseline MX data/scale panel exceeds one half-L1 slot");
    static_assert(L0_K % 64U == 0U, "TMATMUL_MX reduction K must be a multiple of 64");
    static_assert(L0_M * L0_K * sizeof(ElementA) <= L0_PINGPONG_BYTES, "MX A tile exceeds one L0A slot");
    static_assert(L0_K * L0_N * sizeof(ElementB) <= L0_PINGPONG_BYTES, "MX B tile exceeds one L0B slot");
    static_assert(BUFFER_NUM * L0_PINGPONG_BYTES <= AtlasA5::L0A_SIZE, "MX A ping-pong exceeds L0A");
    static_assert(BUFFER_NUM * L0_PINGPONG_BYTES <= AtlasA5::L0B_SIZE, "MX B ping-pong exceeds L0B");
    static_assert(L0_C_TILE_BYTES == AtlasA5::L0C_SIZE, "MX pipeline expects one full-size L0C accumulator");

    struct TileRun {
        __gm__ ElementA* gmA = nullptr;
        __gm__ ScaleElement* gmAScale = nullptr;
        __gm__ ElementB* gmB = nullptr;
        __gm__ ScaleElement* gmBScale = nullptr;
        uint32_t actualM = 0U;
        uint32_t actualN = 0U;
        uint32_t actualK = 0U;
        uint32_t aLeadingDim = 0U;
        uint32_t aScaleLeadingDim = 0U;
        uint32_t bLeadingDim = 0U;
        uint32_t bScaleLeadingDim = 0U;
        uint32_t dataSlotBase = 0U;
        uint32_t scaleSlotBase = 0U;
        bool firstPanelPreloaded = false;
        // Select matrix tiling using the full expert M, not the tail tile M.
        uint32_t problemM = 0U;
        // True only when this run covers the whole expert; partial-wave slices must keep normal L2 reuse.
        bool coversWholeExpert = false;
        // Production Gate/Up pairing: two N/2 panels from separate branches
        // form one 256-column accumulator/CV tile.
        uint64_t bConcatStrideBytes = 0U;
        uint64_t bScaleConcatStrideBytes = 0U;
        uint32_t concatHalfN = 0U;
    };

    AICORE inline explicit PtoGmmMxPreloadPipeline(uint32_t l1BufAddrStart = 0U)
    {
        InitL1Offsets(l1BufAddrStart);
        InitEvents();
    }

    AICORE inline ~PtoGmmMxPreloadPipeline()
    {
        SynchronizeBlock();
        WaitFlag<PIPE_FIX, PIPE_M>(kCReuseEvent);
        for (uint32_t slot = 0U; slot < BUFFER_NUM; ++slot) {
            WaitFlag<PIPE_MTE1, PIPE_MTE2>(ADataEvent(slot));
            WaitFlag<PIPE_MTE1, PIPE_MTE2>(BDataEvent(slot));
            WaitFlag<PIPE_MTE1, PIPE_MTE2>(AScaleEvent(slot));
            WaitFlag<PIPE_MTE1, PIPE_MTE2>(BScaleEvent(slot));
            WaitFlag<PIPE_M, PIPE_MTE1>(L0AEvent(slot));
            WaitFlag<PIPE_M, PIPE_MTE1>(L0BEvent(slot));
        }
    }

    AICORE static inline uint32_t AlignBlockM(uint32_t m) { return (m + 15U) & ~15U; }

    // Select adaptive L1 data and scale windows for small expert M.
    AICORE static inline L1Params SelectL1Params(uint32_t problemM, uint32_t actualK)
    {
        const L1Params baseline{L1_M, BASE_L1_K, BASE_SCALE_K};
        if (problemM == 0U || problemM >= L1_M || actualK <= BASE_L1_K) {
            return baseline;
        }

        const uint32_t blockM = AlignBlockM(problemM);
        constexpr uint64_t scaleBytesPerBaseK =
            static_cast<uint64_t>(BASE_L1_K / SCALE_GROUP_SIZE) * sizeof(ScaleElement);
        const uint64_t dataBytesPerUnit = static_cast<uint64_t>(blockM) * BASE_L1_K * sizeof(ElementA) +
                                          static_cast<uint64_t>(L1_N) * BASE_L1_K * sizeof(ElementB);
        const uint64_t scaleABytesPerUnit = static_cast<uint64_t>(blockM) * scaleBytesPerBaseK;
        const uint64_t scaleBBytesPerUnit = static_cast<uint64_t>(L1_N) * scaleBytesPerBaseK;
        const uint64_t scaleBytesPerUnit = scaleABytesPerUnit + scaleBBytesPerUnit;

        const uint64_t doubleDataBytes = dataBytesPerUnit * 2U;
        const uint64_t doubleScaleABytes = scaleABytesPerUnit * 2U;
        const uint64_t doubleScaleBBytes = scaleBBytesPerUnit * 2U;
        const uint64_t doubleScaleBytes = scaleBytesPerUnit * 2U;
        const bool canUseDoubleK = doubleDataBytes + doubleScaleBytes <= L1_SLOT_BYTES &&
                                   doubleScaleABytes <= SCALE_TRANSFER_BYTES &&
                                   doubleScaleBBytes <= SCALE_TRANSFER_BYTES;
        const uint32_t dataK = canUseDoubleK ? L1_K : BASE_L1_K;
        const uint32_t kUnits = dataK / BASE_L1_K;
        const uint64_t dataBytes = dataBytesPerUnit * kUnits;
        const uint64_t scaleABytes = scaleABytesPerUnit * kUnits;
        const uint64_t scaleBBytes = scaleBBytesPerUnit * kUnits;
        const uint64_t scaleBytes = scaleBytesPerUnit * kUnits;

        uint64_t scaleFactor = (L1_SLOT_BYTES - dataBytes) / scaleBytes;
        scaleFactor = MinU64(scaleFactor, SCALE_TRANSFER_BYTES / scaleABytes);
        scaleFactor = MinU64(scaleFactor, SCALE_TRANSFER_BYTES / scaleBBytes);
        const uint64_t fullKWindowCount = static_cast<uint64_t>(actualK) / dataK;
        scaleFactor = MinU64(scaleFactor, fullKWindowCount == 0U ? 1U : fullKWindowCount);
        scaleFactor = MinU64(scaleFactor, 127U);
        const uint32_t scaleK = static_cast<uint32_t>(scaleFactor) * dataK;
        const L1Params adaptive{blockM, dataK, scaleK};

        const uint32_t adaptiveScaleLoops = static_cast<uint32_t>(ceilDiv(actualK, adaptive.scaleK));
        const uint32_t baselineScaleLoops = static_cast<uint32_t>(ceilDiv(actualK, baseline.scaleK));
        if (adaptive.dataK > baseline.dataK) {
            // Reject wider data panels if they eliminate scale reuse and increase reloads.
            const bool losesAllScaleReuse =
                adaptive.scaleK == adaptive.dataK && adaptiveScaleLoops > baselineScaleLoops;
            if (!losesAllScaleReuse) {
                return adaptive;
            }
        }
        return adaptiveScaleLoops < baselineScaleLoops ? adaptive : baseline;
    }

    AICORE static inline uint32_t DataPanelK(uint32_t problemM, uint32_t actualK)
    {
        return SelectL1Params(problemM, actualK).dataK;
    }

    AICORE static inline uint32_t ScalePanelK(uint32_t problemM, uint32_t actualK)
    {
        return SelectL1Params(problemM, actualK).scaleK;
    }

    AICORE static inline uint32_t AdvanceDataSlotBase(uint32_t slotBase, uint32_t actualK, uint32_t problemM)
    {
        return slotBase ^ (static_cast<uint32_t>(ceilDiv(actualK, DataPanelK(problemM, actualK))) & 1U);
    }

    AICORE static inline uint32_t AdvanceScaleSlotBase(uint32_t slotBase, uint32_t actualK, uint32_t problemM)
    {
        return slotBase ^ (static_cast<uint32_t>(ceilDiv(actualK, ScalePanelK(problemM, actualK))) & 1U);
    }

    template <typename PanelProbe>
    AICORE inline void ComputePairDirect(
        Gmm1SwigluCvPipe& cvPipe, const TileRun& x, const TileRun& gate, PanelProbe& panelProbe)
    {
        WaitFlag<PIPE_FIX, PIPE_M>(kCReuseEvent);
        using AccTile = pto::TileAccCompact<float, L0_M, L0_N, pto::DYNAMIC, pto::DYNAMIC>;
        AccTile cTile(x.actualM, x.actualN);
        pto::TASSIGN(cTile, 0U);

        ComputeResultRange<false>(cTile, x, 0U, x.actualK / L0_K);
        Gmm1SwigluProducerAllocate(cvPipe);
        EnqueuePairHalfToAiv1(cTile, x.actualM, x.actualN, kGmm1SwigluCvXOffset);

        // The 256x256 accumulator fills L0C. Order the x FIX transfer before
        // reusing the same L0C address for gate.
        SetFlag<PIPE_FIX, PIPE_M>(kCReuseEvent);
        WaitFlag<PIPE_FIX, PIPE_M>(kCReuseEvent);

        // Keep the successor-ticket probe on the final gate K tile.
        ComputeResultRangeWithProbe<false>(cTile, gate, 0U, gate.actualK / L0_K, panelProbe);
    }

    AICORE inline void EnqueuePairDirect(Gmm1SwigluCvPipe& cvPipe, uint32_t actualM, uint32_t actualN)
    {
        using AccTile = pto::TileAccCompact<float, L0_M, L0_N, pto::DYNAMIC, pto::DYNAMIC>;
        AccTile gateTile(actualM, actualN);
        // Rebind a handle to the gate result left at L0C offset zero. TASSIGN
        // does not allocate or initialize a second accumulator.
        pto::TASSIGN(gateTile, 0U);
        EnqueuePairHalfToAiv1(gateTile, actualM, actualN, kGmm1SwigluCvGateOffset);
    }

    AICORE inline void RecordPairDirect(Gmm1SwigluCvPipe& cvPipe)
    {
        // TMOV is asynchronous on FIX; signal AIV only after both CV halves are stored.
        pipe_barrier(PIPE_FIX);
        Gmm1SwigluProducerRecord(cvPipe);
        SetFlag<PIPE_FIX, PIPE_M>(kCReuseEvent);
    }

    template <bool EnableUnitFlag = false, typename PanelProbe>
    AICORE inline void ComputeDirect(const TileRun& run, PanelProbe& panelProbe)
    {
        WaitFlag<PIPE_FIX, PIPE_M>(kCReuseEvent);

        using AccTile = pto::TileAccCompact<float, L0_M, L0_N, pto::DYNAMIC, pto::DYNAMIC>;
        AccTile cTile(run.actualM, run.actualN);
        pto::TASSIGN(cTile, 0U);
        ComputeResultRangeWithProbe<EnableUnitFlag>(cTile, run, 0U, run.actualK / L0_K, panelProbe);
    }

    AICORE inline void EnqueueDirectReserved(Gmm2CombineCvPipe& cvPipe, uint32_t actualM, uint32_t actualN)
    {
        using AccTile = pto::TileAccCompact<float, L0_M, L0_N, pto::DYNAMIC, pto::DYNAMIC>;
        AccTile cTile(actualM, actualN);
        pto::TASSIGN(cTile, 0U);
        SetFlag<PIPE_M, PIPE_FIX>(kStoreReadyEvent);
        WaitFlag<PIPE_M, PIPE_FIX>(kStoreReadyEvent);
        // Wait for the Combine slot only after issuing the CUBE work.
        Gmm2CombineProducerAllocate(cvPipe);
        Gmm2CombineCvTile cvTile(actualM, actualN);
        pto::TASSIGN(cvTile, kGmm2CombineCvBufferOffset);
        pto::TMOV<pto::STPhase::Final, Gmm2CombineCvTile, AccTile, pto::AccToVecMode::SingleModeVec1>(cvTile, cTile);
    }

    AICORE inline void DrainDirectStore() { pipe_barrier(PIPE_FIX); }

    AICORE inline void RecordDirect(Gmm2CombineCvPipe& cvPipe)
    {
        Gmm2CombineProducerRecord(cvPipe);
        SetFlag<PIPE_FIX, PIPE_M>(kCReuseEvent);
    }

    AICORE inline void SynchronizeBlock() { pipe_barrier(PIPE_ALL); }

private:
    static constexpr uint32_t kCReuseEvent = 0U;
    static constexpr uint32_t kStoreReadyEvent = 0U;
    static constexpr uint32_t kL0ReadyEvent = 0U;

    AICORE static inline uint32_t MinU32(uint32_t lhs, uint32_t rhs) { return lhs < rhs ? lhs : rhs; }

    AICORE static inline uint64_t MinU64(uint64_t lhs, uint64_t rhs) { return lhs < rhs ? lhs : rhs; }

    AICORE static inline uint32_t ADataEvent(uint32_t slot) { return slot; }

    AICORE static inline uint32_t BDataEvent(uint32_t slot) { return BUFFER_NUM + slot; }

    AICORE static inline uint32_t AScaleEvent(uint32_t slot) { return 2U * BUFFER_NUM + slot; }

    AICORE static inline uint32_t BScaleEvent(uint32_t slot) { return 3U * BUFFER_NUM + slot; }

    AICORE static inline uint32_t L0AEvent(uint32_t slot) { return slot; }

    AICORE static inline uint32_t L0BEvent(uint32_t slot) { return BUFFER_NUM + slot; }

    template <pipe_t SrcPipe, pipe_t DstPipe>
    AICORE static inline void SetFlag(uint32_t eventId)
    {
        set_flag(SrcPipe, DstPipe, static_cast<event_t>(eventId));
    }

    template <pipe_t SrcPipe, pipe_t DstPipe>
    AICORE static inline void WaitFlag(uint32_t eventId)
    {
        wait_flag(SrcPipe, DstPipe, static_cast<event_t>(eventId));
    }

    AICORE inline void InitL1Offsets(uint32_t start) { l1Start_ = start; }

    AICORE inline void InitEvents()
    {
        SetFlag<PIPE_FIX, PIPE_M>(kCReuseEvent);
        for (uint32_t slot = 0U; slot < BUFFER_NUM; ++slot) {
            SetFlag<PIPE_MTE1, PIPE_MTE2>(ADataEvent(slot));
            SetFlag<PIPE_MTE1, PIPE_MTE2>(BDataEvent(slot));
            SetFlag<PIPE_MTE1, PIPE_MTE2>(AScaleEvent(slot));
            SetFlag<PIPE_MTE1, PIPE_MTE2>(BScaleEvent(slot));
            SetFlag<PIPE_M, PIPE_MTE1>(L0AEvent(slot));
            SetFlag<PIPE_M, PIPE_MTE1>(L0BEvent(slot));
        }
    }

    template <bool EnableUnitFlag, bool EnablePanelProbe, typename AccTile, typename PanelProbe>
    AICORE inline void ComputeResultRangeImpl(
        AccTile& cTile, const TileRun& run, uint32_t beginKTile, uint32_t endKTile, PanelProbe& panelProbe)
    {
        const uint32_t problemM = run.problemM == 0U ? run.actualM : run.problemM;
        const L1Params params = SelectL1Params(problemM, run.actualK);
        EnsureL1Layout(params);
        const uint32_t kTileCount = run.actualK / L0_K;
        const uint32_t dataPanelK = params.dataK;
        const uint32_t scalePanelK = params.scaleK;
        const bool bypassWeightL2 = CanBypassWeightL2(run);
        const bool bypassScaleL2 = CanBypassScaleL2(run);
        for (uint32_t kTileIdx = beginKTile; kTileIdx < endKTile; ++kTileIdx) {
            const uint32_t kStart = kTileIdx * L0_K;
            const uint32_t dataPanelStart = kStart / dataPanelK * dataPanelK;
            const uint32_t dataPanelIdx = dataPanelStart / dataPanelK;
            const uint32_t dataSlot = run.dataSlotBase ^ (dataPanelIdx & 1U);
            const uint32_t scalePanelStart = kStart / scalePanelK * scalePanelK;
            const uint32_t scalePanelIdx = scalePanelStart / scalePanelK;
            const uint32_t scaleSlot = run.scaleSlotBase ^ (scalePanelIdx & 1U);

            if (kStart == dataPanelStart && !(run.firstPanelPreloaded && dataPanelIdx == 0U)) {
                const uint32_t panelK = MinU32(dataPanelK, run.actualK - dataPanelStart);
                LoadDataPanel(
                    dataSlot, run.gmA + dataPanelStart, run.gmB + dataPanelStart, run.actualM, run.actualN, panelK,
                    run.aLeadingDim, run.bLeadingDim, params, bypassWeightL2, run.bConcatStrideBytes, run.concatHalfN);
            }
            if (kStart == scalePanelStart && !(run.firstPanelPreloaded && scalePanelIdx == 0U)) {
                const uint32_t panelK = MinU32(scalePanelK, run.actualK - scalePanelStart);
                LoadScalePanel(
                    scaleSlot, run.gmAScale + scalePanelStart / SCALE_GROUP_SIZE,
                    run.gmBScale + scalePanelStart / SCALE_GROUP_SIZE, run.actualM, run.actualN,
                    panelK / SCALE_GROUP_SIZE, run.aScaleLeadingDim, run.bScaleLeadingDim, params, bypassScaleL2,
                    run.bScaleConcatStrideBytes, run.concatHalfN);
            }

            const uint32_t l0Slot = kTileIdx & 1U;
            const uint32_t dataKOffset = kStart - dataPanelStart;
            const uint32_t scaleKOffset = (kStart - scalePanelStart) / SCALE_GROUP_SIZE;
            ExtractL0Tiles(
                dataSlot, scaleSlot, l0Slot, run.actualM, run.actualN, dataKOffset, scaleKOffset,
                kStart + L0_K == MinU32(dataPanelStart + dataPanelK, run.actualK),
                kStart + L0_K == MinU32(scalePanelStart + scalePanelK, run.actualK), params);
            LaunchMatmul<EnableUnitFlag>(
                cTile, l0Slot, run.actualM, run.actualN, kTileIdx == 0U, kTileIdx + 1U == kTileCount);
            panelProbe.ProbeProgress(kTileIdx, kTileCount);
            if constexpr (EnablePanelProbe) {
                if (kTileIdx + 1U == kTileCount) {
                    panelProbe.Probe();
                }
            }
        }
    }

    struct NoopPanelProbe {
        AICORE inline void Probe() {}
        AICORE inline void ProbeProgress(uint32_t, uint32_t) {}
    };

    template <bool EnableUnitFlag, typename AccTile>
    AICORE inline void ComputeResultRange(AccTile& cTile, const TileRun& run, uint32_t beginKTile, uint32_t endKTile)
    {
        NoopPanelProbe panelProbe;
        ComputeResultRangeImpl<EnableUnitFlag, false>(cTile, run, beginKTile, endKTile, panelProbe);
    }

    template <bool EnableUnitFlag, typename AccTile, typename PanelProbe>
    AICORE inline void ComputeResultRangeWithProbe(
        AccTile& cTile, const TileRun& run, uint32_t beginKTile, uint32_t endKTile, PanelProbe& panelProbe)
    {
        ComputeResultRangeImpl<EnableUnitFlag, true>(cTile, run, beginKTile, endKTile, panelProbe);
    }

    template <typename AccTile>
    AICORE inline void EnqueuePairHalfToAiv1(AccTile& cTile, uint32_t actualM, uint32_t actualPairN, uint64_t dstOffset)
    {
        Gmm1SwigluCvHalfTile cvTile(actualM, actualPairN);
        pto::TASSIGN(cvTile, kGmm1SwigluCvBufferOffset + dstOffset);
        SetFlag<PIPE_M, PIPE_FIX>(kStoreReadyEvent);
        WaitFlag<PIPE_M, PIPE_FIX>(kStoreReadyEvent);
        pto::TMOV<Gmm1SwigluCvHalfTile, AccTile, pto::AccToVecMode::SingleModeVec1>(cvTile, cTile);
    }

    AICORE static inline bool IsBaselineL1Params(const L1Params& params)
    {
        return params.blockM == L1_M && params.dataK == BASE_L1_K && params.scaleK == BASE_SCALE_K;
    }

    AICORE static inline uint32_t ScaleCols(const L1Params& params) { return params.scaleK / SCALE_GROUP_SIZE; }

    AICORE static inline bool HasNoLaterWeightReuse(const TileRun& run)
    {
        const uint32_t problemM = run.problemM == 0U ? run.actualM : run.problemM;
        return run.coversWholeExpert && problemM != 0U && problemM <= L1_M;
    }

    AICORE static inline bool CanBypassWeightL2(const TileRun& run)
    {
        // Bypass weight L2 only for full-expert tiles with aligned K strides.
        return HasNoLaterWeightReuse(run) && run.bLeadingDim % 256U == 0U;
    }

    AICORE static inline bool CanBypassScaleL2(const TileRun& run)
    {
        // MX weight-scale loads require 128-byte aligned source strides.
        constexpr uint64_t cacheLineBytes = 128U;
        const uint64_t sourceStrideBytes = static_cast<uint64_t>(run.bScaleLeadingDim) * sizeof(ScaleElement);
        return HasNoLaterWeightReuse(run) && sourceStrideBytes % cacheLineBytes == 0U;
    }

    AICORE inline uint64_t L1SlotBase(uint32_t slot) const
    {
        return l1Start_ + static_cast<uint64_t>(slot) * L1_SLOT_BYTES;
    }

    AICORE inline uint64_t ADataAddress(uint32_t slot, const L1Params&) const { return L1SlotBase(slot); }

    AICORE inline uint64_t BDataAddress(uint32_t slot, const L1Params& params) const
    {
        return ADataAddress(slot, params) + static_cast<uint64_t>(params.blockM) * params.dataK * sizeof(ElementA);
    }

    AICORE inline uint64_t AScaleAddress(uint32_t slot, const L1Params& params) const
    {
        return BDataAddress(slot, params) + static_cast<uint64_t>(params.dataK) * L1_N * sizeof(ElementB);
    }

    AICORE inline uint64_t BScaleAddress(uint32_t slot, const L1Params& params) const
    {
        return AScaleAddress(slot, params) +
               static_cast<uint64_t>(params.blockM) * ScaleCols(params) * sizeof(ScaleElement);
    }

    // Drain previous panel transfers only when the L1 layout changes.
    AICORE inline void EnsureL1Layout(const L1Params& params)
    {
        if (l1ParamsValid_ && !(params != activeL1Params_)) {
            return;
        }
        if (l1ParamsValid_) {
            for (uint32_t slot = 0U; slot < BUFFER_NUM; ++slot) {
                WaitFlag<PIPE_MTE1, PIPE_MTE2>(ADataEvent(slot));
                WaitFlag<PIPE_MTE1, PIPE_MTE2>(BDataEvent(slot));
                WaitFlag<PIPE_MTE1, PIPE_MTE2>(AScaleEvent(slot));
                WaitFlag<PIPE_MTE1, PIPE_MTE2>(BScaleEvent(slot));
                SetFlag<PIPE_MTE1, PIPE_MTE2>(ADataEvent(slot));
                SetFlag<PIPE_MTE1, PIPE_MTE2>(BDataEvent(slot));
                SetFlag<PIPE_MTE1, PIPE_MTE2>(AScaleEvent(slot));
                SetFlag<PIPE_MTE1, PIPE_MTE2>(BScaleEvent(slot));
            }
        }
        activeL1Params_ = params;
        l1ParamsValid_ = true;
    }

    AICORE static inline void SetNzDstStride(uint32_t boxedStride)
    {
        uint64_t mte2NzPara = static_cast<uint64_t>(boxedStride) << 32;
        mte2NzPara |= static_cast<uint64_t>(1U) << 16;
        mte2NzPara |= 1U;
        set_mte2_nz_para(mte2NzPara);
    }

    AICORE inline void LoadADataRaw(
        uint64_t dstAddr, __gm__ ElementA* gmA, uint32_t actualM, uint32_t panelK, uint32_t aLeadingDim,
        const L1Params& params)
    {
        SetNzDstStride(params.blockM);
        copy_gm_to_cbuf_multi_nd2nz(
            reinterpret_cast<__cbuf__ uint8_t*>(dstAddr), reinterpret_cast<__gm__ uint8_t*>(gmA), 0,
            aLeadingDim * sizeof(ElementA), 0, actualM, panelK, 0, false);
    }

    AICORE static inline void SetNzConcatDstStride(uint32_t rowStride, uint32_t matrixStride)
    {
        uint64_t para = static_cast<uint64_t>(matrixStride) << 48;
        para |= static_cast<uint64_t>(rowStride) << 32;
        para |= static_cast<uint64_t>(1U) << 16;
        para |= 2U;
        set_mte2_nz_para(para);
    }

    AICORE inline void LoadBDataConcatRaw(
        uint64_t dstAddr, __gm__ ElementB* gmB, uint32_t halfN, uint32_t panelK, uint32_t bLeadingDim,
        uint64_t branchStrideBytes, uint8_t l2CacheCtl)
    {
        static_assert(sizeof(ElementB) == 1U, "Gate/Up concat requires 1-byte weights");
        SetNzConcatDstStride(L1_N, halfN);
        pto_copy_gm_to_cbuf_multi_nd2nz(
            reinterpret_cast<__cbuf__ int8_t*>(dstAddr), reinterpret_cast<__gm__ int8_t*>(gmB), 0,
            bLeadingDim * sizeof(ElementB), l2CacheCtl, static_cast<uint16_t>(halfN), panelK, branchStrideBytes, false);
    }

    AICORE inline void LoadBScaleConcatRaw(
        uint64_t dstAddr, __gm__ ScaleElement* gmBScale, uint32_t halfN, uint32_t panelScaleK,
        uint32_t bScaleLeadingDim, uint64_t branchStrideBytes, const L1Params& params, uint8_t l2CacheCtl)
    {
        static_assert(sizeof(ScaleElement) == 1U, "Gate/Up concat requires 1-byte MX scales");
        const uint32_t fullScalePairs = ScaleCols(params) >> 1U;
        SetNzConcatDstStride(fullScalePairs, halfN * fullScalePairs / 16U);
        copy_gm_to_cbuf_multi_dn2nz(
            reinterpret_cast<__cbuf__ uint16_t*>(dstAddr), reinterpret_cast<__gm__ uint16_t*>(gmBScale), 0,
            bScaleLeadingDim * sizeof(ScaleElement), l2CacheCtl, static_cast<uint16_t>(panelScaleK >> 1U), halfN,
            branchStrideBytes, false);
    }

    AICORE inline void LoadBDataRaw(
        uint64_t dstAddr, __gm__ ElementB* gmB, uint32_t actualN, uint32_t panelK, uint32_t bLeadingDim,
        uint8_t l2CacheCtl)
    {
        // Same instruction geometry as PTO TLoadCubeDN2ZN: DN source, ZN destination,
        // so the hardware copy opcode is nd2nz with N/K arguments exchanged by the view.
        SetNzDstStride(L1_N);
        copy_gm_to_cbuf_multi_nd2nz(
            reinterpret_cast<__cbuf__ uint8_t*>(dstAddr), reinterpret_cast<__gm__ uint8_t*>(gmB), 0,
            bLeadingDim * sizeof(ElementB), l2CacheCtl, actualN, panelK, 0, false);
    }

    template <typename LeftTile, typename RightTile, typename LeftScaleTile, typename RightScaleTile>
    __tf__ PTO_INTERNAL static void ExtractL0Raw(
        typename LeftTile::TileDType __out__ aDst, typename RightTile::TileDType __out__ bDst,
        typename LeftScaleTile::TileDType __out__ aScaleDst, typename RightScaleTile::TileDType __out__ bScaleDst,
        uint64_t aL1Addr, uint64_t bL1Addr, uint64_t aScaleL1Addr, uint64_t bScaleL1Addr, uint16_t mAlign,
        uint16_t nAlign, uint16_t dataKOffset, uint16_t scaleKOffset, uint16_t blockM, uint16_t scaleCols)
    {
        auto aL0 = reinterpret_cast<__ca__ ElementA*>(__cce_get_tile_ptr(aDst));
        auto bL0 = reinterpret_cast<__cb__ ElementB*>(__cce_get_tile_ptr(bDst));
        auto aL1 = reinterpret_cast<__cbuf__ ElementA*>(aL1Addr);
        auto bL1 = reinterpret_cast<__cbuf__ ElementB*>(bL1Addr);
        const uint16_t kStart = dataKOffset >> 5U;
        const uint8_t kStep = static_cast<uint8_t>(L0_K >> 5U);
        const uint8_t mStep = static_cast<uint8_t>(mAlign >> 4U);
        const uint8_t nStep = static_cast<uint8_t>(nAlign >> 4U);
        load_cbuf_to_ca(aL0, aL1, 0, kStart, mStep, kStep, blockM >> 4U, mStep, 0);
        pto_load_cbuf_to_cb<false>(bL0, bL1, 0, kStart, nStep, kStep, L1_N >> 4U, nStep);

        const uint16_t scaleColStart = scaleKOffset >> 1U;
        const uint8_t scaleColStep = static_cast<uint8_t>(L0_SCALE_K >> 1U);
        const uint16_t scaleSrcStride = scaleCols >> 1U;
        const uint16_t scaleDstStride = L0_SCALE_K >> 1U;
        const uint64_t aScaleL0 = reinterpret_cast<uint64_t>(__cce_get_tile_ptr(aScaleDst));
        const uint64_t bScaleL0 = reinterpret_cast<uint64_t>(__cce_get_tile_ptr(bScaleDst));
        load_cbuf_to_ca_mx(
            aScaleL0, reinterpret_cast<__cbuf__ void*>(aScaleL1Addr), 0, scaleColStart, mStep, scaleColStep,
            scaleSrcStride, scaleDstStride);
        load_cbuf_to_cb_mx(
            bScaleL0, reinterpret_cast<__cbuf__ void*>(bScaleL1Addr), 0, scaleColStart, nStep, scaleColStep,
            scaleSrcStride, scaleDstStride);
    }

    template <typename APanel, typename BPanel>
    AICORE inline void LoadDataPanelTyped(
        uint32_t slot, __gm__ ElementA* gmA, __gm__ ElementB* gmB, uint32_t actualM, uint32_t actualN, uint32_t panelK,
        uint32_t aLeadingDim, uint32_t bLeadingDim, const L1Params& params, bool bypassWeightL2,
        uint64_t bConcatStrideBytes, uint32_t concatHalfN)
    {
        using AShape = pto::TileShape2D<ElementA, pto::DYNAMIC, pto::DYNAMIC, pto::Layout::ND>;
        using AStride = pto::BaseShape2D<ElementA, pto::DYNAMIC, pto::DYNAMIC, pto::Layout::ND>;
        using AGlobal = pto::GlobalTensor<ElementA, AShape, AStride, pto::Layout::ND>;
        using BShape = pto::TileShape2D<ElementB, pto::DYNAMIC, pto::DYNAMIC, pto::Layout::DN>;
        using BStride = pto::BaseShape2D<ElementB, pto::DYNAMIC, pto::DYNAMIC, pto::Layout::DN>;
        using BGlobal = pto::GlobalTensor<ElementB, BShape, BStride, pto::Layout::DN>;

        APanel aPanel(actualM, panelK);
        BPanel bPanel(panelK, actualN);
        pto::TASSIGN(aPanel, ADataAddress(slot, params));
        pto::TASSIGN(bPanel, BDataAddress(slot, params));

        AShape aShape(actualM, panelK);
        AStride aStride(actualM, aLeadingDim);
        BShape bShape(panelK, actualN);
        BStride bStride(bLeadingDim, actualN);
        AGlobal aGlobal(gmA, aShape, aStride);
        BGlobal bGlobal(gmB, bShape, bStride);

        WaitFlag<PIPE_MTE1, PIPE_MTE2>(ADataEvent(slot));

        pto::TLOAD(aPanel, aGlobal);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(ADataEvent(slot));
        WaitFlag<PIPE_MTE1, PIPE_MTE2>(BDataEvent(slot));
        if (concatHalfN != 0U) {
            LoadBDataConcatRaw(
                BDataAddress(slot, params), gmB, concatHalfN, panelK, bLeadingDim, bConcatStrideBytes,
                bypassWeightL2 ? L2_CACHE_DISABLE : L2_CACHE_NORMAL);
        } else if (bypassWeightL2) {
            LoadBDataRaw(BDataAddress(slot, params), gmB, actualN, panelK, bLeadingDim, L2_CACHE_DISABLE);
        } else {
            pto::TLOAD(bPanel, bGlobal);
        }
        SetFlag<PIPE_MTE2, PIPE_MTE1>(BDataEvent(slot));
    }

    AICORE inline void LoadDataPanel(
        uint32_t slot, __gm__ ElementA* gmA, __gm__ ElementB* gmB, uint32_t actualM, uint32_t actualN, uint32_t panelK,
        uint32_t aLeadingDim, uint32_t bLeadingDim, const L1Params& params, bool bypassWeightL2,
        uint64_t bConcatStrideBytes, uint32_t concatHalfN)
    {
        if (IsBaselineL1Params(params)) {
            using APanel = pto::Tile<
                pto::TileType::Mat, ElementA, L1_M, BASE_L1_K, pto::BLayout::ColMajor, pto::DYNAMIC, pto::DYNAMIC,
                pto::SLayout::RowMajor>;
            using BPanel = pto::Tile<
                pto::TileType::Mat, ElementB, BASE_L1_K, L1_N, pto::BLayout::RowMajor, pto::DYNAMIC, pto::DYNAMIC,
                pto::SLayout::ColMajor>;
            LoadDataPanelTyped<APanel, BPanel>(
                slot, gmA, gmB, actualM, actualN, panelK, aLeadingDim, bLeadingDim, params, bypassWeightL2,
                bConcatStrideBytes, concatHalfN);
            return;
        }

        WaitFlag<PIPE_MTE1, PIPE_MTE2>(ADataEvent(slot));

        LoadADataRaw(ADataAddress(slot, params), gmA, actualM, panelK, aLeadingDim, params);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(ADataEvent(slot));
        WaitFlag<PIPE_MTE1, PIPE_MTE2>(BDataEvent(slot));
        if (concatHalfN != 0U) {
            LoadBDataConcatRaw(
                BDataAddress(slot, params), gmB, concatHalfN, panelK, bLeadingDim, bConcatStrideBytes,
                bypassWeightL2 ? L2_CACHE_DISABLE : L2_CACHE_NORMAL);
        } else {
            LoadBDataRaw(
                BDataAddress(slot, params), gmB, actualN, panelK, bLeadingDim,
                bypassWeightL2 ? L2_CACHE_DISABLE : L2_CACHE_NORMAL);
        }
        SetFlag<PIPE_MTE2, PIPE_MTE1>(BDataEvent(slot));
    }

    AICORE inline void LoadScalePanel(
        uint32_t slot, __gm__ ScaleElement* gmAScale, __gm__ ScaleElement* gmBScale, uint32_t actualM, uint32_t actualN,
        uint32_t panelScaleK, uint32_t aScaleLeadingDim, uint32_t bScaleLeadingDim, const L1Params& params,
        bool bypassScaleL2, uint64_t bScaleConcatStrideBytes, uint32_t concatHalfN)
    {
        if (IsBaselineL1Params(params)) {
            constexpr uint32_t BASE_SCALE_COLS = BASE_SCALE_K / SCALE_GROUP_SIZE;
            using AScalePanel = pto::Tile<
                pto::TileType::Mat, ScaleElement, L1_M, BASE_SCALE_COLS, pto::BLayout::RowMajor, pto::DYNAMIC,
                pto::DYNAMIC, pto::SLayout::RowMajor, 32>;
            using BScalePanel = pto::Tile<
                pto::TileType::Mat, ScaleElement, BASE_SCALE_COLS, L1_N, pto::BLayout::ColMajor, pto::DYNAMIC,
                pto::DYNAMIC, pto::SLayout::ColMajor, 32>;
            using AShape = pto::TileShape2D<ScaleElement, pto::DYNAMIC, pto::DYNAMIC, pto::Layout::MX_A_ND>;
            using AStride = pto::BaseShape2D<ScaleElement, pto::DYNAMIC, pto::DYNAMIC, pto::Layout::MX_A_ND>;
            using AGlobal = pto::GlobalTensor<ScaleElement, AShape, AStride, pto::Layout::MX_A_ND>;
            using BShape = pto::TileShape2D<ScaleElement, pto::DYNAMIC, pto::DYNAMIC, pto::Layout::MX_B_DN>;
            using BStride = pto::BaseShape2D<ScaleElement, pto::DYNAMIC, pto::DYNAMIC, pto::Layout::MX_B_DN>;
            using BGlobal = pto::GlobalTensor<ScaleElement, BShape, BStride, pto::Layout::MX_B_DN>;

            AScalePanel aPanel(actualM, panelScaleK);
            BScalePanel bPanel(panelScaleK, actualN);
            pto::TASSIGN(aPanel, AScaleAddress(slot, params));
            pto::TASSIGN(bPanel, BScaleAddress(slot, params));

            AShape aShape(actualM, panelScaleK);
            AStride aStride(actualM, aScaleLeadingDim);
            BShape bShape(panelScaleK, actualN);
            BStride bStride(bScaleLeadingDim, actualN);
            AGlobal aGlobal(gmAScale, aShape, aStride);
            BGlobal bGlobal(gmBScale, bShape, bStride);

            WaitFlag<PIPE_MTE1, PIPE_MTE2>(AScaleEvent(slot));
            pto::TLOAD(aPanel, aGlobal);
            SetFlag<PIPE_MTE2, PIPE_MTE1>(AScaleEvent(slot));
            WaitFlag<PIPE_MTE1, PIPE_MTE2>(BScaleEvent(slot));
            if (concatHalfN != 0U) {
                LoadBScaleConcatRaw(
                    BScaleAddress(slot, params), gmBScale, concatHalfN, panelScaleK, bScaleLeadingDim,
                    bScaleConcatStrideBytes, params, bypassScaleL2 ? L2_CACHE_DISABLE : L2_CACHE_NORMAL);
            } else if (bypassScaleL2) {
                constexpr uint32_t BASE_SCALE_PAIRS = BASE_SCALE_COLS >> 1U;
                SetNzDstStride(BASE_SCALE_PAIRS);
                copy_gm_to_cbuf_multi_dn2nz(
                    reinterpret_cast<__cbuf__ uint16_t*>(BScaleAddress(slot, params)),
                    reinterpret_cast<__gm__ uint16_t*>(gmBScale), 0, bScaleLeadingDim, L2_CACHE_DISABLE,
                    panelScaleK >> 1U, actualN, 0, false);
            } else {
                pto::TLOAD(bPanel, bGlobal);
            }
            SetFlag<PIPE_MTE2, PIPE_MTE1>(BScaleEvent(slot));
            return;
        }

        const uint32_t fullScalePairs = ScaleCols(params) >> 1U;
        const uint32_t panelScalePairs = panelScaleK >> 1U;
        WaitFlag<PIPE_MTE1, PIPE_MTE2>(AScaleEvent(slot));
        SetNzDstStride(fullScalePairs);
        copy_gm_to_cbuf_multi_dn2nz(
            reinterpret_cast<__cbuf__ uint16_t*>(AScaleAddress(slot, params)),
            reinterpret_cast<__gm__ uint16_t*>(gmAScale), 0, aScaleLeadingDim, 0, panelScalePairs, actualM, 0, false);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(AScaleEvent(slot));
        WaitFlag<PIPE_MTE1, PIPE_MTE2>(BScaleEvent(slot));
        if (concatHalfN != 0U) {
            LoadBScaleConcatRaw(
                BScaleAddress(slot, params), gmBScale, concatHalfN, panelScaleK, bScaleLeadingDim,
                bScaleConcatStrideBytes, params, bypassScaleL2 ? L2_CACHE_DISABLE : L2_CACHE_NORMAL);
        } else {
            SetNzDstStride(fullScalePairs);
            copy_gm_to_cbuf_multi_dn2nz(
                reinterpret_cast<__cbuf__ uint16_t*>(BScaleAddress(slot, params)),
                reinterpret_cast<__gm__ uint16_t*>(gmBScale), 0, bScaleLeadingDim,
                bypassScaleL2 ? L2_CACHE_DISABLE : L2_CACHE_NORMAL, panelScalePairs, actualN, 0, false);
        }
        SetFlag<PIPE_MTE2, PIPE_MTE1>(BScaleEvent(slot));
    }

    template <typename APanel, typename BPanel, typename LeftTile, typename RightTile>
    AICORE inline void ExtractDataL0Tiles(
        uint32_t dataSlot, uint32_t actualM, uint32_t actualN, uint32_t dataPanelK, uint32_t dataKOffset,
        const L1Params& params, LeftTile& aTile, RightTile& bTile)
    {
        APanel aPanel(actualM, MinU32(dataPanelK, dataKOffset + L0_K));
        BPanel bPanel(MinU32(dataPanelK, dataKOffset + L0_K), actualN);
        pto::TASSIGN(aPanel, ADataAddress(dataSlot, params));
        pto::TASSIGN(bPanel, BDataAddress(dataSlot, params));
        pto::TEXTRACT(aTile, aPanel, 0U, dataKOffset);
        pto::TEXTRACT(bTile, bPanel, dataKOffset, 0U);
    }

    AICORE inline void ExtractL0Tiles(
        uint32_t dataSlot, uint32_t scaleSlot, uint32_t l0Slot, uint32_t actualM, uint32_t actualN,
        uint32_t dataKOffset, uint32_t scaleKOffset, bool releaseDataPanel, bool releaseScalePanel,
        const L1Params& params)
    {
        using LeftTile = pto::TileLeftCompact<ElementA, L0_M, L0_K, pto::DYNAMIC, L0_K>;
        using RightTile = pto::TileRightCompact<ElementB, L0_K, L0_N, L0_K, pto::DYNAMIC>;
        using LeftScaleTile = pto::TileLeftScaleCompact<ScaleElement, L0_M, L0_SCALE_K, pto::DYNAMIC, L0_SCALE_K>;
        using RightScaleTile = pto::TileRightScaleCompact<ScaleElement, L0_SCALE_K, L0_N, L0_SCALE_K, pto::DYNAMIC>;

        LeftTile aTile(actualM);
        RightTile bTile(actualN);
        LeftScaleTile aScaleTile(actualM);
        RightScaleTile bScaleTile(actualN);
        pto::TASSIGN(aTile, static_cast<uint64_t>(l0Slot) * L0_PINGPONG_BYTES);
        pto::TASSIGN(bTile, static_cast<uint64_t>(l0Slot) * L0_PINGPONG_BYTES);
        pto::TASSIGN(aScaleTile, pto::GetScaleAddr(aTile.data()));
        pto::TASSIGN(bScaleTile, pto::GetScaleAddr(bTile.data()));

        WaitFlag<PIPE_M, PIPE_MTE1>(L0AEvent(l0Slot));
        WaitFlag<PIPE_M, PIPE_MTE1>(L0BEvent(l0Slot));
        if (dataKOffset == 0U) {
            WaitFlag<PIPE_MTE2, PIPE_MTE1>(ADataEvent(dataSlot));
            WaitFlag<PIPE_MTE2, PIPE_MTE1>(BDataEvent(dataSlot));
        }
        if (scaleKOffset == 0U) {
            WaitFlag<PIPE_MTE2, PIPE_MTE1>(AScaleEvent(scaleSlot));
            WaitFlag<PIPE_MTE2, PIPE_MTE1>(BScaleEvent(scaleSlot));
        }

        if (IsBaselineL1Params(params)) {
            constexpr uint32_t BASE_SCALE_COLS = BASE_SCALE_K / SCALE_GROUP_SIZE;
            using APanel = pto::Tile<
                pto::TileType::Mat, ElementA, L1_M, BASE_L1_K, pto::BLayout::ColMajor, pto::DYNAMIC, pto::DYNAMIC,
                pto::SLayout::RowMajor>;
            using BPanel = pto::Tile<
                pto::TileType::Mat, ElementB, BASE_L1_K, L1_N, pto::BLayout::RowMajor, pto::DYNAMIC, pto::DYNAMIC,
                pto::SLayout::ColMajor>;
            using AScalePanel = pto::Tile<
                pto::TileType::Mat, ScaleElement, L1_M, BASE_SCALE_COLS, pto::BLayout::RowMajor, pto::DYNAMIC,
                pto::DYNAMIC, pto::SLayout::RowMajor, 32>;
            using BScalePanel = pto::Tile<
                pto::TileType::Mat, ScaleElement, BASE_SCALE_COLS, L1_N, pto::BLayout::ColMajor, pto::DYNAMIC,
                pto::DYNAMIC, pto::SLayout::ColMajor, 32>;

            ExtractDataL0Tiles<APanel, BPanel>(
                dataSlot, actualM, actualN, params.dataK, dataKOffset, params, aTile, bTile);
            AScalePanel aScalePanel(actualM, MinU32(BASE_SCALE_COLS, scaleKOffset + L0_SCALE_K));
            BScalePanel bScalePanel(MinU32(BASE_SCALE_COLS, scaleKOffset + L0_SCALE_K), actualN);
            pto::TASSIGN(aScalePanel, AScaleAddress(scaleSlot, params));
            pto::TASSIGN(bScalePanel, BScaleAddress(scaleSlot, params));
            pto::TEXTRACT(aScaleTile, aScalePanel, 0U, scaleKOffset);
            pto::TEXTRACT(bScaleTile, bScalePanel, scaleKOffset, 0U);
        } else {
            const uint16_t mAlign = static_cast<uint16_t>(AlignBlockM(actualM));
            const uint16_t nAlign = static_cast<uint16_t>(AlignBlockM(actualN));
            ExtractL0Raw<LeftTile, RightTile, LeftScaleTile, RightScaleTile>(
                aTile.data(), bTile.data(), aScaleTile.data(), bScaleTile.data(), ADataAddress(dataSlot, params),
                BDataAddress(dataSlot, params), AScaleAddress(scaleSlot, params), BScaleAddress(scaleSlot, params),
                mAlign, nAlign, static_cast<uint16_t>(dataKOffset), static_cast<uint16_t>(scaleKOffset),
                static_cast<uint16_t>(params.blockM), static_cast<uint16_t>(ScaleCols(params)));
        }

        if (releaseDataPanel) {
            SetFlag<PIPE_MTE1, PIPE_MTE2>(ADataEvent(dataSlot));
            SetFlag<PIPE_MTE1, PIPE_MTE2>(BDataEvent(dataSlot));
        }
        if (releaseScalePanel) {
            SetFlag<PIPE_MTE1, PIPE_MTE2>(AScaleEvent(scaleSlot));
            SetFlag<PIPE_MTE1, PIPE_MTE2>(BScaleEvent(scaleSlot));
        }
        SetFlag<PIPE_MTE1, PIPE_M>(kL0ReadyEvent);
    }

    template <bool EnableUnitFlag, typename AccTile>
    AICORE inline void LaunchMatmul(
        AccTile& cTile, uint32_t l0Slot, uint32_t actualM, uint32_t actualN, bool firstK, bool lastK)
    {
        using LeftTile = pto::TileLeftCompact<ElementA, L0_M, L0_K, pto::DYNAMIC, L0_K>;
        using RightTile = pto::TileRightCompact<ElementB, L0_K, L0_N, L0_K, pto::DYNAMIC>;
        using LeftScaleTile = pto::TileLeftScaleCompact<ScaleElement, L0_M, L0_SCALE_K, pto::DYNAMIC, L0_SCALE_K>;
        using RightScaleTile = pto::TileRightScaleCompact<ScaleElement, L0_SCALE_K, L0_N, L0_SCALE_K, pto::DYNAMIC>;

        LeftTile aTile(actualM);
        RightTile bTile(actualN);
        LeftScaleTile aScaleTile(actualM);
        RightScaleTile bScaleTile(actualN);
        pto::TASSIGN(aTile, static_cast<uint64_t>(l0Slot) * L0_PINGPONG_BYTES);
        pto::TASSIGN(bTile, static_cast<uint64_t>(l0Slot) * L0_PINGPONG_BYTES);
        pto::TASSIGN(aScaleTile, pto::GetScaleAddr(aTile.data()));
        pto::TASSIGN(bScaleTile, pto::GetScaleAddr(bTile.data()));

        WaitFlag<PIPE_MTE1, PIPE_M>(kL0ReadyEvent);
        if constexpr (EnableUnitFlag) {
            if (firstK) {
                if (lastK) {
                    pto::TMATMUL_MX<pto::AccPhase::Final>(cTile, aTile, aScaleTile, bTile, bScaleTile);
                } else {
                    pto::TMATMUL_MX<pto::AccPhase::Partial>(cTile, aTile, aScaleTile, bTile, bScaleTile);
                }
            } else if (lastK) {
                pto::TMATMUL_MX<pto::AccPhase::Final>(cTile, cTile, aTile, aScaleTile, bTile, bScaleTile);
            } else {
                pto::TMATMUL_MX<pto::AccPhase::Partial>(cTile, cTile, aTile, aScaleTile, bTile, bScaleTile);
            }
        } else if (firstK) {
            pto::TMATMUL_MX(cTile, aTile, aScaleTile, bTile, bScaleTile);
        } else {
            pto::TMATMUL_MX(cTile, cTile, aTile, aScaleTile, bTile, bScaleTile);
        }
        SetFlag<PIPE_M, PIPE_MTE1>(L0AEvent(l0Slot));
        SetFlag<PIPE_M, PIPE_MTE1>(L0BEvent(l0Slot));
    }

    uint64_t l1Start_ = 0U;
    L1Params activeL1Params_{};
    bool l1ParamsValid_ = false;
};

#endif // DISPATCH_MEGA_COMBINE_PTO_GMM_MX_PRELOAD_HPP
