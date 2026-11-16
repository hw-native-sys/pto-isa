/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TMATMUL_HPP
#define TMATMUL_HPP

#include <pto/npu/kirin9050/custom/TMatmulCore.hpp>

namespace pto {

template <typename PtrT>
__tf__ PTO_INTERNAL PtrT GetCbufTilePtr(PtrT ptr)
{
    return (PtrT)__cce_get_tile_ptr(ptr);
}

PTO_INTERNAL void CheckDynamicMmad(uint16_t aMatrixRow, uint16_t aMatrixCol, uint16_t bMatrixCol)
{
    PTO_ASSERT(
        aMatrixRow >= 1 && aMatrixRow <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid aMatrixRow is [1, 4095].");
    PTO_ASSERT(
        aMatrixCol >= 1 && aMatrixCol <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid aMatrixCol is [1, 4095].");
    PTO_ASSERT(
        bMatrixCol >= 1 && bMatrixCol <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid bMatrixCol is [1, 4095].");
}

template <typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void CheckMadValid()
{
    using AType = typename TileLeft::DType;
    using BType = typename TileRight::DType;
    using CType = typename TileRes::DType;
    if constexpr (std::is_same_v<CType, half>) {
        // half 输出：支持 f16*f16；以及 s8/s16 输入（mac 先做 s32 累加，经 pre-stage dequant 落到 fp16）。
        static_assert(
            (std::is_same_v<AType, half> && std::is_same_v<BType, half>) ||
                (std::is_same_v<AType, int8_t> && std::is_same_v<BType, int8_t>) ||
                (std::is_same_v<AType, int16_t> && std::is_same_v<BType, int16_t>) ||
                (std::is_same_v<AType, int16_t> && std::is_same_v<BType, int8_t>),
            "TMATMUL: fp16 Acc supports f16f16 or s8s8/s16s16/s16s8 inputs (pre-stage dequant).");
    } else if constexpr (std::is_same_v<CType, int32_t>) {
        static_assert(
            (std::is_same_v<AType, int8_t> && std::is_same_v<BType, int8_t>) ||
                (std::is_same_v<AType, int16_t> && std::is_same_v<BType, int16_t>) ||
                (std::is_same_v<AType, int16_t> && std::is_same_v<BType, int8_t>) ||
                (std::is_same_v<AType, half> && std::is_same_v<BType, half>),
            "TMATMUL: For int32 accumulation, supported input pairs are s8s8/s16s16/s16s8/f16f16.");
    } else if constexpr (std::is_same_v<CType, int16_t> || std::is_same_v<CType, int8_t>) {
        // int8/int16 输出：支持 f16*f16；以及 s8/s16 输入（pre-stage requant）。
        static_assert(
            (std::is_same_v<AType, half> && std::is_same_v<BType, half>) ||
                (std::is_same_v<AType, int8_t> && std::is_same_v<BType, int8_t>) ||
                (std::is_same_v<AType, int16_t> && std::is_same_v<BType, int16_t>) ||
                (std::is_same_v<AType, int16_t> && std::is_same_v<BType, int8_t>),
            "TMATMUL: int8/int16 Acc supports f16f16 or s8s8/s16s16/s16s8 inputs (pre-stage requant).");
    } else {
        static_assert(sizeof(CType) == 0, "TMATMUL: Acc Type only supports int32_t/half/int16_t/int8_t.");
    }

    static_assert(
        TileLeft::Loc == TileType::Left || TileLeft::Loc == TileType::Mat,
        "TMATMUL: TileLeft TileType must be TileType::Left or TileType::Mat (L1 simulating L0A).");
    static_assert(TileRight::Loc == TileType::Right, "TMATMUL: TileRight TileType must be TileType::Right.");
    static_assert(
        TileRes::Loc == TileType::Acc || TileRes::Loc == TileType::Mat,
        "TMATMUL: TileRes TileType must be TileType::Acc or TileType::Mat (L1 simulating L0C).");
    static_assert(
        (!TileLeft::isRowMajor) && (TileRight::isRowMajor) && (!TileRes::isRowMajor) &&
            (TileLeft::SFractal == SLayout::RowMajor) && (TileRight::SFractal == SLayout::ColMajor) &&
            (TileRes::SFractal == SLayout::RowMajor),
        "TMATMUL: Non-conforming matrix fractal.");
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_IMPL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix)
{
    CheckMadValid<TileRes, TileLeft, TileRight>();
    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    CheckDynamicMmad(m, k, n);
    TMatmul<TileRes, TileLeft, TileRight>(cMatrix.data(), aMatrix.data(), bMatrix.data(), m, k, n, true);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_ACC_IMPL(TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileRight& bMatrix)
{
    CheckMadValid<TileRes, TileLeft, TileRight>();
    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    CheckDynamicMmad(m, k, n);
    TMatmul<TileRes, TileLeft, TileRight>(cOutMatrix.data(), aMatrix.data(), bMatrix.data(), m, k, n, false);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_ACC_IMPL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix)
{
    TMATMUL_ACC_IMPL<Phase>(cMatrix, cMatrix, aMatrix, bMatrix);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight, typename TileBias>
PTO_INTERNAL void TMATMUL_BIAS_IMPL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, TileBias& biasData)
{
    CheckMadValid<TileRes, TileLeft, TileRight>();
    static_assert(std::is_same_v<typename TileRes::DType, typename TileBias::DType>, "No supported bias data type.");
    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    CheckDynamicMmad(m, k, n);

    if constexpr (TileBias::Loc == TileType::Bias) {
        static_assert(TileBias::Rows == 1, "Broadcast bias must be single row.");
        TMatmulBiasBroadcast<TileRes, TileLeft, TileRight>(
            cMatrix.data(), aMatrix.data(), bMatrix.data(), (uint64_t)biasData.data(), m, k, n, false);
    } else if constexpr (TileBias::Loc == TileType::Mat) {
        static_assert(
            (TileBias::Rows == 1) && (TileBias::isRowMajor),
            "Non-broadcast bias must be Mat tile (L1) with single row, row major.");
        TMatmulBiasNonBroadcast<TileRes, TileLeft, TileRight, TileBias>(
            cMatrix.data(), aMatrix.data(), bMatrix.data(), biasData.data(), m, k, n, false);
    } else {
        static_assert(sizeof(typename TileBias::DType) == 0, "Bias tile must be TileType::Bias or TileType::Mat.");
    }
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TGEMV_IMPL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix)
{
    CheckMadValid<TileRes, TileLeft, TileRight>();
    uint16_t k = bMatrix.GetValidRow();
    uint16_t n = bMatrix.GetValidCol();
    PTO_ASSERT(k >= 1 && k <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid aMatrixCol is [1, 4095].");
    PTO_ASSERT(n >= 1 && n <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid bMatrixCol is [1, 4095].");
    TMatmulGemv<TileRes, TileLeft, TileRight>(cMatrix.data(), aMatrix.data(), bMatrix.data(), k, n, true);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TGEMV_ACC_IMPL(TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileRight& bMatrix)
{
    CheckMadValid<TileRes, TileLeft, TileRight>();
    uint16_t k = bMatrix.GetValidRow();
    uint16_t n = bMatrix.GetValidCol();
    PTO_ASSERT(k >= 1 && k <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid aMatrixCol is [1, 4095].");
    PTO_ASSERT(n >= 1 && n <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid bMatrixCol is [1, 4095].");
    TMatmulGemv<TileRes, TileLeft, TileRight>(cOutMatrix.data(), aMatrix.data(), bMatrix.data(), k, n, false);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight, typename TileBias>
PTO_INTERNAL void TGEMV_BIAS_IMPL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, TileBias& biasData)
{
    CheckMadValid<TileRes, TileLeft, TileRight>();
    static_assert(std::is_same_v<typename TileRes::DType, typename TileBias::DType>, "No supported bias data type.");
    // pypto 下发的是单行 Bias tile（TileType::Bias，驻留 BT buffer）；沿用广播 bias 路径，
    // 经 btAddr 传地址（原 TMatmulFull 走 __cbuf__ 指针 cast，在 aicore 中非法且 __tf__ 调 __tf__）。
    static_assert(
        (TileBias::Loc == TileType::Bias || TileBias::Loc == TileType::Mat) && (TileBias::Rows == 1),
        "TileBias must be a single-row Bias/Mat tile.");
    uint16_t k = bMatrix.GetValidRow();
    uint16_t n = bMatrix.GetValidCol();
    PTO_ASSERT(k >= 1 && k <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid aMatrixCol is [1, 4095].");
    PTO_ASSERT(n >= 1 && n <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid bMatrixCol is [1, 4095].");
    uint64_t btAddr = (uint64_t)biasData.data();
    MatmulConfig cfg;
    cfg.initCtrl = false;
    cfg.broadcastEn = true;
    cfg.gemvCtrl = true;
    cfg.sizeN = n;
    if constexpr (std::is_same_v<typename TileRes::DType, half>) {
        cfg.preQuant = MatmulPreQuant::DeqF16Scalar;
    }
    SetMFpcForFp16<typename TileRes::DType>();
    SetMQuantPre<typename TileRes::DType>(0.0f);
    TMatmulCore<TileRes, TileLeft, TileRight>(
        cMatrix.data(), aMatrix.data(), bMatrix.data(), nullptr, btAddr, 1, k, n, cfg.Build());
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TGEMV_MACRO_IMPL(
    TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, const MatmulMacroConfig& cfg)
{
    CheckMadValid<TileRes, TileLeft, TileRight>();
    uint16_t k = bMatrix.GetValidRow();
    uint16_t n = bMatrix.GetValidCol();
    PTO_ASSERT(k >= 1 && k <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid aMatrixCol is [1, 4095].");
    PTO_ASSERT(n >= 1 && n <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid bMatrixCol is [1, 4095].");
    MatmulMacroConfig gemvCfg = cfg;
    gemvCfg.gemvCtrl = true;
    TMatmulMacro<TileRes, TileLeft, TileRight, false, false, false>(
        cMatrix.data(), aMatrix.data(), bMatrix.data(), nullptr, 0, 1, k, n, gemvCfg);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight,
    bool isClear = false>
PTO_INTERNAL void TGEMV_MACRO_ACC_IMPL(
    TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileRight& bMatrix, const MatmulMacroConfig& cfg)
{
    CheckMadValid<TileRes, TileLeft, TileRight>();
    uint16_t k = bMatrix.GetValidRow();
    uint16_t n = bMatrix.GetValidCol();
    PTO_ASSERT(k >= 1 && k <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid aMatrixCol is [1, 4095].");
    PTO_ASSERT(n >= 1 && n <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid bMatrixCol is [1, 4095].");
    MatmulMacroConfig gemvCfg = cfg;
    gemvCfg.gemvCtrl = true;
    TMatmulMacroAcc<TileRes, TileLeft, TileRight, false, false, isClear>(
        cOutMatrix.data(), aMatrix.data(), bMatrix.data(), nullptr, 0, 1, k, n, gemvCfg);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight, typename TileBias>
PTO_INTERNAL void TGEMV_MACRO_BIAS_IMPL(
    TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, TileBias& biasData, const MatmulMacroConfig& cfg)
{
    CheckMadValid<TileRes, TileLeft, TileRight>();
    static_assert(std::is_same_v<typename TileRes::DType, typename TileBias::DType>, "No supported bias data type.");
    static_assert(
        (TileBias::Loc == TileType::Bias || TileBias::Loc == TileType::Mat) && (TileBias::Rows == 1),
        "TileBias must be a single-row Bias/Mat tile.");
    uint16_t k = bMatrix.GetValidRow();
    uint16_t n = bMatrix.GetValidCol();
    PTO_ASSERT(k >= 1 && k <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid aMatrixCol is [1, 4095].");
    PTO_ASSERT(n >= 1 && n <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid bMatrixCol is [1, 4095].");
    MatmulMacroConfig gemvCfg = cfg;
    gemvCfg.gemvCtrl = true;
    TMatmulMacro<TileRes, TileLeft, TileRight, true, true, false>(
        cMatrix.data(), aMatrix.data(), bMatrix.data(), nullptr, (uint64_t)biasData.data(), 1, k, n, gemvCfg);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight,
    typename TileBias = void, bool isClear = false>
PTO_INTERNAL void TMATMUL_MACRO_ACC_IMPL(
    TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, TileBias* biasData, const MatmulMacroConfig& cfg)
{
    CheckMadValid<TileRes, TileLeft, TileRight>();
    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    CheckDynamicMmad(m, k, n);

    constexpr bool hasBias = !std::is_same_v<TileBias, void>;
    __cbuf__ int32_t* biasPtr = nullptr;
    uint64_t btAddr = 0;
    if constexpr (hasBias) {
        if constexpr (TileBias::Loc == TileType::Bias) {
            static_assert(TileBias::Rows == 1, "Broadcast bias must be single row.");
            btAddr = (uint64_t)biasData->data();
        } else if constexpr (TileBias::Loc == TileType::Mat) {
            static_assert(
                (TileBias::Rows == TileRes::Rows) && (TileBias::Cols == TileRes::Cols),
                "Non-broadcast bias must have same shape as TileRes.");
            biasPtr = (__cbuf__ int32_t*)GetCbufTilePtr(biasData->data());
        } else {
            static_assert(sizeof(typename TileBias::DType) == 0, "Bias tile must be TileType::Bias or TileType::Mat.");
        }
    } else if constexpr (!isClear) {
        biasPtr = (__cbuf__ int32_t*)GetCbufTilePtr(cMatrix.data());
    }

    if constexpr (hasBias) {
        if constexpr (TileBias::Loc == TileType::Bias) {
            TMatmulMacroAcc<TileRes, TileLeft, TileRight, true, true, false>(
                cMatrix.data(), aMatrix.data(), bMatrix.data(), biasPtr, btAddr, m, k, n, cfg);
        } else {
            TMatmulMacroAcc<TileRes, TileLeft, TileRight, true, false, false>(
                cMatrix.data(), aMatrix.data(), bMatrix.data(), biasPtr, btAddr, m, k, n, cfg);
        }
    } else {
        TMatmulMacroAcc<TileRes, TileLeft, TileRight, false, false, isClear>(
            cMatrix.data(), aMatrix.data(), bMatrix.data(), biasPtr, btAddr, m, k, n, cfg);
    }
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight, typename TileBias,
    bool isAcc = false>
PTO_INTERNAL void TMATMUL_MACRO_IMPL(
    TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, TileBias& biasData, const MatmulMacroConfig& cfg)
{
    CheckMadValid<TileRes, TileLeft, TileRight>();
    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    CheckDynamicMmad(m, k, n);

    constexpr bool isBroadcast = (TileBias::Loc == TileType::Bias);
    __cbuf__ int32_t* biasPtr = nullptr;
    uint64_t btAddr = 0;
    if constexpr (TileBias::Loc == TileType::Bias) {
        static_assert(TileBias::Rows == 1, "Broadcast bias must be single row.");
        btAddr = (uint64_t)biasData.data();
    } else if constexpr (TileBias::Loc == TileType::Mat) {
        static_assert(
            (TileBias::Rows == TileRes::Rows) && (TileBias::Cols == TileRes::Cols),
            "Non-broadcast bias must have same shape as TileRes.");
        biasPtr = (__cbuf__ int32_t*)GetCbufTilePtr(biasData.data());
    } else {
        static_assert(sizeof(typename TileBias::DType) == 0, "Bias tile must be TileType::Bias or TileType::Mat.");
    }
    TMatmulMacro<TileRes, TileLeft, TileRight, true, isBroadcast, isAcc>(
        cMatrix.data(), aMatrix.data(), bMatrix.data(), biasPtr, btAddr, m, k, n, cfg);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight, bool isAcc = false>
PTO_INTERNAL void TMATMUL_MACRO_IMPL(
    TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, const MatmulMacroConfig& cfg)
{
    CheckMadValid<TileRes, TileLeft, TileRight>();
    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    CheckDynamicMmad(m, k, n);

    __cbuf__ int32_t* biasPtr = nullptr;
    if constexpr (isAcc) {
        biasPtr = (__cbuf__ int32_t*)GetCbufTilePtr(cMatrix.data());
    }
    TMatmulMacro<TileRes, TileLeft, TileRight, false, false, isAcc>(
        cMatrix.data(), aMatrix.data(), bMatrix.data(), biasPtr, 0, m, k, n, cfg);
}

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale>
PTO_INTERNAL void TMATMUL_MX_IMPL(
    TileRes& cMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix, TileRightScale& bScaleMatrix)
{
    static_assert(sizeof(TileRes::DType) == 0, "no support instruction.");
}

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale>
PTO_INTERNAL void TMATMUL_MX_IMPL(
    TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix,
    TileRightScale& bScaleMatrix)
{
    static_assert(sizeof(TileRes::DType) == 0, "no support instruction.");
}

template <
    typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
    typename TileBias>
PTO_INTERNAL void TMATMUL_MX_IMPL(
    TileRes& cMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix, TileRightScale& bScaleMatrix,
    TileBias& biasData)
{
    static_assert(sizeof(TileRes::DType) == 0, "no support instruction.");
}

template <bool isEnable, RoundMode tf32TransMode = RoundMode::CAST_ROUND>
PTO_INTERNAL void TSETTF32MODE_IMPL()
{
    static_assert(!isEnable, "Fix: Kirin9050 does not support setting the TF32 mode to enabled.");
    set_ctrl(sbitset0(get_ctrl(), TF32_MODE_BIT));
}
} // namespace pto
#endif
