/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TMATMULCORE_HPP
#define TMATMULCORE_HPP

#include <pto/npu/kirin9050/custom/TMatmulConfig.hpp>

namespace pto {

template <typename TileRes, typename TileLeft, typename TileRight>
__tf__ PTO_INTERNAL void TMatmulCore(
    typename TileRes::TileDType __out__ cData, typename TileLeft::TileDType __in__ aData,
    typename TileRight::TileDType __in__ bData, __cbuf__ int32_t* biasAddr, uint64_t btAddr, uint16_t m, uint16_t k,
    uint16_t n, uint64_t config)
{
    constexpr uint32_t BLOCK_BYTE = 32;
    constexpr uint32_t FRACTAL_ROW = 16;

    uint64_t matrixPara = (static_cast<uint64_t>(m) & 0xFFFF) | ((static_cast<uint64_t>(k) & 0xFFFF) << 16);
    set_matrix_para(matrixPara);

    constexpr uint32_t srcStride = TileLeft::Rows * TileLeft::InnerCols * sizeof(typename TileLeft::DType) / BLOCK_BYTE;
    constexpr uint32_t dstStride = TileRes::Rows * TileRes::InnerCols * sizeof(typename TileRes::DType) / BLOCK_BYTE;
    uint64_t cubeStridePara =
        (static_cast<uint64_t>(srcStride) & 0xFFFF) | ((static_cast<uint64_t>(dstStride) & 0xFFFF) << 16);
    set_cube_stride_para(cubeStridePara);

    using T = typename TileRes::DType;
    __cbuf__ typename TileLeft::DType* a = (__cbuf__ typename TileLeft::DType*)__cce_get_tile_ptr(aData);
    __cb__ typename TileRight::DType* b = (__cb__ typename TileRight::DType*)__cce_get_tile_ptr(bData);
    __cbuf__ void* c = (__cbuf__ void*)__cce_get_tile_ptr(cData);

    // The hardware clears the accumulator through the initCtrl bit (config & 0x1); the software
    // clear below raced with the M-pipe macro instruction (a delayed clear wiped the first K slice
    // of every C tile after the first one in multi-batch GEMV), so it stays disabled.
    // if (config & 0x1) {
    //     ZeroInitCTilePtr<TileRes>(c);
    //     // The L1 clear above is issued outside the M (macro instruction) pipe. Without an
    //     // explicit barrier the first matmul of a C tile can be queued before the clear
    //     // retires and the clear then wipes that slice (observed as a missing K slice for
    //     // every C tile after the first one in multi-batch GEMV).
    //     pipe_barrier(PIPE_ALL);
    // }

    matmul_to_cbuf(c, a, b, biasAddr, btAddr, config);
}

template <typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMatmul(
    typename TileRes::TileDType __out__ cData, typename TileLeft::TileDType __in__ aData,
    typename TileRight::TileDType __in__ bData, uint16_t m, uint16_t k, uint16_t n, bool initCtrl)
{
    using T = typename TileRes::DType;
    MatmulConfig cfg;
    cfg.initCtrl = initCtrl;
    cfg.sizeN = n;
    if constexpr (std::is_same_v<T, half>) {
        cfg.preQuant = MatmulPreQuant::DeqF16Scalar;
    }
    SetMFpcForFp16<T>();
    SetMQuantPre<T>(0.0f);
    TMatmulCore<TileRes, TileLeft, TileRight>(cData, aData, bData, nullptr, 0, m, k, n, cfg.Build());
}

template <typename TileRes, typename TileLeft, typename TileRight, typename TileBias>
__tf__ PTO_INTERNAL void TMatmulBiasNonBroadcast(
    typename TileRes::TileDType __out__ cData, typename TileLeft::TileDType __in__ aData,
    typename TileRight::TileDType __in__ bData, typename TileBias::TileDType __in__ biasData, uint16_t m, uint16_t k,
    uint16_t n, bool initCtrl)
{
    __cbuf__ typename TileBias::DType* biasPtr = (__cbuf__ typename TileBias::DType*)__cce_get_tile_ptr(biasData);
    MatmulConfig cfg;
    cfg.initCtrl = initCtrl;
    cfg.broadcastEn = false;
    cfg.sizeN = n;
    // 与 TMatmul / TMatmulBiasBroadcast 对齐：half 输出需显式开启 fp16 反量化（preQuant），
    // 否则 fixpipe 不按 fp16 处理累加结果（kirin9050 非宏指令 non-broadcast bias 路径）。
    if constexpr (std::is_same_v<typename TileRes::DType, half>) {
        cfg.preQuant = MatmulPreQuant::DeqF16Scalar;
    }
    SetMFpcForFp16<typename TileRes::DType>();
    SetMQuantPre<typename TileRes::DType>(0.0f);
    TMatmulCore<TileRes, TileLeft, TileRight>(cData, aData, bData, (__cbuf__ int32_t*)biasPtr, 0, m, k, n, cfg.Build());
}

template <typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMatmulBiasBroadcast(
    typename TileRes::TileDType __out__ cData, typename TileLeft::TileDType __in__ aData,
    typename TileRight::TileDType __in__ bData, uint64_t btAddr, uint16_t m, uint16_t k, uint16_t n, bool initCtrl)
{
    using T = typename TileRes::DType;
    // 广播 bias 路径由硬件用 btAddr 初始化 C，无需软件清零；软件清零与 M 管宏指令
    // 无同步，会与后续 tile 的 matmul 竞态（同 TMatmulCore 中已注释的 initCtrl 清零）。
    // ZeroInitCTile<TileRes>(cData);
    MatmulConfig cfg;
    cfg.initCtrl = initCtrl;
    cfg.broadcastEn = true;
    cfg.sizeN = n;
    if constexpr (std::is_same_v<T, half>) {
        cfg.preQuant = MatmulPreQuant::DeqF16Scalar;
    }
    SetMFpcForFp16<T>();
    SetMQuantPre<T>(0.0f);
    TMatmulCore<TileRes, TileLeft, TileRight>(cData, aData, bData, nullptr, btAddr, m, k, n, cfg.Build());
}

template <
    typename TileRes, typename TileLeft, typename TileRight, bool hasBias, bool isBroadcast = false,
    bool isClear = false>
PTO_INTERNAL void TMatmulMacroAcc(
    typename TileRes::TileDType __out__ cData, typename TileLeft::TileDType __in__ aData,
    typename TileRight::TileDType __in__ bData, __cbuf__ int32_t* biasAddr, uint64_t btAddr, uint16_t m, uint16_t k,
    uint16_t n, const MatmulMacroConfig& cfg)
{
    using OutType = typename TileRes::DType;
    constexpr auto preQuant = MatmulPreQuant::NoQuant;
    constexpr auto preRelu = MatmulPreRelu::NoRelu;
    constexpr bool broadcastEn = hasBias && isBroadcast;
    constexpr bool initCtrl = isClear;

    if constexpr (hasBias) {
        // 广播 bias 路径由硬件用 btAddr 初始化 C，无需软件清零（清零与宏指令无同步，会竞态）。
        // ZeroInitCTile<TileRes>(cData);
    }

    using InputType = typename TileLeft::DType;
    uint64_t mFpc = 0;
    if constexpr (std::is_same_v<InputType, half>) {
        mFpc |= ((uint64_t)42) << 40;
    }
    set_m_fpc(mFpc);
    SetMQuantPre<InputType>(0.0f);

    MatmulConfig matmulCfg;
    matmulCfg.initCtrl = initCtrl;
    matmulCfg.broadcastEn = broadcastEn;
    matmulCfg.preQuant = preQuant;
    matmulCfg.preRelu = preRelu;
    matmulCfg.gemvCtrl = cfg.gemvCtrl;
    matmulCfg.sizeN = n;
    TMatmulCore<TileRes, TileLeft, TileRight>(cData, aData, bData, biasAddr, btAddr, m, k, n, matmulCfg.Build());
}

template <
    typename TileRes, typename TileLeft, typename TileRight, bool hasBias, bool isBroadcast = false, bool isAcc = false>
PTO_INTERNAL void TMatmulMacro(
    typename TileRes::TileDType __out__ cData, typename TileLeft::TileDType __in__ aData,
    typename TileRight::TileDType __in__ bData, __cbuf__ int32_t* biasAddr, uint64_t btAddr, uint16_t m, uint16_t k,
    uint16_t n, const MatmulMacroConfig& cfg)
{
    static_assert(!(hasBias && isAcc), "bias and acc are mutually exclusive");
    using OutType = typename TileRes::DType;
    constexpr auto preQuant = InferPreQuant<int32_t, OutType>();
    if (cfg.quantMode == MatmulQuantMode::ScalarQuant) {
        PTO_ASSERT(false, "kirin9050 does not support scalar pre-quant.");
    }
    const auto preRelu = static_cast<MatmulPreRelu>(cfg.reluMode);
    constexpr bool broadcastEn = hasBias && isBroadcast;

    bool initCtrl;
    if constexpr (hasBias) {
        initCtrl = false;
        // 广播 bias 路径由硬件用 btAddr 初始化 C，无需软件清零（清零与宏指令无同步，会竞态）。
        // ZeroInitCTile<TileRes>(cData);
    } else if constexpr (isAcc) {
        initCtrl = false;
    } else {
        initCtrl = true;
    }

    SetMFpcFromConfig<typename TileLeft::DType, preQuant>(preRelu, cfg);

    if (preRelu == MatmulPreRelu::ScalarRelu || preRelu == MatmulPreRelu::VectorRelu) {
        SetMReluAlpha(cfg.reluScalar);
    }
    SetMQuantPre<OutType>(cfg.clipReluVal);

    MatmulConfig matmulCfg;
    matmulCfg.initCtrl = initCtrl;
    matmulCfg.broadcastEn = broadcastEn;
    matmulCfg.preQuant = preQuant;
    matmulCfg.preRelu = preRelu;
    matmulCfg.clipReluEn = (cfg.clipReluVal != 0.0f);
    matmulCfg.gemvCtrl = cfg.gemvCtrl;
    matmulCfg.sizeN = n;
    TMatmulCore<TileRes, TileLeft, TileRight>(cData, aData, bData, biasAddr, btAddr, m, k, n, matmulCfg.Build());
}

template <typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMatmulGemv(
    typename TileRes::TileDType __out__ cData, typename TileLeft::TileDType __in__ aData,
    typename TileRight::TileDType __in__ bData, uint16_t k, uint16_t n, bool initCtrl)
{
    MatmulConfig cfg;
    cfg.initCtrl = initCtrl;
    cfg.gemvCtrl = true;
    cfg.sizeN = n;
    // 与 TMatmul 对齐：half 输出需显式开启 fp16 反量化（preQuant）。
    // 缺失时 fp16 GEMV（M=1,N>1）输出错误，而 s8->s32 GEMV 正常——与实测现象一致。
    if constexpr (std::is_same_v<typename TileRes::DType, half>) {
        cfg.preQuant = MatmulPreQuant::DeqF16Scalar;
    }
    SetMFpcForFp16<typename TileRes::DType>();
    SetMQuantPre<typename TileRes::DType>(0.0f);
    TMatmulCore<TileRes, TileLeft, TileRight>(cData, aData, bData, nullptr, 0, 1, k, n, cfg.Build());
}

template <typename TileRes, typename TileLeft, typename TileRight, typename TileBias>
__tf__ PTO_INTERNAL void TMatmulFull(
    typename TileRes::TileDType __out__ cData, typename TileLeft::TileDType __in__ aData,
    typename TileRight::TileDType __in__ bData, typename TileBias::TileDType __in__ biasData, uint64_t btAddr,
    uint16_t m, uint16_t k, uint16_t n, const MatmulConfig& cfg)
{
    __cbuf__ int32_t* biasPtr = nullptr;
    if constexpr (!std::is_same_v<TileBias, void>) {
        biasPtr = (__cbuf__ int32_t*)__cce_get_tile_ptr(biasData);
    }
    MatmulConfig fullCfg = cfg;
    fullCfg.sizeN = n;
    SetMFpcForFp16<typename TileRes::DType>();
    SetMQuantPre<typename TileRes::DType>(0.0f);
    TMatmulCore<TileRes, TileLeft, TileRight>(cData, aData, bData, biasPtr, btAddr, m, k, n, fullCfg.Build());
}

} // namespace pto
#endif
