/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_GPU_COMMON_TMATMUL_HPP
#define PTO_GPU_COMMON_TMATMUL_HPP

#include <pto/common/pto_tile.hpp>
#include "pto/gpu/common/tile_offsets.hpp"

namespace pto {

template <typename TileAcc, typename TileLeft, typename TileRight>
PTO_INTERNAL void CheckGpuMatmulValid()
{
    using AType = typename TileLeft::DType;
    using BType = typename TileRight::DType;
    using CType = typename TileAcc::DType;
    static_assert(
        (std::is_same_v<AType, int8_t> && std::is_same_v<BType, int8_t> && std::is_same_v<CType, int32_t>) ||
            (std::is_same_v<AType, half> && std::is_same_v<BType, half> && std::is_same_v<CType, float>) ||
            (std::is_same_v<AType, float> && std::is_same_v<BType, float> && std::is_same_v<CType, float>) ||
            (std::is_same_v<AType, bfloat16_t> && std::is_same_v<BType, bfloat16_t> && std::is_same_v<CType, float>),
        "Fix: GPU TMATMUL currently supports int8->int32, half->float, bfloat16->float, and float->float.");
}

template <typename TileAcc, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_IMPL(TileAcc &cMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    CheckGpuMatmulValid<TileAcc, TileLeft, TileRight>();

    const uint16_t m = aMatrix.GetValidRow();
    const uint16_t k = aMatrix.GetValidCol();
    const uint16_t n = bMatrix.GetValidCol();

    for (uint16_t row = 0; row < m; ++row) {
        for (uint16_t col = 0; col < n; ++col) {
            typename TileAcc::DType acc = typename TileAcc::DType{};
            for (uint16_t kk = 0; kk < k; ++kk) {
                const std::size_t aIdx = gpu::GetTileElementOffset<TileLeft>(row, kk);
                const std::size_t bIdx = gpu::GetTileElementOffset<TileRight>(kk, col);
                acc += static_cast<typename TileAcc::DType>(aMatrix.data()[aIdx]) *
                       static_cast<typename TileAcc::DType>(bMatrix.data()[bIdx]);
            }
            cMatrix.data()[gpu::GetTileElementOffset<TileAcc>(row, col)] = acc;
        }
    }
}

template <typename TileAcc, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_ACC_IMPL(TileAcc &cOutMatrix, TileAcc &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    CheckGpuMatmulValid<TileAcc, TileLeft, TileRight>();

    const uint16_t m = aMatrix.GetValidRow();
    const uint16_t k = aMatrix.GetValidCol();
    const uint16_t n = bMatrix.GetValidCol();

    for (uint16_t row = 0; row < m; ++row) {
        for (uint16_t col = 0; col < n; ++col) {
            const std::size_t cIdx = gpu::GetTileElementOffset<TileAcc>(row, col);
            typename TileAcc::DType acc = cInMatrix.data()[cIdx];
            for (uint16_t kk = 0; kk < k; ++kk) {
                const std::size_t aIdx = gpu::GetTileElementOffset<TileLeft>(row, kk);
                const std::size_t bIdx = gpu::GetTileElementOffset<TileRight>(kk, col);
                acc += static_cast<typename TileAcc::DType>(aMatrix.data()[aIdx]) *
                       static_cast<typename TileAcc::DType>(bMatrix.data()[bIdx]);
            }
            cOutMatrix.data()[cIdx] = acc;
        }
    }
}

template <typename TileAcc, typename TileLeft, typename TileRight, typename TileBias>
PTO_INTERNAL void TMATMUL_BIAS_IMPL(TileAcc &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasMatrix)
{
    TMATMUL_IMPL(cMatrix, aMatrix, bMatrix);
    const uint16_t m = aMatrix.GetValidRow();
    const uint16_t n = bMatrix.GetValidCol();
    for (uint16_t row = 0; row < m; ++row) {
        for (uint16_t col = 0; col < n; ++col) {
            const std::size_t cIdx = gpu::GetTileElementOffset<TileAcc>(row, col);
            const std::size_t bIdx = gpu::GetTileElementOffset<TileBias>(0, col);
            cMatrix.data()[cIdx] += biasMatrix.data()[bIdx];
        }
    }
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TGEMV_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    (void)Phase;
    TMATMUL_IMPL(cMatrix, aMatrix, bMatrix);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TGEMV_ACC_IMPL(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    (void)Phase;
    TMATMUL_ACC_IMPL(cOutMatrix, cInMatrix, aMatrix, bMatrix);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight,
          typename TileBias>
PTO_INTERNAL void TGEMV_BIAS_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData)
{
    (void)Phase;
    TMATMUL_BIAS_IMPL(cMatrix, aMatrix, bMatrix, biasData);
}

} // namespace pto

#endif
