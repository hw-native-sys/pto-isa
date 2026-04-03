/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_GPU_SM121_TMATMUL_HPP
#define PTO_GPU_SM121_TMATMUL_HPP

#include <type_traits>
#include <mma.h>
#include <pto/common/type.hpp>
#include "pto/gpu/common/tile_offsets.hpp"

namespace pto::gpu::sm121 {

namespace wmma = nvcuda::wmma;

constexpr int kMmaM = 16;
constexpr int kMmaN = 16;
constexpr int kMmaK = 16;

template <typename T>
PTO_INTERNAL float ToAccumFloat(T value)
{
    return static_cast<float>(value);
}

template <>
PTO_INTERNAL float ToAccumFloat<half>(half value)
{
    return __half2float(value);
}

template <>
PTO_INTERNAL float ToAccumFloat<bfloat16_t>(bfloat16_t value)
{
    return __bfloat162float(value);
}

PTO_INTERNAL float InlinePtxFma(float a, float b, float c)
{
    float out;
    asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(out) : "f"(a), "f"(b), "f"(c));
    return out;
}

PTO_INTERNAL unsigned LinearThreadId()
{
    return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}

PTO_INTERNAL unsigned ThreadsPerBlock()
{
    return blockDim.x * blockDim.y * blockDim.z;
}

template <typename TileAccOut, typename TileLeft, typename TileRight>
PTO_INTERNAL bool CanUseSm121TensorCoreCore(TileAccOut &cOutMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    using CType = typename TileAccOut::DType;
    using AType = typename TileLeft::DType;
    using BType = typename TileRight::DType;

    constexpr bool supportedTypes =
        std::is_same_v<CType, float> &&
        ((std::is_same_v<AType, half> && std::is_same_v<BType, half>) ||
         (std::is_same_v<AType, bfloat16_t> && std::is_same_v<BType, bfloat16_t>));

    if constexpr (!supportedTypes) {
        (void)cOutMatrix;
        (void)aMatrix;
        (void)bMatrix;
        return false;
    } else {
        if constexpr (!TileLeft::isRowMajor || !TileRight::isRowMajor || !TileAccOut::isRowMajor) {
            return false;
        }

        const uint16_t m = aMatrix.GetValidRow();
        const uint16_t k = aMatrix.GetValidCol();
        const uint16_t n = bMatrix.GetValidCol();
        if (k != bMatrix.GetValidRow()) {
            return false;
        }
        if ((m % kMmaM) != 0 || (n % kMmaN) != 0 || (k % kMmaK) != 0) {
            return false;
        }
        return ThreadsPerBlock() >= warpSize;
    }
}

template <bool UseAcc, bool UseBias, typename TileAccOut, typename TileLeft, typename TileRight,
          typename TileAccIn = TileAccOut, typename TileBias = TileAccOut>
PTO_INTERNAL bool TryTensorCoreTMATMULCore(TileAccOut &cOutMatrix, TileLeft &aMatrix, TileRight &bMatrix,
                                           TileAccIn *cInMatrix = nullptr, TileBias *biasMatrix = nullptr)
{
    using CType = typename TileAccOut::DType;
    using AType = typename TileLeft::DType;
    using BType = typename TileRight::DType;

    constexpr bool supportedTypes =
        std::is_same_v<CType, float> &&
        ((std::is_same_v<AType, half> && std::is_same_v<BType, half>) ||
         (std::is_same_v<AType, bfloat16_t> && std::is_same_v<BType, bfloat16_t>));

    if constexpr (!supportedTypes) {
        (void)cOutMatrix;
        (void)aMatrix;
        (void)bMatrix;
        (void)cInMatrix;
        (void)biasMatrix;
        return false;
    } else {
        if (!CanUseSm121TensorCoreCore(cOutMatrix, aMatrix, bMatrix)) {
            return false;
        }
        if constexpr (UseAcc) {
            if constexpr (!std::is_same_v<typename TileAccIn::DType, float> || !TileAccIn::isRowMajor) {
                return false;
            }
        }
        if constexpr (UseBias) {
            if constexpr (!std::is_same_v<typename TileBias::DType, float> || (TileBias::Rows != 1) ||
                          !TileBias::isRowMajor) {
                return false;
            }
        }

        const unsigned linearTid = LinearThreadId();
        const unsigned warpId = linearTid / warpSize;
        if (warpId != 0) {
            return true;
        }

        const uint16_t m = aMatrix.GetValidRow();
        const uint16_t k = aMatrix.GetValidCol();
        const uint16_t n = bMatrix.GetValidCol();
        const int lda = TileLeft::Cols;
        const int ldb = TileRight::Cols;
        const int ldc = TileAccOut::Cols;

        using FragA = wmma::fragment<wmma::matrix_a, kMmaM, kMmaN, kMmaK, AType, wmma::row_major>;
        using FragB = wmma::fragment<wmma::matrix_b, kMmaM, kMmaN, kMmaK, BType, wmma::row_major>;
        using FragC = wmma::fragment<wmma::accumulator, kMmaM, kMmaN, kMmaK, float>;

        for (uint16_t mBase = 0; mBase < m; mBase += kMmaM) {
            for (uint16_t nBase = 0; nBase < n; nBase += kMmaN) {
                FragC cFrag;
                if constexpr (UseAcc) {
                    float *cInPtr = cInMatrix->data() + gpu::GetTileElementOffset<TileAccIn>(mBase, nBase);
                    wmma::load_matrix_sync(cFrag, cInPtr, ldc, wmma::mem_row_major);
                } else {
                    wmma::fill_fragment(cFrag, 0.0f);
                }

                for (uint16_t kBase = 0; kBase < k; kBase += kMmaK) {
                    FragA aFrag;
                    FragB bFrag;
                    AType *aPtr = aMatrix.data() + gpu::GetTileElementOffset<TileLeft>(mBase, kBase);
                    BType *bPtr = bMatrix.data() + gpu::GetTileElementOffset<TileRight>(kBase, nBase);
                    wmma::load_matrix_sync(aFrag, aPtr, lda);
                    wmma::load_matrix_sync(bFrag, bPtr, ldb);
                    wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
                }

                float *cOutPtr = cOutMatrix.data() + gpu::GetTileElementOffset<TileAccOut>(mBase, nBase);
                wmma::store_matrix_sync(cOutPtr, cFrag, ldc, wmma::mem_row_major);

                if constexpr (UseBias) {
                    __syncwarp();
                    if ((linearTid & (warpSize - 1)) == 0) {
                        for (uint16_t row = 0; row < kMmaM; ++row) {
                            for (uint16_t col = 0; col < kMmaN; ++col) {
                                const std::size_t outIdx = row * ldc + col;
                                const std::size_t biasIdx = gpu::GetTileElementOffset<TileBias>(0, nBase + col);
                                cOutPtr[outIdx] += biasMatrix->data()[biasIdx];
                            }
                        }
                    }
                    __syncwarp();
                }
            }
        }
        return true;
    }
}

template <typename TileAcc, typename TileLeft, typename TileRight>
PTO_INTERNAL bool TryTensorCoreTMATMUL(TileAcc &cMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    return TryTensorCoreTMATMULCore<false, false>(cMatrix, aMatrix, bMatrix);
}

template <typename TileAcc, typename TileLeft, typename TileRight>
PTO_INTERNAL bool TryTensorCoreTMATMULAcc(TileAcc &cOutMatrix, TileAcc &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    return TryTensorCoreTMATMULCore<true, false, TileAcc, TileLeft, TileRight, TileAcc, TileAcc>(
        cOutMatrix, aMatrix, bMatrix, &cInMatrix, static_cast<TileAcc *>(nullptr));
}

template <typename TileAcc, typename TileLeft, typename TileRight>
PTO_INTERNAL bool TryTensorCoreTMATMULAcc(TileAcc &cMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    return TryTensorCoreTMATMULCore<true, false, TileAcc, TileLeft, TileRight, TileAcc, TileAcc>(
        cMatrix, aMatrix, bMatrix, &cMatrix, static_cast<TileAcc *>(nullptr));
}

template <typename TileAcc, typename TileLeft, typename TileRight, typename TileBias>
PTO_INTERNAL bool TryTensorCoreTMATMULBias(TileAcc &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasMatrix)
{
    return TryTensorCoreTMATMULCore<false, true, TileAcc, TileLeft, TileRight, TileAcc, TileBias>(
        cMatrix, aMatrix, bMatrix, static_cast<TileAcc *>(nullptr), &biasMatrix);
}

template <typename TileAcc, typename TileLeft, typename TileRight>
PTO_INTERNAL bool TryInlinePtxF32TMATMUL(TileAcc &cMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    using CType = typename TileAcc::DType;
    using AType = typename TileLeft::DType;
    using BType = typename TileRight::DType;

    constexpr bool supportedTypes = std::is_same_v<CType, float> &&
                                    ((std::is_same_v<AType, float> && std::is_same_v<BType, float>) ||
                                     (std::is_same_v<AType, half> && std::is_same_v<BType, half>) ||
                                     (std::is_same_v<AType, bfloat16_t> && std::is_same_v<BType, bfloat16_t>));
    if constexpr (!supportedTypes) {
        (void)cMatrix;
        (void)aMatrix;
        (void)bMatrix;
        return false;
    } else {
        const unsigned linearTid = LinearThreadId();
        if (linearTid != 0) {
            return ThreadsPerBlock() >= 1;
        }

        const uint16_t m = aMatrix.GetValidRow();
        const uint16_t k = aMatrix.GetValidCol();
        const uint16_t n = bMatrix.GetValidCol();
        if (k != bMatrix.GetValidRow()) {
            return false;
        }

#pragma unroll 1
        for (uint16_t row = 0; row < m; ++row) {
            for (uint16_t col = 0; col < n; ++col) {
                float acc = 0.0f;
#pragma unroll 4
                for (uint16_t kk = 0; kk < k; ++kk) {
                    const std::size_t aIdx = gpu::GetTileElementOffset<TileLeft>(row, kk);
                    const std::size_t bIdx = gpu::GetTileElementOffset<TileRight>(kk, col);
                    acc = InlinePtxFma(ToAccumFloat(aMatrix.data()[aIdx]), ToAccumFloat(bMatrix.data()[bIdx]), acc);
                }
                cMatrix.data()[gpu::GetTileElementOffset<TileAcc>(row, col)] = acc;
            }
        }
        return true;
    }
}

template <typename TileAcc, typename TileLeft, typename TileRight>
PTO_INTERNAL bool TrySm121TMATMUL(TileAcc &cMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    if (TryTensorCoreTMATMUL(cMatrix, aMatrix, bMatrix)) {
        return true;
    }
    return TryInlinePtxF32TMATMUL(cMatrix, aMatrix, bMatrix);
}

template <typename TileAcc, typename TileLeft, typename TileRight>
PTO_INTERNAL bool TrySm121TMATMULAcc(TileAcc &cOutMatrix, TileAcc &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    return TryTensorCoreTMATMULAcc(cOutMatrix, cInMatrix, aMatrix, bMatrix);
}

template <typename TileAcc, typename TileLeft, typename TileRight>
PTO_INTERNAL bool TrySm121TMATMULAcc(TileAcc &cMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    return TryTensorCoreTMATMULAcc(cMatrix, aMatrix, bMatrix);
}

template <typename TileAcc, typename TileLeft, typename TileRight, typename TileBias>
PTO_INTERNAL bool TrySm121TMATMULBias(TileAcc &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasMatrix)
{
    return TryTensorCoreTMATMULBias(cMatrix, aMatrix, bMatrix, biasMatrix);
}

} // namespace pto::gpu::sm121

#endif
