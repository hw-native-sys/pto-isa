/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <pto/pto-inst.hpp>

using namespace pto;

namespace {

template <typename T>
T max_abs_diff(const std::vector<T>& a, const std::vector<T>& b)
{
    T m = T{};
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max<T>(m, std::abs(a[i] - b[i]));
    }
    return m;
}

void gemm_naive(const float* a, const float* b, float* c, int m, int k, int n)
{
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int k_idx = 0; k_idx < k; ++k_idx) {
                acc += a[i * k + k_idx] * b[k_idx * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}

} // namespace

int main()
{
    constexpr int kM = 32;
    constexpr int kK = 16;
    constexpr int kN = 32;
    constexpr int kAInitMod = 13;
    constexpr float kAInitScale = 0.1f;
    constexpr float kAInitBias = -0.5f;
    constexpr int kBInitMod = 7;
    constexpr float kBInitScale = 0.2f;
    constexpr float kBInitBias = -0.3f;
    constexpr float kZero = 0.0f;
    constexpr float kDiffThreshold = 1e-3f;
    constexpr uintptr_t kAddrAMat = 0x0;
    constexpr uintptr_t kAddrBMat = 0x10000;
    constexpr int kTileAlignBytes = 512;
    constexpr int kWarmupIters = 2;
    constexpr int kIters = 50;
    constexpr int kExitSuccess = 0;
    constexpr int kExitFailure = 1;

    std::vector<float> A(kM * kK);
    std::vector<float> B(kK * kN);
    std::vector<float> C(kM * kN, kZero);
    std::vector<float> Ref(kM * kN, kZero);

    for (int i = 0; i < kM * kK; ++i) {
        A[i] = (i % kAInitMod) * kAInitScale + kAInitBias;
    }
    for (int i = 0; i < kK * kN; ++i) {
        B[i] = (i % kBInitMod) * kBInitScale + kBInitBias;
    }

    using GlobalA = GlobalTensor<float, Shape<1, 1, 1, kM, kK>, Stride<1 * kM * kK, 1 * kM * kK, kM * kK, kK, 1>>;
    using GlobalB = GlobalTensor<float, Shape<1, 1, 1, kK, kN>, Stride<1 * kK * kN, 1 * kK * kN, kK * kN, kN, 1>>;
    using GlobalC = GlobalTensor<float, Shape<1, 1, 1, kM, kN>, Stride<1 * kM * kN, 1 * kM * kN, kM * kN, kN, 1>>;

    GlobalA aGlobal(A.data());
    GlobalB bGlobal(B.data());
    GlobalC cGlobal(C.data());

    using TileMatA = Tile<TileType::Mat, float, kM, kK, BLayout::ColMajor, kM, kK, SLayout::RowMajor, kTileAlignBytes>;
    using TileMatB = Tile<TileType::Mat, float, kK, kN, BLayout::ColMajor, kK, kN, SLayout::RowMajor, kTileAlignBytes>;
    using LeftTile = TileLeft<float, kM, kK, kM, kK>;
    using RightTile = TileRight<float, kK, kN, kK, kN>;
    using AccTile = TileAcc<float, kM, kN, kM, kN>;

    TileMatA aMat;
    TileMatB bMat;
    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;

    TASSIGN(aMat, kAddrAMat);
    TASSIGN(bMat, kAddrBMat);
    TASSIGN(aTile, kAddrAMat);
    TASSIGN(bTile, kAddrAMat);
    TASSIGN(cTile, kAddrAMat);

    TLOAD(aMat, aGlobal);
    TLOAD(bMat, bGlobal);
    TMOV(aTile, aMat);
    TMOV(bTile, bMat);
    for (int i = 0; i < kWarmupIters; ++i) {
        TMATMUL(cTile, aTile, bTile);
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; ++i) {
        TMATMUL(cTile, aTile, bTile);
    }
    const auto t1 = std::chrono::steady_clock::now();
    TSTORE(cGlobal, cTile);

    const double elapsed_s =
        std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count() / static_cast<double>(kIters);
    const double matmul_flops = 2.0 * static_cast<double>(kM) * static_cast<double>(kK) * static_cast<double>(kN);
    const double gflops = (elapsed_s > 0.0) ? (matmul_flops / elapsed_s / 1e9) : 0.0;

    gemm_naive(A.data(), B.data(), Ref.data(), kM, kK, kN);
    const float diff = max_abs_diff(C, Ref);

    std::cout << "gemm_demo: M=" << kM << " K=" << kK << " N=" << kN << "\n";
    std::cout << "max_abs_diff=" << diff << "\n";
    std::cout << "perf: avg_ms=" << (elapsed_s * 1e3) << " matmul_flops=" << matmul_flops << " gflops=" << gflops;
    if (const char* peak_env = std::getenv("PTO_CPU_PEAK_GFLOPS")) {
        char* end = nullptr;
        const double peak = std::strtod(peak_env, &end);
        if (end != peak_env && peak > 0.0) {
            std::cout << " peak_gflops=" << peak << " mfu=" << (gflops / peak);
        }
    }
    std::cout << "\n";
    return (diff < kDiffThreshold) ? kExitSuccess : kExitFailure;
}
