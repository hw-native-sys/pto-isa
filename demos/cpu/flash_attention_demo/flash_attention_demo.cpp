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
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <pto/pto-inst.hpp>

using namespace pto;

namespace {

constexpr int kExitSuccess = 0;
constexpr int kExitFailure = 1;

constexpr int kBatch = 1;
constexpr int kHeads = 2;
constexpr int kSeqLen = 64;
constexpr int kHeadDim = 32;

constexpr std::uint32_t kRngSeed = 20251220U;
constexpr float kInitScale = 0.02f;
constexpr float kRefEps = 2e-4f;
constexpr int kWarmupIters = 2;
constexpr int kIters = 20;

inline std::size_t idx4(int b, int h, int s, int d, int heads, int seq_len, int head_dim)
{
    return (static_cast<std::size_t>(b) * static_cast<std::size_t>(heads) + static_cast<std::size_t>(h)) *
               static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(head_dim) +
           static_cast<std::size_t>(s) * static_cast<std::size_t>(head_dim) + static_cast<std::size_t>(d);
}

inline float dot_qk(const float *q, const float *k, int d)
{
    float acc = 0.0f;
    for (int i = 0; i < d; ++i) {
        acc += q[i] * k[i];
    }
    return acc;
}

float max_abs_diff(const std::vector<float> &a, const std::vector<float> &b)
{
    float m = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::abs(a[i] - b[i]));
    }
    return m;
}

struct VerifyStats {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    float mean_abs = 0.0f;
    float rmse = 0.0f;
    std::size_t max_abs_idx = 0;
    std::size_t max_rel_idx = 0;
    std::size_t bad_count = 0;
    bool has_nan_or_inf = false;
};

VerifyStats verify_allclose(const std::vector<float> &actual, const std::vector<float> &ref, float abs_tol,
                            float rel_tol, float denom_eps)
{
    VerifyStats stats;
    if (actual.size() != ref.size() || actual.empty()) {
        stats.bad_count = std::max(actual.size(), ref.size());
        return stats;
    }

    double sum_abs = 0.0;
    double sum_sq = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float a = actual[i];
        const float r = ref[i];
        if (!std::isfinite(a) || !std::isfinite(r)) {
            stats.has_nan_or_inf = true;
            ++stats.bad_count;
            continue;
        }

        const float abs_diff = std::abs(a - r);
        const float denom = std::max(std::abs(r), denom_eps);
        const float rel_diff = abs_diff / denom;

        sum_abs += static_cast<double>(abs_diff);
        sum_sq += static_cast<double>(abs_diff) * static_cast<double>(abs_diff);

        if (abs_diff > stats.max_abs) {
            stats.max_abs = abs_diff;
            stats.max_abs_idx = i;
        }
        if (rel_diff > stats.max_rel) {
            stats.max_rel = rel_diff;
            stats.max_rel_idx = i;
        }

        const float allowed = abs_tol + rel_tol * std::abs(r);
        if (!(abs_diff <= allowed)) {
            ++stats.bad_count;
        }
    }

    const double n = static_cast<double>(actual.size());
    stats.mean_abs = static_cast<float>(sum_abs / n);
    stats.rmse = static_cast<float>(std::sqrt(sum_sq / n));
    return stats;
}

void flash_attention_reference(const std::vector<float> &q, const std::vector<float> &k, const std::vector<float> &v,
                               std::vector<float> &out, int batch, int heads, int seq_len, int head_dim, bool causal)
{
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int i = 0; i < seq_len; ++i) {
                const float *q_row = &q[idx4(b, h, i, 0, heads, seq_len, head_dim)];

                const int j_limit = causal ? (i + 1) : seq_len;
                std::vector<float> scores(static_cast<std::size_t>(j_limit), 0.0f);
                float max_score = -std::numeric_limits<float>::infinity();
                for (int j = 0; j < j_limit; ++j) {
                    const float *k_row = &k[idx4(b, h, j, 0, heads, seq_len, head_dim)];
                    const float score = dot_qk(q_row, k_row, head_dim) * scale;
                    scores[static_cast<std::size_t>(j)] = score;
                    max_score = std::max(max_score, score);
                }

                float denom = 0.0f;
                for (int j = 0; j < j_limit; ++j) {
                    scores[static_cast<std::size_t>(j)] = std::exp(scores[static_cast<std::size_t>(j)] - max_score);
                    denom += scores[static_cast<std::size_t>(j)];
                }
                const float inv_denom = (denom > 0.0f) ? (1.0f / denom) : 0.0f;

                float *o_row = &out[idx4(b, h, i, 0, heads, seq_len, head_dim)];
                for (int d = 0; d < head_dim; ++d) {
                    o_row[d] = 0.0f;
                }
                for (int j = 0; j < j_limit; ++j) {
                    const float w = scores[static_cast<std::size_t>(j)] * inv_denom;
                    const float *v_row = &v[idx4(b, h, j, 0, heads, seq_len, head_dim)];
                    for (int d = 0; d < head_dim; ++d) {
                        o_row[d] += w * v_row[d];
                    }
                }
            }
        }
    }
}

void flash_attention_pto(const std::vector<float> &q, const std::vector<float> &k, const std::vector<float> &v,
                         std::vector<float> &out)
{
    constexpr int kB = kBatch;
    constexpr int kH = kHeads;
    constexpr int kS = kSeqLen;
    constexpr int kD = kHeadDim;

    using GlobalQ = GlobalTensor<float, Shape<1, 1, 1, kS, kD>, Stride<kS * kD, kS * kD, kS * kD, kD, 1>>;
    using GlobalK = GlobalTensor<float, Shape<1, 1, 1, kS, kD>, Stride<kS * kD, kS * kD, kS * kD, kD, 1>>;
    using GlobalV = GlobalTensor<float, Shape<1, 1, 1, kS, kD>, Stride<kS * kD, kS * kD, kS * kD, kD, 1>>;
    using GlobalO = GlobalTensor<float, Shape<1, 1, 1, kS, kD>, Stride<kS * kD, kS * kD, kS * kD, kD, 1>>;

    using QPlain = Tile<TileType::Vec, float, kS, kD, BLayout::RowMajor, kS, kD, SLayout::NoneBox>;
    using KPlain = Tile<TileType::Vec, float, kS, kD, BLayout::RowMajor, kS, kD, SLayout::NoneBox>;
    using KTPlain = Tile<TileType::Vec, float, kD, kS, BLayout::RowMajor, kD, kS, SLayout::NoneBox>;
    using VPlain = Tile<TileType::Vec, float, kS, kD, BLayout::RowMajor, kS, kD, SLayout::NoneBox>;

    using ScoresPlain = Tile<TileType::Vec, float, kS, kS, BLayout::RowMajor, kS, kS, SLayout::NoneBox>;
    using RowReducePlain = Tile<TileType::Vec, float, kS, 1, BLayout::ColMajor, kS, 1, SLayout::NoneBox>;

    using LeftQ = TileLeft<float, kS, kD, kS, kD>;
    using RightKT = TileRight<float, kD, kS, kD, kS>;
    using AccScores = TileAcc<float, kS, kS, kS, kS>;

    using LeftP = TileLeft<float, kS, kS, kS, kS>;
    using RightV = TileRight<float, kS, kD, kS, kD>;
    using AccOut = TileAcc<float, kS, kD, kS, kD>;

    const float scale = 1.0f / std::sqrt(static_cast<float>(kD));

    for (int b = 0; b < kB; ++b) {
        for (int h = 0; h < kH; ++h) {
            const float *q_ptr = &q[idx4(b, h, 0, 0, kH, kS, kD)];
            const float *k_ptr = &k[idx4(b, h, 0, 0, kH, kS, kD)];
            const float *v_ptr = &v[idx4(b, h, 0, 0, kH, kS, kD)];
            float *o_ptr = &out[idx4(b, h, 0, 0, kH, kS, kD)];

            GlobalQ qGlobal(const_cast<float *>(q_ptr));
            GlobalK kGlobal(const_cast<float *>(k_ptr));
            GlobalV vGlobal(const_cast<float *>(v_ptr));
            GlobalO oGlobal(o_ptr);

            QPlain qTile;
            KPlain kTile;
            KTPlain ktTile;
            VPlain vTile;

            LeftQ qLeft;
            RightKT kRight;

            AccScores scoresAcc;
            ScoresPlain scores;

            RowReducePlain rowMax;
            ScoresPlain scoresCentered;
            ScoresPlain expScores;
            RowReducePlain rowSum;
            ScoresPlain probs;

            LeftP pLeft;
            RightV vRight;
            AccOut outAcc;

            TLOAD(qTile, qGlobal);
            TLOAD(kTile, kGlobal);
            TLOAD(vTile, vGlobal);

            TMOV(qLeft, qTile);
            TTRANS(ktTile, kTile, kTile);
            TMOV(kRight, ktTile);

            TMATMUL(scoresAcc, qLeft, kRight);
            TMOV(scores, scoresAcc);
            TMULS(scores, scores, scale);

            TROWMAX(rowMax, scores, scores);
            TROWEXPANDSUB(scoresCentered, scores, rowMax);
            TEXP(expScores, scoresCentered);
            TROWSUM(rowSum, expScores, expScores);
            TROWEXPANDDIV(probs, expScores, rowSum);

            TMOV(pLeft, probs);
            TMOV(vRight, vTile);
            TMATMUL(outAcc, pLeft, vRight);
            TSTORE(oGlobal, outAcc);
        }
    }
}

} // namespace

int main(int argc, char **argv)
{
    const bool causal_requested = (argc >= 2) && (std::string(argv[1]) == "--causal");
    const bool causal = false;
    std::cout << "flash_attention_demo: B=" << kBatch << " H=" << kHeads << " S=" << kSeqLen << " D=" << kHeadDim
              << (causal_requested ? " (--causal ignored; non-causal only)\n" : " (non-causal)\n");

    const std::size_t total = static_cast<std::size_t>(kBatch) * static_cast<std::size_t>(kHeads) *
                              static_cast<std::size_t>(kSeqLen) * static_cast<std::size_t>(kHeadDim);
    std::vector<float> q(total), k(total), v(total);

    std::mt19937 rng(kRngSeed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (std::size_t i = 0; i < total; ++i) {
        q[i] = dist(rng) * kInitScale;
        k[i] = dist(rng) * kInitScale;
        v[i] = dist(rng) * kInitScale;
    }

    std::vector<float> out_fused(total, 0.0f);
    std::vector<float> out_ref(total, 0.0f);

    for (int i = 0; i < kWarmupIters; ++i) {
        flash_attention_pto(q, k, v, out_fused);
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; ++i) {
        flash_attention_pto(q, k, v, out_fused);
    }
    const auto t1 = std::chrono::steady_clock::now();
    flash_attention_reference(q, k, v, out_ref, kBatch, kHeads, kSeqLen, kHeadDim, causal);

    const double elapsed_s =
        std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count() / static_cast<double>(kIters);
    const double matmul_flops = 4.0 * static_cast<double>(kBatch) * static_cast<double>(kHeads) *
                                static_cast<double>(kSeqLen) * static_cast<double>(kSeqLen) *
                                static_cast<double>(kHeadDim);
    const double gflops = (elapsed_s > 0.0) ? (matmul_flops / elapsed_s / 1e9) : 0.0;

    const float diff = max_abs_diff(out_fused, out_ref);
    std::cout << "max_abs_diff(pto, ref) = " << diff << "\n";

    constexpr float kAbsTol = 2e-4f;
    constexpr float kRelTol = 2e-4f;
    constexpr float kRelDenomEps = 1e-6f;
    const auto stats = verify_allclose(out_fused, out_ref, kAbsTol, kRelTol, kRelDenomEps);
    std::cout << "verify_allclose: abs_tol=" << kAbsTol << " rel_tol=" << kRelTol << " (bad=" << stats.bad_count
              << ", max_abs=" << stats.max_abs << " @idx=" << stats.max_abs_idx << ", max_rel=" << stats.max_rel
              << " @idx=" << stats.max_rel_idx << ", mean_abs=" << stats.mean_abs << ", rmse=" << stats.rmse << ")\n";

    if (stats.has_nan_or_inf) {
        std::cerr << "[FAIL] NaN/Inf detected\n";
        return kExitFailure;
    }
    if (!std::isfinite(diff) || diff > kRefEps || stats.bad_count != 0) {
        std::cerr << "[FAIL] verification failed\n";
        return kExitFailure;
    }

    float checksum = 0.0f;
    for (std::size_t i = 0; i < out_fused.size(); ++i) {
        checksum += out_fused[i];
    }
    std::cout << "checksum(out) = " << checksum << "\n";
    std::cout << "perf: avg_ms=" << (elapsed_s * 1e3) << " approx_matmul_flops=" << matmul_flops
              << " gflops=" << gflops;
    if (const char *peak_env = std::getenv("PTO_CPU_PEAK_GFLOPS")) {
        char *end = nullptr;
        const double peak = std::strtod(peak_env, &end);
        if (end != peak_env && peak > 0.0) {
            std::cout << " peak_gflops=" << peak << " mfu=" << (gflops / peak);
        }
    }
    std::cout << "\n";
    std::cout << "[PASS] flash_attention_demo\n";
    return kExitSuccess;
}
