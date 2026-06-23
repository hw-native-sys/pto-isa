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
constexpr int kLatentDim = 16;

static_assert((kHeadDim % 8) == 0, "kHeadDim must satisfy NoneBox alignment");
static_assert((kSeqLen % 8) == 0, "kSeqLen must satisfy NoneBox alignment");
static_assert((kLatentDim % 8) == 0, "kLatentDim must satisfy NoneBox alignment");

constexpr std::uint32_t kRngSeed = 20251220U;
constexpr float kInitScale = 0.02f;

constexpr float kAbsTol = 2e-4f;
constexpr float kRelTol = 2e-4f;
constexpr float kRelDenomEps = 1e-6f;
constexpr int kWarmupIters = 2;
constexpr int kIters = 20;

inline std::size_t idx4(int b, int h, int s, int d, int heads, int seq_len, int head_dim)
{
    return (static_cast<std::size_t>(b) * static_cast<std::size_t>(heads) + static_cast<std::size_t>(h)) *
               static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(head_dim) +
           static_cast<std::size_t>(s) * static_cast<std::size_t>(head_dim) + static_cast<std::size_t>(d);
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

void mla_reference(const std::vector<float> &q, const std::vector<float> &k, const std::vector<float> &v,
                   const std::vector<float> &wq, const std::vector<float> &wk, const std::vector<float> &wv,
                   const std::vector<float> &wo, std::vector<float> &out, int batch, int heads, int seq_len,
                   int head_dim, int latent_dim)
{
    const double scale = 1.0 / std::sqrt(static_cast<double>(latent_dim));

    auto q_idx = [&](int b, int h, int s, int d) { return idx4(b, h, s, d, heads, seq_len, head_dim); };
    auto wq_idx = [&](int d, int r) { return static_cast<std::size_t>(d) * static_cast<std::size_t>(latent_dim) + r; };
    auto wo_idx = [&](int r, int d) { return static_cast<std::size_t>(r) * static_cast<std::size_t>(head_dim) + d; };

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            std::vector<double> ql(static_cast<std::size_t>(seq_len) * latent_dim, 0.0);
            std::vector<double> kl(static_cast<std::size_t>(seq_len) * latent_dim, 0.0);
            std::vector<double> vl(static_cast<std::size_t>(seq_len) * latent_dim, 0.0);

            for (int i = 0; i < seq_len; ++i) {
                for (int r = 0; r < latent_dim; ++r) {
                    double aq = 0.0;
                    double ak = 0.0;
                    double av = 0.0;
                    for (int d = 0; d < head_dim; ++d) {
                        const double qv = static_cast<double>(q[q_idx(b, h, i, d)]);
                        const double kv = static_cast<double>(k[q_idx(b, h, i, d)]);
                        const double vv = static_cast<double>(v[q_idx(b, h, i, d)]);
                        const double wqv = static_cast<double>(wq[wq_idx(d, r)]);
                        const double wkv = static_cast<double>(wk[wq_idx(d, r)]);
                        const double wvv = static_cast<double>(wv[wq_idx(d, r)]);
                        aq += qv * wqv;
                        ak += kv * wkv;
                        av += vv * wvv;
                    }
                    ql[static_cast<std::size_t>(i) * latent_dim + r] = aq;
                    kl[static_cast<std::size_t>(i) * latent_dim + r] = ak;
                    vl[static_cast<std::size_t>(i) * latent_dim + r] = av;
                }
            }

            std::vector<double> scores(static_cast<std::size_t>(seq_len) * seq_len, 0.0);
            for (int i = 0; i < seq_len; ++i) {
                for (int j = 0; j < seq_len; ++j) {
                    double acc = 0.0;
                    for (int r = 0; r < latent_dim; ++r) {
                        acc += ql[static_cast<std::size_t>(i) * latent_dim + r] *
                               kl[static_cast<std::size_t>(j) * latent_dim + r];
                    }
                    scores[static_cast<std::size_t>(i) * seq_len + j] = acc * scale;
                }
            }

            // Softmax over rows.
            for (int i = 0; i < seq_len; ++i) {
                double row_max = -std::numeric_limits<double>::infinity();
                for (int j = 0; j < seq_len; ++j) {
                    row_max = std::max(row_max, scores[static_cast<std::size_t>(i) * seq_len + j]);
                }
                double denom = 0.0;
                for (int j = 0; j < seq_len; ++j) {
                    const double e = std::exp(scores[static_cast<std::size_t>(i) * seq_len + j] - row_max);
                    scores[static_cast<std::size_t>(i) * seq_len + j] = e;
                    denom += e;
                }
                const double inv_denom = (denom > 0.0) ? (1.0 / denom) : 0.0;
                for (int j = 0; j < seq_len; ++j) {
                    scores[static_cast<std::size_t>(i) * seq_len + j] *= inv_denom;
                }
            }

            std::vector<double> ctx(static_cast<std::size_t>(seq_len) * latent_dim, 0.0);
            for (int i = 0; i < seq_len; ++i) {
                for (int r = 0; r < latent_dim; ++r) {
                    double acc = 0.0;
                    for (int j = 0; j < seq_len; ++j) {
                        acc += scores[static_cast<std::size_t>(i) * seq_len + j] *
                               vl[static_cast<std::size_t>(j) * latent_dim + r];
                    }
                    ctx[static_cast<std::size_t>(i) * latent_dim + r] = acc;
                }
            }

            for (int i = 0; i < seq_len; ++i) {
                for (int d = 0; d < head_dim; ++d) {
                    double acc = 0.0;
                    for (int r = 0; r < latent_dim; ++r) {
                        acc +=
                            ctx[static_cast<std::size_t>(i) * latent_dim + r] * static_cast<double>(wo[wo_idx(r, d)]);
                    }
                    out[q_idx(b, h, i, d)] = static_cast<float>(acc);
                }
            }
        }
    }
}

void mla_pto(const std::vector<float> &q, const std::vector<float> &k, const std::vector<float> &v,
             const std::vector<float> &wq, const std::vector<float> &wk, const std::vector<float> &wv,
             const std::vector<float> &wo, std::vector<float> &out)
{
    // Multi-Latent Attention (MLA) in PTO instructions (per batch/head):
    //   Q_lat = Q * Wq   where Q:(S×D), Wq:(D×R) -> Q_lat:(S×R)
    //   K_lat = K * Wk   where K:(S×D), Wk:(D×R) -> K_lat:(S×R)
    //   V_lat = V * Wv   where V:(S×D), Wv:(D×R) -> V_lat:(S×R)
    //   P = softmax((Q_lat * K_lat^T) / sqrt(R))  where P:(S×S)
    //   C_lat = P * V_lat                          where C_lat:(S×R)
    //   O = C_lat * Wo                             where Wo:(R×D), O:(S×D)
    //
    // Notes on PTO tiling:
    // - "Plain" tiles (SLayout::NoneBox) are used as temporary row-major buffers for elementwise ops.
    // - Matmul expects Left/Right/Acc tile roles; TMOV bridges between Plain and Matmul tiles.
    constexpr int kB = kBatch;
    constexpr int kH = kHeads;
    constexpr int kS = kSeqLen;
    constexpr int kD = kHeadDim;
    constexpr int kR = kLatentDim;

    using GlobalQ = GlobalTensor<float, Shape<1, 1, 1, kS, kD>, Stride<kS * kD, kS * kD, kS * kD, kD, 1>>;
    using GlobalK = GlobalTensor<float, Shape<1, 1, 1, kS, kD>, Stride<kS * kD, kS * kD, kS * kD, kD, 1>>;
    using GlobalV = GlobalTensor<float, Shape<1, 1, 1, kS, kD>, Stride<kS * kD, kS * kD, kS * kD, kD, 1>>;
    using GlobalO = GlobalTensor<float, Shape<1, 1, 1, kS, kD>, Stride<kS * kD, kS * kD, kS * kD, kD, 1>>;

    using GlobalWqr = GlobalTensor<float, Shape<1, 1, 1, kD, kR>, Stride<kD * kR, kD * kR, kD * kR, kR, 1>>;
    using GlobalWor = GlobalTensor<float, Shape<1, 1, 1, kR, kD>, Stride<kR * kD, kR * kD, kR * kD, kD, 1>>;

    using QPlain = Tile<TileType::Vec, float, kS, kD, BLayout::RowMajor, kS, kD, SLayout::NoneBox>;
    using KPlain = Tile<TileType::Vec, float, kS, kD, BLayout::RowMajor, kS, kD, SLayout::NoneBox>;
    using VPlain = Tile<TileType::Vec, float, kS, kD, BLayout::RowMajor, kS, kD, SLayout::NoneBox>;

    using WDrPlain = Tile<TileType::Vec, float, kD, kR, BLayout::RowMajor, kD, kR, SLayout::NoneBox>;
    using WRdPlain = Tile<TileType::Vec, float, kR, kD, BLayout::RowMajor, kR, kD, SLayout::NoneBox>;

    using SRPlain = Tile<TileType::Vec, float, kS, kR, BLayout::RowMajor, kS, kR, SLayout::NoneBox>;
    using RSTPlain = Tile<TileType::Vec, float, kR, kS, BLayout::RowMajor, kR, kS, SLayout::NoneBox>;

    using ScoresPlain = Tile<TileType::Vec, float, kS, kS, BLayout::RowMajor, kS, kS, SLayout::NoneBox>;
    using RowReducePlain = Tile<TileType::Vec, float, kS, 1, BLayout::ColMajor, kS, 1, SLayout::NoneBox>;

    using LeftSD = TileLeft<float, kS, kD, kS, kD>;
    using RightDR = TileRight<float, kD, kR, kD, kR>;
    using AccSR = TileAcc<float, kS, kR, kS, kR>;

    using LeftSR = TileLeft<float, kS, kR, kS, kR>;
    using RightRS = TileRight<float, kR, kS, kR, kS>;
    using AccSS = TileAcc<float, kS, kS, kS, kS>;

    using LeftSS = TileLeft<float, kS, kS, kS, kS>;
    using RightSD = TileRight<float, kS, kD, kS, kD>;
    using RightSR = TileRight<float, kS, kR, kS, kR>;
    using AccSD = TileAcc<float, kS, kD, kS, kD>;

    using RightRD = TileRight<float, kR, kD, kR, kD>;

    const float scale = 1.0f / std::sqrt(static_cast<float>(kR));

    // Load projection weights once (shared across all heads/batches in this demo).
    GlobalWqr wqGlobal(const_cast<float *>(wq.data()));
    GlobalWqr wkGlobal(const_cast<float *>(wk.data()));
    GlobalWqr wvGlobal(const_cast<float *>(wv.data()));
    GlobalWor woGlobal(const_cast<float *>(wo.data()));

    WDrPlain wqTile;
    WDrPlain wkTile;
    WDrPlain wvTile;
    WRdPlain woTile;
    RightDR wqRight;
    RightDR wkRight;
    RightDR wvRight;
    RightRD woRight;

    TLOAD(wqTile, wqGlobal);
    TLOAD(wkTile, wkGlobal);
    TLOAD(wvTile, wvGlobal);
    TLOAD(woTile, woGlobal);
    TMOV(wqRight, wqTile);
    TMOV(wkRight, wkTile);
    TMOV(wvRight, wvTile);
    TMOV(woRight, woTile);

    for (int b = 0; b < kB; ++b) {
        for (int h = 0; h < kH; ++h) {
            // 1) Load Q/K/V tiles for this (batch, head).
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
            VPlain vTile;
            TLOAD(qTile, qGlobal);
            TLOAD(kTile, kGlobal);
            TLOAD(vTile, vGlobal);

            LeftSD qLeft;
            LeftSD kLeft;
            LeftSD vLeft;
            TMOV(qLeft, qTile);
            TMOV(kLeft, kTile);
            TMOV(vLeft, vTile);

            // 2) Latent projections: (S×D) * (D×R) -> (S×R).
            AccSR qLatAcc;
            AccSR kLatAcc;
            AccSR vLatAcc;
            TMATMUL(qLatAcc, qLeft, wqRight);
            TMATMUL(kLatAcc, kLeft, wkRight);
            TMATMUL(vLatAcc, vLeft, wvRight);

            LeftSR qLatLeft;
            TMOV(qLatLeft, qLatAcc);

            // 3) Attention scores need K_lat^T (R×S) as the matmul RHS.
            SRPlain kLatPlain;
            RSTPlain kLatTPlain;
            RightRS kLatTRight;
            TMOV(kLatPlain, kLatAcc);
            TTRANS(kLatTPlain, kLatPlain, kLatPlain);
            TMOV(kLatTRight, kLatTPlain);

            // 4) Scores: (S×R) * (R×S) -> (S×S), then apply scaling 1/sqrt(R).
            AccSS scoresAcc;
            ScoresPlain scores;
            TMATMUL(scoresAcc, qLatLeft, kLatTRight);
            TMOV(scores, scoresAcc);
            TMULS(scores, scores, scale);

            // 5) Softmax over each row of scores:
            //    probs = exp(scores - row_max(scores)) / row_sum(exp(scores - row_max(scores))).
            RowReducePlain rowMax;
            ScoresPlain scoresCentered;
            ScoresPlain expScores;
            RowReducePlain rowSum;
            ScoresPlain probs;

            TROWMAX(rowMax, scores, scores);
            TROWEXPANDSUB(scoresCentered, scores, rowMax);
            TEXP(expScores, scoresCentered);
            TROWSUM(rowSum, expScores, expScores);
            TROWEXPANDDIV(probs, expScores, rowSum);

            // 6) Context (latent): (S×S) * (S×R) -> (S×R).
            LeftSS probsLeft;
            RightSR vLatRight;
            TMOV(probsLeft, probs);
            TMOV(vLatRight, vLatAcc);

            AccSR ctxLatAcc;
            TMATMUL(ctxLatAcc, probsLeft, vLatRight);

            // 7) Output projection: (S×R) * (R×D) -> (S×D), then store to global memory.
            LeftSR ctxLatLeft;
            TMOV(ctxLatLeft, ctxLatAcc);

            AccSD outAcc;
            TMATMUL(outAcc, ctxLatLeft, woRight);
            TSTORE(oGlobal, outAcc);
        }
    }
}

} // namespace

int main()
{
    std::cout << "mla_attention_demo: B=" << kBatch << " H=" << kHeads << " S=" << kSeqLen << " D=" << kHeadDim
              << " R=" << kLatentDim << "\n";

    const std::size_t qkv_total = static_cast<std::size_t>(kBatch) * static_cast<std::size_t>(kHeads) *
                                  static_cast<std::size_t>(kSeqLen) * static_cast<std::size_t>(kHeadDim);
    std::vector<float> q(qkv_total), k(qkv_total), v(qkv_total);

    const std::size_t w_dr = static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(kLatentDim);
    const std::size_t w_rd = static_cast<std::size_t>(kLatentDim) * static_cast<std::size_t>(kHeadDim);
    std::vector<float> wq(w_dr), wk(w_dr), wv(w_dr), wo(w_rd);

    std::mt19937 rng(kRngSeed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    auto fill_scaled = [&](std::vector<float> &buf) {
        for (std::size_t i = 0; i < buf.size(); ++i) {
            buf[i] = dist(rng) * kInitScale;
        }
    };

    fill_scaled(q);
    fill_scaled(k);
    fill_scaled(v);
    fill_scaled(wq);
    fill_scaled(wk);
    fill_scaled(wv);
    fill_scaled(wo);

    std::vector<float> out_pto(qkv_total, 0.0f);
    std::vector<float> out_ref(qkv_total, 0.0f);

    for (int i = 0; i < kWarmupIters; ++i) {
        mla_pto(q, k, v, wq, wk, wv, wo, out_pto);
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; ++i) {
        mla_pto(q, k, v, wq, wk, wv, wo, out_pto);
    }
    const auto t1 = std::chrono::steady_clock::now();
    mla_reference(q, k, v, wq, wk, wv, wo, out_ref, kBatch, kHeads, kSeqLen, kHeadDim, kLatentDim);

    const double elapsed_s =
        std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count() / static_cast<double>(kIters);
    const double matmul_flops =
        static_cast<double>(kBatch) * static_cast<double>(kHeads) *
        (8.0 * static_cast<double>(kSeqLen) * static_cast<double>(kHeadDim) * static_cast<double>(kLatentDim) +
         4.0 * static_cast<double>(kSeqLen) * static_cast<double>(kSeqLen) * static_cast<double>(kLatentDim));
    const double gflops = (elapsed_s > 0.0) ? (matmul_flops / elapsed_s / 1e9) : 0.0;

    const auto stats = verify_allclose(out_pto, out_ref, kAbsTol, kRelTol, kRelDenomEps);
    std::cout << "verify_allclose: abs_tol=" << kAbsTol << " rel_tol=" << kRelTol << " (bad=" << stats.bad_count
              << ", max_abs=" << stats.max_abs << " @idx=" << stats.max_abs_idx << ", max_rel=" << stats.max_rel
              << " @idx=" << stats.max_rel_idx << ", mean_abs=" << stats.mean_abs << ", rmse=" << stats.rmse << ")\n";

    if (stats.has_nan_or_inf || stats.bad_count != 0) {
        std::cerr << "[FAIL] mla_attention_demo\n";
        return kExitFailure;
    }

    float checksum = 0.0f;
    for (std::size_t i = 0; i < out_pto.size(); ++i) {
        checksum += out_pto[i];
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
    std::cout << "[PASS] mla_attention_demo\n";
    return kExitSuccess;
}
