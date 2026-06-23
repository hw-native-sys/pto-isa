/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cmath>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace {

constexpr int kSeqLen = 64;
constexpr int kHeadDim = 32;

} // namespace

__global__ AICORE void RunTFLASHATTN(__gm__ float *out, __gm__ float *q, __gm__ float *k, __gm__ float *v)
{
    using GlobalQ = GlobalTensor<float, Shape<1, 1, 1, kSeqLen, kHeadDim>,
                                 Stride<kSeqLen * kHeadDim, kSeqLen * kHeadDim, kSeqLen * kHeadDim, kHeadDim, 1>>;
    using GlobalK = GlobalTensor<float, Shape<1, 1, 1, kSeqLen, kHeadDim>,
                                 Stride<kSeqLen * kHeadDim, kSeqLen * kHeadDim, kSeqLen * kHeadDim, kHeadDim, 1>>;
    using GlobalV = GlobalTensor<float, Shape<1, 1, 1, kSeqLen, kHeadDim>,
                                 Stride<kSeqLen * kHeadDim, kSeqLen * kHeadDim, kSeqLen * kHeadDim, kHeadDim, 1>>;
    using GlobalO = GlobalTensor<float, Shape<1, 1, 1, kSeqLen, kHeadDim>,
                                 Stride<kSeqLen * kHeadDim, kSeqLen * kHeadDim, kSeqLen * kHeadDim, kHeadDim, 1>>;

    GlobalQ qGlobal(q);
    GlobalK kGlobal(k);
    GlobalV vGlobal(v);
    GlobalO oGlobal(out);

    using QPlain =
        Tile<TileType::Vec, float, kSeqLen, kHeadDim, BLayout::RowMajor, kSeqLen, kHeadDim, SLayout::NoneBox>;
    using KPlain =
        Tile<TileType::Vec, float, kSeqLen, kHeadDim, BLayout::RowMajor, kSeqLen, kHeadDim, SLayout::NoneBox>;
    using KTPlain =
        Tile<TileType::Vec, float, kHeadDim, kSeqLen, BLayout::RowMajor, kHeadDim, kSeqLen, SLayout::NoneBox>;
    using VPlain =
        Tile<TileType::Vec, float, kSeqLen, kHeadDim, BLayout::RowMajor, kSeqLen, kHeadDim, SLayout::NoneBox>;

    using ScoresPlain =
        Tile<TileType::Vec, float, kSeqLen, kSeqLen, BLayout::RowMajor, kSeqLen, kSeqLen, SLayout::NoneBox>;
    using RowReducePlain = Tile<TileType::Vec, float, kSeqLen, 1, BLayout::ColMajor, kSeqLen, 1, SLayout::NoneBox>;

    using LeftQ = TileLeft<float, kSeqLen, kHeadDim, kSeqLen, kHeadDim>;
    using RightKT = TileRight<float, kHeadDim, kSeqLen, kHeadDim, kSeqLen>;
    using AccScores = TileAcc<float, kSeqLen, kSeqLen, kSeqLen, kSeqLen>;

    using LeftP = TileLeft<float, kSeqLen, kSeqLen, kSeqLen, kSeqLen>;
    using RightV = TileRight<float, kSeqLen, kHeadDim, kSeqLen, kHeadDim>;
    using AccOut = TileAcc<float, kSeqLen, kHeadDim, kSeqLen, kHeadDim>;

    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

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

    // No direct Tile memory assignment is made (via TASSIGN)
    // So, __PTO_AUTO__ macro should be enabled in compiler definitions for auto memory assignment

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

    out = oGlobal.data();
}

void LaunchTFLASHATTN(float *out, float *q, float *k, float *v, void *stream)
{
    (void)stream;
    RunTFLASHATTN(out, q, k, v);
}
