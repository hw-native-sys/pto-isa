/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

template <typename T, typename U, typename S, int M, int N, int K, bool isAtranspose, bool isBtranspose>
AICORE inline void runTMOV(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    using GlobalDataSrc0 = std::conditional_t<
        isAtranspose,
        GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, 1, M>, Layout::DN>,
        GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>, Layout::ND>>;
    using GlobalDataSrc1 = std::conditional_t<
        isBtranspose,
        GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, 1, K>, Layout::DN>,
        GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, N, 1>, Layout::ND>>;
    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, M, N>, pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        std::conditional_t<isAtranspose, Tile<TileType::Mat, U, M, K, BLayout::RowMajor, M, K, SLayout::ColMajor, 512>,
                           Tile<TileType::Mat, U, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>>;
    using TileMatBData =
        std::conditional_t<isBtranspose, Tile<TileType::Mat, S, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>,
                           Tile<TileType::Mat, S, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>>;

    using LeftTile = TileLeft<U, M, K, M, K>;
    using RightTile = TileRight<S, K, N, K, N>;
    using AccTile = TileAcc<T, M, N, M, N>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
#endif
    /**********************************TMOV && TEXTRACT**********************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif
    TMATMUL(cTile, aTile, bTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
    /****************************************TSTORE*****************************************/
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, int M, int N, int K, bool isAtranspose, bool isBtranspose, int baseM,
          int baseN, int baseK, bool isKAlign = false>
AICORE inline void runTMOV_UNALIGN(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    // static shape
    using GlobalDataSrc0 = std::conditional_t<
        isAtranspose,
        GlobalTensor<U, pto::Shape<1, 1, 1, baseM, baseK>,
                     pto::Stride<baseM * baseK, baseM * baseK, baseM * baseK, 1, baseM>, Layout::DN>,
        GlobalTensor<U, pto::Shape<1, 1, 1, baseM, baseK>,
                     pto::Stride<baseM * baseK, baseM * baseK, baseM * baseK, baseK, 1>, Layout::ND>>;
    using GlobalDataSrc1 = std::conditional_t<
        isBtranspose,
        GlobalTensor<S, pto::Shape<1, 1, 1, baseK, baseN>,
                     pto::Stride<baseN * baseK, baseN * baseK, baseN * baseK, 1, baseK>, Layout::DN>,
        GlobalTensor<S, pto::Shape<1, 1, 1, baseK, baseN>,
                     pto::Stride<baseN * baseK, baseN * baseK, baseN * baseK, baseN, 1>, Layout::ND>>;
    using GlobalDataOut = GlobalTensor<T, Shape<1, 1, 1, M, N>, Stride<1 * M * N, 1 * M * N, M * N, N, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = std::conditional_t<
        isAtranspose, Tile<TileType::Mat, U, baseM, baseK, BLayout::RowMajor, baseM, baseK, SLayout::ColMajor, 512>,
        Tile<TileType::Mat, U, baseM, baseK, BLayout::ColMajor, baseM, baseK, SLayout::RowMajor, 512>>;
    using TileMatBData = std::conditional_t<
        isBtranspose, Tile<TileType::Mat, S, baseK, baseN, BLayout::RowMajor, baseK, baseN, SLayout::ColMajor, 512>,
        Tile<TileType::Mat, S, baseK, baseN, BLayout::ColMajor, baseK, baseN, SLayout::RowMajor, 512>>;

    using LeftTile = TileLeft<U, baseM, baseK, baseM, K>;
    using RightTile = TileRight<S, baseK, baseN, K, baseN>;
    using AccTile = TileAcc<T, baseM, baseN, M, N>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    LeftTile aTile;
    aTile.SetKAligned(isKAlign);
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    using AType = typename LeftTile::DType;
    using BType = typename RightTile::DType;
    using CType = typename AccTile::DType;

    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif
    /**********************************TMOV*******************************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif
    TMATMUL(cTile, aTile, bTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
    /****************************************TSTORE*****************************************/
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, int M, int N, int K, uint16_t indexM, uint16_t indexN, uint16_t indexK,
          bool isAtranspose, bool isBtranspose>
AICORE inline void runTEXTRACT(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    constexpr int mValid = M - indexM;
    constexpr int nValid = N - indexN;
    constexpr int kValid = K - indexK;
    // static shape
    using GlobalDataSrc0 = std::conditional_t<
        isAtranspose,
        GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, 1, M>, Layout::DN>,
        GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>, Layout::ND>>;
    using GlobalDataSrc1 = std::conditional_t<
        isBtranspose,
        GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, 1, K>, Layout::DN>,
        GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, N, 1>, Layout::ND>>;
    using GlobalDataOut =
        GlobalTensor<T, pto::Shape<1, 1, 1, mValid, nValid>,
                     pto::Stride<1 * mValid * nValid, 1 * mValid * nValid, mValid * nValid, nValid, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        std::conditional_t<isAtranspose, Tile<TileType::Mat, U, M, K, BLayout::RowMajor, M, K, SLayout::ColMajor, 512>,
                           Tile<TileType::Mat, U, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>>;
    using TileMatBData =
        std::conditional_t<isBtranspose, Tile<TileType::Mat, S, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>,
                           Tile<TileType::Mat, S, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>>;

    using LeftTile = TileLeft<U, mValid, kValid, mValid, kValid>;
    using RightTile = TileRight<S, kValid, nValid, kValid, nValid>;
    using AccTile = TileAcc<T, mValid, nValid, mValid, nValid>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif
    /**********************************TMOV && TEXTRACT**********************************/
    TEXTRACT(aTile, aMatTile, indexM, indexK);
    TEXTRACT(bTile, bMatTile, indexK, indexN);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif
    TMATMUL(cTile, aTile, bTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, int M, int N, int K, uint16_t indexM, uint16_t indexN, uint16_t indexK,
          bool isAtranspose, bool isBtranspose, int baseM, int baseN, int baseK, bool isKAlign = false>
AICORE inline void runTEXTRACT_UNALIGN(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    constexpr int mValid = M - indexM;
    constexpr int nValid = N - indexN;
    constexpr int kValid = K - indexK;

    using GlobalDataSrc0 = std::conditional_t<
        isAtranspose,
        GlobalTensor<U, pto::Shape<1, 1, 1, baseM, baseK>,
                     pto::Stride<baseM * baseK, baseM * baseK, baseM * baseK, 1, baseM>, Layout::DN>,
        GlobalTensor<U, pto::Shape<1, 1, 1, baseM, baseK>,
                     pto::Stride<baseM * baseK, baseM * baseK, baseM * baseK, baseK, 1>, Layout::ND>>;
    using GlobalDataSrc1 = std::conditional_t<
        isBtranspose,
        GlobalTensor<S, pto::Shape<1, 1, 1, baseK, baseN>,
                     pto::Stride<baseN * baseK, baseN * baseK, baseN * baseK, 1, baseK>, Layout::DN>,
        GlobalTensor<S, pto::Shape<1, 1, 1, baseK, baseN>,
                     pto::Stride<baseN * baseK, baseN * baseK, baseN * baseK, baseN, 1>, Layout::ND>>;
    using GlobalDataOut =
        GlobalTensor<T, pto::Shape<1, 1, 1, mValid, nValid>,
                     pto::Stride<1 * mValid * nValid, 1 * mValid * nValid, mValid * nValid, nValid, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = std::conditional_t<
        isAtranspose, Tile<TileType::Mat, U, baseM, baseK, BLayout::RowMajor, baseM, baseK, SLayout::ColMajor, 512>,
        Tile<TileType::Mat, U, baseM, baseK, BLayout::ColMajor, baseM, baseK, SLayout::RowMajor, 512>>;
    using TileMatBData = std::conditional_t<
        isBtranspose, Tile<TileType::Mat, S, baseK, baseN, BLayout::RowMajor, baseK, baseN, SLayout::ColMajor, 512>,
        Tile<TileType::Mat, S, baseK, baseN, BLayout::ColMajor, baseK, baseN, SLayout::RowMajor, 512>>;

    using LeftTile = TileLeft<U, baseM - indexM, baseK - indexK, baseM - indexM, kValid>;
    using RightTile = TileRight<S, baseK - indexK, baseN - indexN, kValid, baseN - indexN>;
    using AccTile = TileAcc<T, baseM - indexM, baseN - indexN, mValid, nValid>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    LeftTile aTile;
    aTile.SetKAligned(isKAlign);
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    using AType = typename LeftTile::DType;
    using BType = typename RightTile::DType;
    using CType = typename AccTile::DType;

    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
#endif
    /**********************************TEXTRACT*******************************************/
    TEXTRACT(aTile, aMatTile, indexM, indexK);
    TEXTRACT(bTile, bMatTile, indexK, indexN);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
#endif
    TMATMUL(cTile, aTile, bTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
#endif
    /****************************************TSTORE*****************************************/
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, int M, int N, int K, uint16_t indexM, uint16_t indexN, uint16_t indexK,
          bool isAtranspose, bool isBtranspose>
AICORE inline void runTEXTRACT_DYNAMIC(__gm__ T *out, __gm__ U *src0, __gm__ S *src1, int m, int n, int k)
{
    constexpr int mValid = M - indexM;
    constexpr int nValid = N - indexN;
    constexpr int kValid = K - indexK;
    // static shape
    using GlobalDataSrc0 = std::conditional_t<
        isAtranspose,
        GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, 1, M>, Layout::DN>,
        GlobalTensor<U, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>, Layout::ND>>;
    using GlobalDataSrc1 = std::conditional_t<
        isBtranspose,
        GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, 1, K>, Layout::DN>,
        GlobalTensor<S, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, N, 1>, Layout::ND>>;
    using GlobalDataOut =
        GlobalTensor<T, pto::Shape<1, 1, 1, mValid, nValid>,
                     pto::Stride<1 * mValid * nValid, 1 * mValid * nValid, mValid * nValid, nValid, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        std::conditional_t<isAtranspose,
                           Tile<TileType::Mat, U, M, K, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>,
                           Tile<TileType::Mat, U, M, K, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>>;
    using TileMatBData =
        std::conditional_t<isBtranspose,
                           Tile<TileType::Mat, S, K, N, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512>,
                           Tile<TileType::Mat, S, K, N, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>>;

    using LeftTile = TileLeft<U, mValid, kValid, -1, -1>;
    using RightTile = TileRight<S, kValid, nValid, kValid, -1>;
    using AccTile = TileAcc<T, mValid, nValid, -1, nValid>;

    TileMatAData aMatTile(m, k);
    TileMatBData bMatTile(k, n);
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    LeftTile aTile(mValid, kValid);
    RightTile bTile(nValid);
    AccTile cTile(mValid);
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif
    /**********************************TMOV && TEXTRACT**********************************/
    TEXTRACT(aTile, aMatTile, indexM, indexK);
    TEXTRACT(bTile, bMatTile, indexK, indexN);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif
    TMATMUL(cTile, aTile, bTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

template <typename T, typename U, typename S, int M, int N, int K, uint16_t indexM, uint16_t indexN, uint16_t indexK,
          bool isAtranspose, bool isBtranspose, int baseM, int baseN, int baseK, bool isKAlign = false>
AICORE inline void runTEXTRACT_COMPACT(__gm__ T *out, __gm__ U *src0, __gm__ S *src1)
{
    constexpr int mValid = M - indexM;
    constexpr int nValid = N - indexN;
    constexpr int kValid = K - indexK;

    using GlobalDataSrc0 = std::conditional_t<
        isAtranspose,
        GlobalTensor<U, pto::Shape<1, 1, 1, baseM, baseK>,
                     pto::Stride<baseM * baseK, baseM * baseK, baseM * baseK, 1, baseM>, Layout::DN>,
        GlobalTensor<U, pto::Shape<1, 1, 1, baseM, baseK>,
                     pto::Stride<baseM * baseK, baseM * baseK, baseM * baseK, baseK, 1>, Layout::ND>>;
    using GlobalDataSrc1 = std::conditional_t<
        isBtranspose,
        GlobalTensor<S, pto::Shape<1, 1, 1, baseK, baseN>,
                     pto::Stride<baseN * baseK, baseN * baseK, baseN * baseK, 1, baseK>, Layout::DN>,
        GlobalTensor<S, pto::Shape<1, 1, 1, baseK, baseN>,
                     pto::Stride<baseN * baseK, baseN * baseK, baseN * baseK, baseN, 1>, Layout::ND>>;
    using GlobalDataOut =
        GlobalTensor<T, pto::Shape<1, 1, 1, mValid, nValid>,
                     pto::Stride<1 * mValid * nValid, 1 * mValid * nValid, mValid * nValid, nValid, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = std::conditional_t<
        isAtranspose, Tile<TileType::Mat, U, baseM, baseK, BLayout::RowMajor, baseM, baseK, SLayout::ColMajor, 512>,
        Tile<TileType::Mat, U, baseM, baseK, BLayout::ColMajor, baseM, baseK, SLayout::RowMajor, 512>>;
    using TileMatBData = std::conditional_t<
        isBtranspose, Tile<TileType::Mat, S, baseK, baseN, BLayout::RowMajor, baseK, baseN, SLayout::ColMajor, 512>,
        Tile<TileType::Mat, S, baseK, baseN, BLayout::ColMajor, baseK, baseN, SLayout::RowMajor, 512>>;

    using LeftTile = TileLeftCompact<U, baseM - indexM, baseK - indexK, mValid, kValid>;
    using RightTile = TileRightCompact<S, baseK - indexK, baseN - indexN, kValid, nValid>;
    using AccTile = TileAccCompact<T, baseM - indexM, baseN - indexN, mValid, nValid>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    LeftTile aTile;
    aTile.SetKAligned(isKAlign);
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    using AType = typename LeftTile::DType;
    using BType = typename RightTile::DType;
    using CType = typename AccTile::DType;

    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif
    /**********************************TEXTRACT*******************************************/
    TEXTRACT(aTile, aMatTile, indexM, indexK);
    TEXTRACT(bTile, bMatTile, indexK, indexN);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif
    TMATMUL(cTile, aTile, bTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
    /****************************************TSTORE*****************************************/
    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTMOV_1(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 32;
    constexpr uint32_t K = 80;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTMOV<float, half, half, M, N, K, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
                                                                    reinterpret_cast<__gm__ half *>(src0),
                                                                    reinterpret_cast<__gm__ half *>(src1));
}

extern "C" __global__ AICORE void launchTMOV_2(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTMOV<int32_t, int8_t, int8_t, M, N, K, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ int32_t *>(out),
                                                                          reinterpret_cast<__gm__ int8_t *>(src0),
                                                                          reinterpret_cast<__gm__ int8_t *>(src1));
}

extern "C" __global__ AICORE void launchTMOV_3(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 48;
    constexpr uint32_t K = 64;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTMOV<float, float, float, M, N, K, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
                                                                      reinterpret_cast<__gm__ float *>(src0),
                                                                      reinterpret_cast<__gm__ float *>(src1));
}

extern "C" __global__ AICORE void launchTMOV_4(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 48;
    constexpr uint32_t K = 96;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTMOV<float, bfloat16_t, bfloat16_t, M, N, K, isAtranspose, isBtranspose>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ bfloat16_t *>(src0),
        reinterpret_cast<__gm__ bfloat16_t *>(src1));
}

extern "C" __global__ AICORE void launchTMOV_11(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;

    runTMOV<float, half, half, M, N, K, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
                                                                    reinterpret_cast<__gm__ half *>(src0),
                                                                    reinterpret_cast<__gm__ half *>(src1));
}

extern "C" __global__ AICORE void launchTMOV_12(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;

    runTMOV<int32_t, int8_t, int8_t, M, N, K, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ int32_t *>(out),
                                                                          reinterpret_cast<__gm__ int8_t *>(src0),
                                                                          reinterpret_cast<__gm__ int8_t *>(src1));
}

extern "C" __global__ AICORE void launchTMOV_13(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 32;
    constexpr uint32_t K = 96;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;

    runTMOV<float, float, float, M, N, K, isAtranspose, isBtranspose>(reinterpret_cast<__gm__ float *>(out),
                                                                      reinterpret_cast<__gm__ float *>(src0),
                                                                      reinterpret_cast<__gm__ float *>(src1));
}

extern "C" __global__ AICORE void launchTMOV_14(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 96;
    constexpr uint32_t N = 80;
    constexpr uint32_t K = 96;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;

    runTMOV<float, bfloat16_t, bfloat16_t, M, N, K, isAtranspose, isBtranspose>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ bfloat16_t *>(src0),
        reinterpret_cast<__gm__ bfloat16_t *>(src1));
}

extern "C" __global__ AICORE void launchTMOV_21(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 29;
    constexpr uint32_t N = 29;
    constexpr uint32_t K = 44;

    constexpr uint16_t baseM = 32;
    constexpr uint16_t baseN = 32;
    constexpr uint16_t baseK = 48;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;

    runTMOV_UNALIGN<float, float, float, M, N, K, isAtranspose, isBtranspose, baseM, baseN, baseK>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}

extern "C" __global__ AICORE void launchTMOV_22(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 29;
    constexpr uint32_t N = 29;
    constexpr uint32_t K = 36;

    constexpr uint16_t baseM = 32;
    constexpr uint16_t baseN = 32;
    constexpr uint16_t baseK = 48;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;
    constexpr bool isKAlign = true;

    runTMOV_UNALIGN<float, float, float, M, N, K, isAtranspose, isBtranspose, baseM, baseN, baseK, isKAlign>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}

extern "C" __global__ AICORE void launchTMOV_23(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 65;
    constexpr uint32_t N = 66;
    constexpr uint32_t K = 40;

    constexpr uint16_t baseM = 80;
    constexpr uint16_t baseN = 96;
    constexpr uint16_t baseK = 64;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;

    runTMOV_UNALIGN<int32_t, int8_t, int8_t, M, N, K, isAtranspose, isBtranspose, baseM, baseN, baseK>(
        reinterpret_cast<__gm__ int32_t *>(out), reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}

extern "C" __global__ AICORE void launchTMOV_24(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 65;
    constexpr uint32_t N = 82;
    constexpr uint32_t K = 40;

    constexpr uint16_t baseM = 80;
    constexpr uint16_t baseN = 96;
    constexpr uint16_t baseK = 64;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;

    runTMOV_UNALIGN<int32_t, int8_t, int8_t, M, N, K, isAtranspose, isBtranspose, baseM, baseN, baseK>(
        reinterpret_cast<__gm__ int32_t *>(out), reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}

extern "C" __global__ AICORE void launchTMOV_25(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 44;
    constexpr uint32_t N = 39;
    constexpr uint32_t K = 39;

    constexpr uint16_t baseM = 48;
    constexpr uint16_t baseN = 48;
    constexpr uint16_t baseK = 48;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;

    runTMOV_UNALIGN<float, bfloat16_t, bfloat16_t, M, N, K, isAtranspose, isBtranspose, baseM, baseN, baseK>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ bfloat16_t *>(src0),
        reinterpret_cast<__gm__ bfloat16_t *>(src1));
}

extern "C" __global__ AICORE void launchTMOV_31(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 29;
    constexpr uint32_t N = 29;
    constexpr uint32_t K = 44;

    constexpr uint16_t baseM = 32;
    constexpr uint16_t baseN = 32;
    constexpr uint16_t baseK = 48;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;

    runTMOV_UNALIGN<float, float, float, M, N, K, isAtranspose, isBtranspose, baseM, baseN, baseK>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}

extern "C" __global__ AICORE void launchTMOV_32(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 29;
    constexpr uint32_t N = 29;
    constexpr uint32_t K = 36;

    constexpr uint16_t baseM = 32;
    constexpr uint16_t baseN = 32;
    constexpr uint16_t baseK = 48;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;
    constexpr bool isKAlign = true;

    runTMOV_UNALIGN<float, float, float, M, N, K, isAtranspose, isBtranspose, baseM, baseN, baseK, isKAlign>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}

extern "C" __global__ AICORE void launchTMOV_33(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 65;
    constexpr uint32_t N = 66;
    constexpr uint32_t K = 40;

    constexpr uint16_t baseM = 96;
    constexpr uint16_t baseN = 80;
    constexpr uint16_t baseK = 64;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;

    runTMOV_UNALIGN<int32_t, int8_t, int8_t, M, N, K, isAtranspose, isBtranspose, baseM, baseN, baseK>(
        reinterpret_cast<__gm__ int32_t *>(out), reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}

extern "C" __global__ AICORE void launchTMOV_34(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 65;
    constexpr uint32_t N = 82;
    constexpr uint32_t K = 40;

    constexpr uint16_t baseM = 96;
    constexpr uint16_t baseN = 96;
    constexpr uint16_t baseK = 64;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;

    runTMOV_UNALIGN<int32_t, int8_t, int8_t, M, N, K, isAtranspose, isBtranspose, baseM, baseN, baseK>(
        reinterpret_cast<__gm__ int32_t *>(out), reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}

extern "C" __global__ AICORE void launchTMOV_35(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 44;
    constexpr uint32_t N = 39;
    constexpr uint32_t K = 39;

    constexpr uint16_t baseM = 48;
    constexpr uint16_t baseN = 48;
    constexpr uint16_t baseK = 48;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;

    runTMOV_UNALIGN<float, bfloat16_t, bfloat16_t, M, N, K, isAtranspose, isBtranspose, baseM, baseN, baseK>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ bfloat16_t *>(src0),
        reinterpret_cast<__gm__ bfloat16_t *>(src1));
}

template <int32_t tilingKey>
void launchTMOV(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        launchTMOV_1<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 2) {
        launchTMOV_2<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 3) {
        launchTMOV_3<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 4) {
        launchTMOV_4<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 11) {
        launchTMOV_11<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 12) {
        launchTMOV_12<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 13) {
        launchTMOV_13<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 14) {
        launchTMOV_14<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 21) {
        launchTMOV_21<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 22) {
        launchTMOV_22<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 23) {
        launchTMOV_23<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 24) {
        launchTMOV_24<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 25) {
        launchTMOV_25<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 31) {
        launchTMOV_31<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 32) {
        launchTMOV_32<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 33) {
        launchTMOV_33<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 34) {
        launchTMOV_34<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 35) {
        launchTMOV_35<<<1, nullptr, stream>>>(out, src0, src1);
    }
}
template void launchTMOV<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<11>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<12>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<13>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<14>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<21>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<22>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<23>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<24>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<25>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<31>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<32>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<33>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<34>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTMOV<35>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

extern "C" __global__ AICORE void launchTEXTRACT_1(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 32;
    constexpr uint32_t K = 80;

    constexpr uint16_t indexM = 16;
    constexpr uint16_t indexN = 16;
    constexpr uint16_t indexK = 32;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTEXTRACT<float, half, half, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}

extern "C" __global__ AICORE void launchTEXTRACT_2(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    constexpr uint16_t indexM = 48;
    constexpr uint16_t indexN = 32;
    constexpr uint16_t indexK = 64;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTEXTRACT<int32_t, int8_t, int8_t, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose>(
        reinterpret_cast<__gm__ int32_t *>(out), reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}

extern "C" __global__ AICORE void launchTEXTRACT_3(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 96;
    constexpr uint32_t N = 48;
    constexpr uint32_t K = 64;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexN = 16;
    constexpr uint16_t indexK = 48;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTEXTRACT<float, float, float, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_4(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 48;
    constexpr uint32_t K = 96;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexN = 32;
    constexpr uint16_t indexK = 16;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTEXTRACT<float, bfloat16_t, bfloat16_t, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ bfloat16_t *>(src0),
        reinterpret_cast<__gm__ bfloat16_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_11(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    constexpr uint16_t indexM = 96;
    constexpr uint16_t indexN = 32;
    constexpr uint16_t indexK = 64;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;

    runTEXTRACT<float, half, half, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1));
}

extern "C" __global__ AICORE void launchTEXTRACT_12(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexN = 32;
    constexpr uint16_t indexK = 32;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;

    runTEXTRACT<int32_t, int8_t, int8_t, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose>(
        reinterpret_cast<__gm__ int32_t *>(out), reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1));
}

extern "C" __global__ AICORE void launchTEXTRACT_13(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 32;
    constexpr uint32_t K = 96;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexN = 16;
    constexpr uint16_t indexK = 16;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;

    runTEXTRACT<float, float, float, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src0),
        reinterpret_cast<__gm__ float *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_14(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 96;
    constexpr uint32_t N = 80;
    constexpr uint32_t K = 96;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexN = 64;
    constexpr uint16_t indexK = 48;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;

    runTEXTRACT<float, bfloat16_t, bfloat16_t, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ bfloat16_t *>(src0),
        reinterpret_cast<__gm__ bfloat16_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_21(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 29;
    constexpr uint32_t N = 29;
    constexpr uint32_t K = 36;

    constexpr uint16_t indexM = 16;
    constexpr uint16_t indexN = 16;
    constexpr uint16_t indexK = 32;

    constexpr uint16_t baseM = 32;
    constexpr uint16_t baseN = 32;
    constexpr uint16_t baseK = 48;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;

    runTEXTRACT_UNALIGN<float, float, float, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose, baseM, baseN,
                        baseK>(reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src0),
                               reinterpret_cast<__gm__ float *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_22(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 65;
    constexpr uint32_t N = 66;
    constexpr uint32_t K = 40;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexN = 64;
    constexpr uint16_t indexK = 32;

    constexpr uint16_t baseM = 80;
    constexpr uint16_t baseN = 96;
    constexpr uint16_t baseK = 64;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;

    runTEXTRACT_UNALIGN<int32_t, int8_t, int8_t, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose, baseM,
                        baseN, baseK>(reinterpret_cast<__gm__ int32_t *>(out), reinterpret_cast<__gm__ int8_t *>(src0),
                                      reinterpret_cast<__gm__ int8_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_23(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 44;
    constexpr uint32_t N = 39;
    constexpr uint32_t K = 39;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexN = 16;
    constexpr uint16_t indexK = 32;

    constexpr uint16_t baseM = 48;
    constexpr uint16_t baseN = 48;
    constexpr uint16_t baseK = 48;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;

    runTEXTRACT_UNALIGN<float, bfloat16_t, bfloat16_t, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose,
                        baseM, baseN, baseK>(reinterpret_cast<__gm__ float *>(out),
                                             reinterpret_cast<__gm__ bfloat16_t *>(src0),
                                             reinterpret_cast<__gm__ bfloat16_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_31(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 29;
    constexpr uint32_t N = 29;
    constexpr uint32_t K = 36;

    constexpr uint16_t indexM = 16;
    constexpr uint16_t indexN = 16;
    constexpr uint16_t indexK = 32;

    constexpr uint16_t baseM = 32;
    constexpr uint16_t baseN = 32;
    constexpr uint16_t baseK = 48;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;

    runTEXTRACT_UNALIGN<float, float, float, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose, baseM, baseN,
                        baseK>(reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src0),
                               reinterpret_cast<__gm__ float *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_32(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 65;
    constexpr uint32_t N = 66;
    constexpr uint32_t K = 40;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexN = 64;
    constexpr uint16_t indexK = 32;

    constexpr uint16_t baseM = 96;
    constexpr uint16_t baseN = 80;
    constexpr uint16_t baseK = 64;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;

    runTEXTRACT_UNALIGN<int32_t, int8_t, int8_t, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose, baseM,
                        baseN, baseK>(reinterpret_cast<__gm__ int32_t *>(out), reinterpret_cast<__gm__ int8_t *>(src0),
                                      reinterpret_cast<__gm__ int8_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_33(__gm__ uint8_t *out, __gm__ uint8_t *src0, __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 44;
    constexpr uint32_t N = 39;
    constexpr uint32_t K = 39;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexN = 16;
    constexpr uint16_t indexK = 32;

    constexpr uint16_t baseM = 48;
    constexpr uint16_t baseN = 48;
    constexpr uint16_t baseK = 48;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;

    runTEXTRACT_UNALIGN<float, bfloat16_t, bfloat16_t, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose,
                        baseM, baseN, baseK>(reinterpret_cast<__gm__ float *>(out),
                                             reinterpret_cast<__gm__ bfloat16_t *>(src0),
                                             reinterpret_cast<__gm__ bfloat16_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_DYNAMIC_41(__gm__ uint8_t *out, __gm__ uint8_t *src0,
                                                            __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 32;
    constexpr uint32_t K = 80;

    constexpr uint16_t indexM = 16;
    constexpr uint16_t indexN = 0;
    constexpr uint16_t indexK = 32;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTEXTRACT_DYNAMIC<float, half, half, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ half *>(src0),
        reinterpret_cast<__gm__ half *>(src1), M, N, K);
}

extern "C" __global__ AICORE void launchTEXTRACT_DYNAMIC_42(__gm__ uint8_t *out, __gm__ uint8_t *src0,
                                                            __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 128;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexN = 0;
    constexpr uint16_t indexK = 32;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;

    runTEXTRACT_DYNAMIC<int32_t, int8_t, int8_t, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose>(
        reinterpret_cast<__gm__ int32_t *>(out), reinterpret_cast<__gm__ int8_t *>(src0),
        reinterpret_cast<__gm__ int8_t *>(src1), M, N, K);
}

template <int32_t tilingKey>
void launchTEXTRACT(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        launchTEXTRACT_1<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 2) {
        launchTEXTRACT_2<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 3) {
        launchTEXTRACT_3<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 4) {
        launchTEXTRACT_4<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 11) {
        launchTEXTRACT_11<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 12) {
        launchTEXTRACT_12<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 13) {
        launchTEXTRACT_13<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 14) {
        launchTEXTRACT_14<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 21) {
        launchTEXTRACT_21<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 22) {
        launchTEXTRACT_22<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 23) {
        launchTEXTRACT_23<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 31) {
        launchTEXTRACT_31<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 32) {
        launchTEXTRACT_32<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 33) {
        launchTEXTRACT_33<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 41) {
        launchTEXTRACT_DYNAMIC_41<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 42) {
        launchTEXTRACT_DYNAMIC_42<<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template void launchTEXTRACT<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<11>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<12>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<13>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<14>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<21>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<22>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<23>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<31>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<32>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<33>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<41>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT<42>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

extern "C" __global__ AICORE void launchTEXTRACT_COMPACT_1(__gm__ uint8_t *out, __gm__ uint8_t *src0,
                                                           __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 20;
    constexpr uint32_t N = 215;
    constexpr uint32_t K = 22;

    constexpr uint16_t indexM = 0;
    constexpr uint16_t indexN = 0;
    constexpr uint16_t indexK = 0;

    constexpr uint16_t baseM = 128;
    constexpr uint16_t baseN = 256;
    constexpr uint16_t baseK = 128;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;

    runTEXTRACT_COMPACT<float, float, float, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose, baseM, baseN,
                        baseK, true>(reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src0),
                                     reinterpret_cast<__gm__ float *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_COMPACT_2(__gm__ uint8_t *out, __gm__ uint8_t *src0,
                                                           __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 46;
    constexpr uint32_t N = 36;
    constexpr uint32_t K = 203;

    constexpr uint16_t indexM = 0;
    constexpr uint16_t indexN = 0;
    constexpr uint16_t indexK = 0;

    constexpr uint16_t baseM = 128;
    constexpr uint16_t baseN = 128;
    constexpr uint16_t baseK = 256;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;

    runTEXTRACT_COMPACT<int32_t, int8_t, int8_t, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose, baseM,
                        baseN, baseK>(reinterpret_cast<__gm__ int32_t *>(out), reinterpret_cast<__gm__ int8_t *>(src0),
                                      reinterpret_cast<__gm__ int8_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_COMPACT_3(__gm__ uint8_t *out, __gm__ uint8_t *src0,
                                                           __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 220;
    constexpr uint32_t N = 25;
    constexpr uint32_t K = 30;

    constexpr uint16_t indexM = 0;
    constexpr uint16_t indexN = 0;
    constexpr uint16_t indexK = 0;

    constexpr uint16_t baseM = 256;
    constexpr uint16_t baseN = 128;
    constexpr uint16_t baseK = 128;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = false;

    runTEXTRACT_COMPACT<float, bfloat16_t, bfloat16_t, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose,
                        baseM, baseN, baseK>(reinterpret_cast<__gm__ float *>(out),
                                             reinterpret_cast<__gm__ bfloat16_t *>(src0),
                                             reinterpret_cast<__gm__ bfloat16_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_COMPACT_11(__gm__ uint8_t *out, __gm__ uint8_t *src0,
                                                            __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 20;
    constexpr uint32_t N = 215;
    constexpr uint32_t K = 22;

    constexpr uint16_t indexM = 0;
    constexpr uint16_t indexN = 0;
    constexpr uint16_t indexK = 0;

    constexpr uint16_t baseM = 128;
    constexpr uint16_t baseN = 256;
    constexpr uint16_t baseK = 128;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTEXTRACT_COMPACT<float, float, float, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose, baseM, baseN,
                        baseK>(reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src0),
                               reinterpret_cast<__gm__ float *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_COMPACT_12(__gm__ uint8_t *out, __gm__ uint8_t *src0,
                                                            __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 46;
    constexpr uint32_t N = 36;
    constexpr uint32_t K = 203;

    constexpr uint16_t indexM = 0;
    constexpr uint16_t indexN = 0;
    constexpr uint16_t indexK = 0;

    constexpr uint16_t baseM = 128;
    constexpr uint16_t baseN = 128;
    constexpr uint16_t baseK = 256;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTEXTRACT_COMPACT<int32_t, int8_t, int8_t, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose, baseM,
                        baseN, baseK>(reinterpret_cast<__gm__ int32_t *>(out), reinterpret_cast<__gm__ int8_t *>(src0),
                                      reinterpret_cast<__gm__ int8_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_COMPACT_13(__gm__ uint8_t *out, __gm__ uint8_t *src0,
                                                            __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 220;
    constexpr uint32_t N = 25;
    constexpr uint32_t K = 30;

    constexpr uint16_t indexM = 0;
    constexpr uint16_t indexN = 0;
    constexpr uint16_t indexK = 0;

    constexpr uint16_t baseM = 256;
    constexpr uint16_t baseN = 128;
    constexpr uint16_t baseK = 128;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = true;

    runTEXTRACT_COMPACT<float, bfloat16_t, bfloat16_t, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose,
                        baseM, baseN, baseK>(reinterpret_cast<__gm__ float *>(out),
                                             reinterpret_cast<__gm__ bfloat16_t *>(src0),
                                             reinterpret_cast<__gm__ bfloat16_t *>(src1));
}

extern "C" __global__ AICORE void launchTEXTRACT_COMPACT_21(__gm__ uint8_t *out, __gm__ uint8_t *src0,
                                                            __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 36;
    constexpr uint32_t N = 215;
    constexpr uint32_t K = 22;

    constexpr uint16_t indexM = 16;
    constexpr uint16_t indexN = 16;
    constexpr uint16_t indexK = 16;

    constexpr uint16_t baseM = 128;
    constexpr uint16_t baseN = 256;
    constexpr uint16_t baseK = 128;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;

    runTEXTRACT_COMPACT<float, float, float, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose, baseM, baseN,
                        baseK, true>(reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src0),
                                     reinterpret_cast<__gm__ float *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_COMPACT_22(__gm__ uint8_t *out, __gm__ uint8_t *src0,
                                                            __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 46;
    constexpr uint32_t N = 36;
    constexpr uint32_t K = 203;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexN = 32;
    constexpr uint16_t indexK = 32;

    constexpr uint16_t baseM = 128;
    constexpr uint16_t baseN = 128;
    constexpr uint16_t baseK = 256;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;

    runTEXTRACT_COMPACT<int32_t, int8_t, int8_t, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose, baseM,
                        baseN, baseK>(reinterpret_cast<__gm__ int32_t *>(out), reinterpret_cast<__gm__ int8_t *>(src0),
                                      reinterpret_cast<__gm__ int8_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_COMPACT_23(__gm__ uint8_t *out, __gm__ uint8_t *src0,
                                                            __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 220;
    constexpr uint32_t N = 25;
    constexpr uint32_t K = 30;

    constexpr uint16_t indexM = 16;
    constexpr uint16_t indexN = 16;
    constexpr uint16_t indexK = 16;

    constexpr uint16_t baseM = 256;
    constexpr uint16_t baseN = 128;
    constexpr uint16_t baseK = 128;

    constexpr bool isAtranspose = false;
    constexpr bool isBtranspose = false;

    runTEXTRACT_COMPACT<float, bfloat16_t, bfloat16_t, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose,
                        baseM, baseN, baseK>(reinterpret_cast<__gm__ float *>(out),
                                             reinterpret_cast<__gm__ bfloat16_t *>(src0),
                                             reinterpret_cast<__gm__ bfloat16_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_COMPACT_31(__gm__ uint8_t *out, __gm__ uint8_t *src0,
                                                            __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 20;
    constexpr uint32_t N = 215;
    constexpr uint32_t K = 22;

    constexpr uint16_t indexM = 16;
    constexpr uint16_t indexN = 16;
    constexpr uint16_t indexK = 16;

    constexpr uint16_t baseM = 128;
    constexpr uint16_t baseN = 256;
    constexpr uint16_t baseK = 128;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;

    runTEXTRACT_COMPACT<float, float, float, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose, baseM, baseN,
                        baseK, true>(reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src0),
                                     reinterpret_cast<__gm__ float *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_COMPACT_32(__gm__ uint8_t *out, __gm__ uint8_t *src0,
                                                            __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 46;
    constexpr uint32_t N = 36;
    constexpr uint32_t K = 203;

    constexpr uint16_t indexM = 32;
    constexpr uint16_t indexN = 32;
    constexpr uint16_t indexK = 32;

    constexpr uint16_t baseM = 128;
    constexpr uint16_t baseN = 128;
    constexpr uint16_t baseK = 256;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;

    runTEXTRACT_COMPACT<int32_t, int8_t, int8_t, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose, baseM,
                        baseN, baseK>(reinterpret_cast<__gm__ int32_t *>(out), reinterpret_cast<__gm__ int8_t *>(src0),
                                      reinterpret_cast<__gm__ int8_t *>(src1));
}
extern "C" __global__ AICORE void launchTEXTRACT_COMPACT_33(__gm__ uint8_t *out, __gm__ uint8_t *src0,
                                                            __gm__ uint8_t *src1)
{
    constexpr uint32_t M = 220;
    constexpr uint32_t N = 25;
    constexpr uint32_t K = 30;

    constexpr uint16_t indexM = 16;
    constexpr uint16_t indexN = 16;
    constexpr uint16_t indexK = 16;

    constexpr uint16_t baseM = 256;
    constexpr uint16_t baseN = 128;
    constexpr uint16_t baseK = 128;

    constexpr bool isAtranspose = true;
    constexpr bool isBtranspose = true;

    runTEXTRACT_COMPACT<float, bfloat16_t, bfloat16_t, M, N, K, indexM, indexN, indexK, isAtranspose, isBtranspose,
                        baseM, baseN, baseK>(reinterpret_cast<__gm__ float *>(out),
                                             reinterpret_cast<__gm__ bfloat16_t *>(src0),
                                             reinterpret_cast<__gm__ bfloat16_t *>(src1));
}

template <int32_t tilingKey>
void launchTEXTRACT_COMPACT(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        launchTEXTRACT_COMPACT_1<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 2) {
        launchTEXTRACT_COMPACT_2<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 3) {
        launchTEXTRACT_COMPACT_3<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 11) {
        launchTEXTRACT_COMPACT_11<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 12) {
        launchTEXTRACT_COMPACT_12<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 13) {
        launchTEXTRACT_COMPACT_13<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 21) {
        launchTEXTRACT_COMPACT_21<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 22) {
        launchTEXTRACT_COMPACT_22<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 23) {
        launchTEXTRACT_COMPACT_23<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 31) {
        launchTEXTRACT_COMPACT_31<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 32) {
        launchTEXTRACT_COMPACT_32<<<1, nullptr, stream>>>(out, src0, src1);
    } else if constexpr (tilingKey == 33) {
        launchTEXTRACT_COMPACT_33<<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template void launchTEXTRACT_COMPACT<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT_COMPACT<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT_COMPACT<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT_COMPACT<11>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT_COMPACT<12>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT_COMPACT<13>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT_COMPACT<21>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT_COMPACT<22>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT_COMPACT<23>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT_COMPACT<31>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT_COMPACT<32>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void launchTEXTRACT_COMPACT<33>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
