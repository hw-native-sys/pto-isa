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

constexpr uint16_t BLOCK_CUBE_M_N = 16;
constexpr uint16_t BLOCK_ALIGN_BYTE = 32;

template <typename T>
AICORE constexpr inline T CeilAlign(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}

template <typename T>
AICORE constexpr inline T CeilDiv(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2;
}

template <Layout layoutType>
AICORE inline constexpr BLayout GetTileBLayout()
{
    if constexpr (layoutType == Layout::NZ || layoutType == Layout::DN) {
        return BLayout::ColMajor;
    } else {
        return BLayout::RowMajor;
    }
}

template <typename T>
using CType = typename std::conditional<std::is_same<T, int8_t>::value, int32_t, float>::type;

template <Layout layoutType>
AICORE inline constexpr SLayout GetTileSLayout()
{
    if constexpr (layoutType == Layout::NZ) {
        return SLayout::RowMajor;
    } else {
        return SLayout::NoneBox;
    }
}

template <typename DstTileData, typename SrcTileData>
__tf__ PTO_INTERNAL void tf_copy_cbuf_to_ubuf(typename DstTileData::TileDType __out__ dst,
                                              typename SrcTileData::TileDType __in__ src, uint16_t nBurst,
                                              uint16_t lenBurst)
{
    __ubuf__ typename DstTileData::DType *dstTileAddr = __cce_get_tile_ptr(dst);
    __cbuf__ typename SrcTileData::DType *srcTileAddr = __cce_get_tile_ptr(src);

    copy_cbuf_to_ubuf(dstTileAddr, srcTileAddr, 0, nBurst, lenBurst, 0, 0);
}

template <typename GlobalData, typename TileData>
__tf__ PTO_INTERNAL void tf_copy_ubuf_to_gm(typename GlobalData::DType __out__ *dst,
                                            typename TileData::TileDType __in__ src, int startDstAddr, int gShape0,
                                            int gStride0, uint16_t nBurst, uint32_t lenBurst, uint64_t burstDstStride,
                                            uint32_t burstSrcStride, int64_t tileStride)
{
    typename GlobalData::DType *dstAddr = dst;
    __ubuf__ typename TileData::DType *srcAddr = __cce_get_tile_ptr(src);
    typename GlobalData::DType *dstGlobalAddr = dstAddr;
    __ubuf__ typename TileData::DType *srcTileAddr = srcAddr;
    for (uint32_t k = 0; k < gShape0; k++) {
        dstGlobalAddr = dstAddr + k * gStride0;
        srcTileAddr = srcAddr + k * tileStride + startDstAddr;
        copy_ubuf_to_gm_align_v2(dstGlobalAddr, srcTileAddr, 0, nBurst, lenBurst, 0, burstDstStride, burstSrcStride);
    }
}

template <typename T, typename GlobalData, typename TileData>
AICORE inline void VecCopyOut(GlobalData &dst, TileData &src, int rows, int cols, int startDstAddr)
{
    constexpr uint32_t c0Size = 64;
    int gShape0 = dst.GetShape(0);
    int gShape1 = dst.GetShape(1);
    int gShape4 = dst.GetShape(4);
    int gStride0 = dst.GetStride(0);
    int gStride1 = dst.GetStride(1);

    uint16_t nBurst = gShape1;
    uint32_t lenBurst = rows * c0Size;
    uint64_t burstDstStride = gStride1 * sizeof(typename TileData::DType);
    uint32_t burstSrcStride = TileData::Rows * c0Size;
    int64_t tileStride = gShape1 * TileData::Rows * gShape4;
    tf_copy_ubuf_to_gm<GlobalData, TileData>(dst.data(), src.data(), startDstAddr, gShape0, gStride0, nBurst, lenBurst,
                                             burstDstStride, burstSrcStride, tileStride);
}

template <typename OutType, typename SrcTileData, int validM, int validN, Layout layoutType = Layout::ND,
          int sfractalSize = 512>
AICORE inline void RunTSTORE(__gm__ OutType *out, SrcTileData &srcTile)
{
    if constexpr (layoutType == Layout::ND) {
        using GlobalDataOut =
            GlobalTensor<OutType, pto::Shape<1, 1, 1, validM, validN>,
                         pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;
        GlobalDataOut dstGlobal(out);
        TSTORE(dstGlobal, srcTile);
    } else if constexpr (layoutType == Layout::DN) {
        using GlobalDataOut =
            GlobalTensor<OutType, pto::Shape<1, 1, 1, validM, validN>,
                         pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, 1, validM>, layoutType>;
        GlobalDataOut dstGlobal(out);
        TSTORE(dstGlobal, srcTile);
    } else if constexpr (layoutType == Layout::NZ) {
        constexpr uint16_t sGRows_ = 16;
        constexpr uint16_t sGCols_ = CeilDiv<uint16_t>(sfractalSize, sGRows_ * sizeof(OutType));
        constexpr uint16_t kGRows_ = CeilDiv<uint16_t>(validM, sGRows_);
        constexpr uint16_t kGCols_ = CeilDiv<uint16_t>(validN, sGCols_);
        using DynShapeDim5 = Shape<1, kGCols_, kGRows_, sGRows_, sGCols_>;
        using DynStridDim5 = pto::Stride<kGCols_ * kGRows_ * sGCols_ * sGRows_, kGRows_ * sGCols_ * sGRows_,
                                         sGCols_ * sGRows_, sGCols_, 1>;

        using GlobalDataOut = GlobalTensor<OutType, DynShapeDim5, DynStridDim5, layoutType>;
        GlobalDataOut dstGlobal(out);
        if (sfractalSize == 512) {
            TSTORE(dstGlobal, srcTile);
        } else {
            VecCopyOut<OutType, GlobalDataOut, SrcTileData>(dstGlobal, srcTile, validM, validN, 0);
        }
    }
}

template <typename T, typename DstTileData, typename SrcTileData, int row, int col>
AICORE inline void TMOVMat2Vec(DstTileData &dst, SrcTileData &src)
{
    uint16_t nBurst = 1;
    uint16_t lenBurst = row * col * sizeof(T) / 32;
    tf_copy_cbuf_to_ubuf<DstTileData, SrcTileData>(dst.data(), src.data(), nBurst, lenBurst);
}

template <typename OutType, typename AType, typename BType, int validM, int validK, int validN, int row, int col,
          bool isNZUnalign = false, bool isRelu = false, Layout layoutType = Layout::ND, int sfractalSize = 512,
          int indexRow = 0, int indexCol = 0, bool isInsert = false, int dstRow = 0, int dstCol = 0>
__global__ AICORE void RunTMOV(__gm__ OutType *out, __gm__ AType *src0, __gm__ BType *src1, __gm__ OutType *src2)
{
    constexpr int blockAlign = std::is_same_v<AType, int8_t> ? 32 : 16;
    constexpr int M = CeilAlign<int>(validM, blockAlign);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, blockAlign);
    constexpr int copyOutM = isInsert ? dstRow : (validM - indexRow);
    constexpr int copyOutN = isInsert ? dstCol : (validN - indexCol);

    constexpr int staticRow = isInsert ? copyOutM : row;
    constexpr int staticCol = isInsert ? copyOutN : col;

    using SrcTileData =
        std::conditional_t<isNZUnalign,
                           Tile<TileType::Mat, OutType, row, col, GetTileBLayout<layoutType>(), row, col,
                                GetTileSLayout<layoutType>(), sfractalSize>,
                           Tile<TileType::Mat, OutType, staticRow, staticCol, GetTileBLayout<layoutType>(), copyOutM,
                                copyOutN, GetTileSLayout<layoutType>(), sfractalSize>>;
    using DstTileData =
        std::conditional_t<isNZUnalign,
                           Tile<TileType::Vec, OutType, row, col, GetTileBLayout<layoutType>(), row, col,
                                GetTileSLayout<layoutType>(), sfractalSize>,
                           Tile<TileType::Vec, OutType, staticRow, staticCol, GetTileBLayout<layoutType>(), copyOutM,
                                copyOutN, GetTileSLayout<layoutType>(), sfractalSize>>;

    using GlobalDataSrc0 = std::conditional_t<
        isNZUnalign, GlobalTensor<AType, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>>,
        GlobalTensor<AType, pto::Shape<1, 1, 1, validM, validK>,
                     pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>>;

    using GlobalDataSrc1 = std::conditional_t<
        isNZUnalign, GlobalTensor<BType, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, N, 1>>,
        GlobalTensor<BType, pto::Shape<1, 1, 1, validK, validN>,
                     pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>>;

    using TileMatAData =
        std::conditional_t<isNZUnalign,
                           Tile<TileType::Mat, AType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>,
                           Tile<TileType::Mat, AType, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>>;

    using TileMatBData =
        std::conditional_t<isNZUnalign,
                           Tile<TileType::Mat, BType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>,
                           Tile<TileType::Mat, BType, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>>;

    using LeftTile =
        std::conditional_t<isNZUnalign, TileLeft<AType, M, K, M, K>, TileLeft<AType, M, K, validM, validK>>;

    using RightTile =
        std::conditional_t<isNZUnalign, TileRight<BType, K, N, K, N>, TileRight<BType, K, N, validK, validN>>;

    using AccTile = TileAcc<CType<AType>, M, N, validM, validN>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);

    TileMatAData aMatTile;
    TileMatBData bMatTile;

    AccTile cTile;

    LeftTile aTile;

    RightTile bTile;

    DstTileData dstTileData;

    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    TASSIGN(aTile, 0x0);

    TASSIGN(bTile, 0x0);

    TASSIGN(cTile, 0x0);

    TASSIGN(dstTileData, 0x0);

#if defined(__DAV_CUBE__)
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

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
#endif

    uint8_t syncId = 0;

#if defined(__DAV_CUBE__)
    SrcTileData srcTileData;
    TASSIGN(srcTileData, 0x0);
    if constexpr (isRelu) {
        TMOV<SrcTileData, AccTile, ReluPreMode::NormalRelu>(srcTileData, cTile);
    } else {
        if constexpr (indexRow == 0 && indexCol == 0) {
            TMOV(srcTileData, cTile);
        } else if constexpr (!isInsert) {
            TEXTRACT(srcTileData, cTile, indexRow, indexCol);
        } else {
            using GlobalDataSrc2 = GlobalTensor<
                OutType, pto::Shape<1, 1, 1, copyOutM, copyOutN>,
                pto::Stride<1 * copyOutM * copyOutN, 1 * copyOutM * copyOutN, copyOutM * copyOutN, copyOutN, 1>>;
            GlobalDataSrc2 src2Global(src2);
#ifndef __PTO_AUTO__
            set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
#endif
            TLOAD(srcTileData, src2Global);
#ifndef __PTO_AUTO__
            set_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID0);
#endif
            TINSERT(srcTileData, cTile, indexRow, indexCol);
        }
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
#endif

    TMOVMat2Vec<OutType, DstTileData, SrcTileData, staticRow, staticCol>(dstTileData, srcTileData);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_FIX, EVENT_ID0);
#endif
    set_intra_block(PIPE_MTE1, syncId);
    set_intra_block(PIPE_MTE1, syncId + 16);
#endif
#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, syncId);
    int64_t idx = get_block_idx() * get_subblockdim() + get_subblockid();
    if (idx == 0) {
        RunTSTORE<OutType, DstTileData, copyOutM, copyOutN, layoutType, sfractalSize>(out, dstTileData);
    }
#endif
}

template <typename OutType, typename AType, typename BType, typename FbType, int validM, int validK, int validN,
          int row, int col, bool isNZUnalign = false, bool isRelu = false, Layout layoutType = Layout::ND,
          int sfractalSize = 512, int indexRow = 0, int indexCol = 0, bool isInsert = false, int dstRow = 0,
          int dstCol = 0>
__global__ AICORE void RunTMOVFBQuant(__gm__ OutType *out, __gm__ AType *src0, __gm__ BType *src1, __gm__ FbType *src2,
                                      __gm__ OutType *src3)
{
    constexpr int blockAlign = std::is_same_v<AType, int8_t> ? 32 : 16;
    constexpr int M = CeilAlign<int>(validM, blockAlign);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, blockAlign);
    constexpr int copyOutM = isInsert ? dstRow : (validM - indexRow);
    constexpr int copyOutN = isInsert ? dstCol : (validN - indexCol);
    constexpr int staticRow = isInsert ? copyOutM : row;
    constexpr int staticCol = isInsert ? copyOutN : col;
    using SrcTileData =
        std::conditional_t<isNZUnalign,
                           Tile<TileType::Mat, OutType, row, col, GetTileBLayout<layoutType>(), row, col,
                                GetTileSLayout<layoutType>(), sfractalSize>,
                           Tile<TileType::Mat, OutType, staticRow, staticCol, GetTileBLayout<layoutType>(), copyOutM,
                                copyOutN, GetTileSLayout<layoutType>(), sfractalSize>>;
    using DstTileData =
        std::conditional_t<isNZUnalign,
                           Tile<TileType::Vec, OutType, row, col, GetTileBLayout<layoutType>(), row, col,
                                GetTileSLayout<layoutType>(), sfractalSize>,
                           Tile<TileType::Vec, OutType, staticRow, staticCol, GetTileBLayout<layoutType>(), copyOutM,
                                copyOutN, GetTileSLayout<layoutType>(), sfractalSize>>;

    using GlobalDataSrc0 = std::conditional_t<
        isNZUnalign, GlobalTensor<AType, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>>,
        GlobalTensor<AType, pto::Shape<1, 1, 1, validM, validK>,
                     pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>>;

    using GlobalDataSrc1 = std::conditional_t<
        isNZUnalign, GlobalTensor<BType, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, N, 1>>,
        GlobalTensor<BType, pto::Shape<1, 1, 1, validK, validN>,
                     pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>>;

    using TileMatAData =
        std::conditional_t<isNZUnalign,
                           Tile<TileType::Mat, AType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>,
                           Tile<TileType::Mat, AType, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>>;

    using TileMatBData =
        std::conditional_t<isNZUnalign,
                           Tile<TileType::Mat, BType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>,
                           Tile<TileType::Mat, BType, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>>;

    using LeftTile =
        std::conditional_t<isNZUnalign, TileLeft<AType, M, K, M, K>, TileLeft<AType, M, K, validM, validK>>;

    using RightTile =
        std::conditional_t<isNZUnalign, TileRight<BType, K, N, K, N>, TileRight<BType, K, N, validK, validN>>;

    using AccTile = TileAcc<CType<AType>, M, N, validM, validN>;

    using GlobalDataSrc2 =
        std::conditional_t<isNZUnalign,
                           GlobalTensor<FbType, pto::Shape<1, 1, 1, 1, N>, pto::Stride<1 * N, 1 * N, 1 * N, N, 1>>,
                           GlobalTensor<FbType, pto::Shape<1, 1, 1, 1, validN>,
                                        pto::Stride<1 * validN, 1 * validN, 1 * validN, validN, 1>>>;

    using TileMatFbData =
        std::conditional_t<isNZUnalign, Tile<TileType::Mat, FbType, 1, N, BLayout::RowMajor, 1, N, SLayout::NoneBox>,
                           Tile<TileType::Mat, FbType, 1, N, BLayout::RowMajor, 1, validN, SLayout::NoneBox>>;

    using FbTile = Tile<TileType::Scaling, FbType, 1, N, BLayout::RowMajor, 1, validN, SLayout::NoneBox>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileMatFbData fbMatTile;

    AccTile cTile;

    LeftTile aTile;

    RightTile bTile;

    DstTileData dstTileData;

    FbTile fbTile;

    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);
    TASSIGN(fbMatTile, 0x20000);

    TASSIGN(aTile, 0x0);

    TASSIGN(bTile, 0x0);

    TASSIGN(cTile, 0x0);

    TASSIGN(dstTileData, 0x0);

    TASSIGN(fbTile, 0x0);

#if defined(__DAV_CUBE__)
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

    if (src2 != nullptr) {
        GlobalDataSrc2 src2Global(src2);
        TLOAD(fbMatTile, src2Global);
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID0);
#endif

    TMOV(fbTile, fbMatTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

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

#endif

    uint8_t syncId = 0;

#if defined(__DAV_CUBE__)
    SrcTileData srcTileData;
    TASSIGN(srcTileData, 0x0);
    if constexpr (isRelu) {
        TMOV_FP<SrcTileData, AccTile, FbTile, ReluPreMode::NormalRelu>(srcTileData, cTile, fbTile);
    } else {
        TMOV_FP<SrcTileData, AccTile, FbTile>(srcTileData, cTile, fbTile);
        if constexpr (indexRow == 0 && indexCol == 0) {
            TMOV_FP<SrcTileData, AccTile, FbTile>(srcTileData, cTile, fbTile);
        } else if constexpr (!isInsert) {
            TEXTRACT_FP<SrcTileData, AccTile, FbTile>(srcTileData, cTile, fbTile, indexRow, indexCol);
        } else {
            using GlobalDataSrc3 = GlobalTensor<
                OutType, pto::Shape<1, 1, 1, copyOutM, copyOutN>,
                pto::Stride<1 * copyOutM * copyOutN, 1 * copyOutM * copyOutN, copyOutM * copyOutN, copyOutN, 1>>;
            GlobalDataSrc3 src3Global(src3);
#ifndef __PTO_AUTO__
            set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
#endif
            TLOAD(srcTileData, src3Global);
#ifndef __PTO_AUTO__
            set_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID0);
#endif
            TINSERT_FP<SrcTileData, AccTile, FbTile>(srcTileData, cTile, fbTile, indexRow, indexCol);
        }
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
#endif

    TMOVMat2Vec<OutType, DstTileData, SrcTileData, staticRow, staticCol>(dstTileData, srcTileData);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_FIX, EVENT_ID0);
#endif
    set_intra_block(PIPE_MTE1, syncId);
    set_intra_block(PIPE_MTE1, syncId + 16);
#endif
#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, syncId);
    int64_t idx = get_block_idx() * get_subblockdim() + get_subblockid();
    if (idx == 0) {
        RunTSTORE<OutType, DstTileData, copyOutM, copyOutN, layoutType, sfractalSize>(out, dstTileData);
    }
#endif
}

template <typename OutType, typename AType, typename BType, int validM, int validK, int validN, int row, int col,
          bool isNZUnalign = false, bool isRelu = false, Layout layoutType = Layout::ND, int sfractalSize = 512,
          int indexRow = 0, int indexCol = 0, bool isInsert = false, int dstRow = 0, int dstCol = 0>
__global__ AICORE void RunTMOVSCQuant(__gm__ OutType *out, __gm__ AType *src0, __gm__ BType *src1, __gm__ OutType *src2,
                                      float scalar)
{
    constexpr int blockAlign = std::is_same_v<AType, int8_t> ? 32 : 16;
    constexpr int M = CeilAlign<int>(validM, blockAlign);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, blockAlign);
    constexpr int copyOutM = isInsert ? dstRow : (validM - indexRow);
    constexpr int copyOutN = isInsert ? dstCol : (validN - indexCol);

    constexpr int staticRow = isInsert ? copyOutM : row;
    constexpr int staticCol = isInsert ? copyOutN : col;
    using SrcTileData =
        std::conditional_t<isNZUnalign,
                           Tile<TileType::Mat, OutType, row, col, GetTileBLayout<layoutType>(), row, col,
                                GetTileSLayout<layoutType>(), sfractalSize>,
                           Tile<TileType::Mat, OutType, staticRow, staticCol, GetTileBLayout<layoutType>(), copyOutM,
                                copyOutN, GetTileSLayout<layoutType>(), sfractalSize>>;
    using DstTileData =
        std::conditional_t<isNZUnalign,
                           Tile<TileType::Vec, OutType, row, col, GetTileBLayout<layoutType>(), row, col,
                                GetTileSLayout<layoutType>(), sfractalSize>,
                           Tile<TileType::Vec, OutType, staticRow, staticCol, GetTileBLayout<layoutType>(), copyOutM,
                                copyOutN, GetTileSLayout<layoutType>(), sfractalSize>>;

    using SrcTileData =
        std::conditional_t<isNZUnalign,
                           Tile<TileType::Mat, OutType, row, col, GetTileBLayout<layoutType>(), row, col,
                                GetTileSLayout<layoutType>(), sfractalSize>,
                           Tile<TileType::Mat, OutType, staticRow, staticCol, GetTileBLayout<layoutType>(), copyOutM,
                                copyOutN, GetTileSLayout<layoutType>(), sfractalSize>>;
    using DstTileData =
        std::conditional_t<isNZUnalign,
                           Tile<TileType::Vec, OutType, row, col, GetTileBLayout<layoutType>(), row, col,
                                GetTileSLayout<layoutType>(), sfractalSize>,
                           Tile<TileType::Vec, OutType, staticRow, staticCol, GetTileBLayout<layoutType>(), copyOutM,
                                copyOutN, GetTileSLayout<layoutType>(), sfractalSize>>;

    using GlobalDataSrc0 = std::conditional_t<
        isNZUnalign, GlobalTensor<AType, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>>,
        GlobalTensor<AType, pto::Shape<1, 1, 1, validM, validK>,
                     pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>>;

    using GlobalDataSrc1 = std::conditional_t<
        isNZUnalign, GlobalTensor<BType, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, N, 1>>,
        GlobalTensor<BType, pto::Shape<1, 1, 1, validK, validN>,
                     pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>>;

    using TileMatAData =
        std::conditional_t<isNZUnalign,
                           Tile<TileType::Mat, AType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>,
                           Tile<TileType::Mat, AType, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>>;

    using TileMatBData =
        std::conditional_t<isNZUnalign,
                           Tile<TileType::Mat, BType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>,
                           Tile<TileType::Mat, BType, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>>;

    using LeftTile =
        std::conditional_t<isNZUnalign, TileLeft<AType, M, K, M, K>, TileLeft<AType, M, K, validM, validK>>;

    using RightTile =
        std::conditional_t<isNZUnalign, TileRight<BType, K, N, K, N>, TileRight<BType, K, N, validK, validN>>;

    using AccTile = TileAcc<CType<AType>, M, N, validM, validN>;

    using GlobalDataSrc2 =
        std::conditional_t<isNZUnalign,
                           GlobalTensor<OutType, pto::Shape<1, 1, 1, 1, N>, pto::Stride<1 * N, 1 * N, 1 * N, N, 1>>,
                           GlobalTensor<OutType, pto::Shape<1, 1, 1, 1, validN>,
                                        pto::Stride<1 * validN, 1 * validN, 1 * validN, validN, 1>>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);

    TileMatAData aMatTile;
    TileMatBData bMatTile;

    AccTile cTile;

    LeftTile aTile;

    RightTile bTile;

    SrcTileData srcTileData;
    DstTileData dstTileData;

    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x10000);

    TASSIGN(aTile, 0x0);

    TASSIGN(bTile, 0x0);

    TASSIGN(cTile, 0x0);

    TASSIGN(srcTileData, 0x0);
    TASSIGN(dstTileData, 0x0);

#if defined(__DAV_CUBE__)
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

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
#endif

    uint8_t syncId = 0;

#if defined(__DAV_CUBE__)
    uint64_t preScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&scalar));
    if (sizeof(OutType) == 1) {
        constexpr bool sign = (std::is_same_v<typename DstTileData::DType, int8_t>) ? true : false;
        preScalar = (preScalar & ~(static_cast<uint64_t>(1) << 46)) | (static_cast<uint64_t>(sign) << 46);
    }
    if constexpr (isRelu) {
        TMOV<SrcTileData, AccTile, ReluPreMode::NormalRelu>(srcTileData, cTile, preScalar);
    } else {
        if constexpr (indexRow == 0 && indexCol == 0) {
            TMOV(srcTileData, cTile, preScalar);
        } else if constexpr (!isInsert) {
            TEXTRACT(srcTileData, cTile, preScalar, indexRow, indexCol);
        } else {
            using GlobalDataSrc2 = GlobalTensor<
                OutType, pto::Shape<1, 1, 1, copyOutM, copyOutN>,
                pto::Stride<1 * copyOutM * copyOutN, 1 * copyOutM * copyOutN, copyOutM * copyOutN, copyOutN, 1>>;
            GlobalDataSrc2 src2Global(src2);
#ifndef __PTO_AUTO__
            set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
#endif
            TLOAD(srcTileData, src2Global);
#ifndef __PTO_AUTO__
            set_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID0);
#endif
            TINSERT(srcTileData, cTile, preScalar, indexRow, indexCol);
        }
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
#endif

    TMOVMat2Vec<OutType, DstTileData, SrcTileData, staticRow, staticCol>(dstTileData, srcTileData);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_FIX, EVENT_ID0);
#endif
    set_intra_block(PIPE_MTE1, syncId);
    set_intra_block(PIPE_MTE1, syncId + 16);
#endif
#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, syncId);
    int64_t idx = get_block_idx() * get_subblockdim() + get_subblockid();
    if (idx == 0) {
        RunTSTORE<OutType, DstTileData, copyOutM, copyOutN, layoutType, sfractalSize>(out, dstTileData);
    }
#endif
}

template <int32_t tilingKey>
void LaunchTMOVAcc2MatNZ2NZ(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMOV<half, half, half, 96, 80, 112, 96, 112, false, false, Layout::NZ>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<half *>(src2));
    } else if constexpr (tilingKey == 2) {
        RunTMOV<float, half, half, 128, 64, 128, 128, 128, false, false, Layout::NZ, 1024>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<float *>(src2));
    } else if constexpr (tilingKey == 3) {
        RunTMOV<float, half, half, 13, 16, 9, 16, 16, true, true, Layout::NZ>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<float *>(src2));
    } else if constexpr (tilingKey == 4) {
        RunTMOV<bfloat16_t, half, half, 30, 128, 61, 32, 64, true, false, Layout::NZ>
            <<<1, nullptr, stream>>>(reinterpret_cast<bfloat16_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<bfloat16_t *>(src2));
    } else if constexpr (tilingKey == 5) {
        RunTMOV<half, half, half, 64, 64, 64, 64, 64, false, false, Layout::NZ, 512, 16, 16>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<half *>(src2));
    } else if constexpr (tilingKey == 6) {
        RunTMOV<half, half, half, 32, 32, 32, 32, 32, false, false, Layout::NZ, 512, 32, 32, true, 128, 128>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<half *>(src2));
    }
}

template void LaunchTMOVAcc2MatNZ2NZ<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMOVAcc2MatNZ2NZ<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMOVAcc2MatNZ2NZ<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMOVAcc2MatNZ2NZ<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMOVAcc2MatNZ2NZ<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMOVAcc2MatNZ2NZ<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);

template <int32_t tilingKey>
void LaunchTMOVAcc2MatNZ2ND(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMOV<half, half, half, 65, 40, 80, 80, 80, false, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<half *>(out));
    } else if constexpr (tilingKey == 2) {
        RunTMOV<float, half, half, 111, 48, 88, 112, 96>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<float *>(out));
    } else if constexpr (tilingKey == 3) {
        RunTMOV<bfloat16_t, half, half, 80, 128, 112, 80, 112>
            <<<1, nullptr, stream>>>(reinterpret_cast<bfloat16_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<bfloat16_t *>(out));
    } else if constexpr (tilingKey == 4) {
        RunTMOV<float, half, half, 6, 7, 8, 32, 32>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<float *>(out));
    }
}

template void LaunchTMOVAcc2MatNZ2ND<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2MatNZ2ND<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2MatNZ2ND<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2MatNZ2ND<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int32_t tilingKey>
void LaunchTMOVAcc2MatNZ2DN(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMOV<half, half, half, 80, 40, 66, 80, 80, false, false, Layout::DN>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<half *>(out));
    } else if constexpr (tilingKey == 2) {
        RunTMOV<float, half, half, 88, 48, 95, 96, 96, false, false, Layout::DN>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<float *>(out));
    } else if constexpr (tilingKey == 3) {
        RunTMOV<bfloat16_t, half, half, 48, 80, 60, 48, 64, false, true, Layout::DN>
            <<<1, nullptr, stream>>>(reinterpret_cast<bfloat16_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<bfloat16_t *>(out));
    } else if constexpr (tilingKey == 4) {
        RunTMOV<float, half, half, 8, 7, 6, 32, 32, false, true, Layout::DN>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<float *>(out));
    }
}

template void LaunchTMOVAcc2MatNZ2DN<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2MatNZ2DN<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2MatNZ2DN<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2MatNZ2DN<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int32_t tilingKey>
void LaunchTMOVAcc2MatFBQuantNZ2NZ(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                   void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMOVFBQuant<half, half, half, uint64_t, 128, 64, 64, 128, 64, false, false, Layout::NZ>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<uint64_t *>(src2),
                                     reinterpret_cast<half *>(src3));
    } else if constexpr (tilingKey == 2) {
        RunTMOVFBQuant<int8_t, half, half, uint64_t, 128, 64, 64, 128, 64, false, false, Layout::NZ>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<uint64_t *>(src2),
                                     reinterpret_cast<int8_t *>(src3));
    } else if constexpr (tilingKey == 3) {
        RunTMOVFBQuant<half, int8_t, int8_t, uint64_t, 121, 128, 63, 128, 64, true, true, Layout::NZ>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(src2),
                                     reinterpret_cast<half *>(src3));
    } else if constexpr (tilingKey == 4) {
        RunTMOVFBQuant<int8_t, int8_t, int8_t, uint64_t, 59, 128, 126, 64, 128, true, true, Layout::NZ>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(src2),
                                     reinterpret_cast<int8_t *>(src3));
    } else if constexpr (tilingKey == 5) {
        RunTMOVFBQuant<int8_t, half, half, uint64_t, 128, 64, 128, 128, 128, false, false, Layout::NZ, 512, 32, 32>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<uint64_t *>(src2),
                                     reinterpret_cast<int8_t *>(src3));
    } else if constexpr (tilingKey == 6) {
        RunTMOVFBQuant<int8_t, half, half, uint64_t, 128, 64, 128, 128, 128, false, false, Layout::NZ, 512, 32, 32,
                       true, 256, 256><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1),
            reinterpret_cast<uint64_t *>(src2), reinterpret_cast<int8_t *>(src3));
    }
}

template void LaunchTMOVAcc2MatFBQuantNZ2NZ<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                               void *stream);
template void LaunchTMOVAcc2MatFBQuantNZ2NZ<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                               void *stream);
template void LaunchTMOVAcc2MatFBQuantNZ2NZ<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                               void *stream);
template void LaunchTMOVAcc2MatFBQuantNZ2NZ<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                               void *stream);
template void LaunchTMOVAcc2MatFBQuantNZ2NZ<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                               void *stream);
template void LaunchTMOVAcc2MatFBQuantNZ2NZ<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                               void *stream);

template <int32_t tilingKey>
void LaunchTMOVAcc2MatFBQuantNZ2ND(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMOVFBQuant<half, half, half, uint64_t, 111, 47, 96, 112, 96, false, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1),
            reinterpret_cast<uint64_t *>(src2), reinterpret_cast<half *>(out));
    } else if constexpr (tilingKey == 2) {
        RunTMOVFBQuant<int8_t, half, half, uint64_t, 60, 128, 64, 64, 64, false, true><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1),
            reinterpret_cast<uint64_t *>(src2), reinterpret_cast<int8_t *>(out));
    } else if constexpr (tilingKey == 3) {
        RunTMOVFBQuant<half, int8_t, int8_t, uint64_t, 30, 48, 64, 32, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<int8_t *>(src0), reinterpret_cast<int8_t *>(src1),
            reinterpret_cast<uint64_t *>(src2), reinterpret_cast<half *>(out));
    } else if constexpr (tilingKey == 4) {
        RunTMOVFBQuant<int8_t, int8_t, int8_t, uint64_t, 60, 48, 32, 64, 32><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(src0), reinterpret_cast<int8_t *>(src1),
            reinterpret_cast<uint64_t *>(src2), reinterpret_cast<int8_t *>(out));
    }
}

template void LaunchTMOVAcc2MatFBQuantNZ2ND<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMOVAcc2MatFBQuantNZ2ND<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMOVAcc2MatFBQuantNZ2ND<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMOVAcc2MatFBQuantNZ2ND<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);

template <int32_t tilingKey>
void LaunchTMOVAcc2MatFBQuantNZ2DN(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMOVFBQuant<half, half, half, uint64_t, 80, 80, 80, 80, 80, false, false, Layout::DN>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<uint64_t *>(src2),
                                     reinterpret_cast<half *>(out));
    } else if constexpr (tilingKey == 2) {
        RunTMOVFBQuant<int8_t, half, half, uint64_t, 96, 128, 60, 96, 64, false, false, Layout::DN>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<uint64_t *>(src2),
                                     reinterpret_cast<int8_t *>(out));
    } else if constexpr (tilingKey == 3) {
        RunTMOVFBQuant<half, int8_t, int8_t, uint64_t, 64, 48, 60, 64, 64, false, true, Layout::DN>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(src2),
                                     reinterpret_cast<half *>(out));
    } else if constexpr (tilingKey == 4) {
        RunTMOVFBQuant<int8_t, int8_t, int8_t, uint64_t, 64, 64, 90, 64, 96, false, true, Layout::DN>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(src2),
                                     reinterpret_cast<int8_t *>(out));
    }
}

template void LaunchTMOVAcc2MatFBQuantNZ2DN<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMOVAcc2MatFBQuantNZ2DN<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMOVAcc2MatFBQuantNZ2DN<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMOVAcc2MatFBQuantNZ2DN<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);

template <int32_t tilingKey>
void LaunchTMOVAcc2MatSCQuantNZ2NZ(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMOVSCQuant<half, half, half, 112, 48, 96, 112, 96, false, true, Layout::NZ>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<half *>(src2), 4);
    } else if constexpr (tilingKey == 2) {
        RunTMOVSCQuant<int8_t, half, half, 112, 96, 64, 112, 128, false, true, Layout::NZ>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<int8_t *>(src2), 3);
    } else if constexpr (tilingKey == 3) {
        RunTMOVSCQuant<half, int8_t, int8_t, 27, 128, 58, 32, 64, true, false, Layout::NZ>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<half *>(src2), 5);
    } else if constexpr (tilingKey == 4) {
        RunTMOVSCQuant<int8_t, int8_t, int8_t, 58, 32, 61, 64, 64, true, false, Layout::NZ>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<int8_t *>(src2), 2);
    } else if constexpr (tilingKey == 5) {
        RunTMOVSCQuant<half, int8_t, int8_t, 96, 128, 64, 96, 64, false, false, Layout::NZ, 512, 48, 48>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<half *>(src2), 2);
    } else if constexpr (tilingKey == 6) {
        RunTMOVSCQuant<half, int8_t, int8_t, 96, 128, 64, 96, 64, false, false, Layout::NZ, 512, 48, 48, true, 256, 256>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<half *>(src2), 2);
    }
}

template void LaunchTMOVAcc2MatSCQuantNZ2NZ<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMOVAcc2MatSCQuantNZ2NZ<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMOVAcc2MatSCQuantNZ2NZ<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMOVAcc2MatSCQuantNZ2NZ<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMOVAcc2MatSCQuantNZ2NZ<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMOVAcc2MatSCQuantNZ2NZ<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);

template <int32_t tilingKey>
void LaunchTMOVAcc2MatSCQuantNZ2ND(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMOVSCQuant<half, half, half, 112, 48, 96, 112, 96>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<half *>(out), 4);
    } else if constexpr (tilingKey == 2) {
        RunTMOVSCQuant<int8_t, half, half, 60, 128, 64, 64, 64>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<int8_t *>(out), 3);
    } else if constexpr (tilingKey == 3) {
        RunTMOVSCQuant<half, int8_t, int8_t, 30, 48, 64, 32, 64, false, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<half *>(out), 5);
    } else if constexpr (tilingKey == 4) {
        RunTMOVSCQuant<int8_t, int8_t, int8_t, 60, 48, 32, 64, 32, false, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<int8_t *>(out), 2);
    }
}

template void LaunchTMOVAcc2MatSCQuantNZ2ND<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2MatSCQuantNZ2ND<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2MatSCQuantNZ2ND<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2MatSCQuantNZ2ND<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int32_t tilingKey>
void LaunchTMOVAcc2MatSCQuantNZ2DN(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMOVSCQuant<half, half, half, 80, 40, 66, 80, 80, false, true, Layout::DN>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<half *>(out), 4);
    } else if constexpr (tilingKey == 2) {
        RunTMOVSCQuant<int8_t, half, half, 96, 128, 60, 96, 64, false, true, Layout::DN>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<int8_t *>(out), 3);
    } else if constexpr (tilingKey == 3) {
        RunTMOVSCQuant<half, int8_t, int8_t, 128, 128, 64, 128, 64, false, false, Layout::DN>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<half *>(out), 5);
    } else if constexpr (tilingKey == 4) {
        RunTMOVSCQuant<int8_t, int8_t, int8_t, 64, 64, 90, 64, 96, false, false, Layout::DN>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<int8_t *>(out), 2);
    }
}

template void LaunchTMOVAcc2MatSCQuantNZ2DN<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2MatSCQuantNZ2DN<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2MatSCQuantNZ2DN<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2MatSCQuantNZ2DN<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
