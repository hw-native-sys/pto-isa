/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

template <typename T>
using CType = typename std::conditional<std::is_same<T, int8_t>::value, int32_t, float>::type;

template <int subBlockId, int DualDstCtl>
AICORE inline constexpr uint8_t getMode()
{
    if constexpr (DualDstCtl == 0) {
        return subBlockId;
    }
    return 1 + DualDstCtl;
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

template <typename AType, typename BType, typename fbType, int M, int K, int N, int validM, int validK, int validN>
AICORE inline void RunMATMUL(__gm__ AType *src0, __gm__ BType *src1, __gm__ fbType *src2)
{
    using GlobalDataSrc0 =
        GlobalTensor<AType, pto::Shape<1, 1, 1, validM, validK>,
                     pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 =
        GlobalTensor<BType, pto::Shape<1, 1, 1, validK, validN>,
                     pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);

    using TileMatAData = Tile<TileType::Mat, AType, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, BType, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x20000);
    TASSIGN(bMatTile, 0x10000);

    using LeftTile = TileLeft<AType, M, K, validM, validK>;
    using RightTile = TileRight<BType, K, N, validK, validN>;
    using AccTile = TileAcc<CType<AType>, M, N, validM, validN>;
    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
#if defined(__DAV_CUBE__)
    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    /**********************************TMOV && TEXTRACT**********************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    /**********************************TMATMUL**********************************/
    TMATMUL(cTile, aTile, bTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
#endif
}

template <typename AType, typename BType, typename fbType, int M, int K, int N, int validM, int validK, int validN>
AICORE inline void RunMATMUL_NZUNALIGN(__gm__ AType *src0, __gm__ BType *src1, __gm__ fbType *src2)
{
    using GlobalDataSrc0 =
        GlobalTensor<AType, pto::Shape<1, 1, 1, M, K>, pto::Stride<1 * M * K, 1 * M * K, M * K, K, 1>>;
    using GlobalDataSrc1 =
        GlobalTensor<BType, pto::Shape<1, 1, 1, K, N>, pto::Stride<1 * K * N, 1 * K * N, K * N, N, 1>>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);

    using TileMatAData = Tile<TileType::Mat, AType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, BType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x20000);
    TASSIGN(bMatTile, 0x10000);

    using LeftTile = TileLeft<AType, M, K, M, K>;
    using RightTile = TileRight<BType, K, N, K, N>;
    using AccTile = TileAcc<CType<AType>, M, N, validM, validN>;
    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
#if defined(__DAV_CUBE__)
    /*************************************TLOAD****************************************/
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    /**********************************TMOV && TEXTRACT**********************************/
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif
    /**********************************TMATMUL**********************************/
    TMATMUL(cTile, aTile, bTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
#endif
}

template <typename T, typename GlobalData, typename TileData>
AICORE inline void UBCopyOut(GlobalData &dst, TileData &src, int rows, int cols, int startDstAddr)
{
    constexpr uint32_t c0Size = 64;
    int gShape0 = dst.GetShape(pto::GlobalTensorDim::DIM_0);
    int gShape1 = dst.GetShape(pto::GlobalTensorDim::DIM_1);
    int gShape4 = dst.GetShape(pto::GlobalTensorDim::DIM_4);
    int gStride0 = dst.GetStride(pto::GlobalTensorDim::DIM_0);
    int gStride1 = dst.GetStride(pto::GlobalTensorDim::DIM_1);

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
            UBCopyOut<OutType, GlobalDataOut, SrcTileData>(dstGlobal, srcTile, validM, validN, 0);
        }
    }
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

template <Layout layoutType>
AICORE inline constexpr SLayout GetTileSLayout()
{
    if constexpr (layoutType == Layout::NZ) {
        return SLayout::RowMajor;
    } else {
        return SLayout::NoneBox;
    }
}

template <typename OutType, Layout layoutType, int staticRow, int staticCol, int sfractalSize>
AICORE inline void RunInitLoad(__gm__ OutType *src2)
{
#if defined(__DAV_VEC__)
    using GlobalDataSrc2 = GlobalTensor<
        OutType, pto::Shape<1, 1, 1, staticRow, staticCol>,
        pto::Stride<1 * staticRow * staticCol, 1 * staticRow * staticCol, staticRow * staticCol, staticCol, 1>>;
    GlobalDataSrc2 src2Global(src2);
    using DstInitTileData = Tile<TileType::Vec, OutType, staticRow, staticCol, GetTileBLayout<layoutType>(), staticRow,
                                 staticCol, GetTileSLayout<layoutType>(), sfractalSize>;
    DstInitTileData dstTile1Data;
    TASSIGN(dstTile1Data, 0x0);
    TLOAD(dstTile1Data, src2Global);
#endif
}

template <typename OutType, typename AType, typename BType, int validM, int validK, int validN, int row, int col,
          int subBlockId = 0, bool isNZUnalign = false, bool isRelu = false, Layout layoutType = Layout::ND,
          int sfractalSize = 512, int indexRow = 0, int indexCol = 0, bool isInsert = false, int staticRow = 0,
          int staticCol = 0>
__global__ AICORE void RunTMOV(__gm__ OutType *out, __gm__ AType *src0, __gm__ BType *src1, __gm__ OutType *src2)
{
    constexpr int blockAlign = std::is_same_v<AType, int8_t> ? 32 : 16;
    constexpr int M = CeilAlign<int>(validM, blockAlign);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, blockAlign);
    constexpr int copyOutM = validM - indexRow;
    constexpr int copyOutN = validN - indexCol;
    if constexpr (!isNZUnalign) {
        RunMATMUL<AType, BType, OutType, M, K, N, validM, validK, validN>(src0, src1, nullptr);
    } else {
        RunMATMUL_NZUNALIGN<AType, BType, OutType, M, K, N, validM, validK, validN>(src0, src1, nullptr);
    }
    using AccTile = TileAcc<CType<AType>, M, N, validM, validN>;
    AccTile cTile;
    TASSIGN(cTile, 0x0);

    uint8_t syncId = 0;
    using DstTileData =
        std::conditional_t<isNZUnalign,
                           Tile<TileType::Vec, OutType, row, col, GetTileBLayout<layoutType>(), row, col,
                                GetTileSLayout<layoutType>(), sfractalSize>,
                           Tile<TileType::Vec, OutType, staticRow, staticCol, GetTileBLayout<layoutType>(), copyOutM,
                                copyOutN, GetTileSLayout<layoutType>(), sfractalSize>>;
    DstTileData dstTileData;
    TASSIGN(dstTileData, 0x0);

#if defined(__DAV_VEC__)
    RunInitLoad<OutType, layoutType, staticRow, staticCol, sfractalSize>(src2);
    set_intra_block(PIPE_MTE2, 0x1);
#endif

#if defined(__DAV_CUBE__)
    wait_intra_block(PIPE_FIX, 0x1);
    constexpr uint8_t mode = getMode<subBlockId, 0>();
    if constexpr (subBlockId == 0) {
        if constexpr (isRelu) {
            TEXTRACT<DstTileData, AccTile, ReluPreMode::NormalRelu>(dstTileData, cTile, indexRow, indexCol);
        } else {
            TEXTRACT(dstTileData, cTile, indexRow, indexCol);
        }
    } else {
        if constexpr (isRelu) {
            TEXTRACT<DstTileData, AccTile, static_cast<AccToVecMode>(mode), ReluPreMode::NormalRelu>(
                dstTileData, cTile, indexRow, indexCol);
        } else {
            TEXTRACT<DstTileData, AccTile, static_cast<AccToVecMode>(mode)>(dstTileData, cTile, indexRow, indexCol);
        }
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_FIX, EVENT_ID0);
#endif
    set_intra_block(PIPE_FIX, syncId);
    set_intra_block(PIPE_FIX, syncId + 16);
#endif
#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, syncId);
    int64_t idx = get_block_idx() * get_subblockdim() + get_subblockid();
    if (idx == subBlockId) {
        RunTSTORE<OutType, DstTileData, staticRow, staticCol, layoutType, sfractalSize>(out, dstTileData);
    }
#endif
}

template <typename OutType, typename AType, typename BType, typename fbType, int validM, int validK, int validN,
          int row, int col, int subBlockId = 0, bool isNZUnalign = false, bool isRelu = false,
          Layout layoutType = Layout::ND, int sfractalSize = 512, int indexRow = 0, int indexCol = 0,
          bool isInsert = false, int staticRow = 0, int staticCol = 0>
__global__ AICORE void RunTMOVFBQuant(__gm__ OutType *out, __gm__ AType *src0, __gm__ BType *src1, __gm__ fbType *src2,
                                      __gm__ OutType *src3)
{
    constexpr int blockAlign = std::is_same_v<AType, int8_t> ? 32 : 16;
    constexpr int M = CeilAlign<int>(validM, blockAlign);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, blockAlign);
    if constexpr (!isNZUnalign) {
        RunMATMUL<AType, BType, fbType, M, K, N, validM, validK, validN>(src0, src1, src2);
    } else {
        RunMATMUL_NZUNALIGN<AType, BType, fbType, M, K, N, validM, validK, validN>(src0, src1, src2);
    }
    constexpr int copyOutM = validM - indexRow;
    constexpr int copyOutN = validN - indexCol;

    using DstTileData =
        std::conditional_t<isNZUnalign,
                           Tile<TileType::Vec, OutType, row, col, GetTileBLayout<layoutType>(), row, col,
                                GetTileSLayout<layoutType>(), sfractalSize>,
                           Tile<TileType::Vec, OutType, staticRow, staticCol, GetTileBLayout<layoutType>(), copyOutM,
                                copyOutN, GetTileSLayout<layoutType>(), sfractalSize>>;

#if defined(__DAV_VEC__)
    RunInitLoad<OutType, layoutType, staticRow, staticCol, sfractalSize>(src3);
    set_intra_block(PIPE_MTE2, 0x1);
#endif

#if defined(__DAV_CUBE__)
    wait_intra_block(PIPE_FIX, 0x1);
    using TileMatFbData = Tile<TileType::Mat, fbType, 1, N, BLayout::RowMajor, 1, validN, SLayout::NoneBox>;
    TileMatFbData fbMatTile;
    TASSIGN(fbMatTile, 0x0);
    if (src2 != nullptr) {
        using GlobalDataSrc2 = GlobalTensor<fbType, pto::Shape<1, 1, 1, 1, validN>,
                                            pto::Stride<1 * validN, 1 * validN, 1 * validN, validN, 1>>;
        GlobalDataSrc2 src2Global(src2);
        TLOAD(fbMatTile, src2Global);
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID0);
#endif

#endif

    using AccTile = TileAcc<CType<AType>, M, N, validM, validN>;
    AccTile cTile;
    TASSIGN(cTile, 0x0);
    uint8_t syncId = 0;

    using FbTile = Tile<TileType::Scaling, fbType, 1, N, BLayout::RowMajor, 1, validN, SLayout::NoneBox>;
    FbTile fbTile;
    TASSIGN(fbTile, 0x0);
    DstTileData dstTileData;
    TASSIGN(dstTileData, 0x0);

#if defined(__DAV_CUBE__)
    TMOV(fbTile, fbMatTile);

    constexpr uint8_t mode = getMode<subBlockId, 0>();
    if constexpr (subBlockId == 0) {
        if constexpr (isRelu) {
            TEXTRACT_FP<DstTileData, AccTile, FbTile, ReluPreMode::NormalRelu>(dstTileData, cTile, fbTile, indexRow,
                                                                               indexCol);
        } else {
            TEXTRACT_FP<DstTileData, AccTile, FbTile>(dstTileData, cTile, fbTile, indexRow, indexCol);
        }
    } else {
        if constexpr (isRelu) {
            TEXTRACT<DstTileData, AccTile, FbTile, static_cast<AccToVecMode>(mode), ReluPreMode::NormalRelu>(
                dstTileData, cTile, fbTile, indexRow, indexCol);
        } else {
            TEXTRACT<DstTileData, AccTile, FbTile, static_cast<AccToVecMode>(mode)>(dstTileData, cTile, fbTile,
                                                                                    indexRow, indexCol);
        }
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_FIX, EVENT_ID0);
#endif
    set_intra_block(PIPE_FIX, syncId);
    set_intra_block(PIPE_FIX, syncId + 16);
#endif
#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, syncId);
    int64_t idx = get_block_idx() * get_subblockdim() + get_subblockid();
    if (idx == subBlockId) {
        RunTSTORE<OutType, DstTileData, staticRow, staticCol, layoutType, sfractalSize>(out, dstTileData);
    }
#endif
}

template <typename OutType, typename AType, typename BType, int validM, int validK, int validN, int row, int col,
          int subBlockId = 0, bool isNZUnalign = false, bool isRelu = false, Layout layoutType = Layout::ND,
          int sfractalSize = 512, int indexRow = 0, int indexCol = 0, bool isInsert = false, int staticRow = 0,
          int staticCol = 0>
__global__ AICORE void RunTMOVSCQuant(__gm__ OutType *out, __gm__ AType *src0, __gm__ BType *src1, __gm__ OutType *src2,
                                      float scalar)
{
    constexpr int blockAlign = std::is_same_v<AType, int8_t> ? 32 : 16;
    constexpr int M = CeilAlign<int>(validM, blockAlign);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, blockAlign);
    if constexpr (!isNZUnalign) {
        RunMATMUL<AType, BType, OutType, M, K, N, validM, validK, validN>(src0, src1, nullptr);
    } else {
        RunMATMUL_NZUNALIGN<AType, BType, OutType, M, K, N, validM, validK, validN>(src0, src1, nullptr);
    }
    using AccTile = TileAcc<CType<AType>, M, N, validM, validN>;
    AccTile cTile;
    TASSIGN(cTile, 0x0);
    uint8_t syncId = 0;
    constexpr int copyOutM = validM - indexRow;
    constexpr int copyOutN = validN - indexCol;
    using DstTileData =
        std::conditional_t<isNZUnalign,
                           Tile<TileType::Vec, OutType, row, col, GetTileBLayout<layoutType>(), row, col,
                                GetTileSLayout<layoutType>(), sfractalSize>,
                           Tile<TileType::Vec, OutType, staticRow, staticCol, GetTileBLayout<layoutType>(), copyOutM,
                                copyOutN, GetTileSLayout<layoutType>(), sfractalSize>>;
    DstTileData dstTileData;
    TASSIGN(dstTileData, 0x0);

#if defined(__DAV_VEC__)
    RunInitLoad<OutType, layoutType, staticRow, staticCol, sfractalSize>(src2);
    set_intra_block(PIPE_MTE2, 0x1);
#endif

#if defined(__DAV_CUBE__)
    wait_intra_block(PIPE_FIX, 0x1);
    uint64_t preScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&scalar));
    if (sizeof(OutType) == 1) {
        constexpr bool sign = (std::is_same_v<typename DstTileData::DType, int8_t>) ? true : false;
        preScalar = (preScalar & ~(static_cast<uint64_t>(1) << 46)) | (static_cast<uint64_t>(sign) << 46);
    }
    constexpr uint8_t mode = getMode<subBlockId, 0>();
    if constexpr (subBlockId == 0) {
        if constexpr (isRelu) {
            TEXTRACT<DstTileData, AccTile, ReluPreMode::NormalRelu>(dstTileData, cTile, preScalar, indexRow, indexCol);
        } else {
            TEXTRACT<DstTileData, AccTile>(dstTileData, cTile, preScalar, indexRow, indexCol);
        }
    } else {
        if constexpr (isRelu) {
            TEXTRACT<DstTileData, AccTile, static_cast<AccToVecMode>(mode), ReluPreMode::NormalRelu>(
                dstTileData, cTile, preScalar, indexRow, indexCol);
        } else {
            TEXTRACT<DstTileData, AccTile, static_cast<AccToVecMode>(mode)>(dstTileData, cTile, preScalar, indexRow,
                                                                            indexCol);
        }
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_FIX, EVENT_ID0);
#endif
    set_intra_block(PIPE_FIX, syncId);
    set_intra_block(PIPE_FIX, syncId + 16);
#endif
#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, syncId);
    int64_t idx = get_block_idx() * get_subblockdim() + get_subblockid();
    if (idx == subBlockId) {
        RunTSTORE<OutType, DstTileData, staticRow, staticCol, layoutType, sfractalSize>(out, dstTileData);
    }
#endif
}

template <int32_t tilingKey>
void LaunchTMOVAcc2VecNZ2ND(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream)
{
    if constexpr (tilingKey == 1) {
        // OutType, AType, BType, validM, validK, validN, row, col,
        // subBlockId = 0, isNZUnalign = false, isRelu = false, layoutType = Layout::ND, sfractalSize = 512,
        // indexRow = 0, indexCol = 0, isInsert = false, dstRow = 0, dstCol = 0
        RunTMOV<float, half, half, 60, 127, 120, 0, 0, 0, false, true, Layout::ND, 512, 0, 16, false, 64, 128>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<float *>(src2));
    } else if constexpr (tilingKey == 2) {
        RunTMOV<half, half, half, 110, 100, 80, 112, 80, 0, false, true, Layout::ND, 512, 5, 0, false, 120, 96>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<half *>(src2));
    } else if constexpr (tilingKey == 3) {
        RunTMOV<float, half, half, 6, 7, 8, 32, 32, 1, false, true, Layout::ND, 512, 2, 0, false, 10, 16>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<float *>(src2));
    } else if constexpr (tilingKey == 4) {
        RunTMOV<bfloat16_t, half, half, 111, 47, 96, 112, 96, 0, false, true, Layout::ND, 512, 3, 32, false, 150, 160>
            <<<1, nullptr, stream>>>(reinterpret_cast<bfloat16_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<bfloat16_t *>(src2));
    }
}

template void LaunchTMOVAcc2VecNZ2ND<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMOVAcc2VecNZ2ND<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMOVAcc2VecNZ2ND<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMOVAcc2VecNZ2ND<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);

template <int32_t tilingKey>
void LaunchTMOVAcc2VecFBQuantNZ2ND(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                   void *stream)
{
    // OutType, AType, BType, fbType, validM, validK, validN,
    // row, col, subBlockId = 0, isNZUnalign = false, isRelu = false,
    // layoutType = Layout::ND, sfractalSize = 512,
    // indexRow = 0, indexCol = 0, isInsert = false, dstRow = 0, dstCol = 0
    if constexpr (tilingKey == 1) {
        RunTMOVFBQuant<int8_t, int8_t, int8_t, uint64_t, 30, 48, 64, 32, 64, 0, false, false, Layout::ND, 512, 0, 32,
                       false, 40, 96><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(src0), reinterpret_cast<int8_t *>(src1),
            reinterpret_cast<uint64_t *>(src2), reinterpret_cast<int8_t *>(src3));
    } else if constexpr (tilingKey == 2) {
        RunTMOVFBQuant<half, int8_t, int8_t, uint64_t, 60, 128, 32, 64, 32, 0, false, false, Layout::ND, 512, 5, 0,
                       false, 70, 96><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<int8_t *>(src0), reinterpret_cast<int8_t *>(src1),
            reinterpret_cast<uint64_t *>(src2), reinterpret_cast<half *>(src3));
    } else if constexpr (tilingKey == 3) {
        RunTMOVFBQuant<bfloat16_t, int8_t, int8_t, uint64_t, 128, 64, 96, 128, 96, 0, false, false, Layout::ND, 512, 0,
                       0, false, 128, 96><<<1, nullptr, stream>>>(
            reinterpret_cast<bfloat16_t *>(out), reinterpret_cast<int8_t *>(src0), reinterpret_cast<int8_t *>(src1),
            reinterpret_cast<uint64_t *>(src2), reinterpret_cast<bfloat16_t *>(src3));
    } else if constexpr (tilingKey == 4) {
        RunTMOVFBQuant<int8_t, float, float, uint64_t, 60, 128, 64, 64, 64, 0, false, true, Layout::ND, 512, 7, 32,
                       false, 80, 256><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1),
            reinterpret_cast<uint64_t *>(src2), reinterpret_cast<int8_t *>(src3));
    } else if constexpr (tilingKey == 5) {
        RunTMOVFBQuant<half, float, float, uint64_t, 31, 128, 128, 31, 128, 0, false, true, Layout::ND, 512, 0, 64,
                       false, 40, 256><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1),
            reinterpret_cast<uint64_t *>(src2), reinterpret_cast<half *>(src3));
    }
}

template void LaunchTMOVAcc2VecFBQuantNZ2ND<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                               void *stream);
template void LaunchTMOVAcc2VecFBQuantNZ2ND<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                               void *stream);
template void LaunchTMOVAcc2VecFBQuantNZ2ND<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                               void *stream);
template void LaunchTMOVAcc2VecFBQuantNZ2ND<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                               void *stream);
template void LaunchTMOVAcc2VecFBQuantNZ2ND<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, uint8_t *src3,
                                               void *stream);

template <int32_t tilingKey>
void LaunchTMOVAcc2VecSCQuantNZ2ND(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream)
{
    // OutType, AType, BType, validM, validK, validN, row, col,
    // subBlockId = 0, isNZUnalign = false, isRelu = false, layoutType = Layout::ND,
    // sfractalSize = 512,
    // indexRow = 0, indexCol = 0, isInsert = false, dstRow = 0, dstCol = 0
    if constexpr (tilingKey == 1) {
        RunTMOVSCQuant<half, float, float, 128, 48, 96, 128, 96, 0, false, true, Layout::ND, 512, 0, 0, false, 128, 96>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<float *>(src0),
                                     reinterpret_cast<float *>(src1), reinterpret_cast<half *>(src2), 2);
    } else if constexpr (tilingKey == 2) {
        RunTMOVSCQuant<int8_t, float, float, 60, 128, 64, 64, 64, 0, false, true, Layout::ND, 512, 0, 32, false, 128,
                       128><<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<float *>(src0),
                                                    reinterpret_cast<float *>(src1), reinterpret_cast<int8_t *>(src2),
                                                    5);
    } else if constexpr (tilingKey == 3) {
        RunTMOVSCQuant<half, int8_t, int8_t, 30, 48, 64, 32, 64, 0, false, false, Layout::ND, 512, 5, 32, false, 40,
                       128><<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<int8_t *>(src0),
                                                    reinterpret_cast<int8_t *>(src1), reinterpret_cast<half *>(src2),
                                                    3);
    } else if constexpr (tilingKey == 4) {
        RunTMOVSCQuant<int8_t, int8_t, int8_t, 60, 128, 32, 64, 32, 0, false, false, Layout::ND, 512, 3, 0, false, 64,
                       64><<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(src0),
                                                   reinterpret_cast<int8_t *>(src1), reinterpret_cast<int8_t *>(src2),
                                                   1);
    }
}

template void LaunchTMOVAcc2VecSCQuantNZ2ND<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMOVAcc2VecSCQuantNZ2ND<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMOVAcc2VecSCQuantNZ2ND<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMOVAcc2VecSCQuantNZ2ND<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
