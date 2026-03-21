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

template <typename OutType, typename AType, typename BType, int validM, int validK, int validN, int row, int col,
          int subBlockId = 0, bool isNZUnalign = false, bool isRelu = false, Layout layoutType = Layout::ND,
          int sfractalSize = 512>
__global__ AICORE void RunTMOV(__gm__ OutType *out, __gm__ AType *src0, __gm__ BType *src1)
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
    using DstTileData = std::conditional_t<isNZUnalign,
                                           Tile<TileType::Vec, OutType, row, col, GetTileBLayout<layoutType>(), row,
                                                col, GetTileSLayout<layoutType>(), sfractalSize>,
                                           Tile<TileType::Vec, OutType, row, col, GetTileBLayout<layoutType>(), validM,
                                                validN, GetTileSLayout<layoutType>(), sfractalSize>>;
    DstTileData dstTileData;
    TASSIGN(dstTileData, 0x0);

#if defined(__DAV_CUBE__)
    constexpr uint8_t mode = getMode<subBlockId, 0>();
    if constexpr (subBlockId == 0) {
        if constexpr (isRelu) {
            TMOV<DstTileData, AccTile, ReluPreMode::NormalRelu>(dstTileData, cTile);
        } else {
            TMOV(dstTileData, cTile);
        }
    } else {
        if constexpr (isRelu) {
            TMOV<DstTileData, AccTile, static_cast<AccToVecMode>(mode), ReluPreMode::NormalRelu>(dstTileData, cTile);
        } else {
            TMOV<DstTileData, AccTile, static_cast<AccToVecMode>(mode)>(dstTileData, cTile);
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
        RunTSTORE<OutType, DstTileData, validM, validN, layoutType, sfractalSize>(out, dstTileData);
    }
#endif
}

template <typename OutType, typename AType, typename BType, typename fbType, int validM, int validK, int validN,
          int row, int col, int subBlockId = 0, bool isNZUnalign = false, bool isRelu = false,
          Layout layoutType = Layout::ND, int sfractalSize = 512>
__global__ AICORE void RunTMOVFBQuant(__gm__ OutType *out, __gm__ AType *src0, __gm__ BType *src1, __gm__ fbType *src2)
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

#if defined(__DAV_CUBE__)
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
    using DstTileData = std::conditional_t<isNZUnalign,
                                           Tile<TileType::Vec, OutType, row, col, GetTileBLayout<layoutType>(), row,
                                                col, GetTileSLayout<layoutType>(), sfractalSize>,
                                           Tile<TileType::Vec, OutType, row, col, GetTileBLayout<layoutType>(), validM,
                                                validN, GetTileSLayout<layoutType>(), sfractalSize>>;
    DstTileData dstTileData;
    TASSIGN(dstTileData, 0x0);

#if defined(__DAV_CUBE__)
    TMOV(fbTile, fbMatTile);

    constexpr uint8_t mode = getMode<subBlockId, 0>();
    if constexpr (subBlockId == 0) {
        if constexpr (isRelu) {
            TMOV_FP<DstTileData, AccTile, FbTile, ReluPreMode::NormalRelu>(dstTileData, cTile, fbTile);
        } else {
            TMOV_FP<DstTileData, AccTile, FbTile>(dstTileData, cTile, fbTile);
        }
    } else {
        if constexpr (isRelu) {
            TMOV<DstTileData, AccTile, FbTile, static_cast<AccToVecMode>(mode), ReluPreMode::NormalRelu>(dstTileData,
                                                                                                         cTile, fbTile);
        } else {
            TMOV<DstTileData, AccTile, FbTile, static_cast<AccToVecMode>(mode)>(dstTileData, cTile, fbTile);
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
        RunTSTORE<OutType, DstTileData, validM, validN, layoutType, sfractalSize>(out, dstTileData);
    }
#endif
}

template <typename OutType, typename AType, typename BType, int validM, int validK, int validN, int row, int col,
          int subBlockId = 0, bool isNZUnalign = false, bool isRelu = false, Layout layoutType = Layout::ND,
          int sfractalSize = 512>
__global__ AICORE void RunTMOVSCQuant(__gm__ OutType *out, __gm__ AType *src0, __gm__ BType *src1, float scalar)
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
    using DstTileData = std::conditional_t<isNZUnalign,
                                           Tile<TileType::Vec, OutType, row, col, GetTileBLayout<layoutType>(), row,
                                                col, GetTileSLayout<layoutType>(), sfractalSize>,
                                           Tile<TileType::Vec, OutType, row, col, GetTileBLayout<layoutType>(), validM,
                                                validN, GetTileSLayout<layoutType>(), sfractalSize>>;
    DstTileData dstTileData;
    TASSIGN(dstTileData, 0x0);

#if defined(__DAV_CUBE__)
    uint64_t preScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&scalar));
    if (sizeof(OutType) == 1) {
        constexpr bool sign = (std::is_same_v<typename DstTileData::DType, int8_t>) ? true : false;
        preScalar = (preScalar & ~(static_cast<uint64_t>(1) << 46)) | (static_cast<uint64_t>(sign) << 46);
    }
    constexpr uint8_t mode = getMode<subBlockId, 0>();
    if constexpr (subBlockId == 0) {
        if constexpr (isRelu) {
            TMOV<DstTileData, AccTile, ReluPreMode::NormalRelu>(dstTileData, cTile, preScalar);
        } else {
            TMOV<DstTileData, AccTile>(dstTileData, cTile, preScalar);
        }
    } else {
        if constexpr (isRelu) {
            TMOV<DstTileData, AccTile, static_cast<AccToVecMode>(mode), ReluPreMode::NormalRelu>(dstTileData, cTile,
                                                                                                 preScalar);
        } else {
            TMOV<DstTileData, AccTile, static_cast<AccToVecMode>(mode)>(dstTileData, cTile, preScalar);
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
        RunTSTORE<OutType, DstTileData, validM, validN, layoutType, sfractalSize>(out, dstTileData);
    }
#endif
}

template <typename OutType, typename AType, typename BType, int M, int K, int N, int validM, int validN, bool splitM,
          bool isNZUnalign = false>
__global__ AICORE void RunSplitNTMOVNz2Nz(__gm__ OutType *out, __gm__ AType *src0, __gm__ BType *src1)
{
    constexpr int mSize = splitM ? M / 2 : M;
    constexpr int nSize = splitM ? N : N / 2;
    constexpr int sFractalSize = std::is_same_v<OutType, float> ? 1024 : 512; // float:1024, other:512
    constexpr uint16_t sGRows_ = 16;
    constexpr uint16_t sGCols_ = CeilDiv<uint16_t>(sFractalSize, sGRows_ * sizeof(OutType));
    constexpr uint16_t kGRows_ = CeilDiv<uint16_t>(mSize, sGRows_);
    constexpr uint16_t kGCols_ = CeilDiv<uint16_t>(nSize, sGCols_);
    using DynShapeDim5 = Shape<1, kGCols_, kGRows_, sGRows_, sGCols_>;

    constexpr uint32_t gShape2 = CeilDiv<uint16_t>(M, sGRows_);
    constexpr uint32_t gShape3 = CeilDiv<uint16_t>(N, sGRows_);
    using DynStrideDim5 =
        pto::Stride<gShape3 * gShape2 * sGCols_ * sGRows_, gShape2 * sGCols_ * sGRows_, sGCols_ * sGRows_, sGCols_, 1>;

    using GlobalDataOut = GlobalTensor<OutType, DynShapeDim5, DynStrideDim5, Layout::NZ>;
    GlobalDataOut dstGlobal1(out);
    constexpr int stride = splitM ? mSize * sGCols_ : M * nSize;
    GlobalDataOut dstGlobal2(out + stride);
    if constexpr (!isNZUnalign) {
        RunMATMUL<AType, BType, OutType, M, K, N, validM, K, validN>(src0, src1, nullptr);
    } else {
        RunMATMUL_NZUNALIGN<AType, BType, OutType, M, K, N, validM, K, validN>(src0, src1, nullptr);
    }
    using AccTile = TileAcc<CType<AType>, M, N, validM, validN>;
    AccTile cTile;
    TASSIGN(cTile, 0x0);
    uint8_t syncId = 0;

    using DstTileData =
        Tile<TileType::Vec, OutType, M, N, BLayout::ColMajor, mSize, nSize, SLayout::RowMajor, sFractalSize>; // nz
    DstTileData dstTileData;
    TASSIGN(dstTileData, 0x0);

#if defined(__DAV_CUBE__)

    constexpr int dualDstCtl = splitM ? 1 : 2;
    constexpr uint64_t mode = getMode<0, dualDstCtl>();
    TMOV<DstTileData, AccTile, static_cast<AccToVecMode>(mode)>(dstTileData, cTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
#endif
    set_intra_block(PIPE_FIX, syncId);
    set_intra_block(PIPE_FIX, syncId + 16);

#endif
#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, syncId);
    int64_t idx = get_block_idx() * get_subblockdim() + get_subblockid();

    if (idx == 0) {
        if (sFractalSize == 512) {
            TSTORE(dstGlobal1, dstTileData);
        } else {
            UBCopyOut<OutType, GlobalDataOut, DstTileData>(dstGlobal1, dstTileData, M, N, 0);
        }
    } else {
        if (sFractalSize == 512) {
            TSTORE(dstGlobal2, dstTileData);
        } else {
            UBCopyOut<OutType, GlobalDataOut, DstTileData>(dstGlobal2, dstTileData, M, N, 0);
        }
    }
#endif
}

template <typename OutType, typename AType, typename BType, int M, int K, int N, int validM, int validN, bool splitM,
          bool isNZUnalign = false>
__global__ AICORE void RunSplitMTMOVNz2Nz(__gm__ OutType *out, __gm__ AType *src0, __gm__ BType *src1)
{
    constexpr int mSize = splitM ? M / 2 : M;
    constexpr int nSize = splitM ? N : N / 2;
    constexpr int sFractalSize = std::is_same_v<OutType, float> ? 1024 : 512; // float:1024, other:512
    constexpr uint16_t sGRows_ = 16;
    constexpr uint16_t sGCols_ = CeilDiv<uint16_t>(sFractalSize, sGRows_ * sizeof(OutType));
    constexpr uint16_t kGRows_ = CeilDiv<uint16_t>(mSize, sGRows_);
    constexpr uint16_t kGCols_ = CeilDiv<uint16_t>(nSize, sGRows_);
    using DynShapeDim5 = Shape<1, 1, 1, sGRows_, sGCols_>;

    using DynStrideDim5 = pto::Stride<sGCols_ * sGRows_, sGCols_ * sGRows_, sGCols_ * sGRows_, sGCols_, 1>;

    using GlobalDataOut = GlobalTensor<OutType, DynShapeDim5, DynStrideDim5, Layout::NZ>;
    constexpr int stride = splitM ? mSize * sGCols_ : M * nSize;

    using TileMatAData = Tile<TileType::Mat, AType, M, K, BLayout::RowMajor, M, K, SLayout::ColMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, BType, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>;
    if constexpr (!isNZUnalign) {
        RunMATMUL<AType, BType, OutType, M, K, N, validM, K, validN>(src0, src1, nullptr);
    } else {
        RunMATMUL_NZUNALIGN<AType, BType, OutType, M, K, N, validM, K, validN>(src0, src1, nullptr);
    }
    using AccTile = TileAcc<CType<AType>, M, N, validM, validN>;
    AccTile cTile;
    TASSIGN(cTile, 0x0);
    uint8_t syncId = 0;

    using DstTileData =
        Tile<TileType::Vec, OutType, M, N, BLayout::ColMajor, mSize, nSize, SLayout::RowMajor, sFractalSize>;
    DstTileData dstTileData;
    TASSIGN(dstTileData, 0x0);

#if defined(__DAV_CUBE__)

    constexpr int dualDstCtl = splitM ? 1 : 2;
    constexpr uint64_t mode = getMode<0, dualDstCtl>();
    TMOV<DstTileData, AccTile, static_cast<AccToVecMode>(mode)>(dstTileData, cTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
#endif
    set_intra_block(PIPE_FIX, syncId);
    set_intra_block(PIPE_FIX, syncId + 16);

#endif
#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, syncId);
    int64_t idx = get_block_idx() * get_subblockdim() + get_subblockid();

    constexpr uint16_t sFractalColNum = CeilDiv<uint16_t>(nSize, sGRows_);
    if (idx == 0) {
        for (int i = 0; i < sFractalColNum; i++) {
            GlobalDataOut dstGlobal1(out + 2 * stride * i);
            if (sFractalSize == 512) {
                TSTORE(dstGlobal1, dstTileData);
            } else {
                uint16_t startDstAddr = 2 * i * stride;
                UBCopyOut<OutType, GlobalDataOut, DstTileData>(dstGlobal1, dstTileData, mSize, nSize, startDstAddr);
            }
        }

    } else {
        for (int i = 0; i < sFractalColNum; i++) {
            GlobalDataOut dstGlobal2(out + stride + 2 * stride * i);
            if (sFractalSize == 512) {
                TSTORE(dstGlobal2, dstTileData);
            } else {
                uint16_t startDstAddr = 2 * i * stride;
                UBCopyOut<OutType, GlobalDataOut, DstTileData>(dstGlobal2, dstTileData, mSize, nSize, startDstAddr);
            }
        }
    }
#endif
}

template <typename OutType, typename AType, typename BType, int M, int K, int N, bool splitM>
__global__ AICORE void RunSplitTMOV(__gm__ OutType *out, __gm__ AType *src0, __gm__ BType *src1)
{
    constexpr int mSize = splitM ? M / 2 : M;
    constexpr int nSize = splitM ? N : N / 2;
    using GlobalDataOut = GlobalTensor<OutType, pto::Shape<1, 1, 1, mSize, nSize>,
                                       pto::Stride<1 * M * N, 1 * M * N, M * N, N, 1>, Layout::ND>;
    GlobalDataOut dstGlobal1(out);
    constexpr int stride = splitM ? mSize * nSize : nSize;
    GlobalDataOut dstGlobal2(out + stride);

    RunMATMUL<AType, BType, OutType, M, K, N, M, K, N>(src0, src1, nullptr);

    using AccTile = TileAcc<CType<AType>, M, N, M, N>;
    AccTile cTile;
    TASSIGN(cTile, 0x0);
    uint8_t syncId = 0;

    using DstTileData = Tile<TileType::Vec, OutType, M, N, BLayout::RowMajor, mSize, nSize>;
    DstTileData dstTileData;
    TASSIGN(dstTileData, 0x0);

#if defined(__DAV_CUBE__)

    constexpr int dualDstCtl = splitM ? 1 : 2;
    constexpr uint8_t mode = getMode<0, dualDstCtl>();
    TMOV<DstTileData, AccTile, static_cast<AccToVecMode>(mode)>(dstTileData, cTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
#endif
    set_intra_block(PIPE_FIX, syncId);
    set_intra_block(PIPE_FIX, syncId + 16);

#endif
#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, syncId);
    int64_t idx = get_block_idx() * get_subblockdim() + get_subblockid();

    if (idx == 0) {
        TSTORE(dstGlobal1, dstTileData);
    } else {
        TSTORE(dstGlobal2, dstTileData);
    }
#endif
}

template <int32_t tilingKey>
void LaunchTMOVAcc2VecNZ2ND(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMOV<float, half, half, 60, 127, 120, 64, 128, 0, false, true><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 2) {
        RunTMOV<half, half, half, 110, 100, 80, 112, 80, 0, false, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 3) {
        RunTMOV<float, half, half, 6, 7, 8, 32, 32, 1><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 4) {
        RunTMOV<bfloat16_t, half, half, 111, 47, 96, 112, 96, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<bfloat16_t *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 5) {
        RunSplitTMOV<float, half, half, 96, 32, 48, true><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 6) {
        RunSplitTMOV<float, half, half, 48, 32, 128, false><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    }
}

template void LaunchTMOVAcc2VecNZ2ND<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2VecNZ2ND<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2VecNZ2ND<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2VecNZ2ND<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2VecNZ2ND<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2VecNZ2ND<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int32_t tilingKey>
void LaunchTMOVAcc2VecNZ2NZ(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMOV<half, half, half, 96, 80, 112, 96, 112, 0, false, false, Layout::NZ, 512><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 2) {
        RunTMOV<float, half, half, 80, 112, 96, 80, 96, 0, false, false, Layout::NZ, 1024><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 3) {
        RunTMOV<float, float, float, 13, 16, 9, 16, 16, 0, true, true, Layout::NZ, 512><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1));
    } else if constexpr (tilingKey == 4) {
        RunTMOV<bfloat16_t, half, half, 45, 112, 43, 48, 48, 1, true, true, Layout::NZ, 512><<<1, nullptr, stream>>>(
            reinterpret_cast<bfloat16_t *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 5) {
        RunSplitNTMOVNz2Nz<float, float, float, 48, 80, 128, 45, 125, false, true><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1));
    } else if constexpr (tilingKey == 6) {
        RunSplitNTMOVNz2Nz<float, half, half, 80, 16, 96, 75, 90, false, true><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 7) {
        RunSplitMTMOVNz2Nz<float, half, half, 112, 48, 80, 110, 78, true, true><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 8) {
        RunSplitMTMOVNz2Nz<float, float, float, 16, 112, 112, 13, 110, true, true><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1));
    }
}

template void LaunchTMOVAcc2VecNZ2NZ<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2VecNZ2NZ<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2VecNZ2NZ<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2VecNZ2NZ<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2VecNZ2NZ<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2VecNZ2NZ<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2VecNZ2NZ<7>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2VecNZ2NZ<8>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int32_t tilingKey>
void LaunchTMOVAcc2VecFBQuantNZ2NZ(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMOVFBQuant<int8_t, int8_t, int8_t, uint64_t, 128, 48, 128, 128, 128, 0, false, false, Layout::NZ>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(src2));
    } else if constexpr (tilingKey == 2) {
        RunTMOVFBQuant<half, int8_t, int8_t, uint64_t, 64, 80, 96, 128, 128, 0, false, false, Layout::NZ>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(src2));
    } else if constexpr (tilingKey == 3) {
        RunTMOVFBQuant<int8_t, float, float, uint64_t, 125, 32, 91, 128, 96, 0, true, true, Layout::NZ>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<float *>(src0),
                                     reinterpret_cast<float *>(src1), reinterpret_cast<uint64_t *>(src2));
    } else if constexpr (tilingKey == 4) {
        RunTMOVFBQuant<half, float, float, uint64_t, 73, 16, 110, 80, 112, 0, true, true, Layout::NZ>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<float *>(src0),
                                     reinterpret_cast<float *>(src1), reinterpret_cast<uint64_t *>(src2));
    }
}

template void LaunchTMOVAcc2VecFBQuantNZ2NZ<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMOVAcc2VecFBQuantNZ2NZ<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMOVAcc2VecFBQuantNZ2NZ<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMOVAcc2VecFBQuantNZ2NZ<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);

template <int32_t tilingKey>
void LaunchTMOVAcc2VecSCQuantNZ2NZ(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMOVSCQuant<half, float, float, 48, 32, 80, 48, 80, 0, false, true, Layout::NZ><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1), 2);
    } else if constexpr (tilingKey == 2) {
        RunTMOVSCQuant<half, int8_t, int8_t, 96, 48, 128, 96, 128, 0, false, true, Layout::NZ><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<int8_t *>(src0), reinterpret_cast<int8_t *>(src1), 4);
    } else if constexpr (tilingKey == 3) {
        RunTMOVSCQuant<int8_t, int8_t, int8_t, 125, 64, 124, 128, 128, 0, true, false, Layout::NZ>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), 5);
    } else if constexpr (tilingKey == 4) {
        RunTMOVSCQuant<int8_t, float, float, 61, 80, 93, 64, 96, 0, true, false, Layout::NZ><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1), 7);
    }
}

template void LaunchTMOVAcc2VecSCQuantNZ2NZ<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2VecSCQuantNZ2NZ<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2VecSCQuantNZ2NZ<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2VecSCQuantNZ2NZ<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int32_t tilingKey>
void LaunchTMOVAcc2VecFBQuantNZ2ND(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMOVFBQuant<int8_t, int8_t, int8_t, uint64_t, 30, 48, 64, 32, 64, 0>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(src2));
    } else if constexpr (tilingKey == 2) {
        RunTMOVFBQuant<half, int8_t, int8_t, uint64_t, 60, 128, 32, 64, 32, 0>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(src2));
    } else if constexpr (tilingKey == 3) {
        RunTMOVFBQuant<bfloat16_t, int8_t, int8_t, uint64_t, 128, 64, 96, 128, 96, 0>
            <<<1, nullptr, stream>>>(reinterpret_cast<bfloat16_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(src2));
    } else if constexpr (tilingKey == 4) {
        RunTMOVFBQuant<int8_t, float, float, uint64_t, 60, 128, 64, 64, 64, 0, false, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<float *>(src0),
                                     reinterpret_cast<float *>(src1), reinterpret_cast<uint64_t *>(src2));
    } else if constexpr (tilingKey == 5) {
        RunTMOVFBQuant<half, float, float, uint64_t, 31, 128, 128, 31, 128, 0, false, true>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<float *>(src0),
                                     reinterpret_cast<float *>(src1), reinterpret_cast<uint64_t *>(src2));
    }
}

template void LaunchTMOVAcc2VecFBQuantNZ2ND<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMOVAcc2VecFBQuantNZ2ND<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMOVAcc2VecFBQuantNZ2ND<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMOVAcc2VecFBQuantNZ2ND<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMOVAcc2VecFBQuantNZ2ND<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);

template <int32_t tilingKey>
void LaunchTMOVAcc2VecSCQuantNZ2ND(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMOVSCQuant<half, float, float, 128, 48, 96, 128, 96, 0, false, true><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1), 2);
    } else if constexpr (tilingKey == 2) {
        RunTMOVSCQuant<int8_t, float, float, 60, 128, 64, 64, 64, 0, false, true><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1), 5);
    } else if constexpr (tilingKey == 3) {
        RunTMOVSCQuant<half, int8_t, int8_t, 30, 48, 64, 32, 64><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<int8_t *>(src0), reinterpret_cast<int8_t *>(src1), 3);
    } else if constexpr (tilingKey == 4) {
        RunTMOVSCQuant<int8_t, int8_t, int8_t, 60, 128, 32, 64, 32><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(src0), reinterpret_cast<int8_t *>(src1), 1);
    }
}

template void LaunchTMOVAcc2VecSCQuantNZ2ND<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2VecSCQuantNZ2ND<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2VecSCQuantNZ2ND<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2VecSCQuantNZ2ND<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int32_t tilingKey>
void LaunchTMOVAcc2VecNZ2DN(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMOV<float, float, float, 8, 7, 6, 32, 32, 0, false, false, Layout::DN><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1));
    } else if constexpr (tilingKey == 2) {
        RunTMOV<half, half, half, 112, 48, 95, 112, 96, 0, false, false, Layout::DN><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 3) {
        RunTMOV<bfloat16_t, half, half, 48, 31, 31, 64, 32, 1, false, true, Layout::DN><<<1, nullptr, stream>>>(
            reinterpret_cast<bfloat16_t *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 4) {
        RunTMOV<float, half, half, 88, 48, 95, 96, 96, 0, false, true, Layout::DN, 1024><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    }
}

template void LaunchTMOVAcc2VecNZ2DN<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2VecNZ2DN<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2VecNZ2DN<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2VecNZ2DN<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int32_t tilingKey>
void LaunchTMOVAcc2VecFBQuantNZ2DN(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMOVFBQuant<int8_t, int8_t, int8_t, uint64_t, 96, 128, 60, 96, 64, 0, false, false, Layout::DN>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(src2));
    } else if constexpr (tilingKey == 2) {
        RunTMOVFBQuant<half, int8_t, int8_t, uint64_t, 32, 48, 64, 32, 64, 0, false, false, Layout::DN>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(src2));
    } else if constexpr (tilingKey == 3) {
        RunTMOVFBQuant<int8_t, half, half, uint64_t, 32, 128, 60, 32, 64, 0, false, true, Layout::DN>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<uint64_t *>(src2));
    } else if constexpr (tilingKey == 4) {
        RunTMOVFBQuant<half, half, half, uint64_t, 64, 64, 90, 64, 96, 0, false, true, Layout::DN>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<uint64_t *>(src2));
    }
}

template void LaunchTMOVAcc2VecFBQuantNZ2DN<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMOVAcc2VecFBQuantNZ2DN<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMOVAcc2VecFBQuantNZ2DN<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);
template void LaunchTMOVAcc2VecFBQuantNZ2DN<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);

template <int32_t tilingKey>
void LaunchTMOVAcc2VecSCQuantNZ2DN(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        RunTMOVSCQuant<half, float, float, 80, 40, 66, 80, 80, 0, false, true, Layout::DN><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1), 2);
    } else if constexpr (tilingKey == 2) {
        RunTMOVSCQuant<int8_t, float, float, 96, 128, 60, 96, 64, 0, false, true, Layout::DN><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1), 5);
    } else if constexpr (tilingKey == 3) {
        RunTMOVSCQuant<half, int8_t, int8_t, 32, 128, 64, 32, 64, 0, false, false, Layout::DN><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<int8_t *>(src0), reinterpret_cast<int8_t *>(src1), 3);
    } else if constexpr (tilingKey == 4) {
        RunTMOVSCQuant<int8_t, int8_t, int8_t, 64, 64, 90, 64, 96, 0, false, false, Layout::DN><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(src0), reinterpret_cast<int8_t *>(src1), 1);
    }
}

template void LaunchTMOVAcc2VecSCQuantNZ2DN<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2VecSCQuantNZ2DN<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2VecSCQuantNZ2DN<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTMOVAcc2VecSCQuantNZ2DN<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
