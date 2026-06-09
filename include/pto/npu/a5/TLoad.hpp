/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TLOAD_HPP
#define TLOAD_HPP
#include <pto/common/utils.hpp>
#include "common.hpp"
#include "pto/common/arch/register/tload_common.hpp"

namespace pto {
struct A5LoadOp : LoadOpBase {
    using LoadOpBase::TLoadCubeInstr;
    template <Layout Layout = Layout::ND, typename T>
    PTO_INTERNAL static void TLoadCubeInstr(__cbuf__ T *dst, __gm__ T *src, uint64_t loop1SrcStride, uint16_t nValue,
                                            uint32_t dValue, uint64_t loop4SrcStride)
    {
        if constexpr (Layout == Layout::ND) {
            pto_copy_gm_to_cbuf_multi_nd2nz(dst, src, 0 /*sid*/, loop1SrcStride, 0, nValue, dValue, loop4SrcStride);
        } else {
            using LoadT = LoadTypeBySize_t<T>;
            copy_gm_to_cbuf_multi_dn2nz(reinterpret_cast<__cbuf__ LoadT *>(dst), reinterpret_cast<__gm__ LoadT *>(src),
                                        0 /*sid*/, loop1SrcStride, 0, nValue, dValue, loop4SrcStride, false);
        }
    }
};

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadMxCubeCheck()
{
    // support ZZ2ZZ NN2NN
    static_assert(
        ((GlobalData::layout == pto::Layout::MX_A_ZZ || GlobalData::layout == pto::Layout::MX_A_ND ||
          GlobalData::layout == pto::Layout::MX_A_DN || GlobalData::layout == pto::Layout::ND) &&
         (TileData::isRowMajor && (TileData::SFractal == SLayout::RowMajor))) ||
            ((GlobalData::layout == pto::Layout::MX_B_NN || GlobalData::layout == pto::Layout::MX_B_ND ||
              GlobalData::layout == pto::Layout::MX_B_DN) &&
             (!TileData::isRowMajor && (TileData::SFractal == SLayout::ColMajor))),
        "Fix: now only support MX_A_ZZ2ZZ/MX_A_ND2ZZ/MX_A_DN2ZZ or MX_B_NN2NN/MX_B_ND2NN/MX_B_DN2NN in current "
        "platform");

    static_assert(caps::IsFP8E8M0<typename TileData::DType>() && caps::IsFP8E8M0<typename GlobalData::RawDType>(),
                  "Fix: DType only support float8_e8m0_t in MX_A_ZZ or MX_B_NN");
    static_assert(TileData::SFractalSize == 32, "Fix: TileData SFractalSize must be 32 of Zz or Nn format in L1");

    // L1 space check
    static_assert(TileData::Rows * TileData::Cols <= 512 * 1024,
                  "Fix: TileData static shape must less than 512KB in L1");
    // ZZ2ZZ and NN2NN check SFractal shape
    if constexpr (GlobalData::layout == pto::Layout::MX_A_ZZ || GlobalData::layout == pto::Layout::MX_B_NN) {
        // globaltensor only support [16,2] fractal
        static_assert((GlobalData::staticShape[3] == 16 || GlobalData::staticShape[3] == -1) &&
                          (GlobalData::staticShape[4] == 2 || GlobalData::staticShape[4] == -1),
                      "Fix: GlobalTensor input SFractal is [16,2] when Layout is MX_AZZ or MX_BNN");
    }
    // check shape
    if constexpr (GlobalData::layout == pto::Layout::MX_A_ZZ) {
        static_assert(
            (TileData::Rows >= GlobalData::staticShape[0] * GlobalData::staticShape[1] * GlobalData::staticShape[3]) &&
                (TileData::Cols >= GlobalData::staticShape[2] * GlobalData::staticShape[4]),
            "Fix: TileData::Rows need >= GlobalTensor inputShape[0] * inputShape[1] * inputShape[3] and TileData::Cols "
            "need "
            ">= GlobalTensor inputShape[2] * inputShape[4], when Layout is MX_A_ZZ");
    }

    if constexpr (GlobalData::layout == pto::Layout::MX_B_NN) {
        static_assert((TileData::Rows >= GlobalData::staticShape[2] * GlobalData::staticShape[4]) &&
                          (TileData::Cols >=
                           GlobalData::staticShape[0] * GlobalData::staticShape[1] * GlobalData::staticShape[3]),
                      "Fix: TileData::Rows need >= GlobalTensor inputShape[2] * inputShape[4] and TileData::Cols need "
                      ">= GlobalTensor inputShape[0] * inputShape[1] * inputShape[3], when Layout is MX_B_NN");
    }

    if constexpr (GlobalData::layout == pto::Layout::MX_A_ND || GlobalData::layout == pto::Layout::MX_A_DN ||
                  GlobalData::layout == pto::Layout::MX_B_ND || GlobalData::layout == pto::Layout::MX_B_DN) {
        static_assert(((GlobalData::staticShape[0] == 1 || GlobalData::staticShape[0] == -1) &&
                       (GlobalData::staticShape[1] == 1 || GlobalData::staticShape[1] == -1)),
                      "expect 3D for layout MX_A_ND/MX_A_DN/MX_B_ND/MX_B_DN.");
        static_assert((GlobalData::staticShape[4] == 2 || GlobalData::staticShape[4] == -1),
                      "expect gShape4 == 2 for layout MX_A_ND/MX_A_DN/MX_B_ND/MX_B_DN.");
    }
}

template <typename Op, typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadMxCubeNN2NN(__cbuf__ typename TileData::DType *dst, typename GlobalData::DType *src, int gShape0,
                                   int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1,
                                   int gStride2, int gStride3, int gStride4)
{
    // [0   1       2      3   4]
    // [1, N/16, scaleK/2, 16, 2]
    __cbuf__ typename TileData::DType *dstAddrP = dst;
    typename GlobalData::DType *srcAddrP = src;
    uint32_t nBurst = gShape1;
    uint32_t lenBurst = gShape2 * gShape4 * BLOCK_LEN;
    uint64_t gmStride = gStride1;
    uint32_t dstStride = BLOCK_LEN * TileData::Rows;

    int64_t tileStride = TileData::Rows * gShape1 * gShape3; // stitching along the col direction
    set_loop_size_outtol1(1ULL << 21 | 1ULL);
    for (uint32_t i = 0; i < gShape0; i++) {
        srcAddrP = src + i * gStride0;
        dstAddrP = dst + i * tileStride;
        Op::TLoadCubeInstr(dstAddrP, srcAddrP, nBurst, lenBurst, gmStride, dstStride, 0);
    }
}

template <typename Op, typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadMxCubeZZ2ZZ(__cbuf__ typename TileData::DType *dst, typename GlobalData::DType *src, int gShape0,
                                   int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1,
                                   int gStride2, int gStride3, int gStride4)
{
    // [0   1       2      3   4]
    // [1, M/16, scaleK/2, 16, 2]
    __cbuf__ typename TileData::DType *dstAddrP = dst;
    typename GlobalData::DType *srcAddrP = src;
    uint32_t nBurst = gShape1;
    uint32_t lenBurst = BLOCK_LEN * gShape2 * gShape4;
    uint64_t gmStride = gStride1;
    uint32_t dstStride = BLOCK_LEN * TileData::Cols;

    int64_t tileStride = gShape1 * gShape3 * TileData::Cols; // stitching along the row direction
    set_loop_size_outtol1(1ULL << 21 | 1ULL);
    for (uint32_t i = 0; i < gShape0; i++) {
        srcAddrP = src + i * gStride0;
        dstAddrP = dst + i * tileStride;
        Op::TLoadCubeInstr(dstAddrP, srcAddrP, nBurst, lenBurst, gmStride, dstStride, 0);
    }
}

// DN for AND2ZZ && BDN2NN
// ND for ADN2ZZ && BND2NN
template <typename Op, typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadMxCubeAND2ZZ(__cbuf__ typename TileData::DType *dst, typename GlobalData::DType *src,
                                    int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                    int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    uint16_t nValue = validCol >> 1;
    uint32_t dValue = validRow;

    constexpr uint16_t loop3DstStride = TileData::Cols >> 1; // unit is 32B
    uint64_t loop1SrcStride = GetByteSize<typename TileData::DType>(gStride3);

    // MTE2_NZ_PARA[63:48]  loop4DstStride,  MTE2_NZ_PARA[47:32]  loop3DstStride
    // MTE2_NZ_PARA[31:16]  loop2DstStride   MTE2_NZ_PARA[15:0]   ndNum
    uint64_t mte2NzPara = static_cast<uint64_t>(0) << 48 | static_cast<uint64_t>(loop3DstStride) << 32;
    mte2NzPara |= static_cast<uint64_t>(1) << 16 | static_cast<uint64_t>(1);
    set_mte2_nz_para(mte2NzPara); // only set once

    Op::template TLoadCubeInstr<pto::Layout::DN>(reinterpret_cast<__cbuf__ uint16_t *>(dst),
                                                 reinterpret_cast<__gm__ uint16_t *>(src), loop1SrcStride, nValue,
                                                 dValue, 0);
}

template <typename Op, typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoadMxCubeAVector(typename TileData::TileDType __out__ dst, typename GlobalData::DType *src,
                                            int gShape0, int gShape1, int gShape2, int gShape3, int gShape4,
                                            int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
                                            int validRow, int validCol)
{
    // gm: shape <1,1,1,1,k> stride <k,k,k,k,1>
    static_assert((GlobalData::staticShape[0] == 1 && GlobalData::staticShape[1] == 1 &&
                   GlobalData::staticShape[2] == 1 && GlobalData::staticShape[3] == 1),
                  "Vector input must have the first 4 dimensions of staticShpae all equal to 1.");
    using L1Type = typename TileData::TileDType;
    __cbuf__ typename TileData::DType *dstAddrP = __cce_get_tile_ptr(dst);
    typename GlobalData::DType *srcAddrP = src;

    uint32_t lenBurst = validCol * sizeof(L1Type);
    uint64_t gmStride = gStride3 * sizeof(L1Type);
    constexpr uint32_t dstStride = TileData::Cols * sizeof(L1Type);

    constexpr uint32_t blockSizeElem = BLOCK_BYTE_SIZE / sizeof(L1Type);
    uint32_t gapElement = (TileData::Cols - validCol);
    uint32_t padCount = gapElement % blockSizeElem;
    pto_set_tload_pad_val<TileType::Mat>(GetPadValue<TileData>());

    constexpr uint64_t loop2 = 1;
    constexpr uint64_t loop1 = 1;
    uint64_t loop2SrcStride = GetByteSize<L1Type>(gStride1);
    uint64_t loop1SrcStride = GetByteSize<L1Type>(gStride2);
    constexpr uint64_t loop2DstStride = TileData::Cols * sizeof(L1Type);
    constexpr uint64_t loop1DstStride = TileData::Cols * sizeof(L1Type);

    set_loop2_stride_outtol1(loop2DstStride << 40 | loop2SrcStride);
    set_loop1_stride_outtol1(loop1DstStride << 40 | loop1SrcStride);
    set_loop_size_outtol1(loop2 << 21 | loop1);

    Op::TLoadCubeInstr(dstAddrP, srcAddrP, 1, lenBurst, gmStride, dstStride, padCount);
}

template <typename Op, typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadMxCubeADN2ZZ(__cbuf__ typename TileData::DType *dst, typename GlobalData::DType *src,
                                    int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                    int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    uint16_t nValue = validCol >> 1;
    uint32_t dValue = validRow;

    uint64_t loop1SrcStride = GetByteSize<typename TileData::DType>(gStride4) * sizeof(uint16_t);
    constexpr uint16_t loop3DstStride = TileData::Cols >> 1; // unit is 32B

    // MTE2_NZ_PARA[63:48]  loop4DstStride,  MTE2_NZ_PARA[47:32]  loop3DstStride
    // MTE2_NZ_PARA[31:16]  loop2DstStride   MTE2_NZ_PARA[15:0]   ndNum
    uint64_t mte2NzPara = static_cast<uint64_t>(0) << 48 | static_cast<uint64_t>(loop3DstStride) << 32;
    mte2NzPara |= static_cast<uint64_t>(1) << 16 | static_cast<uint64_t>(1);
    set_mte2_nz_para(mte2NzPara); // only set once
    Op::TLoadCubeInstr(dst, src, loop1SrcStride, nValue, dValue, 0);
}

template <typename Op, typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadMxCubeBND2NN(__cbuf__ typename TileData::DType *dst, typename GlobalData::DType *src,
                                    int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                    int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    uint16_t nValue = validRow >> 1;
    uint32_t dValue = validCol;

    constexpr uint16_t loop3DstStride = TileData::Rows >> 1; // unit is 32B
    uint64_t loop1SrcStride = GetByteSize<typename TileData::DType>(gStride3) * sizeof(uint16_t);

    // MTE2_NZ_PARA[63:48]  loop4DstStride,  MTE2_NZ_PARA[47:32]  loop3DstStride
    // MTE2_NZ_PARA[31:16]  loop2DstStride   MTE2_NZ_PARA[15:0]   ndNum
    uint64_t mte2NzPara = static_cast<uint64_t>(0) << 48 | static_cast<uint64_t>(loop3DstStride) << 32;
    mte2NzPara |= static_cast<uint64_t>(1) << 16 | static_cast<uint64_t>(1);
    set_mte2_nz_para(mte2NzPara); // only set once

    Op::TLoadCubeInstr(dst, src, loop1SrcStride, nValue, dValue, 0);
}

template <typename Op, typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadMxCubeBDN2NN(__cbuf__ typename TileData::DType *dst, typename GlobalData::DType *src,
                                    int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                    int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    uint16_t nValue = validRow >> 1;
    uint32_t dValue = validCol;

    uint64_t loop1SrcStride = GetByteSize<typename TileData::DType>(gStride4);
    constexpr uint16_t loop3DstStride = TileData::Rows >> 1; // unit is 32B

    // MTE2_NZ_PARA[63:48]  loop4DstStride,  MTE2_NZ_PARA[47:32]  loop3DstStride
    // MTE2_NZ_PARA[31:16]  loop2DstStride   MTE2_NZ_PARA[15:0]   ndNum
    uint64_t mte2NzPara = static_cast<uint64_t>(0) << 48 | static_cast<uint64_t>(loop3DstStride) << 32;
    mte2NzPara |= static_cast<uint64_t>(1) << 16 | static_cast<uint64_t>(1);
    set_mte2_nz_para(mte2NzPara); // only set once

    Op::template TLoadCubeInstr<pto::Layout::DN>(reinterpret_cast<__cbuf__ uint16_t *>(dst),
                                                 reinterpret_cast<__gm__ uint16_t *>(src), loop1SrcStride, nValue,
                                                 dValue, 0);
}

template <typename Op, typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoadMxCube(typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__ *src,
                                     int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                     int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    using L1Type = __cbuf__ typename TileData::DType *;
    L1Type dstAddr = (L1Type)__cce_get_tile_ptr(dst);

    // ZZ2ZZ or NN2NN
    if constexpr (GlobalData::layout == pto::Layout::MX_A_ZZ &&
                  (TileData::isRowMajor && TileData::SFractal == SLayout::RowMajor)) {
        TLoadMxCubeZZ2ZZ<Op, TileData, GlobalData>(dstAddr, src, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                                   gStride1, gStride2, gStride3, gStride4);
    } else if constexpr (GlobalData::layout == pto::Layout::MX_B_NN &&
                         (!TileData::isRowMajor && TileData::SFractal == SLayout::ColMajor)) {
        TLoadMxCubeNN2NN<Op, TileData, GlobalData>(dstAddr, src, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                                   gStride1, gStride2, gStride3, gStride4);
    } else if constexpr (GlobalData::layout == pto::Layout::MX_A_ND &&
                         (TileData::isRowMajor && (TileData::SFractal == SLayout::RowMajor))) {
        // newgStride3 -> gStride2;
        TLoadMxCubeAND2ZZ<Op, TileData, GlobalData>(dstAddr, src, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                                    gStride1, gStride2, gStride2, gStride4, validRow, validCol);
    } else if constexpr (GlobalData::layout == pto::Layout::MX_A_DN &&
                         (TileData::isRowMajor && (TileData::SFractal == SLayout::RowMajor))) {
        // newgStride4 -> gStride2 / gStride3;
        TLoadMxCubeADN2ZZ<Op, TileData, GlobalData>(dstAddr, src, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                                    gStride1, gStride2, gStride3, gStride2 / gStride3, validRow,
                                                    validCol);
    } else if constexpr (GlobalData::layout == pto::Layout::MX_B_ND &&
                         (!TileData::isRowMajor && (TileData::SFractal == SLayout::ColMajor))) {
        // newgStride3 -> gStride2 / gStride3;
        TLoadMxCubeBND2NN<Op, TileData, GlobalData>(dstAddr, src, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                                    gStride1, gStride2, gStride2 / gStride3, gStride4, validRow,
                                                    validCol);
    } else if constexpr (GlobalData::layout == pto::Layout::MX_B_DN &&
                         (!TileData::isRowMajor && (TileData::SFractal == SLayout::ColMajor))) {
        // newgStride4 -> gStride2;
        TLoadMxCubeBDN2NN<Op, TileData, GlobalData>(dstAddr, src, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                                    gStride1, gStride2, gStride3, gStride2, validRow, validCol);
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLOAD_TILE_IMPL(TileData &dst, GlobalData &src)
{
    StaticCheck<TileData, GlobalData>();
    if constexpr (TileData::Loc == pto::TileType::Vec) {
        TLoad<A5LoadOp, TileData, GlobalData>(
            dst.data(), src.data(), src.GetShape(pto::GlobalTensorDim::DIM_0),
            src.GetShape(pto::GlobalTensorDim::DIM_1), src.GetShape(pto::GlobalTensorDim::DIM_2),
            src.GetShape(pto::GlobalTensorDim::DIM_3), src.GetShape(pto::GlobalTensorDim::DIM_4),
            src.GetStride(pto::GlobalTensorDim::DIM_0), src.GetStride(pto::GlobalTensorDim::DIM_1),
            src.GetStride(pto::GlobalTensorDim::DIM_2), src.GetStride(pto::GlobalTensorDim::DIM_3),
            src.GetStride(pto::GlobalTensorDim::DIM_4), dst.GetValidRow(), dst.GetValidCol());
    } else if constexpr (TileData::Loc == pto::TileType::Mat) {
        if constexpr ((TileData::Rows == 1) && (TileData::SFractal == SLayout::RowMajor && TileData::isRowMajor)) {
            TLoadMxCubeCheck<TileData, GlobalData>();
            TLoadMxCubeAVector<A5LoadOp, TileData, GlobalData>(
                dst.data(), src.data(), src.GetShape(0), src.GetShape(1), src.GetShape(2), src.GetShape(3),
                src.GetShape(4), src.GetStride(0), src.GetStride(1), src.GetStride(2), src.GetStride(3),
                src.GetStride(4), dst.GetValidRow(), dst.GetValidCol());
        } else if constexpr (!IsScale<TileData, GlobalData>()) {
            TLoadCubeCheck<TileData, GlobalData>();
            TLoadCube<A5LoadOp, TileData, GlobalData>(
                dst.data(), src.data(), src.GetShape(pto::GlobalTensorDim::DIM_0),
                src.GetShape(pto::GlobalTensorDim::DIM_1), src.GetShape(pto::GlobalTensorDim::DIM_2),
                src.GetShape(pto::GlobalTensorDim::DIM_3), src.GetShape(pto::GlobalTensorDim::DIM_4),
                src.GetStride(pto::GlobalTensorDim::DIM_0), src.GetStride(pto::GlobalTensorDim::DIM_1),
                src.GetStride(pto::GlobalTensorDim::DIM_2), src.GetStride(pto::GlobalTensorDim::DIM_3),
                src.GetStride(pto::GlobalTensorDim::DIM_4), dst.GetValidRow(), dst.GetValidCol());
        } else if constexpr (IsScale<TileData, GlobalData>()) {
            TLoadMxCubeCheck<TileData, GlobalData>();
            TLoadMxCube<A5LoadOp, TileData, GlobalData>(
                dst.data(), src.data(), src.GetShape(0), src.GetShape(1), src.GetShape(2), src.GetShape(3),
                src.GetShape(4), src.GetStride(0), src.GetStride(1), src.GetStride(2), src.GetStride(3),
                src.GetStride(4), dst.GetValidRow(), dst.GetValidCol());
        }
    }
}

template <typename Op, typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoadNHWC(typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__ *src,
                                   int srcShape0, int srcShape1, int srcShape2, int srcShape3, int srcShape4,
                                   int gStride0, int gStride1, int gStride2, int gStride3, int gStride4, int dstShape0,
                                   int dstShape1, int dstShape2, int dstShape3)
{
#if defined(__DAV_CUBE__)
    __cbuf__ typename TileData::DType *dstAddr = (__cbuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);
    typename GlobalData::DType *srcAddr = src;

    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    typename GlobalData::DType *srcAddrP = srcAddr;
    __cbuf__ typename TileData::DType *dstAddrP = dstAddr;
    __cbuf__ typename TileData::DType *dstAddrO = dstAddr;

    // ConvTile layout is [N,C1,H,W,C0] = [dstShape0, dstShape1, dstShape2, dstShape3, c0ElemCount]
    // GlobalTensor layout is [1,N,H,W,C] = [1, srcShape1, srcShape2, srcShape3, srcShape4]
    PTO_ASSERT(srcShape1 == dstShape0 && srcShape2 == dstShape2 && srcShape3 == dstShape3,
               "Fix: src layout is [1,N,H,W,C],dst layout [N,C1,H,W,C0], srcShape dstShape should be same!");
    PTO_ASSERT(srcShape4 <= dstShape1 * c0ElemCount,
               "Fix: src layout is [1,N,H,W,C],dst layout [N,C1,H,W,C0], srcC should <= dstC1 * dstC0!");

    uint16_t ndNum = srcShape2;  // srcH is ndNum
    uint16_t nValue = srcShape3; // srcW is nValue
    uint32_t dValue = srcShape4; // srcC is dValue

    uint64_t loop1SrcStride = GetByteSize<typename TileData::DType>(gStride3); // unit Byte
    uint64_t loop4SrcStride = GetByteSize<typename TileData::DType>(gStride2); // unit Byte
    constexpr uint16_t loop2DstStride = 1;
    uint16_t loop3DstStride = dstShape2 * dstShape3;                   // unit is 32B
    uint16_t loop4DstStride = dstShape3;                               // dstW
    uint64_t mte2NzPara = static_cast<uint64_t>(loop4DstStride) << 48; // MTE2_NZ_PARA[63:48]
    mte2NzPara |= static_cast<uint64_t>(loop3DstStride) << 32;         // MTE2_NZ_PARA[47:32]
    mte2NzPara |= static_cast<uint64_t>(loop2DstStride) << 16;         // MTE2_NZ_PARA[31:16]
    mte2NzPara |= static_cast<uint64_t>(ndNum);                        // MTE2_NZ_PARA[15:0]
    set_mte2_nz_para(mte2NzPara);                                      // only set once

    for (uint32_t i = 0; i < dstShape0; i++) { // use nd2nz
        srcAddrP = src + i * gStride1;
        dstAddrP = dstAddrO + i * dstShape1 * dstShape2 * dstShape3 * c0ElemCount;
        Op::TLoadCubeInstr(dstAddrP, srcAddrP, loop1SrcStride, nValue, dValue, loop4SrcStride);
    }
#endif
}

template <typename Op, typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoadNCHW(typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__ *src,
                                   int srcShape0, int srcShape1, int srcShape2, int srcShape3, int srcShape4,
                                   int gStride0, int gStride1, int gStride2, int gStride3, int gStride4, int dstShape0,
                                   int dstShape1, int dstShape2, int dstShape3)
{
#if defined(__DAV_CUBE__)
    __cbuf__ typename TileData::DType *dstAddr = (__cbuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);

    typename GlobalData::DType *srcAddrP = src;
    __cbuf__ typename TileData::DType *dstAddrP = dstAddr;
    __cbuf__ typename TileData::DType *dstAddrO = dstAddr;
    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);

    // ConvTile layout is [N,C1,H,W,C0] = [dstShape0, dstShape1, dstShape2, dstShape3, c0ElemCount]
    // GlobalTensor layout is [1,N,C,H,W] = [1, srcShape1, srcShape2, srcShape3, srcShape4]
    PTO_ASSERT(srcShape1 == dstShape0 && srcShape3 == dstShape2 && srcShape4 == dstShape3,
               "Fix: src layout is [1,N,C,H,W],dst layout [N,C1,H,W,C0], srcShape dstShape should be same!");
    PTO_ASSERT(srcShape2 <= dstShape1 * c0ElemCount,
               "Fix: src layout is [1,N,C,H,W],dst layout [N,C1,H,W,C0], srcC should <= dstC1 * dstC0!");

    uint16_t nValue = srcShape4; // srcW is nValue
    uint32_t dValue = srcShape2; // srcC is dValue
    uint64_t loop4SrcStride = 0;
    uint64_t loop1SrcStride = GetByteSize<typename TileData::DType>(gStride2);
    constexpr uint16_t loop2DstStride = 1;
    uint16_t loop3DstStride = dstShape2 * dstShape3;                   // unit is 32B
    uint16_t loop4DstStride = dstShape3;                               // dstW
    uint64_t mte2NzPara = static_cast<uint64_t>(loop4DstStride) << 48; // MTE2_NZ_PARA[63:48]
    mte2NzPara |= static_cast<uint64_t>(loop3DstStride) << 32;         // MTE2_NZ_PARA[47:32]
    mte2NzPara |= static_cast<uint64_t>(loop2DstStride) << 16;         // MTE2_NZ_PARA[31:16]

    if (dstShape3 == gStride3) {                // w direction all load
        mte2NzPara |= static_cast<uint64_t>(1); // MTE2_NZ_PARA[15:0] dnNum = 1
        set_mte2_nz_para(mte2NzPara);
        nValue = srcShape4 * srcShape3;
    } else if (dstShape3 < gStride3) {
        mte2NzPara |= static_cast<uint64_t>(srcShape3); // MTE2_NZ_PARA[15:0] dnNum = srcH
        set_mte2_nz_para(mte2NzPara);
        loop4SrcStride = GetByteSize<typename TileData::DType>(gStride3);
    }

    for (uint32_t i = 0; i < dstShape0; i++) {
        srcAddrP = src + i * gStride1;
        dstAddrP = dstAddrO + i * dstShape1 * dstShape2 * dstShape3 * c0ElemCount;
        Op::template TLoadCubeInstr<pto::Layout::DN>(dstAddrP, srcAddrP, loop1SrcStride, nValue, dValue,
                                                     loop4SrcStride);
    }
#endif
}

template <typename Op, typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoadNCHW2FractalZ(typename TileData::TileDType __out__ dst,
                                            typename GlobalData::DType __in__ *src, int srcShape0, int srcShape1,
                                            int srcShape2, int srcShape3, int srcShape4, int gStride0, int gStride1,
                                            int gStride2, int gStride3, int gStride4, int dstShape0, int dstShape1,
                                            int dstShape2, int dstShape3)
{
#if defined(__DAV_CUBE__)
    __cbuf__ typename TileData::DType *dstAddr = (__cbuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);
    typename GlobalData::DType *srcAddr = src;

    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    typename GlobalData::DType *srcAddrP = srcAddr;
    __cbuf__ typename TileData::DType *dstAddrP = dstAddr;

    // ConvTile layout is [C1HW,N/16,16,C0] = [dstShape0, dstShape1, dstShape2, dstShape3]
    // GlobalTensor layout is [1,N,C,H,W] = [1, srcShape1, srcShape2, srcShape3, srcShape4]
    PTO_ASSERT(gStride2 == srcShape3 * srcShape4,
               "Fix: src layout is [1,N,C,H,W],dst layout [N,C1,H,W,C0], H*W should be all load");

    uint16_t dnNum = srcShape1;
    uint16_t nValue = gStride2;  // H*W all load
    uint32_t dValue = srcShape2; // srcC is dValue

    uint32_t c1Size = CeilDivision(srcShape2, c0ElemCount);
    uint32_t dstHW = dstShape0 / c1Size;

    uint64_t loop1SrcStride = GetByteSize<typename TileData::DType>(gStride2); // global H*W, unit Byte
    uint64_t loop4SrcStride = GetByteSize<typename TileData::DType>(gStride1); // global C*H*W, unit Byte
    uint16_t loop2DstStride = dstShape1 * dstShape2;
    uint16_t loop3DstStride = loop2DstStride * dstHW;                  // unit is 32B
    constexpr uint16_t loop4DstStride = 1;                             // each c0 of contiguous dNnum save continously
    uint64_t mte2NzPara = static_cast<uint64_t>(loop4DstStride) << 48; // MTE2_NZ_PARA[63:48]
    mte2NzPara |= static_cast<uint64_t>(loop3DstStride) << 32;         // MTE2_NZ_PARA[47:32]
    mte2NzPara |= static_cast<uint64_t>(loop2DstStride) << 16;         // MTE2_NZ_PARA[31:16]
    mte2NzPara |= static_cast<uint64_t>(dnNum);                        // MTE2_NZ_PARA[15:0]
    set_mte2_nz_para(mte2NzPara);                                      // only set once

    Op::template TLoadCubeInstr<pto::Layout::DN>(dstAddrP, srcAddrP, loop1SrcStride, nValue, dValue, loop4SrcStride);
#endif
}

template <typename Op, typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoadNCDHW2NDC1HWC0(typename TileData::TileDType __out__ dst,
                                             typename GlobalData::DType __in__ *src, int srcShape0, int srcShape1,
                                             int srcShape2, int srcShape3, int srcShape4, int gStride0, int gStride1,
                                             int gStride2, int gStride3, int gStride4, int dstShape0, int dstShape1,
                                             int dstShape2, int dstShape3, int dstShape4)
{
#if defined(__DAV_CUBE__)
    __cbuf__ typename TileData::DType *dstAddr = (__cbuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);
    typename GlobalData::DType *srcAddr = src;

    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    typename GlobalData::DType *srcAddrP = srcAddr;
    __cbuf__ typename TileData::DType *dstAddrP = dstAddr;
    __cbuf__ typename TileData::DType *dstAddrO = dstAddr;

    // ConvTile layout is [N,D,C1,H,W,C0] = [dstShape0, dstShape1, dstShape2, dstShape3, dstShape4, C0]
    // GlobalTensor layout is [N,C,D,H,W] = [srcShape0, srcShape1, srcShape2, srcShape3, srcShape4]
    PTO_ASSERT(srcShape0 == dstShape0 && srcShape3 == dstShape3 && srcShape4 == dstShape4,
               "Fix: src layout is [N,C,D,H,W],dst layout [N,D,C1,H,W,C0], srcShape dstShape should be same!");

    uint16_t nValue = srcShape3 * srcShape4; // srcH*srcW is nValue
    uint32_t dValue = srcShape1;             // srcC is dValue
    uint64_t loop4SrcStride = 0;
    uint64_t loop1SrcStride = GetByteSize<typename TileData::DType>(gStride1); // unit Byte
    constexpr uint16_t loop2DstStride = 1;
    uint16_t loop3DstStride = dstShape3 * dstShape4;                   // unit is 32B
    uint16_t loop4DstStride = dstShape4;                               // dstW
    uint64_t mte2NzPara = static_cast<uint64_t>(loop4DstStride) << 48; // MTE2_NZ_PARA[63:48]
    mte2NzPara |= static_cast<uint64_t>(loop3DstStride) << 32;         // MTE2_NZ_PARA[47:32]
    mte2NzPara |= static_cast<uint64_t>(loop2DstStride) << 16;         // MTE2_NZ_PARA[31:16]

    if (dstShape4 == gStride3) {                // w direction all load
        mte2NzPara |= static_cast<uint64_t>(1); // MTE2_NZ_PARA[15:0] dnNum = 1
        set_mte2_nz_para(mte2NzPara);
    } else if (dstShape4 < gStride3) {
        nValue = srcShape4;
        mte2NzPara |= static_cast<uint64_t>(srcShape3); // MTE2_NZ_PARA[15:0] dnNum = srcW
        set_mte2_nz_para(mte2NzPara);
        loop4SrcStride = GetByteSize<typename TileData::DType>(gStride3);
    }

    for (uint32_t i = 0; i < dstShape0; i++) {
        srcAddr = src + i * gStride0;
        dstAddr = dstAddrO + i * dstShape1 * dstShape2 * dstShape3 * dstShape4 * c0ElemCount;
        for (uint32_t j = 0; j < srcShape2; j++) { // use dn2nz, inner iterations : srcD
            srcAddrP = srcAddr + j * gStride2;
            dstAddrP = dstAddr + j * dstShape2 * dstShape3 * dstShape4 * c0ElemCount;
            Op::template TLoadCubeInstr<pto::Layout::DN>(dstAddrP, srcAddrP, loop1SrcStride, nValue, dValue,
                                                         loop4SrcStride);
        }
    }
#endif
}

template <typename Op, typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoadNCDHW2FractalZ3D(typename TileData::TileDType __out__ dst,
                                               typename GlobalData::DType __in__ *src, int srcShape0, int srcShape1,
                                               int srcShape2, int srcShape3, int srcShape4, int gStride0, int gStride1,
                                               int gStride2, int gStride3, int gStride4, int dstShape0, int dstShape1,
                                               int dstShape2, int dstShape3)
{
#if defined(__DAV_CUBE__)
    __cbuf__ typename TileData::DType *dstAddr = (__cbuf__ typename TileData::DType *)__cce_get_tile_ptr(dst);
    typename GlobalData::DType *srcAddr = src;

    typename GlobalData::DType *srcAddrP = srcAddr;
    __cbuf__ typename TileData::DType *dstAddrP = dstAddr;
    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    // ConvTile layout is [C1DHW,N/16,16,C0] = [dstShape0, dstShape1, dstShape2, dstShape3]
    // GlobalTensor layout is [N,C,D,H,W] = [srcShape0, srcShape1, srcShape2, srcShape3, srcShape4]
    PTO_ASSERT(gStride2 == srcShape3 * srcShape4,
               "Fix: src layout is [N,C,D,H,W],dst layout [N,C1,H,W,C0], H*W should be all load");

    uint16_t dnNum = srcShape0;
    uint16_t nValue = gStride2 * srcShape2;
    uint32_t dValue = srcShape1; // srcC is dValue

    uint32_t c1Size = CeilDivision(srcShape1, c0ElemCount);
    uint32_t dstDHW = dstShape0 / c1Size;

    constexpr uint16_t loop4DstStride = 1; // each c0 of contiguous dNnum save continously
    uint64_t loop1SrcStride = GetByteSize<typename TileData::DType>(gStride1); // global D*H*W, unit Byte
    uint64_t loop4SrcStride = GetByteSize<typename TileData::DType>(gStride0); // global C*D*H*W, unit Byte
    uint16_t loop2DstStride = dstShape1 * dstShape2;
    uint16_t loop3DstStride = loop2DstStride * dstDHW;               // unit is 32B
    uint64_t mte2Para = static_cast<uint64_t>(loop4DstStride) << 48; // MTE2_NZ_PARA[63:48]
    mte2Para |= static_cast<uint64_t>(loop3DstStride) << 32;         // MTE2_NZ_PARA[47:32]
    mte2Para |= static_cast<uint64_t>(loop2DstStride) << 16;         // MTE2_NZ_PARA[31:16]
    mte2Para |= static_cast<uint64_t>(dnNum);                        // MTE2_NZ_PARA[15:0]
    set_mte2_nz_para(mte2Para);                                      // only set once

    Op::template TLoadCubeInstr<pto::Layout::DN>(dstAddrP, srcAddrP, loop1SrcStride, nValue, dValue, loop4SrcStride);
#endif
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void CheckConvTileData(TileData &dst, GlobalData &src)
{
    static_assert(
        caps::IsInt8<typename TileData::DType>() || caps::IsInt16<typename TileData::DType>() ||
            caps::IsInt32<typename TileData::DType>() || caps::IsFP8E4M3<typename TileData::DType>() ||
            caps::IsFP8E5M2<typename TileData::DType>() || caps::IsFP16<typename TileData::DType>() ||
            caps::IsBF16<typename TileData::DType>() || caps::IsFP32<typename TileData::DType>(),
        "Fix: Data type must be "
        "int8_t/uint8_t/float8_e4m3_t/float8_e5m2_t/int16_t/uint16_t/int32_t/uint32_t/half/bfloat16_t/float!");
    static_assert(TileData::Loc == pto::TileType::Mat, "Fix: Dst TileType must be Mat!");
    static_assert(sizeof(typename TileData::DType) == sizeof(typename GlobalData::DType),
                  "Fix: Source dtype must be same with dst dtype!");

    constexpr bool isSameLayout =
        (GlobalData::layout == pto::Layout::NC1HWC0 && TileData::layout == pto::Layout::NC1HWC0) ||
        (GlobalData::layout == pto::Layout::FRACTAL_Z && TileData::layout == pto::Layout::FRACTAL_Z) ||
        (GlobalData::layout == pto::Layout::NHWC && TileData::layout == pto::Layout::NC1HWC0) ||
        (GlobalData::layout == pto::Layout::NCHW && TileData::layout == pto::Layout::NC1HWC0) ||
        (GlobalData::layout == pto::Layout::NCHW && TileData::layout == pto::Layout::FRACTAL_Z) ||
        (GlobalData::layout == pto::Layout::NCDHW && TileData::layout == pto::Layout::FRACTAL_Z_3D) ||
        (GlobalData::layout == pto::Layout::NCDHW && TileData::layout == pto::Layout::NDC1HWC0);
    static_assert(isSameLayout == true, "Fix: Src layout must be NC1HWC0 or FRACTAL_Z or NHWC or NCHW or NCDHW!");
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLOAD_CONVTILE_IMPL(TileData &dst, GlobalData &src)
{
    CheckConvTileData<TileData, GlobalData>(dst, src);
    if constexpr (GlobalData::layout == pto::Layout::NC1HWC0) { // layout is [N,C1,H,W,C0]
        TLoad5HD<A5LoadOp, TileData, GlobalData>(dst.data(), src.data(), src.GetShape(0), src.GetShape(1),
                                                 src.GetShape(2), src.GetShape(3), src.GetStride(0), src.GetStride(1),
                                                 src.GetStride(2), src.GetStride(3), src.GetStride(4), dst.GetShape(0),
                                                 dst.GetShape(1), dst.GetShape(2), dst.GetShape(3));
    } else if constexpr (GlobalData::layout == pto::Layout::FRACTAL_Z) {
        if constexpr (TileData::totalDimCount == 4) { // layout is [C1HW,N/16,16,C0]
            TLoadFractalZ<A5LoadOp, TileData, GlobalData>(
                dst.data(), src.data(), src.GetShape(0), src.GetShape(1), src.GetShape(2), src.GetShape(3),
                src.GetShape(4), src.GetStride(0), src.GetStride(1), src.GetStride(2), src.GetStride(3),
                src.GetStride(4), dst.GetShape(0), dst.GetShape(1), dst.GetShape(2), dst.GetShape(3));
        } else { // layout is [C1,H,W,N,C0]
            TLoad5HD<A5LoadOp, TileData, GlobalData>(
                dst.data(), src.data(), src.GetShape(0), src.GetShape(1), src.GetShape(2), src.GetShape(3),
                src.GetStride(0), src.GetStride(1), src.GetStride(2), src.GetStride(3), src.GetStride(4),
                dst.GetShape(0), dst.GetShape(1), dst.GetShape(2), dst.GetShape(3));
        }
    } else if constexpr (GlobalData::layout == pto::Layout::NHWC) { // NHWC->NC1HWC0
        TLoadNHWC<A5LoadOp, TileData, GlobalData>(
            dst.data(), src.data(), src.GetShape(0), src.GetShape(1), src.GetShape(2), src.GetShape(3), src.GetShape(4),
            src.GetStride(0), src.GetStride(1), src.GetStride(2), src.GetStride(3), src.GetStride(4), dst.GetShape(0),
            dst.GetShape(1), dst.GetShape(2), dst.GetShape(3));
    } else if constexpr (GlobalData::layout == pto::Layout::NCHW &&
                         TileData::layout == pto::Layout::NC1HWC0) { // NCHW->NC1HWC0
        TLoadNCHW<A5LoadOp, TileData, GlobalData>(
            dst.data(), src.data(), src.GetShape(0), src.GetShape(1), src.GetShape(2), src.GetShape(3), src.GetShape(4),
            src.GetStride(0), src.GetStride(1), src.GetStride(2), src.GetStride(3), src.GetStride(4), dst.GetShape(0),
            dst.GetShape(1), dst.GetShape(2), dst.GetShape(3));
    } else if constexpr (GlobalData::layout == pto::Layout::NCHW &&
                         TileData::layout == pto::Layout::FRACTAL_Z) { // NCHW->[C1HW,N/16,16,C0]
        TLoadNCHW2FractalZ<A5LoadOp, TileData, GlobalData>(
            dst.data(), src.data(), src.GetShape(0), src.GetShape(1), src.GetShape(2), src.GetShape(3), src.GetShape(4),
            src.GetStride(0), src.GetStride(1), src.GetStride(2), src.GetStride(3), src.GetStride(4), dst.GetShape(0),
            dst.GetShape(1), dst.GetShape(2), dst.GetShape(3));
    } else if constexpr (GlobalData::layout == pto::Layout::NCDHW && TileData::layout == pto::Layout::NDC1HWC0) {
        TLoadNCDHW2NDC1HWC0<A5LoadOp, TileData, GlobalData>(
            dst.data(), src.data(), src.GetShape(0), src.GetShape(1), src.GetShape(2), src.GetShape(3), src.GetShape(4),
            src.GetStride(0), src.GetStride(1), src.GetStride(2), src.GetStride(3), src.GetStride(4), dst.GetShape(0),
            dst.GetShape(1), dst.GetShape(2), dst.GetShape(3), dst.GetShape(4));
    } else if constexpr (GlobalData::layout == pto::Layout::NCDHW && TileData::layout == pto::Layout::FRACTAL_Z_3D) {
        TLoadNCDHW2FractalZ3D<A5LoadOp, TileData, GlobalData>(
            dst.data(), src.data(), src.GetShape(0), src.GetShape(1), src.GetShape(2), src.GetShape(3), src.GetShape(4),
            src.GetStride(0), src.GetStride(1), src.GetStride(2), src.GetStride(3), src.GetStride(4), dst.GetShape(0),
            dst.GetShape(1), dst.GetShape(2), dst.GetShape(3));
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLOAD_IMPL(TileData &dst, GlobalData &src)
{
    if constexpr (is_conv_tile_v<TileData>) {
        TLOAD_CONVTILE_IMPL(dst, src);
    } else {
        TLOAD_TILE_IMPL(dst, src);
    }
}
} // namespace pto
#endif // TLOAD_HPP
