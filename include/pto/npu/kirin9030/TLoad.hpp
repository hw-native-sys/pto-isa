/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifdef PTO_NPU_ARCH_KIRINX90
#include "pto/npu/a2a3/TLoad.hpp"
#elif defined(PTO_NPU_ARCH_KIRIN9030)
#ifndef TLOAD_HPP
#define TLOAD_HPP
#include "common.hpp"
#include <pto/common/utils.hpp>

namespace pto {
template <typename DstTile, typename SrcGlobal>
PTO_INTERNAL void TLoadInstr(__ubuf__ typename DstTile::DType *dst, typename SrcGlobal::DType *src, uint32_t nBurst,
                             uint32_t lenBurst, uint64_t gmStride, uint32_t ubStride, bool enableUBPad)
{
    using LoadT = LoadTypeBySize_t<typename DstTile::DType>;
    pto_copy_gm_to_ubuf_align_v2(reinterpret_cast<__ubuf__ LoadT *>(dst), reinterpret_cast<__gm__ LoadT *>(src), 0,
                                 nBurst, lenBurst, 0 /*left padding count*/, 0 /*right padding count*/,
                                 enableUBPad /*data select bit*/, 0, gmStride, ubStride);
}

template <typename DstTile, typename SrcGlobal>
PTO_INTERNAL void TLoadVecND2ND(__ubuf__ typename DstTile::DType *dstAddr, typename SrcGlobal::DType *srcAddr,
                                int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol,
                                bool enableUBPad)
{
    typename SrcGlobal::DType *srcAddrP = srcAddr;
    __ubuf__ typename DstTile::DType *dstAddrP = dstAddr;
    uint32_t nBurst = gShape3;
    uint32_t lenBurst = GetByteSize<typename DstTile::DType>(validCol);
    uint64_t gmStride = GetByteSize<typename DstTile::DType>(gStride3);
    uint32_t ubStride = GetByteSize<typename DstTile::DType>(DstTile::Cols);

    int64_t dstStride2 = gShape3 * DstTile::Cols;
    int64_t dstStride1 = gShape2 * dstStride2;
    int64_t dstStride0 = gShape1 * dstStride1;
    uint64_t loop2 = gShape1;
    uint64_t loop1 = gShape2;
    uint64_t loop2_src_stride = GetByteSize<typename DstTile::DType>(gStride1);
    uint64_t loop1_src_stride = GetByteSize<typename DstTile::DType>(gStride2);
    uint64_t loop2_dst_stride = GetByteSize<typename DstTile::DType>(dstStride1);
    uint64_t loop1_dst_stride = GetByteSize<typename DstTile::DType>(dstStride2);
    set_loop2_stride_outtoub(loop2_dst_stride << 40 | loop2_src_stride);
    set_loop1_stride_outtoub(loop1_dst_stride << 40 | loop1_src_stride);
    set_loop_size_outtoub(loop2 << 21 | loop1);
    for (uint32_t i = 0; i < gShape0; i++) {
        int64_t dstAddr0 = i * dstStride0;
        int64_t srcAddr0 = i * gStride0;
        dstAddrP = dstAddr + dstAddr0;
        srcAddrP = srcAddr + srcAddr0;
        TLoadInstr<DstTile, SrcGlobal>(dstAddrP, srcAddrP, nBurst, lenBurst, gmStride, ubStride, enableUBPad);
    }
}
template <typename DstTile, typename SrcGlobal>
PTO_INTERNAL void TLoadVecDN2DN(__ubuf__ typename DstTile::DType *dstAddr, typename SrcGlobal::DType *srcAddr,
                                int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol,
                                bool enableUBPad)
{
    uint32_t nBurst = gShape4;
    uint32_t lenBurst = GetByteSize<typename DstTile::DType>(validRow);
    uint64_t gmStride = GetByteSize<typename DstTile::DType>(gStride4);
    uint32_t ubStride = GetByteSize<typename DstTile::DType>(DstTile::Rows);

    typename SrcGlobal::DType *srcAddrP = srcAddr;
    __ubuf__ typename DstTile::DType *dstAddrP = dstAddr;

    int64_t dstStride2 = gShape4 * DstTile::Rows;
    int64_t dstStride1 = gShape2 * dstStride2;
    int64_t dstStride0 = gShape1 * dstStride1;

    uint64_t loop2 = gShape1;
    uint64_t loop1 = gShape2;
    uint64_t loop2_src_stride = GetByteSize<typename DstTile::DType>(gStride1);
    uint64_t loop1_src_stride = GetByteSize<typename DstTile::DType>(gStride2);
    uint64_t loop2_dst_stride = GetByteSize<typename DstTile::DType>(dstStride1);
    uint64_t loop1_dst_stride = GetByteSize<typename DstTile::DType>(dstStride2);
    set_loop2_stride_outtoub(loop2_dst_stride << 40 | loop2_src_stride);
    set_loop1_stride_outtoub(loop1_dst_stride << 40 | loop1_src_stride);
    set_loop_size_outtoub(loop2 << 21 | loop1);
    for (uint32_t i = 0; i < gShape0; i++) {
        int64_t dstAddr0 = i * dstStride0;
        int64_t srcAddr0 = i * gStride0;
        dstAddrP = dstAddr + dstAddr0;
        srcAddrP = srcAddr + srcAddr0;
        TLoadInstr<DstTile, SrcGlobal>(dstAddrP, srcAddrP, nBurst, lenBurst, gmStride, ubStride, enableUBPad);
    }
}
template <typename DstTile, typename SrcGlobal>
PTO_INTERNAL void TLoadVecNZ2NZ(__ubuf__ typename DstTile::DType *dstAddr, typename SrcGlobal::DType *srcAddr,
                                int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    uint32_t nBurst = gShape1;
    uint32_t lenBurst = validRow * C0_SIZE_BYTE;
    uint32_t gmStride = GetByteSize<typename DstTile::DType>(gStride1);
    uint32_t ubStride = DstTile::Rows * C0_SIZE_BYTE;

    typename SrcGlobal::DType *srcAddrP = srcAddr;
    __ubuf__ typename DstTile::DType *dstAddrP = dstAddr;

    int64_t tileStride = gShape1 * DstTile::Rows * gShape4;
    set_loop_size_outtoub(1ULL << 21 | 1ULL);
    for (uint32_t i = 0; i < gShape0; i++) {
        srcAddrP = srcAddr + i * gStride0;
        dstAddrP = dstAddr + i * tileStride;
        TLoadInstr<DstTile, SrcGlobal>(dstAddrP, srcAddrP, nBurst, lenBurst, gmStride, ubStride, 0);
    }
}

template <typename DstTile, typename SrcGlobal>
__tf__ PTO_INTERNAL OP_NAME(TLOAD)
    OP_TYPE(memory) void TLoad(typename DstTile::TileDType __out__ dst, typename SrcGlobal::DType __in__ *src,
                               int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                               int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    __ubuf__ typename DstTile::DType *dstAddr = (__ubuf__ typename DstTile::DType *)__cce_get_tile_ptr(dst);
    typename SrcGlobal::DType *srcAddr = src;
    constexpr bool enableUBPad = DstTile::PadVal != PadValue::Null;
    if constexpr (enableUBPad) {
        set_pad_val_outtoub(GetPadValue<DstTile>());
    }
    if constexpr (DstTile::isRowMajor && (DstTile::SFractal == SLayout::NoneBox)) {
        TLoadVecND2ND<DstTile, SrcGlobal>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                          gStride1, gStride2, gStride3, gStride4, validRow, validCol, enableUBPad);
    } else if constexpr (!DstTile::isRowMajor && (DstTile::SFractal == SLayout::NoneBox)) {
        TLoadVecDN2DN<DstTile, SrcGlobal>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                          gStride1, gStride2, gStride3, gStride4, validRow, validCol, enableUBPad);
    } else if constexpr (!DstTile::isRowMajor && (DstTile::SFractal == SLayout::RowMajor)) {
        TLoadVecNZ2NZ<DstTile, SrcGlobal>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                          gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    }
}

template <typename DstTile, typename SrcGlobal>
PTO_INTERNAL void TLoadCubeCheck()
{
    // support ND2NZ DN2NZ ND2ND DN2DN NZ2NZ DN2ZN
    static_assert(((SrcGlobal::layout == pto::Layout::ND) &&
                   (!DstTile::isRowMajor && (DstTile::SFractal == SLayout::RowMajor))) ||
                      ((SrcGlobal::layout == pto::Layout::DN) &&
                       (!DstTile::isRowMajor && (DstTile::SFractal == SLayout::RowMajor))) ||
                      ((SrcGlobal::layout == pto::Layout::NZ) &&
                       (!DstTile::isRowMajor && (DstTile::SFractal == SLayout::RowMajor))) ||
                      (((SrcGlobal::layout == pto::Layout::ND) &&
                        (DstTile::isRowMajor && (DstTile::SFractal == SLayout::NoneBox)))) ||
                      (((SrcGlobal::layout == pto::Layout::DN) &&
                        (!DstTile::isRowMajor && (DstTile::SFractal == SLayout::NoneBox)))) ||
                      (((SrcGlobal::layout == pto::Layout::DN) &&
                        (DstTile::isRowMajor && (DstTile::SFractal == SLayout::ColMajor)))),
                  "Fix: now only support ND2NZ DN2NZ ND2ND DN2DN NZ2NZ DN2ZN in current platform");

    // L1 space check
    static_assert(DstTile::Rows <= 16384, "Fix: DstTile::Rows must less than 16384 in L1");
    static_assert(DstTile::Rows * DstTile::Cols <= 512 * 1024, "Fix: DstTile static shape must less than 512KB in L1");

    // ND2NZ or DN2NZ
    if constexpr ((SrcGlobal::layout == pto::Layout::ND || SrcGlobal::layout == pto::Layout::DN) &&
                  (!DstTile::isRowMajor && (DstTile::SFractal == SLayout::RowMajor))) {
        static_assert(DstTile::SFractalSize == 512, "Fix: DstTile SFractalSize must be 512 of NZ format in L1");
        static_assert(sizeof(typename DstTile::DType) != 8, "Fix: DType not support b64 in ND2NZ or DN2NZ");
        // globaltensor only support 2 dim
        static_assert(
            SrcGlobal::staticShape[0] == 1 && SrcGlobal::staticShape[1] == 1 && SrcGlobal::staticShape[2] == 1,
            "Fix: GlobalTensor input shape now only support 2 dim");
    }

    // NZ2NZ
    if constexpr ((SrcGlobal::layout == pto::Layout::NZ) &&
                  (!DstTile::isRowMajor && (DstTile::SFractal == SLayout::RowMajor))) {
        static_assert(BLOCK_BYTE_SIZE / sizeof(typename SrcGlobal::DType) == SrcGlobal::staticShape[4] &&
                          BLOCK_LEN == SrcGlobal::staticShape[3],
                      "Fix: Src GlobalTensor staticShape[3][4] must be satisfied with NZ format require!");
    }
}

template <typename DstTile, typename SrcGlobal, Layout Layout = Layout::ND>
PTO_INTERNAL void TLoadCubeInstr(__cbuf__ typename DstTile::DType *dst, typename SrcGlobal::DType *src,
                                 uint64_t loop1SrcStride, uint16_t nValue, uint32_t dValue)
{
    if constexpr (Layout == Layout::ND) {
        if constexpr (sizeof(typename DstTile::DType) == 1) {
            copy_gm_to_cbuf_multi_nd2nz(reinterpret_cast<__cbuf__ uint8_t *>(dst),
                                        reinterpret_cast<__gm__ uint8_t *>(src), 0 /*sid*/, loop1SrcStride, nValue,
                                        dValue, 0, false, false);
        } else if constexpr (sizeof(typename DstTile::DType) == 2) {
            copy_gm_to_cbuf_multi_nd2nz(reinterpret_cast<__cbuf__ uint16_t *>(dst),
                                        reinterpret_cast<__gm__ uint16_t *>(src), 0 /*sid*/, loop1SrcStride, nValue,
                                        dValue, 0, false, false);
        } else if constexpr (sizeof(typename DstTile::DType) == 4) {
            copy_gm_to_cbuf_multi_nd2nz(reinterpret_cast<__cbuf__ uint32_t *>(dst),
                                        reinterpret_cast<__gm__ uint32_t *>(src), 0 /*sid*/, loop1SrcStride, nValue,
                                        dValue, 0, false, false);
        }
    } else {
        static_assert(sizeof(DstTile::DType) == 0, "Fix: TLoad does not support DN2NZ.");
    }
}
template <typename DstTile, typename SrcGlobal>
PTO_INTERNAL void TLoadCubeInstr(__cbuf__ typename DstTile::DType *dst, typename SrcGlobal::DType *src, uint32_t nBurst,
                                 uint32_t lenBurst, uint64_t srcStride, uint32_t dstStride, uint32_t padCount)
{
    if constexpr (sizeof(typename DstTile::DType) == 1) {
        copy_gm_to_cbuf_align_v2(reinterpret_cast<__cbuf__ uint8_t *>(dst), reinterpret_cast<__gm__ uint8_t *>(src),
                                 0 /*sid*/, nBurst, lenBurst, 0 /*left padding count*/,
                                 padCount /*right padding count*/, true /*data select bit*/, srcStride, dstStride);
    } else if constexpr (sizeof(typename DstTile::DType) == 2) {
        copy_gm_to_cbuf_align_v2(reinterpret_cast<__cbuf__ uint16_t *>(dst), reinterpret_cast<__gm__ uint16_t *>(src),
                                 0 /*sid*/, nBurst, lenBurst, 0 /*left padding count*/,
                                 padCount /*right padding count*/, true /*data select bit*/, srcStride, dstStride);
    } else if constexpr (sizeof(typename DstTile::DType) == 4) {
        copy_gm_to_cbuf_align_v2(reinterpret_cast<__cbuf__ uint32_t *>(dst), reinterpret_cast<__gm__ uint32_t *>(src),
                                 0 /*sid*/, nBurst, lenBurst, 0 /*left padding count*/,
                                 padCount /*right padding count*/, true /*data select bit*/, srcStride, dstStride);
    } else if constexpr (sizeof(typename DstTile::DType) == 8) {
        copy_gm_to_cbuf_align_v2(reinterpret_cast<__cbuf__ uint32_t *>(dst), reinterpret_cast<__gm__ uint32_t *>(src),
                                 0 /*sid*/, nBurst, lenBurst, 0 /*left padding count*/,
                                 padCount * 2 /*right padding count*/, true /*data select bit*/, srcStride, dstStride);
    }
}

template <typename DstTile, typename SrcGlobal>
PTO_INTERNAL void TLoadCubeND2NZ(__cbuf__ typename DstTile::DType *dst, typename SrcGlobal::DType *src, int gShape0,
                                 int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1,
                                 int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    uint16_t nValue = gShape3;
    uint32_t dValue = validCol;
    uint64_t loop1SrcStride = GetByteSize<typename DstTile::DType>(gStride3);
    if constexpr (SrcGlobal::layout == pto::Layout::DN) {
        loop1SrcStride = GetByteSize<typename DstTile::DType>(gStride4);
    }
    constexpr uint16_t ndNum = 1;
    uint16_t loop2DstStride = 1;
    uint16_t loop3DstStride = DstTile::Rows;                           // unit is 32B
    uint16_t loop4DstStride = 0;                                       // because ndNum = 1
    uint64_t mte2NzPara = static_cast<uint64_t>(loop4DstStride) << 48; // MTE2_NZ_PARA[63:48]
    mte2NzPara |= static_cast<uint64_t>(loop3DstStride) << 32;         // MTE2_NZ_PARA[47:32]
    mte2NzPara |= static_cast<uint64_t>(loop2DstStride) << 16;         // MTE2_NZ_PARA[31:16]
    mte2NzPara |= static_cast<uint64_t>(ndNum);                        // MTE2_NZ_PARA[15:0]
    set_mte2_nz_para(mte2NzPara);                                      // only set once

    TLoadCubeInstr<DstTile, SrcGlobal, SrcGlobal::layout>(dst, src, loop1SrcStride, nValue, dValue);
}
template <typename DstTile, typename SrcGlobal>
PTO_INTERNAL void TLoadCubeNZ2NZ(__cbuf__ typename DstTile::DType *dst, typename SrcGlobal::DType *src, int gShape0,
                                 int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1,
                                 int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    __cbuf__ typename DstTile::DType *dstAddrP = dst;
    typename SrcGlobal::DType *srcAddrP = src;
    uint32_t nBurst = gShape1;
    uint32_t lenBurst = validRow * BLOCK_BYTE_SIZE;
    uint64_t gmStride = GetByteSize<typename DstTile::DType>(gStride1);
    uint32_t dstStride = DstTile::Rows * BLOCK_BYTE_SIZE;

    int64_t tileStride = gShape1 * DstTile::Rows * gShape4;
    set_loop_size_outtol1(1ULL << 21 | 1ULL);
    for (uint32_t i = 0; i < gShape0; i++) {
        srcAddrP = src + i * gStride0;
        dstAddrP = dst + i * tileStride;
        TLoadCubeInstr<DstTile, SrcGlobal>(dstAddrP, srcAddrP, nBurst, lenBurst, gmStride, dstStride, 0);
    }
}

template <typename DstTile, typename SrcGlobal>
PTO_INTERNAL void TLoadCubeND2ND(__cbuf__ typename DstTile::DType *dst, typename SrcGlobal::DType *src, int gShape0,
                                 int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1,
                                 int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    __cbuf__ typename DstTile::DType *dstAddrP = dst;
    typename SrcGlobal::DType *srcAddrP = src;
    uint32_t nBurst = gShape3;
    uint32_t lenBurst = GetByteSize<typename DstTile::DType>(validCol);
    uint64_t gmStride = GetByteSize<typename DstTile::DType>(gStride3);
    uint32_t dstStride = GetByteSize<typename DstTile::DType>(DstTile::Cols);

    constexpr uint32_t blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename DstTile::DType);
    uint32_t gapElement = (DstTile::Cols - validCol);
    uint32_t padCount = gapElement % blockSizeElem;
    set_pad_val_outtol1(GetPadValue<DstTile>());

    int64_t dstStride2 = gShape3 * DstTile::Cols;
    int64_t dstStride1 = gShape2 * dstStride2;
    int64_t dstStride0 = gShape1 * dstStride1;

    uint64_t loop2 = gShape1;
    uint64_t loop1 = gShape2;
    uint64_t loop2SrcStride = GetByteSize<typename DstTile::DType>(gStride1);
    uint64_t loop1SrcStride = GetByteSize<typename DstTile::DType>(gStride2);
    uint64_t loop2DstStride = GetByteSize<typename DstTile::DType>(dstStride1);
    uint64_t loop1DstStride = GetByteSize<typename DstTile::DType>(dstStride2);

    set_loop2_stride_outtol1(loop2DstStride << 40 | loop2SrcStride);
    set_loop1_stride_outtol1(loop1DstStride << 40 | loop1SrcStride);
    set_loop_size_outtol1(loop2 << 21 | loop1);
    for (uint32_t i = 0; i < gShape0; i++) {
        int64_t dstAddr0 = i * dstStride0;
        int64_t srcAddr0 = i * gStride0;
        dstAddrP = dst + dstAddr0;
        srcAddrP = src + srcAddr0;
        TLoadCubeInstr<DstTile, SrcGlobal>(dstAddrP, srcAddrP, nBurst, lenBurst, gmStride, dstStride, padCount);
    }
}

template <typename DstTile, typename SrcGlobal>
PTO_INTERNAL void TLoadCubeDN2DN(__cbuf__ typename DstTile::DType *dst, typename SrcGlobal::DType *src, int gShape0,
                                 int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1,
                                 int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    __cbuf__ typename DstTile::DType *dstAddrP = dst;
    typename SrcGlobal::DType *srcAddrP = src;
    uint32_t nBurst = gShape4;
    uint32_t lenBurst = GetByteSize<typename DstTile::DType>(validRow);
    uint64_t gmStride = GetByteSize<typename DstTile::DType>(gStride4);
    uint32_t dstStride = GetByteSize<typename DstTile::DType>(DstTile::Rows);

    constexpr uint32_t blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename DstTile::DType);
    uint32_t gapElement = (DstTile::Rows - validRow);
    uint32_t padCount = gapElement % blockSizeElem;
    set_pad_val_outtol1(GetPadValue<DstTile>());

    int64_t dstStride2 = gShape4 * DstTile::Rows;
    int64_t dstStride1 = gShape2 * dstStride2;
    int64_t dstStride0 = gShape1 * dstStride1;
    uint64_t loop2 = gShape1;
    uint64_t loop1 = gShape2;
    uint64_t loop2SrcStride = GetByteSize<typename DstTile::DType>(gStride1);
    uint64_t loop1SrcStride = GetByteSize<typename DstTile::DType>(gStride2);
    uint64_t loop2DstStride = GetByteSize<typename DstTile::DType>(dstStride1);
    uint64_t loop1DstStride = GetByteSize<typename DstTile::DType>(dstStride2);

    set_loop2_stride_outtol1(loop2DstStride << 40 | loop2SrcStride);
    set_loop1_stride_outtol1(loop1DstStride << 40 | loop1SrcStride);
    set_loop_size_outtol1(loop2 << 21 | loop1);
    for (uint32_t i = 0; i < gShape0; i++) {
        int64_t dstAddr0 = i * dstStride0;
        int64_t srcAddr0 = i * gStride0;
        dstAddrP = dst + dstAddr0;
        srcAddrP = src + srcAddr0;
        TLoadCubeInstr<DstTile, SrcGlobal>(dstAddrP, srcAddrP, nBurst, lenBurst, gmStride, dstStride, padCount);
    }
}

template <typename DstTile, typename SrcGlobal>
PTO_INTERNAL void TLoadCubeDN2ZN(__cbuf__ typename DstTile::DType *dst, typename SrcGlobal::DType *src, int gShape0,
                                 int gShape1, int gShape2, int gShape3, int gShape4, int gStride0, int gStride1,
                                 int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    uint16_t nValue = gShape4;
    uint32_t dValue = validRow;
    uint64_t loop1SrcStride = GetByteSize<typename DstTile::DType>(gStride4);

    constexpr uint16_t ndNum = 1;
    uint16_t loop2DstStride = 1;
    uint16_t loop3DstStride = DstTile::Cols;                           // unit is 32B
    uint16_t loop4DstStride = 0;                                       // because ndNum = 1
    uint64_t mte2NzPara = static_cast<uint64_t>(loop4DstStride) << 48; // MTE2_NZ_PARA[63:48]
    mte2NzPara |= static_cast<uint64_t>(loop3DstStride) << 32;         // MTE2_NZ_PARA[47:32]
    mte2NzPara |= static_cast<uint64_t>(loop2DstStride) << 16;         // MTE2_NZ_PARA[31:16]
    mte2NzPara |= static_cast<uint64_t>(ndNum);                        // MTE2_NZ_PARA[15:0]
    set_mte2_nz_para(mte2NzPara);                                      // only set once

    // use nd2nz
    TLoadCubeInstr<DstTile, SrcGlobal, pto::Layout::ND>(dst, src, loop1SrcStride, nValue, dValue);
}

template <typename DstTile, typename SrcGlobal>
__tf__ PTO_INTERNAL void TLoadCube(typename DstTile::TileDType __out__ dst, typename SrcGlobal::DType __in__ *src,
                                   int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                   int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
#if defined(__DAV_CUBE__)
    using L1Type = typename DstTile::TileDType;
    L1Type dstAddr = (L1Type)__cce_get_tile_ptr(dst);

    // ND2NZ or DN2NZ
    if constexpr ((SrcGlobal::layout == pto::Layout::ND || SrcGlobal::layout == pto::Layout::DN) &&
                  (!DstTile::isRowMajor && (DstTile::SFractal == SLayout::RowMajor))) {
        TLoadCubeND2NZ<DstTile, SrcGlobal>(dstAddr, src, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                           gStride1, gStride2, gStride3, gStride4, validRow, validCol);

    } else if constexpr ((SrcGlobal::layout == pto::Layout::NZ) &&
                         (!DstTile::isRowMajor && (DstTile::SFractal == SLayout::RowMajor))) {
        TLoadCubeNZ2NZ<DstTile, SrcGlobal>(dstAddr, src, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                           gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    } else if constexpr ((SrcGlobal::layout == pto::Layout::ND &&
                          (DstTile::isRowMajor && (DstTile::SFractal == SLayout::NoneBox)))) {
        // ND2ND support cols padding
        TLoadCubeND2ND<DstTile, SrcGlobal>(dstAddr, src, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                           gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    } else if constexpr (SrcGlobal::layout == pto::Layout::DN &&
                         (!DstTile::isRowMajor && (DstTile::SFractal == SLayout::NoneBox))) {
        // dn support rows padding
        TLoadCubeDN2DN<DstTile, SrcGlobal>(dstAddr, src, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                           gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    } else if constexpr (SrcGlobal::layout == pto::Layout::DN &&
                         (DstTile::isRowMajor && (DstTile::SFractal == SLayout::ColMajor))) {
        TLoadCubeDN2ZN<DstTile, SrcGlobal>(dstAddr, src, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                           gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    }
#endif
}

template <typename DstTile, typename SrcGlobal>
PTO_INTERNAL constexpr bool IsScale()
{
    if constexpr (SrcGlobal::layout == pto::Layout::MX_A_ND || SrcGlobal::layout == pto::Layout::MX_A_DN ||
                  SrcGlobal::layout == pto::Layout::MX_A_ZZ || SrcGlobal::layout == pto::Layout::MX_B_ND ||
                  SrcGlobal::layout == pto::Layout::MX_B_DN || SrcGlobal::layout == pto::Layout::MX_B_NN) {
        return true;
    }
    return false;
}

template <typename DstTile, typename SrcGlobal>
PTO_INTERNAL void StaticCheck()
{
    static_assert((sizeof(typename DstTile::DType) == 1) || (sizeof(typename DstTile::DType) == 2) ||
                      (sizeof(typename DstTile::DType) == 4) || (sizeof(typename DstTile::DType) == 8),
                  "Fix: Data type must be b8/b16/b32/b64");
    if constexpr (std::is_same<typename DstTile::DType, int64_t>::value ||
                  std::is_same<typename DstTile::DType, uint64_t>::value) {
        static_assert(DstTile::PadVal == PadValue::Null || DstTile::PadVal == PadValue::Zero,
                      "Fix: DstTile::PadVal only support Null or Zero in B64 mode");
    }
    static_assert(sizeof(typename DstTile::DType) == sizeof(typename SrcGlobal::DType),
                  "Fix: Source dtype must be same with dst dtype!");

    // for static shape case, enforce the global tensor (tiled) shape matching with vecTile valid shape for xfer
    if constexpr (DstTile::Loc == pto::TileType::Vec) {
        static_assert(((SrcGlobal::layout == pto::Layout::ND) &&
                       (DstTile::isRowMajor && (DstTile::SFractal == SLayout::NoneBox))) ||
                          ((SrcGlobal::layout == pto::Layout::DN) &&
                           (!DstTile::isRowMajor && (DstTile::SFractal == SLayout::NoneBox))) ||
                          ((SrcGlobal::layout == pto::Layout::NZ) &&
                           (!DstTile::isRowMajor && (DstTile::SFractal == SLayout::RowMajor))),
                      "Src and dst layout must be same!");
        if constexpr (DstTile::isRowMajor && (DstTile::SFractal == SLayout::NoneBox)) {
            if constexpr (DstTile::ValidCol > 0 && SrcGlobal::staticShape[4] > 0) {
                static_assert(DstTile::ValidCol == SrcGlobal::staticShape[4],
                              "Fix: Src GlobalTensor Col and Tile ValidCol must be the same!");
            }
            if constexpr (DstTile::ValidRow > 0 && SrcGlobal::staticShape[0] > 0 && SrcGlobal::staticShape[1] > 0 &&
                          SrcGlobal::staticShape[2] > 0 && SrcGlobal::staticShape[3] > 0) {
                constexpr const int mergedRows = SrcGlobal::staticShape[0] * SrcGlobal::staticShape[1] *
                                                 SrcGlobal::staticShape[2] * SrcGlobal::staticShape[3];
                static_assert(DstTile::ValidRow == mergedRows,
                              "Fix: Src GlobalTensor Row Products and Tile ValidRow must be the same!");
            }
        }
        if constexpr ((SrcGlobal::layout == pto::Layout::NZ) &&
                      (!DstTile::isRowMajor && (DstTile::SFractal == SLayout::RowMajor))) {
            static_assert(BLOCK_BYTE_SIZE / sizeof(typename SrcGlobal::DType) == SrcGlobal::staticShape[4] &&
                              BLOCK_LEN == SrcGlobal::staticShape[3],
                          "Fix: Src GlobalTensor staticShape[3][4] must be satisfied with NZ format require!");
        }
    }
}

template <typename DstTile, typename SrcGlobal>
PTO_INTERNAL void TLOAD_TILE_IMPL(DstTile &dst, SrcGlobal &src)
{
    StaticCheck<DstTile, SrcGlobal>();
    if constexpr (DstTile::Loc == pto::TileType::Vec) {
        TLoad<DstTile, SrcGlobal>(
            dst.data(), src.data(), src.GetShape(pto::GlobalTensorDim::DIM_0),
            src.GetShape(pto::GlobalTensorDim::DIM_1), src.GetShape(pto::GlobalTensorDim::DIM_2),
            src.GetShape(pto::GlobalTensorDim::DIM_3), src.GetShape(pto::GlobalTensorDim::DIM_4),
            src.GetStride(pto::GlobalTensorDim::DIM_0), src.GetStride(pto::GlobalTensorDim::DIM_1),
            src.GetStride(pto::GlobalTensorDim::DIM_2), src.GetStride(pto::GlobalTensorDim::DIM_3),
            src.GetStride(pto::GlobalTensorDim::DIM_4), dst.GetValidRow(), dst.GetValidCol());
    } else if constexpr (DstTile::Loc == pto::TileType::Mat) {
        static_assert(!IsScale<DstTile, SrcGlobal>(), "Fix: TLOAD not supported Mx cube.");
        TLoadCubeCheck<DstTile, SrcGlobal>();
        TLoadCube<DstTile, SrcGlobal>(
            dst.data(), src.data(), src.GetShape(pto::GlobalTensorDim::DIM_0),
            src.GetShape(pto::GlobalTensorDim::DIM_1), src.GetShape(pto::GlobalTensorDim::DIM_2),
            src.GetShape(pto::GlobalTensorDim::DIM_3), src.GetShape(pto::GlobalTensorDim::DIM_4),
            src.GetStride(pto::GlobalTensorDim::DIM_0), src.GetStride(pto::GlobalTensorDim::DIM_1),
            src.GetStride(pto::GlobalTensorDim::DIM_2), src.GetStride(pto::GlobalTensorDim::DIM_3),
            src.GetStride(pto::GlobalTensorDim::DIM_4), dst.GetValidRow(), dst.GetValidCol());
    }
}

template <typename DstTile, typename SrcGlobal>
__tf__ PTO_INTERNAL void TLoad5HD(typename DstTile::TileDType __out__ dst, typename SrcGlobal::DType __in__ *src,
                                  int srcShape0, int srcShape1, int srcShape2, int srcShape3, int gStride0,
                                  int gStride1, int gStride2, int gStride3, int gStride4, int dstShape0, int dstShape1,
                                  int dstShape2, int dstShape3)
{
    __cbuf__ typename DstTile::DType *dstAddr = (__cbuf__ typename DstTile::DType *)__cce_get_tile_ptr(dst);
    typename SrcGlobal::DType *srcAddr = src;

    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename DstTile::DType);

    PTO_ASSERT(srcShape1 == dstShape1 && srcShape2 == dstShape2 && srcShape0 == dstShape0 && srcShape3 == dstShape3,
               "Fix: when layout is NC1HWC0 or C1HWNC0, srcShape dstShape should be same!");

    uint32_t nBurst = dstShape2;
    // lenBurst gmStride dstStride unit is byte
    uint32_t lenBurst = GetByteSize<typename DstTile::DType>(dstShape3 * c0ElemCount);
    uint64_t gmStride = GetByteSize<typename DstTile::DType>(gStride2);
    uint32_t dstStride = lenBurst;

    uint64_t loop2 = dstShape0;
    uint64_t loop1 = dstShape1;
    uint64_t loop2SrcStride = GetByteSize<typename DstTile::DType>(gStride0);
    uint64_t loop2DstStride = GetByteSize<typename DstTile::DType>(dstShape1 * dstShape2 * dstShape3 * c0ElemCount);

    uint64_t loop1SrcStride = GetByteSize<typename DstTile::DType>(gStride1);
    uint64_t loop1DstStride = GetByteSize<typename DstTile::DType>(dstShape2 * dstShape3 * c0ElemCount);
#if defined(__DAV_CUBE__)
    set_loop2_stride_outtol1(loop2DstStride << 40 | loop2SrcStride); // [39:0] is loop2 src stride,[60:40] is dst stride
    set_loop1_stride_outtol1(loop1DstStride << 40 | loop1SrcStride); // [39:0] is loop1 src stride,[60:40] is dst stride
    set_loop_size_outtol1(loop2 << 21 | loop1);                      // [20:0] is loop1 size, [40:21] is loop2 size
    TLoadCubeInstr<DstTile, SrcGlobal>(dstAddr, srcAddr, nBurst, lenBurst, gmStride, dstStride, 0);
#endif
}

template <typename DstTile, typename SrcGlobal>
__tf__ PTO_INTERNAL void TLoadFractalZ(typename DstTile::TileDType __out__ dst, typename SrcGlobal::DType __in__ *src,
                                       int srcShape0, int srcShape1, int srcShape2, int srcShape3, int srcShape4,
                                       int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
                                       int dstShape0, int dstShape1, int dstShape2, int dstShape3)
{
    __cbuf__ typename DstTile::DType *dstAddr = (__cbuf__ typename DstTile::DType *)__cce_get_tile_ptr(dst);
    typename SrcGlobal::DType *srcAddr = src;

    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename DstTile::DType);
    typename SrcGlobal::DType *srcAddrP = srcAddr;
    __cbuf__ typename DstTile::DType *dstAddrP = dstAddr;

    // ConvTile layout is [C1HW,N/16,16,C0]
    static_assert(DstTile::staticShape[2] == FRACTAL_NZ_ROW && DstTile::staticShape[3] == c0ElemCount,
                  "Fix: The DstTile last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");
    static_assert(SrcGlobal::staticShape[3] == FRACTAL_NZ_ROW && SrcGlobal::staticShape[4] == c0ElemCount,
                  "Fix: The GlobalTensor last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");

    PTO_ASSERT(srcShape1 == dstShape0 && srcShape2 == dstShape1 && srcShape3 == dstShape2 && srcShape4 == dstShape3,
               "Fix: layout is [C1HW,N/16,16,C0], srcShape dstShape should be same!");
    uint32_t nBurst = dstShape0; // [1, srcShape1, srcShape2, 16, c0]
    uint32_t lenBurst = dstShape1 * dstShape2 * BLOCK_BYTE_SIZE;
    uint64_t gmStride = GetByteSize<typename DstTile::DType>(gStride1);
    uint32_t dstStride = dstShape1 * dstShape2 * BLOCK_BYTE_SIZE;
#if defined(__DAV_CUBE__)
    set_loop_size_outtol1(1ULL << 21 | 1ULL); // use normal mode
    TLoadCubeInstr<DstTile, SrcGlobal>(dstAddrP, srcAddrP, nBurst, lenBurst, gmStride, dstStride, 0);
#endif
}

template <typename DstTile, typename SrcGlobal>
PTO_INTERNAL void CheckConvTileData(DstTile &dst, SrcGlobal &src)
{
    static_assert(
        std::is_same_v<typename DstTile::DType, int8_t> || std::is_same_v<typename DstTile::DType, uint8_t> ||
            std::is_same_v<typename DstTile::DType, int16_t> || std::is_same_v<typename DstTile::DType, uint16_t> ||
            std::is_same_v<typename DstTile::DType, int32_t> || std::is_same_v<typename DstTile::DType, uint32_t> ||
            std::is_same_v<typename DstTile::DType, half> || std::is_same_v<typename DstTile::DType, float>,
        "Fix: Data type must be int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/half/float!");
    static_assert(DstTile::Loc == pto::TileType::Mat, "Fix: Dst TileType must be Mat!");
    static_assert(sizeof(typename DstTile::DType) == sizeof(typename SrcGlobal::DType),
                  "Fix: Source dtype must be same with dst dtype!");

    constexpr bool isSameLayout =
        (SrcGlobal::layout == pto::Layout::NC1HWC0 && DstTile::layout == pto::Layout::NC1HWC0) ||
        (SrcGlobal::layout == pto::Layout::FRACTAL_Z && DstTile::layout == pto::Layout::FRACTAL_Z);
    static_assert(isSameLayout == true, "Fix: Src Dst layout must be NC1HWC0 or FRACTAL_Z!");
}

template <typename DstTile, typename SrcGlobal>
PTO_INTERNAL void TLOAD_CONVTILE_IMPL(DstTile &dst, SrcGlobal &src)
{
    CheckConvTileData<DstTile, SrcGlobal>(dst, src);
    if constexpr (SrcGlobal::layout == pto::Layout::NC1HWC0) { // layout is [N,C1,H,W,C0]
        TLoad5HD<DstTile, SrcGlobal>(dst.data(), src.data(), src.GetShape(0), src.GetShape(1), src.GetShape(2),
                                     src.GetShape(3), src.GetStride(0), src.GetStride(1), src.GetStride(2),
                                     src.GetStride(3), src.GetStride(4), dst.GetShape(0), dst.GetShape(1),
                                     dst.GetShape(2), dst.GetShape(3));
    } else if constexpr (SrcGlobal::layout == pto::Layout::FRACTAL_Z) {
        if constexpr (DstTile::totalDimCount == 4) { // layout is [C1HW,N/16,16,C0]
            TLoadFractalZ<DstTile, SrcGlobal>(dst.data(), src.data(), src.GetShape(0), src.GetShape(1), src.GetShape(2),
                                              src.GetShape(3), src.GetShape(4), src.GetStride(0), src.GetStride(1),
                                              src.GetStride(2), src.GetStride(3), src.GetStride(4), dst.GetShape(0),
                                              dst.GetShape(1), dst.GetShape(2), dst.GetShape(3));
        } else { // layout is [C1,H,W,N,C0]
            TLoad5HD<DstTile, SrcGlobal>(dst.data(), src.data(), src.GetShape(0), src.GetShape(1), src.GetShape(2),
                                         src.GetShape(3), src.GetStride(0), src.GetStride(1), src.GetStride(2),
                                         src.GetStride(3), src.GetStride(4), dst.GetShape(0), dst.GetShape(1),
                                         dst.GetShape(2), dst.GetShape(3));
        }
    }
}

template <typename DstTile, typename SrcGlobal>
PTO_INTERNAL void TLOAD_IMPL(DstTile &dst, SrcGlobal &src)
{
    if constexpr (is_conv_tile_v<DstTile>) {
        TLOAD_CONVTILE_IMPL(dst, src);
    } else {
        TLOAD_TILE_IMPL(dst, src);
    }
}
} // namespace pto
#endif // TLOAD_HPP
#endif // PTO_NPU_ARCH_KIRIN9030
