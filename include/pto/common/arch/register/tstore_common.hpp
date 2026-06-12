/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSTORE_COMMON_REGISTER
#define TSTORE_COMMON_REGISTER
#include <pto/common/utils.hpp>

namespace pto {
template <typename GlobalData, typename TileData, QuantMode_t quantPre = QuantMode_t::NoQuant,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TStoreAccND(typename GlobalData::DType *dstGlobalAddr, __cc__ typename TileData::DType *srcTileAddr,
                              int gShape3, int gShape4, int gStride2, int gStride3, int validRow, int validCol)
{
    uint16_t mSize = validRow;
    uint16_t nSize = validCol;

    uint16_t srcStride = TileData::Rows;
    uint32_t dstD = gStride3;

    uint16_t ndNum = validCol / gShape4;
    constexpr uint16_t c0 = 16;
    uint16_t srcNdStride = TileData::Rows * gShape4 * c0;
    if constexpr (TileData::Compact == CompactMode::Normal) {
        srcStride = (validRow + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW * FRACTAL_NZ_ROW;
        srcNdStride = srcStride * gShape4 * c0;
    }
    constexpr uint8_t unitFlagCtrl = static_cast<uint8_t>(Phase);
    constexpr uint8_t nz2ndEn = 1;
    uint16_t dstNdStride = gStride2;

    uint64_t xmReg =
        ((nSize & 0xfff) << 4) |                          // Xm[15:4] the n-direction size of the matrix
        (static_cast<uint64_t>(mSize & 0xffff) << 16) |   // Xm[31:16] the m-direction size of the matrix
        (static_cast<uint64_t>(dstD & 0xffffffff) << 32); // Xm[63:32] destination stride between the start addr
    uint64_t xtReg = srcStride |                          // Xt[15:0] the source stride between the start addr
                     (static_cast<uint64_t>(unitFlagCtrl & 0x3) << 32) | // Xt[33:32] unit flag control bit
                     (((quantPre >> SHIFT_BLOCK_BYTE) & 0x1) << 29) |
                     (static_cast<uint64_t>(quantPre & 0x1f) << 34) | // Xt[29], Xt[38:34] pre-stage quantization mode
                     ((static_cast<uint64_t>(reluPreMode) & 0x7) << 39) | //  Xt[41:39] relu pre mode
                     (static_cast<uint64_t>(nz2ndEn & 0x1) << 43);        //  Xt[43] nz2nd control bit
    uint64_t config =
        ndNum |                                               // ND_PARA[15:0] the number of source nd
        (static_cast<uint64_t>(srcNdStride & 0xffff) << 16) | // ND_PARA[31:16] the stride of source nd
        (static_cast<uint64_t>(dstNdStride & 0xffff) << 32);  // ND_PARA[47:32] the stride of destination nd
    set_loop3_para(config);
    copy_matrix_cc_to_gm(dstGlobalAddr, srcTileAddr, xmReg, xtReg);
}

template <typename GlobalData, typename TileData, QuantMode_t quantPre = QuantMode_t::NoQuant,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TStoreAccNZ(typename GlobalData::DType *dstAddr, __cc__ typename TileData::DType *srcAddr,
                              typename GlobalData::DType *dstGlobalAddr, __cc__ typename TileData::DType *srcTileAddr,
                              int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                              int validRow, int validCol)
{
    PTO_ASSERT(validRow == gShape2 * gShape3, "The validRow of TileData must be equal to Shape2 * Shape3 of NZ shape!");
    PTO_ASSERT(validCol == gShape0 * gShape1 * gShape4,
               "The validCol of TileData must be equal to Shape0 * Shape1 * Shape4 of NZ shape!");
    static_assert(GlobalData::staticShape[3] == FRACTAL_NZ_ROW,
                  "When GlobalData is NZ format, the second-to-last dimension shall be 16.");
    static_assert((caps::IsSInt32<typename GlobalData::RawDType>() && GlobalData::staticShape[4] == 16) ||
                      (GlobalData::staticShape[4] == BLOCK_BYTE_SIZE / sizeof(typename GlobalData::RawDType)) ||
                      (caps::IsFP32<typename GlobalData::RawDType>() &&
                       (GlobalData::staticShape[4] == 8 || GlobalData::staticShape[4] == 16)),
                  "When GlobalData is in NZ format: if DstType is float, the last dimension must be either 8 or 16, "
                  "and the dimension value is 8 if and only if Channel Split is enabled; if DstType is int32_t, the "
                  "last dimension must be exactly 16. In addition, the last dimension must be static and satisfy 32 / "
                  "sizeof(DstType).");

    uint16_t mSize = validRow;
    uint16_t nSize = validCol;
    uint16_t srcStride = TileData::Rows;
    if constexpr (CompactMode::Normal == TileData::Compact) {
        srcStride = (FRACTAL_NZ_ROW - 1 + validRow) / FRACTAL_NZ_ROW * FRACTAL_NZ_ROW;
    }
    constexpr uint8_t unitFlagCtrl = static_cast<uint8_t>(Phase);
    uint8_t channelSplitEn = 0;

    uint16_t c0Size = 16;
    if constexpr (sizeof(typename TileData::DType) == 1) {
        c0Size = 32;
    } else if constexpr (caps::IsFP32<typename TileData::DType>() && caps::IsFP32<typename GlobalData::RawDType>()) {
        if (gShape4 == 8) {
            c0Size = 8;
            channelSplitEn = 1;
        }
    }
    uint32_t dstStride = (gShape2 * gShape3 + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW * FRACTAL_NZ_ROW * c0Size;
    if constexpr (sizeof(typename GlobalData::RawDType) == 1) {
        dstStride <<= 1;
    }
    uint64_t xtReg = srcStride | // Xt[15:0] the source stride between the start addr
                     (static_cast<uint64_t>(unitFlagCtrl & 0x3) << 32) | // Xt[33:32] unit flag control bit
                     (((quantPre >> SHIFT_BLOCK_BYTE) & 0x1) << 29) |
                     (static_cast<uint64_t>(quantPre & 0x1f) << 34) | // Xt[29], Xt[38:34] pre-stage quantization mode
                     ((static_cast<uint64_t>(reluPreMode) & 0x7) << 39) | //  Xt[41:39] relu pre mode
                     (static_cast<uint64_t>(channelSplitEn & 0x1) << 42); // Xt[42] channel split control bit
    uint64_t xmReg =
        ((static_cast<uint64_t>(nSize & 0xfff) << 4) |           // Xm[15:4] the n-direction size of the matrix
         (static_cast<uint64_t>(mSize & 0xffff) << 16) |         // Xm[31:16] the m-direction size of the matrix
         (static_cast<uint64_t>(dstStride & 0xffffffff) << 32)); // Xm[63:32] destination stride between the start addr

    copy_matrix_cc_to_gm(dstAddr, srcAddr, xmReg, xtReg);
}

template <typename GlobalData, typename TileData, QuantMode_t quantPre = QuantMode_t::NoQuant,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TStoreAccNHWC(typename GlobalData::DType *dstAddr, __cc__ typename TileData::DType *srcAddr,
                                int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                int gStride3, int validRow, int validCol)
{
    PTO_ASSERT(validRow == gShape1 * gShape2 * gShape3,
               "The validRow of TileData must be equal to Shape1 * Shape2 * Shape3 of NHWC shape!");
    PTO_ASSERT(validCol == gShape4, "The validCol of TileData must be equal to Shape4 of NHWC shape!");

    uint16_t mSize = validRow;
    uint16_t nSize = validCol;
    uint16_t srcStride = TileData::Rows;
    if constexpr (CompactMode::Normal == TileData::Compact) {
        srcStride = CeilAlignment(validRow, FRACTAL_NZ_ROW);
    }
    uint32_t dstStride = gStride3;

    uint16_t loop3Num = gShape0;
    uint16_t loop3SrcStirde = srcStride * gShape4 / ACC_C0_SIZE;
    uint16_t loop3DstStirde = gStride0;

    constexpr uint8_t unitFlagCtrl = static_cast<uint8_t>(Phase);
    constexpr uint8_t nz2ndEn = 1;

    uint64_t xmReg =
        ((nSize & 0xfff) << 4) |                               // Xm[15:4] the n-direction size of the matrix
        (static_cast<uint64_t>(mSize & 0xffff) << 16) |        // Xm[31:16] the m-direction size of the matrix
        (static_cast<uint64_t>(dstStride & 0xffffffff) << 32); // Xm[63:32] destination stride between the start addr
    uint64_t xtReg = srcStride |                               // Xt[15:0] the source stride between the start addr
                     (static_cast<uint64_t>(unitFlagCtrl & 0x3) << 32) | // Xt[33:32] unit flag control bit
                     (((quantPre >> SHIFT_BLOCK_BYTE) & 0x1) << 29) |
                     (static_cast<uint64_t>(quantPre & 0x1f) << 34) | // Xt[29], Xt[38:34] pre-stage quantization mode
                     ((static_cast<uint64_t>(reluPreMode) & 0x7) << 39) | //  Xt[41:39] relu pre mode
                     (static_cast<uint64_t>(nz2ndEn & 0x1) << 43);        //  Xt[43] nz2nd control bit
    uint64_t loop3Config = loop3Num |                                     // LOOP3_PARA[15:0] the number of source nd
                           (static_cast<uint64_t>(loop3SrcStirde & 0xffff)
                            << 16) | // LOOP3_PARA[31:16] the source stride of loop3 in uint of C0_SIZE
                           (static_cast<uint64_t>(loop3DstStirde & 0xffffffff)
                            << 32); // LOOP3_PARA[63:32] the dst stride of loop3 in uint of element
    set_loop3_para(loop3Config);

    copy_matrix_cc_to_gm(dstAddr, srcAddr, xmReg, xtReg);
}

template <typename GlobalData, typename TileData, QuantMode_t quantPre = QuantMode_t::NoQuant,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TStoreAccNCHW(typename GlobalData::DType *dstAddr, __cc__ typename TileData::DType *srcAddr,
                                int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride1,
                                int gStride2, int validRow, int validCol)
{
    if constexpr (GlobalData::layout == pto::Layout::NCHW) {
        PTO_ASSERT(validRow == gShape1 * gShape3 * gShape4,
                   "The validRow of TileData must be equal to Shape1 * Shape3 * Shape4 of NCHW shape!");
        PTO_ASSERT(validCol == gShape2, "The validCol of TileData must be equal to Shape2 of NCHW shape!");
    } else { // NCDHW
        PTO_ASSERT(validRow == gShape0 * gShape3 * gShape4,
                   "The validRow of TileData must be equal to Shape0 * Shape3 * Shape4 of NCDHW shape!");
        PTO_ASSERT(validCol == gShape1, "The validCol of TileData must be equal to Shape1 of NCDHW shape!");
    }

    uint16_t mSize = validRow;
    uint16_t nSize = validCol;
    uint16_t srcStride = TileData::Rows;
    if constexpr (CompactMode::Normal == TileData::Compact) {
        srcStride = CeilAlignment(validRow, FRACTAL_NZ_ROW);
    }
    uint32_t dstStride = (GlobalData::layout == pto::Layout::NCDHW) ? gStride1 : gStride2;
    uint16_t loop3Num = 1;
    uint16_t loop3SrcStirde = 0;
    uint16_t loop3DstStirde = 0;

    constexpr uint8_t unitFlagCtrl = static_cast<uint8_t>(Phase);
    constexpr uint8_t nz2dnEn = 1;

    uint64_t xmReg =
        ((nSize & 0xfff) << 4) |                               // Xm[15:4] the n-direction size of the matrix
        (static_cast<uint64_t>(mSize & 0xffff) << 16) |        // Xm[31:16] the m-direction size of the matrix
        (static_cast<uint64_t>(dstStride & 0xffffffff) << 32); // Xm[63:32] destination stride between the start addr
    uint64_t xtReg = srcStride |                               // Xt[15:0] the source stride between the start addr
                     (static_cast<uint64_t>(unitFlagCtrl & 0x3) << 32) | // Xt[33:32] unit flag control bit
                     (((quantPre >> SHIFT_BLOCK_BYTE) & 0x1) << 29) |
                     (static_cast<uint64_t>(quantPre & 0x1f) << 34) | // Xt[29], Xt[38:34] pre-stage quantization mode
                     ((static_cast<uint64_t>(reluPreMode) & 0x7) << 39) | //  Xt[41:39] relu pre mode
                     (static_cast<uint64_t>(nz2dnEn & 0x1) << 62);        //  Xt[62] nz2dn control bit
    uint64_t loop3Config = loop3Num |                                     // LOOP3_PARA[15:0] the number of source nd
                           (static_cast<uint64_t>(loop3SrcStirde & 0xffff)
                            << 16) | // LOOP3_PARA[31:16] the source stride of loop3 in uint of C0_SIZE
                           (static_cast<uint64_t>(loop3DstStirde & 0xffffffff)
                            << 32); // LOOP3_PARA[63:32] the dst stride of loop3 in uint of element
    set_loop3_para(loop3Config);
    uint16_t loop0SrcStirde = 1; // loop0SrcStirde is 1 when src layout is NZ
    uint64_t channelConfig = static_cast<uint64_t>(loop0SrcStirde & 0xffff)
                             << 48; // CHANNEL_PARA[63:48] source stride of loop0 in unit of C0_SIZE
    set_channel_para(channelConfig);
    copy_matrix_cc_to_gm(dstAddr, srcAddr, xmReg, xtReg);
}

template <typename GlobalData, typename TileData, typename FpTileData, QuantMode_t quantPre = QuantMode_t::NoQuant,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu>
__tf__ AICORE void TStoreAccFp(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
                               typename FpTileData::TileDType __in__ fp, int gShape0, int gShape1, int gShape2,
                               int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3,
                               int gStride4, int validRow, int validCol)
{
    __cc__ typename TileData::DType *srcAddr = (__cc__ typename TileData::DType *)__cce_get_tile_ptr(src);
    typename GlobalData::DType *dstAddr = dst;
    __fbuf__ typename FpTileData::DType *dstAddrFp = (__fbuf__ typename FpTileData::DType *)__cce_get_tile_ptr(fp);
    uint64_t deqTensorAddr = ((uint64_t)dstAddrFp >> static_cast<uint64_t>(7)) << 8;
    set_fpc(deqTensorAddr);
    if constexpr (GlobalData::layout == pto::Layout::NZ) {
        __cc__ typename TileData::DType *srcTileAddr = srcAddr;
        typename GlobalData::DType *dstGlobalAddr = dstAddr;
        TStoreAccNZ<GlobalData, TileData, quantPre, reluPreMode>(dstAddr, srcAddr, dstGlobalAddr, srcTileAddr, gShape0,
                                                                 gShape1, gShape2, gShape3, gShape4, gStride0, validRow,
                                                                 validCol);
    } else if constexpr (GlobalData::layout == pto::Layout::ND) {
        TStoreAccND<GlobalData, TileData, quantPre, reluPreMode>(dstAddr, srcAddr, gShape3, gShape4, gStride2, gStride3,
                                                                 validRow, validCol);
    } else if constexpr (GlobalData::layout == pto::Layout::NHWC) {
        TStoreAccNHWC<GlobalData, TileData, quantPre, reluPreMode>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3,
                                                                   gShape4, gStride0, gStride3, validRow, validCol);
    } else if constexpr (GlobalData::layout == pto::Layout::NCHW || GlobalData::layout == pto::Layout::NCDHW) {
        TStoreAccNCHW<GlobalData, TileData, quantPre, reluPreMode>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3,
                                                                   gShape4, gStride1, gStride2, validRow, validCol);
    }
}

template <typename GlobalData, typename TileData, QuantMode_t quantPre = QuantMode_t::NoQuant,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu, STPhase Phase = STPhase::Unspecified>
__tf__ AICORE void TStoreAcc(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
                             int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                             int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    __cc__ typename TileData::DType *srcAddr = (__cc__ typename TileData::DType *)__cce_get_tile_ptr(src);
    typename GlobalData::DType *dstAddr = dst;
    if constexpr (GlobalData::layout == pto::Layout::ND) {
        TStoreAccND<GlobalData, TileData, quantPre, reluPreMode, Phase>(dstAddr, srcAddr, gShape3, gShape4, gStride2,
                                                                        gStride3, validRow, validCol);
    } else if constexpr (GlobalData::layout == pto::Layout::NZ) {
        __cc__ typename TileData::DType *srcTileAddr = srcAddr;
        typename GlobalData::DType *dstGlobalAddr = dstAddr;
        TStoreAccNZ<GlobalData, TileData, quantPre, reluPreMode, Phase>(dstAddr, srcAddr, dstGlobalAddr, srcTileAddr,
                                                                        gShape0, gShape1, gShape2, gShape3, gShape4,
                                                                        gStride0, validRow, validCol);
    } else if constexpr (GlobalData::layout == pto::Layout::NHWC) {
        TStoreAccNHWC<GlobalData, TileData, quantPre, reluPreMode, Phase>(
            dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride3, validRow, validCol);
    } else if constexpr (GlobalData::layout == pto::Layout::NCHW || GlobalData::layout == pto::Layout::NCDHW) {
        TStoreAccNCHW<GlobalData, TileData, quantPre, reluPreMode, Phase>(
            dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride1, gStride2, validRow, validCol);
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TStoreInstr(typename GlobalData::DType *dst, __ubuf__ typename TileData::DType *src, uint32_t nBurst,
                              uint32_t lenBurst, uint64_t burstDstStride, uint32_t burstSrcStride)
{
    pto_copy_ubuf_to_gm_align_v2(dst, src, 0, nBurst, lenBurst, 0, burstDstStride, burstSrcStride);
}

template <typename GlobalData, typename TileData>
PTO_INTERNAL void TStoreVecND(typename GlobalData::DType *dstAddr, __ubuf__ typename TileData::DType *srcAddr,
                              int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                              int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    PTO_ASSERT(validCol == gShape4, "The validCol of TileData must be equal to the 5th dim(Shape4) of ND shape!");
    PTO_ASSERT(validRow == gShape0 * gShape1 * gShape2 * gShape3,
               "The validRow of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape3) of ND shape!");
    typename GlobalData::DType *dstGlobalAddr = dstAddr;
    __ubuf__ typename TileData::DType *srcTileAddr = srcAddr;
    uint32_t loop1SrcStride = GetByteSize<typename TileData::DType>(gShape3 * TileData::Cols);
    uint32_t loop1DstStride = GetByteSize<typename TileData::DType>(gStride2);
    uint32_t loop2SrcStride = GetByteSize<typename TileData::DType>(gShape2 * gShape3 * TileData::Cols);
    uint32_t loop2DstStride = GetByteSize<typename TileData::DType>(gStride1);

    uint64_t loopSizeConfig = 0;
    uint64_t loop1Size = gShape2 & 0x1FFFFF;
    loopSizeConfig |= loop1Size;
    uint64_t loop2Size = (static_cast<uint64_t>(gShape1) & 0x3FFFFF) << 21;
    loopSizeConfig |= loop2Size;
    set_loop_size_ubtoout(loopSizeConfig);

    uint64_t loop1Config = 0;
    loop1Config |= ((uint64_t)loop1SrcStride) << 40;
    loop1Config |= (uint64_t)loop1DstStride;
    set_loop1_stride_ubtoout(loop1Config);
    uint64_t loop2Config = 0;
    loop2Config |= ((uint64_t)loop2SrcStride) << 40;
    loop2Config |= (uint64_t)loop2DstStride;
    set_loop2_stride_ubtoout(loop2Config);
    uint64_t srcStride0 = gShape1 * gShape2 * gShape3 * TileData::Cols;
    if constexpr (caps::IsFP4<typename TileData::DType>()) {
        srcStride0 = srcStride0 >> 1; // fp4 srcAddr offset need divide 2 as use b8 to move
        gStride0 = gStride0 >> 1;     // fp4 dstAddr offset need divide 2 as use b8 to move
    }
    uint32_t nBurst = gShape3;

    uint32_t lenBurst = GetByteSize<typename TileData::DType>(validCol);
    uint64_t burstDstStride = GetByteSize<typename TileData::DType>(gStride3);
    uint32_t burstSrcStride = GetByteSize<typename TileData::DType>(TileData::Cols);
    for (uint32_t k = 0; k < gShape0; k++) {
        dstGlobalAddr = dstAddr + k * gStride0;
        srcTileAddr = srcAddr + k * srcStride0;
        TStoreInstr<TileData, GlobalData>(dstGlobalAddr, srcTileAddr, nBurst, lenBurst, burstDstStride, burstSrcStride);
    }
    set_loop_size_ubtoout(1 << 21 | 1); // resume to normal mode
}
template <typename GlobalData, typename TileData>
PTO_INTERNAL void TStoreVecDN(typename GlobalData::DType *dstAddr, __ubuf__ typename TileData::DType *srcAddr,
                              int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                              int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    PTO_ASSERT(validRow == gShape3, "The validCol of TileData must be equal to the 4th dim(Shape3) of DN shape!");
    PTO_ASSERT(validCol == gShape0 * gShape1 * gShape2 * gShape4,
               "The validRow of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape4) of DN shape!");
    typename GlobalData::DType *dstGlobalAddr = dstAddr;
    __ubuf__ typename TileData::DType *srcTileAddr = srcAddr;
    uint32_t loop1SrcStride = GetByteSize<typename TileData::DType>(TileData::Rows * gShape4);
    uint32_t loop1DstStride = GetByteSize<typename TileData::DType>(gStride2);
    uint32_t loop2SrcStride = GetByteSize<typename TileData::DType>(gShape2 * TileData::Rows * gShape4);
    uint32_t loop2DstStride = GetByteSize<typename TileData::DType>(gStride1);

    uint64_t loop1Config = 0;
    loop1Config |= ((uint64_t)loop1SrcStride) << 40;
    loop1Config |= (uint64_t)loop1DstStride;
    set_loop1_stride_ubtoout(loop1Config);
    uint64_t loop2Config = 0;
    loop2Config |= ((uint64_t)loop2SrcStride) << 40;
    loop2Config |= (uint64_t)loop2DstStride;
    set_loop2_stride_ubtoout(loop2Config);

    uint64_t loopSizeConfig = 0;
    uint64_t loop1Size = gShape2 & 0x1FFFFF;
    loopSizeConfig |= loop1Size;
    uint64_t loop2Size = (static_cast<uint64_t>(gShape1) & 0x3FFFFF) << 21;
    loopSizeConfig |= loop2Size;
    set_loop_size_ubtoout(loopSizeConfig);

    uint64_t srcStride0 = gShape1 * gShape2 * gShape4 * TileData::Rows;
    uint32_t nBurst = gShape4;
    uint32_t lenBurst = GetByteSize<typename TileData::DType>(validRow);
    uint64_t burstDstStride = GetByteSize<typename TileData::DType>(gStride4);
    uint32_t burstSrcStride = GetByteSize<typename TileData::DType>(TileData::Rows);
    if constexpr (caps::IsFP4<typename TileData::DType>()) {
        srcStride0 = srcStride0 >> 1; // fp4 srcAddr offset need divide 2 as use b8 to move
        gStride0 = gStride0 >> 1;     // fp4 dstAddr offset need divide 2 as use b8 to move
    }

    for (uint32_t k = 0; k < gShape0; k++) {
        dstGlobalAddr = dstAddr + k * gStride0;
        srcTileAddr = srcAddr + k * srcStride0;
        TStoreInstr<TileData, GlobalData>(dstGlobalAddr, srcTileAddr, nBurst, lenBurst, burstDstStride, burstSrcStride);
    }
    set_loop_size_ubtoout(1 << 21 | 1); // resume to normal mode
}

template <typename GlobalData, typename TileData>
PTO_INTERNAL void TStoreVecNZ(typename GlobalData::DType *dstAddr, __ubuf__ typename TileData::DType *srcAddr,
                              int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                              int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    static_assert(
        (caps::IsSInt32<typename GlobalData::RawDType>() && GlobalData::staticShape[4] == 16) ||
            (GlobalData::staticShape[4] == BLOCK_BYTE_SIZE / sizeof(typename GlobalData::RawDType)) ||
            (caps::IsFP32<typename GlobalData::RawDType>() &&
             (GlobalData::staticShape[4] == 8 || GlobalData::staticShape[4] == 16)) ||
            (caps::IsFP4<typename GlobalData::RawDType>() && GlobalData::staticShape[4] == 64),
        "When GlobalData is in NZ format: if DstType is float, the last dimension must be either 8 or 16, \n"
        "if DstType is float4, the last dimension must be 64, \n"
        "and the dimension value is 8 if and only if Channel Split is enabled; if DstType is int32_t, the \n"
        "last dimension must be exactly 16. In addition, the last dimension must be static and satisfy 32 / \n"
        "sizeof(DstType).");
    static_assert(GlobalData::staticShape[3] == FRACTAL_NZ_ROW,
                  "When GlobalData is NZ format, the second-to-last dimension shall be 16.");
    PTO_ASSERT(validRow == gShape2 * gShape3, "The validRow of TileData must be equal to Shape2 * Shape3 of NZ shape!");
    PTO_ASSERT(validCol == gShape0 * gShape1 * gShape4,
               "The validCol of TileData must be equal to Shape0 * Shape1 * Shape4 of NZ shape!");
    typename GlobalData::DType *dstGlobalAddr = dstAddr;
    __ubuf__ typename TileData::DType *srcTileAddr = srcAddr;
    uint32_t nBurst = gShape1;
    uint32_t lenBurst = validRow * C0_SIZE_BYTE;
    uint64_t burstDstStride = GetByteSize<typename TileData::DType>(gStride1);
    uint32_t burstSrcStride = TileData::Rows * C0_SIZE_BYTE;
    int64_t tileStride = gShape1 * TileData::Rows * gShape4;
    if constexpr (caps::IsFP4<typename TileData::DType>()) {
        tileStride = tileStride >> 1; // fp4 srcAddr offset need divide 2 as use b8 to move
        gStride0 = gStride0 >> 1;     // fp4 dstAddr offset need divide 2 as use b8 to move
    }
    for (uint32_t k = 0; k < gShape0; k++) {
        dstGlobalAddr = dstAddr + k * gStride0;
        srcTileAddr = srcAddr + k * tileStride;
        TStoreInstr<TileData, GlobalData>(dstGlobalAddr, srcTileAddr, nBurst, lenBurst, burstDstStride, burstSrcStride);
    }
}
template <typename GlobalData, typename TileData>
__tf__ AICORE OP_NAME(TSTORE)
    OP_TYPE(memory) void TStore(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
                                int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    __ubuf__ typename TileData::DType *srcAddr = (__ubuf__ typename TileData::DType *)__cce_get_tile_ptr(src);
    typename GlobalData::DType *dstAddr = dst;

    if constexpr (TileData::isRowMajor & (TileData::SFractal == SLayout::NoneBox)) {
        TStoreVecND<GlobalData, TileData>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                          gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    } else if constexpr (!TileData::isRowMajor & (TileData::SFractal == SLayout::NoneBox)) {
        TStoreVecDN<GlobalData, TileData>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                          gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    } else if constexpr (!TileData::isRowMajor & (TileData::SFractal == SLayout::RowMajor)) {
        TStoreVecNZ<GlobalData, TileData>(dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0,
                                          gStride1, gStride2, gStride3, gStride4, validRow, validCol);
    }
}
} // namespace pto
#endif // TSTORE_COMMON_REGISTER