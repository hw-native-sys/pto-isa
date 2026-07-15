/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TLOAD_COMMON_REGISTER
#define TLOAD_COMMON_REGISTER
#include <pto/common/utils.hpp>

namespace pto {
struct LoadOpBase {
    template <typename T, typename U>
    PTO_INTERNAL static void TLoadInstr(
        __ubuf__ T* dst, __gm__ U* src, uint32_t nBurst, uint32_t lenBurst, uint64_t gmStride, uint32_t ubStride,
        bool enableUBPad)
    {
        using LoadT = LoadTypeBySize_t<T>;
        pto_copy_gm_to_ubuf_align_v2(
            reinterpret_cast<__ubuf__ LoadT*>(dst), reinterpret_cast<__gm__ LoadT*>(src), 0 /*sid*/, nBurst, lenBurst,
            0 /*left padding count*/, 0 /*right padding count*/, enableUBPad /*data select bit*/, 0 /*l2 cache ctl*/,
            gmStride, ubStride);
    }

    template <typename T>
    PTO_INTERNAL static void TLoadCubeInstr(
        __cbuf__ T* dst, __gm__ T* src, uint32_t nBurst, uint32_t lenBurst, uint64_t srcStride, uint32_t dstStride,
        uint32_t padCount)
    {
        if constexpr (sizeof(T) == sizeof(uint64_t)) {
            padCount *= sizeof(uint64_t) / sizeof(uint32_t);
        }
        pto_copy_gm_to_cbuf_align_v2(
            dst, src, 0 /*sid*/, nBurst, lenBurst, 0 /*left padding count*/, padCount /*right padding count*/,
            true /*data select bit*/, 0 /*l2 cache ctl*/, srcStride, dstStride);
    }
};

template <typename Op, typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadVecND2ND(
    __ubuf__ typename TileData::DType* dstAddr, typename GlobalData::DType* srcAddr, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
    int validRow, int validCol, bool enableUBPad)
{
    typename GlobalData::DType* srcAddrP = srcAddr;
    __ubuf__ typename TileData::DType* dstAddrP = dstAddr;
    uint32_t nBurst = gShape3;
    uint32_t lenBurst = GetByteSize<typename TileData::DType>(validCol);
    uint64_t gmStride = GetByteSize<typename TileData::DType>(gStride3);
    uint32_t ubStride = GetByteSize<typename TileData::DType>(TileData::Cols);

    int64_t dstStride2 = gShape3 * TileData::Cols;
    int64_t dstStride1 = gShape2 * dstStride2;
    int64_t dstStride0 = gShape1 * dstStride1;
    if constexpr (caps::IsFP4<typename TileData::DType>()) {
        dstStride0 = dstStride0 >> 1; // fp4 dstAddr offset need divide 2 as use b8 to move
        gStride0 = gStride0 >> 1;     // fp4 srcAddr offset need divide 2 as use b8 to move
    }
    uint64_t loop2 = gShape1;
    uint64_t loop1 = gShape2;
    uint64_t loop2_src_stride = GetByteSize<typename TileData::DType>(gStride1);
    uint64_t loop1_src_stride = GetByteSize<typename TileData::DType>(gStride2);
    uint64_t loop2_dst_stride = GetByteSize<typename TileData::DType>(dstStride1);
    uint64_t loop1_dst_stride = GetByteSize<typename TileData::DType>(dstStride2);
    if (loop1 != 1 || loop2 != 1) {
        set_loop2_stride_outtoub(loop2_dst_stride << 40 | loop2_src_stride);
        set_loop1_stride_outtoub(loop1_dst_stride << 40 | loop1_src_stride);
        set_loop_size_outtoub(loop2 << 21 | loop1);
    }

    for (uint32_t i = 0; i < gShape0; i++) {
        int64_t dstAddr0 = i * dstStride0;
        int64_t srcAddr0 = i * gStride0;
        dstAddrP = dstAddr + dstAddr0;
        srcAddrP = srcAddr + srcAddr0;
        Op::TLoadInstr(dstAddrP, srcAddrP, nBurst, lenBurst, gmStride, ubStride, enableUBPad);
    }
    if (loop1 != 1 || loop2 != 1) {
        set_loop_size_outtoub(1 << 21 | 1); // resume to normal mode
    }
}
template <typename Op, typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadVecDN2DN(
    __ubuf__ typename TileData::DType* dstAddr, typename GlobalData::DType* srcAddr, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
    int validRow, int validCol, bool enableUBPad)
{
    uint32_t nBurst = gShape4;
    uint32_t lenBurst = GetByteSize<typename TileData::DType>(validRow);
    uint64_t gmStride = GetByteSize<typename TileData::DType>(gStride4);
    uint32_t ubStride = GetByteSize<typename TileData::DType>(TileData::Rows);

    typename GlobalData::DType* srcAddrP = srcAddr;
    __ubuf__ typename TileData::DType* dstAddrP = dstAddr;

    int64_t dstStride2 = gShape4 * TileData::Rows;
    int64_t dstStride1 = gShape2 * dstStride2;
    int64_t dstStride0 = gShape1 * dstStride1;

    uint64_t loop2 = gShape1;
    uint64_t loop1 = gShape2;
    uint64_t loop2_src_stride = GetByteSize<typename TileData::DType>(gStride1);
    uint64_t loop1_src_stride = GetByteSize<typename TileData::DType>(gStride2);
    uint64_t loop2_dst_stride = GetByteSize<typename TileData::DType>(dstStride1);
    uint64_t loop1_dst_stride = GetByteSize<typename TileData::DType>(dstStride2);
    if (loop1 != 1 || loop2 != 1) {
        set_loop2_stride_outtoub(loop2_dst_stride << 40 | loop2_src_stride);
        set_loop1_stride_outtoub(loop1_dst_stride << 40 | loop1_src_stride);
        set_loop_size_outtoub(loop2 << 21 | loop1);
    }
    if constexpr (caps::IsFP4<typename TileData::DType>()) {
        dstStride0 = dstStride0 >> 1; // fp4 dstAddr offset need divide 2 as use b8 to move
        gStride0 = gStride0 >> 1;     // fp4 srcAddr offset need divide 2 as use b8 to move
    }

    for (uint32_t i = 0; i < gShape0; i++) {
        int64_t dstAddr0 = i * dstStride0;
        int64_t srcAddr0 = i * gStride0;
        dstAddrP = dstAddr + dstAddr0;
        srcAddrP = srcAddr + srcAddr0;
        Op::TLoadInstr(dstAddrP, srcAddrP, nBurst, lenBurst, gmStride, ubStride, enableUBPad);
    }
    if (loop1 != 1 || loop2 != 1) {
        set_loop_size_outtoub(1 << 21 | 1); // resume to normal mode
    }
}
template <typename Op, typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadVecNZ2NZ(
    __ubuf__ typename TileData::DType* dstAddr, typename GlobalData::DType* srcAddr, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
    int validRow, int validCol)
{
    uint32_t nBurst = gShape1;
    uint32_t lenBurst = validRow * C0_SIZE_BYTE;
    uint32_t gmStride = GetByteSize<typename TileData::DType>(gStride1);
    uint32_t ubStride = TileData::Rows * C0_SIZE_BYTE;

    typename GlobalData::DType* srcAddrP = srcAddr;
    __ubuf__ typename TileData::DType* dstAddrP = dstAddr;

    int64_t tileStride = gShape1 * TileData::Rows * gShape4;
    set_loop_size_outtoub(1ULL << 21 | 1ULL);
    if constexpr (caps::IsFP4<typename TileData::DType>()) {
        tileStride = tileStride >> 1; // fp4 dstAddr offset need divide 2 as use b8 to move
        gStride0 = gStride0 >> 1;     // fp4 srcAddr offset need divide 2 as use b8 to move
    }
    for (uint32_t i = 0; i < gShape0; i++) {
        srcAddrP = srcAddr + i * gStride0;
        dstAddrP = dstAddr + i * tileStride;
        Op::TLoadInstr(dstAddrP, srcAddrP, nBurst, lenBurst, gmStride, ubStride, 0);
    }
}

template <typename Op, typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL OP_NAME(TLOAD) OP_TYPE(memory) void TLoad(
    typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__* src, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
    int validRow, int validCol)
{
    __ubuf__ typename TileData::DType* dstAddr = (__ubuf__ typename TileData::DType*)__cce_get_tile_ptr(dst);
    typename GlobalData::DType* srcAddr = src;
    constexpr bool enableUBPad = TileData::PadVal != PadValue::Null;
    if constexpr (enableUBPad) {
        pto_set_tload_pad_val<TileType::Vec>(GetPadValue<TileData>());
    }
    if constexpr (TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox)) {
        TLoadVecND2ND<Op, TileData, GlobalData>(
            dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1, gStride2, gStride3,
            gStride4, validRow, validCol, enableUBPad);
    } else if constexpr (!TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox)) {
        TLoadVecDN2DN<Op, TileData, GlobalData>(
            dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1, gStride2, gStride3,
            gStride4, validRow, validCol, enableUBPad);
    } else if constexpr (!TileData::isRowMajor && (TileData::SFractal == SLayout::RowMajor)) {
        TLoadVecNZ2NZ<Op, TileData, GlobalData>(
            dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1, gStride2, gStride3,
            gStride4, validRow, validCol);
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadCubeCheck()
{
    // support ND2NZ DN2NZ ND2ND DN2DN NZ2NZ DN2ZN
    static_assert(
        ((GlobalData::layout == pto::Layout::ND) &&
         (!TileData::isRowMajor && (TileData::SFractal == SLayout::RowMajor))) ||
            ((GlobalData::layout == pto::Layout::DN) &&
             (!TileData::isRowMajor && (TileData::SFractal == SLayout::RowMajor))) ||
            ((GlobalData::layout == pto::Layout::NZ) &&
             (!TileData::isRowMajor && (TileData::SFractal == SLayout::RowMajor))) ||
            (((GlobalData::layout == pto::Layout::ND) &&
              (TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox)))) ||
            (((GlobalData::layout == pto::Layout::DN) &&
              (!TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox)))) ||
            (((GlobalData::layout == pto::Layout::DN) &&
              (TileData::isRowMajor && (TileData::SFractal == SLayout::ColMajor)))),
        "Fix: now only support ND2NZ DN2NZ ND2ND DN2DN NZ2NZ DN2ZN in current platform");

    // L1 space check
    static_assert(TileData::Rows <= 16384, "Fix: TileData::Rows must less than 16384 in L1");
    static_assert(
        TileData::Rows * TileData::Cols <= 512 * 1024, "Fix: TileData static shape must less than 512KB in L1");

    // ND2NZ or DN2NZ
    if constexpr (
        (GlobalData::layout == pto::Layout::ND || GlobalData::layout == pto::Layout::DN) &&
        (!TileData::isRowMajor && (TileData::SFractal == SLayout::RowMajor))) {
        static_assert(TileData::SFractalSize == 512, "Fix: TileData SFractalSize must be 512 of NZ format in L1");
        static_assert(sizeof(typename TileData::DType) != 8, "Fix: DType not support b64 in ND2NZ or DN2NZ");
        // globaltensor only support 2 dim
        static_assert(
            GlobalData::staticShape[0] == 1 && GlobalData::staticShape[1] == 1 && GlobalData::staticShape[2] == 1,
            "Fix: GlobalTensor input shape now only support 2 dim");
        if constexpr (caps::IsFP4<typename TileData::DType>()) {
            static_assert(
                GlobalData::layout != pto::Layout::DN &&
                    !(TileData::isRowMajor && (TileData::SFractal == SLayout::ColMajor)),
                "Fix: DN2NZ not support if input dtype is fp4");
        }
    }

    // NZ2NZ
    if constexpr (
        (GlobalData::layout == pto::Layout::NZ) &&
        (!TileData::isRowMajor && (TileData::SFractal == SLayout::RowMajor))) {
        if constexpr (caps::IsFP4<typename TileData::DType>()) {
            static_assert(
                BLOCK_BYTE_SIZE * 2 == GlobalData::staticShape[4] && BLOCK_LEN == GlobalData::staticShape[3],
                "Fix: Src GlobalTensor staticShape[3][4] must be satisfied with NZ format require!");
        } else {
            static_assert(
                BLOCK_BYTE_SIZE / sizeof(typename GlobalData::DType) == GlobalData::staticShape[4] &&
                    BLOCK_LEN == GlobalData::staticShape[3],
                "Fix: Src GlobalTensor staticShape[3][4] must be satisfied with NZ format require!");
        }
    }
}

template <typename Op, typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadCubeND2NZ(
    __cbuf__ typename TileData::DType* dst, typename GlobalData::DType* src, int gShape0, int gShape1, int gShape2,
    int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4, int validRow,
    int validCol)
{
    uint16_t nValue = gShape3;
    uint32_t dValue = validCol;
    if constexpr (caps::IsFP4<typename TileData::DType>()) {
        dValue = (dValue + 1) >> 1; // move fp4 as b8, ceil division to include last nibble for odd counts
    }

    uint64_t loop1SrcStride = GetByteSize<typename TileData::DType>(gStride3);
    if constexpr (GlobalData::layout == pto::Layout::DN) {
        loop1SrcStride = GetByteSize<typename TileData::DType>(gStride4);
    }
    constexpr uint16_t ndNum = 1;
    uint16_t loop2DstStride = 1;
    uint16_t loop3DstStride = TileData::Rows;                          // unit is 32B
    uint16_t loop4DstStride = 0;                                       // because ndNum = 1
    uint64_t mte2NzPara = static_cast<uint64_t>(loop4DstStride) << 48; // MTE2_NZ_PARA[63:48]
    mte2NzPara |= static_cast<uint64_t>(loop3DstStride) << 32;         // MTE2_NZ_PARA[47:32]
    mte2NzPara |= static_cast<uint64_t>(loop2DstStride) << 16;         // MTE2_NZ_PARA[31:16]
    mte2NzPara |= static_cast<uint64_t>(ndNum);                        // MTE2_NZ_PARA[15:0]
    set_mte2_nz_para(mte2NzPara);                                      // only set once

    Op::template TLoadCubeInstr<GlobalData::layout>(dst, src, loop1SrcStride, nValue, dValue, 0);
}
template <typename Op, typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadCubeNZ2NZ(
    __cbuf__ typename TileData::DType* dst, typename GlobalData::DType* src, int gShape0, int gShape1, int gShape2,
    int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4, int validRow,
    int validCol)
{
    __cbuf__ typename TileData::DType* dstAddrP = dst;
    typename GlobalData::DType* srcAddrP = src;
    uint32_t nBurst = gShape1;
    uint32_t lenBurst = validRow * BLOCK_BYTE_SIZE;
    uint64_t gmStride = GetByteSize<typename TileData::DType>(gStride1);
    uint32_t dstStride = TileData::Rows * BLOCK_BYTE_SIZE;

    int64_t tileStride = gShape1 * TileData::Rows * gShape4;
    if constexpr (caps::IsFP4<typename TileData::DType>()) {
        gStride0 = gStride0 >> 1;     // fp4 srcAddr offset need divide 2 as use b8 to move
        tileStride = tileStride >> 1; // fp4 dstAddr offset need divide 2 as use b8 to move
    }
    set_loop_size_outtol1(1ULL << 21 | 1ULL);
    for (uint32_t i = 0; i < gShape0; i++) {
        srcAddrP = src + i * gStride0;
        dstAddrP = dst + i * tileStride;
        Op::TLoadCubeInstr(dstAddrP, srcAddrP, nBurst, lenBurst, gmStride, dstStride, 0);
    }
}

template <typename Op, typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadCubeND2ND(
    __cbuf__ typename TileData::DType* dst, typename GlobalData::DType* src, int gShape0, int gShape1, int gShape2,
    int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4, int validRow,
    int validCol)
{
    __cbuf__ typename TileData::DType* dstAddrP = dst;
    typename GlobalData::DType* srcAddrP = src;
    uint32_t nBurst = gShape3;
    uint32_t lenBurst = GetByteSize<typename TileData::DType>(validCol);
    uint64_t gmStride = GetByteSize<typename TileData::DType>(gStride3);
    uint32_t dstStride = GetByteSize<typename TileData::DType>(TileData::Cols);

    constexpr uint32_t blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    uint32_t gapElement = (TileData::Cols - validCol);
    uint32_t padCount = gapElement % blockSizeElem;
    if constexpr (caps::IsFP4<typename TileData::DType>()) {
        padCount = padCount >> 1;
    }
    if constexpr (!(TileData::PadVal == PadValue::Null || TileData::PadVal == PadValue::Zero)) {
        pto_set_tload_pad_val<TileType::Mat>(GetPadValue<TileData>());
    }

    int64_t dstStride2 = gShape3 * TileData::Cols;
    int64_t dstStride1 = gShape2 * dstStride2;
    int64_t dstStride0 = gShape1 * dstStride1;

    uint64_t loop2 = gShape1;
    uint64_t loop1 = gShape2;
    uint64_t loop2SrcStride = GetByteSize<typename TileData::DType>(gStride1);
    uint64_t loop1SrcStride = GetByteSize<typename TileData::DType>(gStride2);
    uint64_t loop2DstStride = GetByteSize<typename TileData::DType>(dstStride1);
    uint64_t loop1DstStride = GetByteSize<typename TileData::DType>(dstStride2);

    if (loop1 != 1 || loop2 != 1) {
        set_loop2_stride_outtol1(loop2DstStride << 40 | loop2SrcStride);
        set_loop1_stride_outtol1(loop1DstStride << 40 | loop1SrcStride);
        set_loop_size_outtol1(loop2 << 21 | loop1);
    }
    if constexpr (caps::IsFP4<typename TileData::DType>()) {
        dstStride0 = dstStride0 >> 1; // fp4 dstAddr offset need divide 2 as use b8 to move
        gStride0 = gStride0 >> 1;     // fp4 srcAddr offset need divide 2 as use b8 to move
    }
    for (uint32_t i = 0; i < gShape0; i++) {
        int64_t dstAddr0 = i * dstStride0;
        int64_t srcAddr0 = i * gStride0;
        dstAddrP = dst + dstAddr0;
        srcAddrP = src + srcAddr0;
        Op::TLoadCubeInstr(dstAddrP, srcAddrP, nBurst, lenBurst, gmStride, dstStride, padCount);
    }
    if (loop1 != 1 || loop2 != 1) {
        set_loop_size_outtol1(1 << 21 | 1); // resume to normal mode
    }
    if constexpr (!(TileData::PadVal == PadValue::Null || TileData::PadVal == PadValue::Zero)) {
        pto_set_tload_pad_val<TileType::Mat>(uint8_t(0));
    }
}

template <typename Op, typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadCubeDN2DN(
    __cbuf__ typename TileData::DType* dst, typename GlobalData::DType* src, int gShape0, int gShape1, int gShape2,
    int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4, int validRow,
    int validCol)
{
    __cbuf__ typename TileData::DType* dstAddrP = dst;
    typename GlobalData::DType* srcAddrP = src;
    uint32_t nBurst = gShape4;
    uint32_t lenBurst = GetByteSize<typename TileData::DType>(validRow);
    uint64_t gmStride = GetByteSize<typename TileData::DType>(gStride4);
    uint32_t dstStride = GetByteSize<typename TileData::DType>(TileData::Rows);

    constexpr uint32_t blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    uint32_t gapElement = (TileData::Rows - validRow);
    uint32_t padCount = gapElement % blockSizeElem;
    if constexpr (caps::IsFP4<typename TileData::DType>()) {
        padCount = padCount >> 1;
    }
    if constexpr (!(TileData::PadVal == PadValue::Null || TileData::PadVal == PadValue::Zero)) {
        pto_set_tload_pad_val<TileType::Mat>(GetPadValue<TileData>());
    }
    int64_t dstStride2 = gShape4 * TileData::Rows;
    int64_t dstStride1 = gShape2 * dstStride2;
    int64_t dstStride0 = gShape1 * dstStride1;
    if constexpr (caps::IsFP4<typename TileData::DType>()) {
        dstStride0 = dstStride0 >> 1; // fp4 dstAddr offset need divide 2 as use b8 to move
        gStride0 = gStride0 >> 1;     // fp4 srcAddr offset need divide 2 as use b8 to move
    }
    uint64_t loop2 = gShape1;
    uint64_t loop1 = gShape2;
    uint64_t loop2SrcStride = GetByteSize<typename TileData::DType>(gStride1);
    uint64_t loop1SrcStride = GetByteSize<typename TileData::DType>(gStride2);
    uint64_t loop2DstStride = GetByteSize<typename TileData::DType>(dstStride1);
    uint64_t loop1DstStride = GetByteSize<typename TileData::DType>(dstStride2);

    if (loop1 != 1 || loop2 != 1) {
        set_loop2_stride_outtol1(loop2DstStride << 40 | loop2SrcStride);
        set_loop1_stride_outtol1(loop1DstStride << 40 | loop1SrcStride);
        set_loop_size_outtol1(loop2 << 21 | loop1);
    }
    for (uint32_t i = 0; i < gShape0; i++) {
        int64_t dstAddr0 = i * dstStride0;
        int64_t srcAddr0 = i * gStride0;
        dstAddrP = dst + dstAddr0;
        srcAddrP = src + srcAddr0;
        Op::TLoadCubeInstr(dstAddrP, srcAddrP, nBurst, lenBurst, gmStride, dstStride, padCount);
    }
    if (loop1 != 1 || loop2 != 1) {
        set_loop_size_outtol1(1 << 21 | 1); // resume to normal mode
    }
    if constexpr (!(TileData::PadVal == PadValue::Null || TileData::PadVal == PadValue::Zero)) {
        pto_set_tload_pad_val<TileType::Mat>(uint8_t(0));
    }
}

template <typename Op, typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadCubeDN2ZN(
    __cbuf__ typename TileData::DType* dst, typename GlobalData::DType* src, int gShape0, int gShape1, int gShape2,
    int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4, int validRow,
    int validCol)
{
    uint16_t nValue = gShape4;
    uint32_t dValue = validRow;
    if constexpr (caps::IsFP4<typename TileData::DType>()) {
        dValue = (dValue + 1) >> 1; // move fp4 as b8, ceil division to include last nibble for odd counts
    }

    uint64_t loop1SrcStride = GetByteSize<typename TileData::DType>(gStride4);

    constexpr uint16_t ndNum = 1;
    uint16_t loop2DstStride = 1;
    uint16_t loop3DstStride = TileData::Cols;                          // unit is 32B
    uint16_t loop4DstStride = 0;                                       // because ndNum = 1
    uint64_t mte2NzPara = static_cast<uint64_t>(loop4DstStride) << 48; // MTE2_NZ_PARA[63:48]
    mte2NzPara |= static_cast<uint64_t>(loop3DstStride) << 32;         // MTE2_NZ_PARA[47:32]
    mte2NzPara |= static_cast<uint64_t>(loop2DstStride) << 16;         // MTE2_NZ_PARA[31:16]
    mte2NzPara |= static_cast<uint64_t>(ndNum);                        // MTE2_NZ_PARA[15:0]
    set_mte2_nz_para(mte2NzPara);                                      // only set once

    // use nd2nz
    Op::template TLoadCubeInstr<pto::Layout::ND>(dst, src, loop1SrcStride, nValue, dValue, 0);
}

template <typename Op, typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoadCube(
    typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__* src, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
    int validRow, int validCol)
{
#if defined(__DAV_CUBE__)
    using L1Type = __cbuf__ typename TileData::DType*;
    L1Type dstAddr = (L1Type)__cce_get_tile_ptr(dst);

    // ND2NZ or DN2NZ
    if constexpr (
        (GlobalData::layout == pto::Layout::ND || GlobalData::layout == pto::Layout::DN) &&
        (!TileData::isRowMajor && (TileData::SFractal == SLayout::RowMajor))) {
        TLoadCubeND2NZ<Op, TileData, GlobalData>(
            dstAddr, src, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1, gStride2, gStride3, gStride4,
            validRow, validCol);

    } else if constexpr (
        (GlobalData::layout == pto::Layout::NZ) &&
        (!TileData::isRowMajor && (TileData::SFractal == SLayout::RowMajor))) {
        TLoadCubeNZ2NZ<Op, TileData, GlobalData>(
            dstAddr, src, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1, gStride2, gStride3, gStride4,
            validRow, validCol);
    } else if constexpr ((GlobalData::layout == pto::Layout::ND &&
                          (TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox)))) {
        // ND2ND support cols padding
        TLoadCubeND2ND<Op, TileData, GlobalData>(
            dstAddr, src, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1, gStride2, gStride3, gStride4,
            validRow, validCol);
    } else if constexpr (
        GlobalData::layout == pto::Layout::DN && (!TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox))) {
        // dn support rows padding
        TLoadCubeDN2DN<Op, TileData, GlobalData>(
            dstAddr, src, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1, gStride2, gStride3, gStride4,
            validRow, validCol);
    } else if constexpr (
        GlobalData::layout == pto::Layout::DN && (TileData::isRowMajor && (TileData::SFractal == SLayout::ColMajor))) {
        TLoadCubeDN2ZN<Op, TileData, GlobalData>(
            dstAddr, src, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1, gStride2, gStride3, gStride4,
            validRow, validCol);
    }
#endif
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL constexpr bool IsScale()
{
    if constexpr (
        GlobalData::layout == pto::Layout::MX_A_ND || GlobalData::layout == pto::Layout::MX_A_DN ||
        GlobalData::layout == pto::Layout::MX_A_ZZ || GlobalData::layout == pto::Layout::MX_B_ND ||
        GlobalData::layout == pto::Layout::MX_B_DN || GlobalData::layout == pto::Layout::MX_B_NN) {
        return true;
    }
    return false;
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void StaticCheck()
{
    static_assert(
        (sizeof(typename TileData::DType) == 1) || (sizeof(typename TileData::DType) == 2) ||
            (sizeof(typename TileData::DType) == 4) || (sizeof(typename TileData::DType) == 8),
        "Fix: Data type must be b8/b16/b32/b64");
    if constexpr (caps::IsInt64<typename TileData::DType>()) {
        static_assert(
            TileData::PadVal == PadValue::Null || TileData::PadVal == PadValue::Zero,
            "Fix: TileData::PadVal only support Null or Zero in B64 mode");
    }
    static_assert(
        sizeof(typename TileData::DType) == sizeof(typename GlobalData::DType),
        "Fix: Source dtype must be same with dst dtype!");

    // for static shape case, enforce the global tensor (tiled) shape matching with vecTile valid shape for xfer
    if constexpr (TileData::Loc == pto::TileType::Vec) {
        static_assert(
            ((GlobalData::layout == pto::Layout::ND) &&
             (TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox))) ||
                ((GlobalData::layout == pto::Layout::DN) &&
                 (!TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox))) ||
                ((GlobalData::layout == pto::Layout::NZ) &&
                 (!TileData::isRowMajor && (TileData::SFractal == SLayout::RowMajor))),
            "Src and dst layout must be same!");
        if constexpr (TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox)) {
            if constexpr (TileData::ValidCol > 0 && GlobalData::staticShape[4] > 0) {
                static_assert(
                    TileData::ValidCol == GlobalData::staticShape[4],
                    "Fix: Src GlobalTensor Col and Tile ValidCol must be the same!");
            }
            if constexpr (
                TileData::ValidRow > 0 && GlobalData::staticShape[0] > 0 && GlobalData::staticShape[1] > 0 &&
                GlobalData::staticShape[2] > 0 && GlobalData::staticShape[3] > 0) {
                constexpr const int mergedRows = GlobalData::staticShape[0] * GlobalData::staticShape[1] *
                                                 GlobalData::staticShape[2] * GlobalData::staticShape[3];
                static_assert(
                    TileData::ValidRow == mergedRows,
                    "Fix: Src GlobalTensor Row Products and Tile ValidRow must be the same!");
            }
        }
        if constexpr (
            (GlobalData::layout == pto::Layout::NZ) &&
            (!TileData::isRowMajor && (TileData::SFractal == SLayout::RowMajor))) {
            if constexpr (caps::IsFP4<typename TileData::DType>()) {
                static_assert(
                    BLOCK_BYTE_SIZE * 2 == GlobalData::staticShape[4] && BLOCK_LEN == GlobalData::staticShape[3],
                    "Fix: Src GlobalTensor staticShape[3][4] must be satisfied with NZ format require!");
            } else {
                static_assert(
                    BLOCK_BYTE_SIZE / sizeof(typename GlobalData::DType) == GlobalData::staticShape[4] &&
                        BLOCK_LEN == GlobalData::staticShape[3],
                    "Fix: Src GlobalTensor staticShape[3][4] must be satisfied with NZ format require!");
            }
        }
    }
}

template <typename Op, typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoad5HD(
    typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__* src, int srcShape0, int srcShape1,
    int srcShape2, int srcShape3, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4, int dstShape0,
    int dstShape1, int dstShape2, int dstShape3)
{
    __cbuf__ typename TileData::DType* dstAddr = (__cbuf__ typename TileData::DType*)__cce_get_tile_ptr(dst);
    typename GlobalData::DType* srcAddr = src;

    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);

    PTO_ASSERT(
        srcShape1 == dstShape1 && srcShape2 == dstShape2 && srcShape0 == dstShape0 && srcShape3 == dstShape3,
        "Fix: when layout is NC1HWC0 or C1HWNC0, srcShape dstShape should be same!");

    uint32_t nBurst = dstShape2;
    // lenBurst gmStride dstStride unit is byte
    uint32_t lenBurst = GetByteSize<typename TileData::DType>(dstShape3 * c0ElemCount);
    uint64_t gmStride = GetByteSize<typename TileData::DType>(gStride2);
    uint32_t dstStride = lenBurst;

    uint64_t loop2 = dstShape0;
    uint64_t loop1 = dstShape1;
    uint64_t loop2SrcStride = GetByteSize<typename TileData::DType>(gStride0);
    uint64_t loop2DstStride = GetByteSize<typename TileData::DType>(dstShape1 * dstShape2 * dstShape3 * c0ElemCount);

    uint64_t loop1SrcStride = GetByteSize<typename TileData::DType>(gStride1);
    uint64_t loop1DstStride = GetByteSize<typename TileData::DType>(dstShape2 * dstShape3 * c0ElemCount);
#if defined(__DAV_CUBE__)
    set_loop2_stride_outtol1(loop2DstStride << 40 | loop2SrcStride); // [39:0] is loop2 src stride,[60:40] is dst stride
    set_loop1_stride_outtol1(loop1DstStride << 40 | loop1SrcStride); // [39:0] is loop1 src stride,[60:40] is dst stride
    set_loop_size_outtol1(loop2 << 21 | loop1);                      // [20:0] is loop1 size, [40:21] is loop2 size
    Op::TLoadCubeInstr(dstAddr, srcAddr, nBurst, lenBurst, gmStride, dstStride, 0);
#endif
}

template <typename Op, typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoadFractalZ(
    typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__* src, int srcShape0, int srcShape1,
    int srcShape2, int srcShape3, int srcShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
    int dstShape0, int dstShape1, int dstShape2, int dstShape3)
{
    __cbuf__ typename TileData::DType* dstAddr = (__cbuf__ typename TileData::DType*)__cce_get_tile_ptr(dst);
    typename GlobalData::DType* srcAddr = src;

    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename TileData::DType);
    typename GlobalData::DType* srcAddrP = srcAddr;
    __cbuf__ typename TileData::DType* dstAddrP = dstAddr;

    // ConvTile layout is [C1HW,N/16,16,C0]
    static_assert(
        TileData::staticShape[2] == FRACTAL_NZ_ROW && TileData::staticShape[3] == c0ElemCount,
        "Fix: The TileData last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");
    static_assert(
        GlobalData::staticShape[3] == FRACTAL_NZ_ROW && GlobalData::staticShape[4] == c0ElemCount,
        "Fix: The GlobalTensor last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");

    PTO_ASSERT(
        srcShape1 == dstShape0 && srcShape2 == dstShape1 && srcShape3 == dstShape2 && srcShape4 == dstShape3,
        "Fix: layout is [C1HW,N/16,16,C0], srcShape dstShape should be same!");
    uint32_t nBurst = dstShape0; // [1, srcShape1, srcShape2, 16, c0]
    uint32_t lenBurst = dstShape1 * dstShape2 * BLOCK_BYTE_SIZE;
    uint64_t gmStride = GetByteSize<typename TileData::DType>(gStride1);
    uint32_t dstStride = dstShape1 * dstShape2 * BLOCK_BYTE_SIZE;
#if defined(__DAV_CUBE__)
    set_loop_size_outtol1(1ULL << 21 | 1ULL); // use normal mode
    Op::TLoadCubeInstr(dstAddrP, srcAddrP, nBurst, lenBurst, gmStride, dstStride, 0);
#endif
}

} // namespace pto
#endif // TLOAD_COMMON_REGISTER
