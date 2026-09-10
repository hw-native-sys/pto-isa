/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TLOAD_COMMON_MEMORY
#define TLOAD_COMMON_MEMORY
#include <pto/common/utils.hpp>
#include <pto/common/arch_capability.hpp>

#if defined(PTO_NPU_ARCH_A2A3)
template <TLoadL2Hint l2Control, typename T>
PTO_INTERNAL T* TLoadSrcAddrWithL2Hint(T* addr)
{
    return ApplyTLoadL2HintAddr<l2Control>(addr);
}
#else
template <TLoadL2Hint l2Control, typename T>
PTO_INTERNAL T* TLoadSrcAddrWithL2Hint(T* addr)
{
    (void)l2Control;
    return addr;
}
#endif

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadNd2nzInstr(
    __cbuf__ typename TileData::DType* dst, typename GlobalData::DType* src, uint16_t ndNum, uint16_t nValue,
    uint16_t dValue, uint16_t srcNdMatrixStride, uint16_t srcDValue, uint16_t dstNzC0Stride, uint16_t dstNzNStride,
    uint16_t dstNzMatrixStride)
{
    pto_copy_gm_to_cbuf_multi_nd2nz(
        dst, src, 0, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue, dstNzC0Stride, dstNzNStride,
        dstNzMatrixStride);
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void CheckNzFormat(
    int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int validRow, int validCol)
{
    static_assert(
        GlobalData::staticShape[3] == FRACTAL_NZ_ROW &&
            GlobalData::staticShape[4] == C0_SIZE_BYTE / sizeof(typename TileData::DType),
        "Fix: When TileData is NZ format, the last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");
    PTO_ASSERT(validRow == gShape2 * gShape3, "The validRow of TileData must be equal to Shape2 * Shape3 of NZ shape!");
    PTO_ASSERT(
        validCol == gShape0 * gShape1 * gShape4,
        "The validCol of TileData must be equal to Shape0 * Shape1 * Shape4 of NZ shape!");
    PTO_ASSERT(gShape1 < 4096, "The gshape1 (which equals nBurst) must be less than 4096");
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2ubNd2nd(
    __ubuf__ typename TileData::DType* dstAddr, typename GlobalData::DType* srcAddr, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
    int validRow, int validCol)
{
    static_assert(TileData::Rows < 4096, "Fix: TLOAD Rows>=4096 not supported in A2/A3");
    PTO_ASSERT(validCol == gShape4, "The validCol of TileData must be equal to the 5th dim(Shape4) of ND shape!");
    PTO_ASSERT(
        validRow == gShape0 * gShape1 * gShape2 * gShape3,
        "The validRow of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape3) of ND shape!");
    PTO_ASSERT(gShape3 < 4096, "The gshape3 (which equals nBurst) must be less than 4096");
    constexpr uint32_t blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    uint16_t nBurst = gShape3;
    uint32_t lenBurst = validCol * sizeof(typename TileData::DType);
    uint64_t gmGapValue = (gStride3 - gShape4) * sizeof(typename TileData::DType);
    uint32_t gmGap = (uint32_t)gmGapValue;
    uint32_t ubGapElement = (TileData::Cols - validCol);
    uint32_t ubGap = (ubGapElement * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    uint32_t ubPad = 0;
    if constexpr (TileData::PadVal != PadValue::Null) {
        ubPad = ubGapElement % blockSizeElem;
        pto_set_tload_pad_val<TileType::Vec>(GetPadValue<TileData>());
    }
    __ubuf__ typename TileData::DType* dstAddrP = dstAddr;
    typename GlobalData::DType* srcAddrP = srcAddr;
    int64_t dstStride2 = gShape3 * TileData::Cols;
    int64_t dstStride1 = gShape2 * dstStride2;
    int64_t dstStride0 = gShape1 * dstStride1;

    for (uint32_t i = 0; i < gShape0; i++) {
        int64_t srcAddr0 = i * gStride0;
        int64_t dstAddr0 = i * dstStride0;
        for (uint32_t j = 0; j < gShape1; j++) {
            int64_t srcAddr1 = j * gStride1;
            int64_t dstAddr1 = j * dstStride1;
            for (uint32_t k = 0; k < gShape2; k++) {
                srcAddrP = srcAddr + srcAddr0 + srcAddr1 + k * gStride2;
                dstAddrP = dstAddr + dstAddr0 + dstAddr1 + k * dstStride2;
                TLoadInstrGm2ub<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, ubGap, ubPad);
            }
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2ubDn2dn(
    __ubuf__ typename TileData::DType* dstAddr, typename GlobalData::DType* srcAddr, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
    int validRow, int validCol)
{
    PTO_ASSERT(validRow == gShape3, "The validCol of TileData must be equal to the 4th dim(Shape3) of DN shape!");
    PTO_ASSERT(
        validCol == gShape0 * gShape1 * gShape2 * gShape4,
        "The validRow of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape4) of DN shape!");
    PTO_ASSERT(gShape4 < 4096, "The gshape4 (which equals nBurst) must be less than 4096");
    constexpr uint32_t blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    uint16_t nBurst = gShape4;
    uint32_t lenBurst = validRow * sizeof(typename TileData::DType);
    uint64_t gmGapValue = (gStride4 - gShape3) * sizeof(typename TileData::DType);
    uint32_t gmGap = (uint32_t)gmGapValue;
    uint32_t ubGapElement = (TileData::Rows - gShape3);
    uint32_t ubGap = (ubGapElement * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    uint32_t ubPad = 0;
    if constexpr (TileData::PadVal != PadValue::Null) {
        ubPad = ubGapElement % blockSizeElem;
        pto_set_tload_pad_val<TileType::Vec>(GetPadValue<TileData>());
    }
    typename GlobalData::DType* srcAddrP = srcAddr;
    __ubuf__ typename TileData::DType* dstAddrP = dstAddr;

    int64_t dstStride2 = gShape4 * TileData::Rows;
    int64_t dstStride1 = gShape2 * dstStride2;
    int64_t dstStride0 = gShape1 * dstStride1;
    for (uint32_t i = 0; i < gShape0; i++) {
        int64_t dstAddr0 = i * dstStride0;
        int64_t srcAddr0 = i * gStride0;
        for (uint32_t j = 0; j < gShape1; j++) {
            int64_t dstAddr1 = j * dstStride1;
            int64_t srcAddr1 = j * gStride1;
            for (uint32_t k = 0; k < gShape2; k++) {
                dstAddrP = dstAddr + dstAddr0 + dstAddr1 + k * dstStride2;
                srcAddrP = srcAddr + srcAddr0 + srcAddr1 + k * gStride2;
                TLoadInstrGm2ub<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, ubGap, ubPad);
            }
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2ubNz2nz(
    __ubuf__ typename TileData::DType* dstAddr, typename GlobalData::DType* srcAddr, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
    int validRow, int validCol)
{
    CheckNzFormat<TileData, GlobalData>(gShape0, gShape1, gShape2, gShape3, gShape4, validRow, validCol);
    uint16_t nBurst = gShape1;
    uint32_t lenBurst = validRow * C0_SIZE_BYTE;
    uint32_t gmGap = (gStride1 - validRow * gShape4) * sizeof(typename TileData::DType);
    uint32_t ubGap = TileData::Rows - validRow;
    typename GlobalData::DType* srcAddrP = srcAddr;
    __ubuf__ typename TileData::DType* dstAddrP = dstAddr;
    int64_t tileStride = TileData::Rows * gShape1 * gShape4;
    for (uint32_t i = 0; i < gShape0; i++) {
        srcAddrP = srcAddr + i * gStride0;
        dstAddrP = dstAddr + i * tileStride;
        TLoadInstrGm2ub<TileData, GlobalData>(dstAddrP, srcAddrP, nBurst, lenBurst, gmGap, ubGap, 0);
    }
}

template <typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoadGm2ub(
    typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__* src, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
    int validRow, int validCol)
{
    __ubuf__ typename TileData::DType* dstAddr = (__ubuf__ typename TileData::DType*)__cce_get_tile_ptr(dst);
    typename GlobalData::DType* srcAddr = src;
    if constexpr (GetTileLayoutCustom<TileData>() == TileLayoutCustom::ND) {
        TLoadGm2ubNd2nd<TileData, GlobalData>(
            dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1, gStride2, gStride3,
            gStride4, validRow, validCol);
    } else if constexpr (GetTileLayoutCustom<TileData>() == TileLayoutCustom::DN) {
        TLoadGm2ubDn2dn<TileData, GlobalData>(
            dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1, gStride2, gStride3,
            gStride4, validRow, validCol);
    } else if constexpr (GetTileLayoutCustom<TileData>() == TileLayoutCustom::NZ) {
        TLoadGm2ubNz2nz<TileData, GlobalData>(
            dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1, gStride2, gStride3,
            gStride4, validRow, validCol);
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2L1VectorInND(
    __cbuf__ typename TileData::DType* dstAddr, typename GlobalData::DType* srcAddr, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
    int validRow, int validCol)
{
    PTO_ASSERT(validCol == gShape4, "The validCol of TileData must be equal to the 5th dim(Shape4) of ND shape!");
    PTO_ASSERT(
        validRow == gShape0 * gShape1 * gShape2 * gShape3,
        "The validRow of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape3) of ND shape!");
    static_assert(
        GlobalData::staticShape[0] == 1 && GlobalData::staticShape[1] == 1 && GlobalData::staticShape[2] == 1,
        "Fix: GlobalTensor only support 2 dim when using vector input!");
    uint16_t nValue = gShape3;
    uint16_t dValue = gShape4;
    uint16_t srcDValue = gStride3;
    typename GlobalData::DType* srcAddrP = srcAddr;
    __cbuf__ typename TileData::DType* dstAddrP = dstAddr;
    TLoadNd2nzInstr<TileData, GlobalData>(dstAddrP, srcAddrP, 1, nValue, dValue, 0, srcDValue, TileData::Rows, 1, 1);
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLoadGm2L1VectorInDn(
    __cbuf__ typename TileData::DType* dstAddr, typename GlobalData::DType* srcAddr, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
    int validRow, int validCol)
{
    static_assert(
        GlobalData::staticShape[0] == 1 && GlobalData::staticShape[1] == 1 && GlobalData::staticShape[2] == 1,
        "Fix: GlobalTensor only support 2 dim when using vector input!");
    PTO_ASSERT(validRow == gShape3, "The validCol of TileData must be equal to the 4th dim(Shape3) of DN shape!");
    PTO_ASSERT(
        validCol == gShape0 * gShape1 * gShape2 * gShape4,
        "The validRow of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape4) of DN shape!");
    uint16_t nValue = gShape4;
    uint16_t dValue = gShape3;
    uint16_t srcDValue = gStride3;
    typename GlobalData::DType* srcAddrP = srcAddr;
    __cbuf__ typename TileData::DType* dstAddrP = dstAddr;
    TLoadNd2nzInstr<TileData, GlobalData>(dstAddrP, srcAddrP, 1, nValue, dValue, 0, srcDValue, TileData::Cols, 1, 1);
}

template <typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoadGm2L1(
    typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__* src, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
    int validRow, int validCol)
{
    __cbuf__ typename TileData::DType* dstAddr = (__cbuf__ typename TileData::DType*)__cce_get_tile_ptr(dst);
    typename GlobalData::DType* srcAddr = src;
    if constexpr (GetTileLayoutCustom<TileData>() == TileLayoutCustom::ND) {
        if constexpr (TileData::Rows == 1) {
            TLoadGm2L1VectorInND<TileData, GlobalData>(
                dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1, gStride2, gStride3,
                gStride4, validRow, validCol);
        } else {
            TLoadGm2L1Nd2nd<TileData, GlobalData>(
                dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1, gStride2, gStride3,
                gStride4, validRow, validCol);
        }
    } else if constexpr (GetTileLayoutCustom<TileData>() == TileLayoutCustom::DN) {
        if constexpr (TileData::Cols == 1) {
            TLoadGm2L1VectorInDn<TileData, GlobalData>(
                dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1, gStride2, gStride3,
                gStride4, validRow, validCol);
        } else {
            TLoadGm2L1Dn2dn<TileData, GlobalData>(
                dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1, gStride2, gStride3,
                gStride4, validRow, validCol);
        }
    } else if constexpr (GetTileLayoutCustom<TileData>() == TileLayoutCustom::NZ) {
        TLoadGm2L1Nz2nz<TileData, GlobalData>(
            dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1, gStride2, gStride3,
            gStride4, validRow, validCol);
    }
}

template <typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoadGm2L1Nd2nz(
    typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__* src, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
    int validRow, int validCol)
{
    __cbuf__ typename TileData::DType* dstAddr = (__cbuf__ typename TileData::DType*)__cce_get_tile_ptr(dst);
    typename GlobalData::DType* srcAddr = src;
    // GlobalTensor is 2 dim, or 3 dim with DIM_2 mapped onto the instruction's ndNum.
    static_assert(
        GlobalData::staticShape[0] == 1 && GlobalData::staticShape[1] == 1,
        "Fix: GlobalTensor only support 2 dim when ND2NZ, or 3 dim with Shape2 as ndNum!");
    constexpr bool multiNd = GlobalData::staticShape[2] != 1;
    static_assert(
        !multiNd || caps::SupportsNd2nzMultiND<>(),
        "Fix: GlobalTensor Shape2 != 1 (multi-ND ND2NZ) is not supported on current platform");
    static_assert(TileData::SFractalSize == 512, "Fix: TileData only support SFractalSize = 512Bytes!");
    PTO_ASSERT(gShape3 > 0 && gShape3 <= 16384, "The Shape3 of GlobalTensor must be in range of [1, 16384]!");
    PTO_ASSERT(gShape4 > 0 && gShape4 <= 65535, "The Shape4 of GlobalTensor must be must be in range of [1, 65535]!");
    PTO_ASSERT(
        gStride3 > 0 && gStride3 <= 65535, "The Stride3 of GlobalTensor must be must be in range of [1, 65535]!");
    static_assert(TileData::Rows <= 16384, "Fix: The Rows of TileData must be less than 16384!");

    uint16_t nValue = gShape3;
    uint16_t dValue = gShape4;
    uint16_t srcDValue = gStride3;
    // ndNum matrices of [nValue, dValue] are stacked along the NZ row direction, so the tile row
    // index is i2 * nValue + i3. srcNdMatrixStride and dstNzMatrixStride are both in elements,
    // while dstNzC0Stride and dstNzNStride stay in C0 units; one NZ row is one C0 of elements.
    // b64 reaches this instruction through the b32s form with dValue/srcDValue doubled, so if ND2NZ
    // is ever opened up for b64 these two element strides must be doubled the same way. Today
    // CheckNormalTileData rejects b64 for anything but ND2ND/DN2DN, which keeps them consistent.
    uint16_t ndNum = 1;
    uint16_t srcNdMatrixStride = 0;
    uint16_t dstNzMatrixStride = 1;
    if constexpr (multiNd) {
        constexpr int c0Elem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        PTO_ASSERT(gShape2 > 0 && gShape2 <= 65535, "The Shape2 (ndNum) of GlobalTensor must be in [1, 65535]!");
        // The two matrix strides only drive the instruction once there is more than one nd matrix, so
        // a dynamic Shape2 that happens to be 1 at run time must not trip their 16-bit range checks.
        PTO_ASSERT(
            gShape2 == 1 || (gStride2 > 0 && gStride2 <= 65535),
            "The Stride2 of GlobalTensor must be in [1, 65535], it is the 16-bit srcNdMatrixStride!");
        PTO_ASSERT(
            gShape2 == 1 || gShape3 * c0Elem <= 65535,
            "The Shape3 * C0 must not exceed 65535, it is the 16-bit dstNzMatrixStride!");
        PTO_ASSERT(gShape2 * gShape3 <= TileData::Rows, "The ndNum * nValue must not exceed TileData::Rows!");
        PTO_ASSERT(validRow == gShape2 * gShape3, "The validRow must be equal to Shape2 * Shape3 in multi-ND ND2NZ!");
        ndNum = static_cast<uint16_t>(gShape2);
        srcNdMatrixStride = static_cast<uint16_t>(gStride2);
        dstNzMatrixStride = static_cast<uint16_t>(gShape3 * c0Elem);
    }
    TLoadNd2nzInstr<TileData, GlobalData>(
        dstAddr, srcAddr, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue, TileData::Rows, 1, dstNzMatrixStride);
}

template <typename TileData, typename GlobalData>
__tf__ PTO_INTERNAL void TLoadGm2L1Dn2zn(
    typename TileData::TileDType __out__ dst, typename GlobalData::DType __in__* src, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
    int validRow, int validCol)
{
    static_assert(
        GlobalData::staticShape[0] == 1 && GlobalData::staticShape[1] == 1 && GlobalData::staticShape[2] == 1,
        "Fix: GlobalTensor only support 2 dim when DN2ZN!");
    static_assert(TileData::SFractalSize == 512, "Fix: TileData only support SFractalSize = 512Bytes!");
    __cbuf__ typename TileData::DType* dstAddr = (__cbuf__ typename TileData::DType*)__cce_get_tile_ptr(dst);
    typename GlobalData::DType* srcAddr = src;
    PTO_ASSERT(gShape4 > 0 && gShape4 <= 16384, "The Shape4 of GlobalTensor must be in range of [1, 16384]!");
    PTO_ASSERT(gShape3 > 0 && gShape3 <= 65535, "The Shape3 of GlobalTensor must be must be in range of [1, 65535]!");
    PTO_ASSERT(
        gStride4 > 0 && gStride4 <= 65535, "The Stride3 of GlobalTensor must be must be in range of [1, 65535]!");
    static_assert(TileData::Cols <= 16384, "Fix: The Cols of TileData must be less than 16384!");

    uint16_t nValue = gShape4;
    uint16_t dValue = gShape3;
    uint16_t srcDValue = gStride4;
    TLoadNd2nzInstr<TileData, GlobalData>(dstAddr, srcAddr, 1, nValue, dValue, 0, srcDValue, TileData::Cols, 1, 1);
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void CheckNormalTileData(TileData& dst, GlobalData& src)
{
    static_assert(
        std::is_same_v<typename TileData::DType, int8_t> || std::is_same_v<typename TileData::DType, uint8_t> ||
            std::is_same_v<typename TileData::DType, int16_t> || std::is_same_v<typename TileData::DType, uint16_t> ||
            std::is_same_v<typename TileData::DType, int32_t> || std::is_same_v<typename TileData::DType, uint32_t> ||
            std::is_same_v<typename TileData::DType, int64_t> || std::is_same_v<typename TileData::DType, uint64_t> ||
            std::is_same_v<typename TileData::DType, half> || std::is_same_v<typename TileData::DType, bfloat16_t> ||
            std::is_same_v<typename TileData::DType, float>,
        "Fix: Data type must be "
        "int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/half/bfloat16_t/float/int64_t/uint64_t!");
    static_assert(
        TileData::Loc == TileType::Vec || TileData::Loc == TileType::Mat, "Fix: Dst TileType must be Vec or Mat!");
    static_assert(
        sizeof(typename TileData::DType) == sizeof(typename GlobalData::DType),
        "Fix: Source dtype must be same with dst dtype!");

    if constexpr (
        std::is_same_v<typename TileData::DType, int64_t> || std::is_same_v<typename TileData::DType, uint64_t>) {
        static_assert(
            (GlobalData::layout == Layout::ND && GetTileLayoutCustom<TileData>() == TileLayoutCustom::ND) ||
                (GlobalData::layout == Layout::DN && GetTileLayoutCustom<TileData>() == TileLayoutCustom::DN),
            "Fix: TLOAD only support ND2ND/DN2DN for b64!");
    }
    PTO_ASSERT(
        src.GetShape(GlobalTensorDim::DIM_0) > 0 && src.GetShape(GlobalTensorDim::DIM_1) > 0 &&
            src.GetShape(GlobalTensorDim::DIM_2) > 0 && src.GetShape(GlobalTensorDim::DIM_3) > 0 &&
            src.GetShape(GlobalTensorDim::DIM_4) > 0 && dst.GetValidRow() > 0 && dst.GetValidCol() > 0,
        "The shape of src and dst must be greater than 0!");
}

template <typename TileData, typename GlobalData, TLoadL2Hint l2Control = TLoadL2Hint::NormalFirstVictim>
PTO_INTERNAL void TLOAD_TILE_IMPL(TileData& dst, GlobalData& src)
{
    CheckNormalTileData<TileData, GlobalData>(dst, src);
    constexpr bool isSameLayout =
        (GlobalData::layout == Layout::ND && GetTileLayoutCustom<TileData>() == TileLayoutCustom::ND) ||
        (GlobalData::layout == Layout::DN && GetTileLayoutCustom<TileData>() == TileLayoutCustom::DN) ||
        (GlobalData::layout == Layout::NZ && GetTileLayoutCustom<TileData>() == TileLayoutCustom::NZ);
    if constexpr (TileData::Loc == TileType::Vec) {
        static_assert(isSameLayout, "Fix: TLOAD(VecTile, GlobalTensor) only support ND2ND/DN2DN/NZ2NZ!");
        TLoadGm2ub<TileData, GlobalData>(
            dst.data(), TLoadSrcAddrWithL2Hint<l2Control>(src.data()), src.GetShape(GlobalTensorDim::DIM_0),
            src.GetShape(GlobalTensorDim::DIM_1), src.GetShape(GlobalTensorDim::DIM_2),
            src.GetShape(GlobalTensorDim::DIM_3), src.GetShape(GlobalTensorDim::DIM_4),
            src.GetStride(GlobalTensorDim::DIM_0), src.GetStride(GlobalTensorDim::DIM_1),
            src.GetStride(GlobalTensorDim::DIM_2), src.GetStride(GlobalTensorDim::DIM_3),
            src.GetStride(GlobalTensorDim::DIM_4), dst.GetValidRow(), dst.GetValidCol());
    } else if constexpr (TileData::Loc == TileType::Mat) {
        static_assert(
            isSameLayout ||
                (GlobalData::layout == Layout::ND && GetTileLayoutCustom<TileData>() == TileLayoutCustom::NZ) ||
                (GlobalData::layout == Layout::DN && GetTileLayoutCustom<TileData>() == TileLayoutCustom::ZN),
            "Fix: TLOAD(MatTile, GlobalTensor) only support ND2ND/DN2DN/NZ2NZ/ND2NZ/DN2ZN!");
        if constexpr (isSameLayout) {
            TLoadGm2L1<TileData, GlobalData>(
                dst.data(), TLoadSrcAddrWithL2Hint<l2Control>(src.data()), src.GetShape(GlobalTensorDim::DIM_0),
                src.GetShape(GlobalTensorDim::DIM_1), src.GetShape(GlobalTensorDim::DIM_2),
                src.GetShape(GlobalTensorDim::DIM_3), src.GetShape(GlobalTensorDim::DIM_4),
                src.GetStride(GlobalTensorDim::DIM_0), src.GetStride(GlobalTensorDim::DIM_1),
                src.GetStride(GlobalTensorDim::DIM_2), src.GetStride(GlobalTensorDim::DIM_3),
                src.GetStride(GlobalTensorDim::DIM_4), dst.GetValidRow(), dst.GetValidCol());
        } else if constexpr (
            GlobalData::layout == Layout::ND && GetTileLayoutCustom<TileData>() == TileLayoutCustom::NZ) {
            TLoadGm2L1Nd2nz<TileData, GlobalData>(
                dst.data(), TLoadSrcAddrWithL2Hint<l2Control>(src.data()), src.GetShape(GlobalTensorDim::DIM_0),
                src.GetShape(GlobalTensorDim::DIM_1), src.GetShape(GlobalTensorDim::DIM_2),
                src.GetShape(GlobalTensorDim::DIM_3), src.GetShape(GlobalTensorDim::DIM_4),
                src.GetStride(GlobalTensorDim::DIM_0), src.GetStride(GlobalTensorDim::DIM_1),
                src.GetStride(GlobalTensorDim::DIM_2), src.GetStride(GlobalTensorDim::DIM_3),
                src.GetStride(GlobalTensorDim::DIM_4), dst.GetValidRow(), dst.GetValidCol());
        } else if constexpr (
            GlobalData::layout == Layout::DN && GetTileLayoutCustom<TileData>() == TileLayoutCustom::ZN) {
            TLoadGm2L1Dn2zn<TileData, GlobalData>(
                dst.data(), TLoadSrcAddrWithL2Hint<l2Control>(src.data()), src.GetShape(GlobalTensorDim::DIM_0),
                src.GetShape(GlobalTensorDim::DIM_1), src.GetShape(GlobalTensorDim::DIM_2),
                src.GetShape(GlobalTensorDim::DIM_3), src.GetShape(GlobalTensorDim::DIM_4),
                src.GetStride(GlobalTensorDim::DIM_0), src.GetStride(GlobalTensorDim::DIM_1),
                src.GetStride(GlobalTensorDim::DIM_2), src.GetStride(GlobalTensorDim::DIM_3),
                src.GetStride(GlobalTensorDim::DIM_4), dst.GetValidRow(), dst.GetValidCol());
        }
    }
}

template <typename TileData, typename GlobalData, TLoadL2Hint l2Control = TLoadL2Hint::NormalFirstVictim>
PTO_INTERNAL void TLOAD_CONVTILE_IMPL(TileData& dst, GlobalData& src)
{
    CheckConvTileData<TileData, GlobalData>(dst, src);
    if constexpr (GlobalData::layout == pto::Layout::NC1HWC0) {
        TLoad5HD<TileData, GlobalData>(
            dst.data(), TLoadSrcAddrWithL2Hint<l2Control>(src.data()), src.GetShape(0), src.GetShape(1),
            src.GetShape(2), src.GetShape(3), src.GetStride(0), src.GetStride(1), src.GetStride(2), src.GetStride(3),
            src.GetStride(4), dst.GetShape(0), dst.GetShape(1), dst.GetShape(2), dst.GetShape(3));
    } else if constexpr (
        GlobalData::layout == pto::Layout::FRACTAL_Z || GlobalData::layout == pto::Layout::FRACTAL_Z_3D) {
        TLoadFractalZ<TileData, GlobalData>(
            dst.data(), TLoadSrcAddrWithL2Hint<l2Control>(src.data()), src.GetShape(0), src.GetShape(1),
            src.GetShape(2), src.GetShape(3), src.GetShape(4), src.GetStride(0), src.GetStride(1), src.GetStride(2),
            src.GetStride(3), src.GetStride(4), dst.GetShape(0), dst.GetShape(1), dst.GetShape(2), dst.GetShape(3));
    } else if constexpr (GlobalData::layout == pto::Layout::NDC1HWC0) {
        TLoadNDC1HWC0<TileData, GlobalData>(
            dst.data(), TLoadSrcAddrWithL2Hint<l2Control>(src.data()), src.GetShape(0), src.GetShape(1),
            src.GetShape(2), src.GetShape(3), src.GetShape(4), src.GetStride(0), src.GetStride(1), src.GetStride(2),
            src.GetStride(3), src.GetStride(4), dst.GetShape(0), dst.GetShape(1), dst.GetShape(2), dst.GetShape(3),
            dst.GetShape(4));
    }
}

#endif
