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
struct Kirin9030LoadOp : LoadOpBase {
    using LoadOpBase::TLoadCubeInstr;
    template <Layout Layout = Layout::ND, typename T>
    PTO_INTERNAL static void TLoadCubeInstr(
        __cbuf__ T* dst, __gm__ T* src, uint64_t loop1SrcStride, uint16_t nValue, uint32_t dValue,
        uint64_t loop4SrcStride)
    {
        if constexpr (Layout == Layout::ND) {
            pto_copy_gm_to_cbuf_multi_nd2nz(dst, src, 0 /*sid*/, loop1SrcStride, 0, nValue, dValue, loop4SrcStride);
        } else {
            static_assert(sizeof(T) == 0, "Fix: TLoad does not support DN2NZ.");
        }
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
        dstStride1 = dstStride1 >> 1;
        gStride1 = gStride1 >> 1;
        dstStride2 = dstStride2 >> 1;
        gStride2 = gStride2 >> 1;
    }
    for (uint32_t i = 0; i < gShape0; i++) {
        for (uint32_t j = 0; j < gShape1; j++) {
            for (uint32_t k = 0; k < gShape2; k++) {
                int64_t dstAddr0 = i * dstStride0 + j * dstStride1 + k * dstStride2;
                int64_t srcAddr0 = i * gStride0 + j * gStride1 + k * gStride2;
                dstAddrP = dstAddr + dstAddr0;
                srcAddrP = srcAddr + srcAddr0;
                Op::TLoadInstr(dstAddrP, srcAddrP, nBurst, lenBurst, gmStride, ubStride, enableUBPad);
            }
        }
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

    if constexpr (caps::IsFP4<typename TileData::DType>()) {
        dstStride0 = dstStride0 >> 1; // fp4 dstAddr offset need divide 2 as use b8 to move
        gStride0 = gStride0 >> 1;     // fp4 srcAddr offset need divide 2 as use b8 to move
        dstStride1 = dstStride1 >> 1;
        gStride1 = gStride1 >> 1;
        dstStride2 = dstStride2 >> 1;
        gStride2 = gStride2 >> 1;
    }

    for (uint32_t i = 0; i < gShape0; i++) {
        for (uint32_t j = 0; j < gShape1; j++) {
            for (uint32_t k = 0; k < gShape2; k++) {
                int64_t dstAddr0 = i * dstStride0 + j * dstStride1 + k * dstStride2;
                int64_t srcAddr0 = i * gStride0 + j * gStride1 + k * gStride2;
                dstAddrP = dstAddr + dstAddr0;
                srcAddrP = srcAddr + srcAddr0;
                Op::TLoadInstr(dstAddrP, srcAddrP, nBurst, lenBurst, gmStride, ubStride, enableUBPad);
            }
        }
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

    if constexpr (caps::IsFP4<typename TileData::DType>()) {
        dstStride0 = dstStride0 >> 1; // fp4 dstAddr offset need divide 2 as use b8 to move
        gStride0 = gStride0 >> 1;     // fp4 srcAddr offset need divide 2 as use b8 to move
        dstStride1 = dstStride1 >> 1;
        gStride1 = gStride1 >> 1;
        dstStride2 = dstStride2 >> 1;
        gStride2 = gStride2 >> 1;
    }
    for (uint32_t i = 0; i < gShape0; i++) {
        for (uint32_t j = 0; j < gShape1; j++) {
            for (uint32_t k = 0; k < gShape2; k++) {
                int64_t dstAddr0 = i * dstStride0 + j * dstStride1 + k * dstStride2;
                int64_t srcAddr0 = i * gStride0 + j * gStride1 + k * gStride2;
                dstAddrP = dst + dstAddr0;
                srcAddrP = src + srcAddr0;
                Op::TLoadCubeInstr(dstAddrP, srcAddrP, nBurst, lenBurst, gmStride, dstStride, padCount);
            }
        }
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
        dstStride1 = dstStride1 >> 1;
        gStride1 = gStride1 >> 1;
        dstStride2 = dstStride2 >> 1;
        gStride2 = gStride2 >> 1;
    }

    for (uint32_t i = 0; i < gShape0; i++) {
        for (uint32_t j = 0; j < gShape1; j++) {
            for (uint32_t k = 0; k < gShape2; k++) {
                int64_t dstAddr0 = i * dstStride0 + j * dstStride1 + k * dstStride2;
                int64_t srcAddr0 = i * gStride0 + j * gStride1 + k * gStride2;
                dstAddrP = dst + dstAddr0;
                srcAddrP = src + srcAddr0;
                Op::TLoadCubeInstr(dstAddrP, srcAddrP, nBurst, lenBurst, gmStride, dstStride, padCount);
            }
        }
    }
    if constexpr (!(TileData::PadVal == PadValue::Null || TileData::PadVal == PadValue::Zero)) {
        pto_set_tload_pad_val<TileType::Mat>(uint8_t(0));
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TLOAD_TILE_IMPL(TileData& dst, GlobalData& src)
{
    StaticCheck<TileData, GlobalData>();
    if constexpr (TileData::Loc == pto::TileType::Vec) {
        TLoad<Kirin9030LoadOp, TileData, GlobalData>(
            dst.data(), src.data(), src.GetShape(pto::GlobalTensorDim::DIM_0),
            src.GetShape(pto::GlobalTensorDim::DIM_1), src.GetShape(pto::GlobalTensorDim::DIM_2),
            src.GetShape(pto::GlobalTensorDim::DIM_3), src.GetShape(pto::GlobalTensorDim::DIM_4),
            src.GetStride(pto::GlobalTensorDim::DIM_0), src.GetStride(pto::GlobalTensorDim::DIM_1),
            src.GetStride(pto::GlobalTensorDim::DIM_2), src.GetStride(pto::GlobalTensorDim::DIM_3),
            src.GetStride(pto::GlobalTensorDim::DIM_4), dst.GetValidRow(), dst.GetValidCol());
    } else if constexpr (TileData::Loc == pto::TileType::Mat) {
        static_assert(!IsScale<TileData, GlobalData>(), "Fix: TLOAD not supported Mx cube.");
        TLoadCubeCheck<TileData, GlobalData>();
        TLoadCube<Kirin9030LoadOp, TileData, GlobalData>(
            dst.data(), src.data(), src.GetShape(pto::GlobalTensorDim::DIM_0),
            src.GetShape(pto::GlobalTensorDim::DIM_1), src.GetShape(pto::GlobalTensorDim::DIM_2),
            src.GetShape(pto::GlobalTensorDim::DIM_3), src.GetShape(pto::GlobalTensorDim::DIM_4),
            src.GetStride(pto::GlobalTensorDim::DIM_0), src.GetStride(pto::GlobalTensorDim::DIM_1),
            src.GetStride(pto::GlobalTensorDim::DIM_2), src.GetStride(pto::GlobalTensorDim::DIM_3),
            src.GetStride(pto::GlobalTensorDim::DIM_4), dst.GetValidRow(), dst.GetValidCol());
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

#if defined(__DAV_CUBE__)
    for (uint32_t i = 0; i < dstShape0; i++) {
        for (uint32_t j = 0; j < dstShape1; j++) {
            int64_t dstAddr0 =
                i * dstShape1 * dstShape2 * dstShape3 * c0ElemCount + j * dstShape2 * dstShape3 * c0ElemCount;
            int64_t srcAddr0 = i * gStride0 + j * gStride1;
            Op::TLoadCubeInstr(dstAddr + dstAddr0, srcAddr + srcAddr0, nBurst, lenBurst, gmStride, dstStride, 0);
        }
    }
#endif
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void CheckConvTileData(TileData& dst, GlobalData& src)
{
    static_assert(
        caps::IsInt8<typename TileData::DType>() || caps::IsInt16<typename TileData::DType>() ||
            caps::IsInt32<typename TileData::DType>() || caps::IsFP16<typename TileData::DType>() ||
            caps::IsFP32<typename TileData::DType>(),
        "Fix: Data type must be int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/half/float!");
    static_assert(TileData::Loc == pto::TileType::Mat, "Fix: Dst TileType must be Mat!");
    static_assert(
        sizeof(typename TileData::DType) == sizeof(typename GlobalData::DType),
        "Fix: Source dtype must be same with dst dtype!");

    constexpr bool isSameLayout =
        (GlobalData::layout == pto::Layout::NC1HWC0 && TileData::layout == pto::Layout::NC1HWC0) ||
        (GlobalData::layout == pto::Layout::FRACTAL_Z && TileData::layout == pto::Layout::FRACTAL_Z);
    static_assert(isSameLayout == true, "Fix: Src Dst layout must be NC1HWC0 or FRACTAL_Z!");
}

template <typename DstTile, typename SrcGlobal>
PTO_INTERNAL void TLOAD_CONVTILE_IMPL(DstTile& dst, SrcGlobal& src)
{
    CheckConvTileData<DstTile, SrcGlobal>(dst, src);
    if constexpr (SrcGlobal::layout == pto::Layout::NC1HWC0) { // layout is [N,C1,H,W,C0]
        TLoad5HD<Kirin9030LoadOp, DstTile, SrcGlobal>(
            dst.data(), src.data(), src.GetShape(0), src.GetShape(1), src.GetShape(2), src.GetShape(3),
            src.GetStride(0), src.GetStride(1), src.GetStride(2), src.GetStride(3), src.GetStride(4), dst.GetShape(0),
            dst.GetShape(1), dst.GetShape(2), dst.GetShape(3));
    } else if constexpr (SrcGlobal::layout == pto::Layout::FRACTAL_Z) {
        if constexpr (DstTile::totalDimCount == 4) { // layout is [C1HW,N/16,16,C0]
            TLoadFractalZ<Kirin9030LoadOp, DstTile, SrcGlobal>(
                dst.data(), src.data(), src.GetShape(0), src.GetShape(1), src.GetShape(2), src.GetShape(3),
                src.GetShape(4), src.GetStride(0), src.GetStride(1), src.GetStride(2), src.GetStride(3),
                src.GetStride(4), dst.GetShape(0), dst.GetShape(1), dst.GetShape(2), dst.GetShape(3));
        } else { // layout is [C1,H,W,N,C0]
            TLoad5HD<Kirin9030LoadOp, DstTile, SrcGlobal>(
                dst.data(), src.data(), src.GetShape(0), src.GetShape(1), src.GetShape(2), src.GetShape(3),
                src.GetStride(0), src.GetStride(1), src.GetStride(2), src.GetStride(3), src.GetStride(4),
                dst.GetShape(0), dst.GetShape(1), dst.GetShape(2), dst.GetShape(3));
        }
    }
}

template <typename DstTile, typename SrcGlobal>
PTO_INTERNAL void TLOAD_IMPL(DstTile& dst, SrcGlobal& src)
{
    if constexpr (is_conv_tile_v<DstTile>) {
        TLOAD_CONVTILE_IMPL(dst, src);
    } else {
        TLOAD_TILE_IMPL(dst, src);
    }
}
} // namespace pto
#endif // TLOAD_HPP
