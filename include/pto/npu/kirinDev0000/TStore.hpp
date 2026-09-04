/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSTORE_HPP
#define TSTORE_HPP
#include "pto/npu/kirin9030/common.hpp"
#include "pto/common/arch_cce_intrinsic.hpp"
#include "pto/common/arch/register/tstore_common.hpp"

namespace pto {

template <typename GlobalData, typename TileData>
PTO_INTERNAL void TStoreMatViaUb(
    typename GlobalData::DType* dstAddr, __cbuf__ typename TileData::DType* srcAddr,
    __ubuf__ typename TileData::DType* ubAddr, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4,
    int gStride0, int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    static_assert(
        GlobalData::staticShape[3] == FRACTAL_NZ_ROW &&
            GlobalData::staticShape[4] == BLOCK_BYTE_SIZE / sizeof(typename TileData::DType),
        "When TileData is NZ format, the last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");
    PTO_ASSERT(gShape1 < 4096, "The gshape1 (which equals nBurst) must be less than 4096");
    PTO_ASSERT(validRow == gShape2 * gShape3, "The validRow of TileData must be equal to Shape2 * Shape3 of NZ shape!");
    PTO_ASSERT(
        validCol == gShape0 * gShape1 * gShape4,
        "The validCol of TileData must be equal to Shape0 * Shape1 * Shape4 of NZ shape!");

    uint32_t totalBytes = TileData::Rows * TileData::Cols * sizeof(typename TileData::DType);
    uint32_t burstLen = totalBytes / BLOCK_BYTE_SIZE;
    uint64_t fixpNzPara =
        static_cast<uint64_t>(1) | (static_cast<uint64_t>(burstLen) << 16) | (static_cast<uint64_t>(1) << 32);
    set_fixp_nz_para(fixpNzPara);
    pto_copy_cbuf_to_ubuf((__ubuf__ void*)ubAddr, (__cbuf__ void*)srcAddr, 0, 1, burstLen, 0, 0);
    set_fixp_nz_para(0);
    pipe_barrier(PIPE_ALL);

    uint16_t nBurst = gShape1;
    uint32_t lenBurstBytes = validRow * sizeof(typename TileData::DType);
    uint32_t srcGap = (TileData::Rows - validRow) * sizeof(typename TileData::DType) / BLOCK_BYTE_SIZE;
    uint32_t dstGap = (gStride1 - gShape2 * gShape3 * gShape4) * sizeof(typename TileData::DType);

    typename GlobalData::DType* dstAddrP = dstAddr;
    __ubuf__ typename TileData::DType* ubSrcP = ubAddr;

    int64_t tileStride = TileData::Rows * gShape1 * gShape4;

    for (uint32_t i = 0; i < gShape0; i++) {
        dstAddrP = dstAddr + i * gStride0;
        ubSrcP = ubAddr + i * tileStride;
        pto_copy_ubuf_to_gm_align_v2(
            reinterpret_cast<__gm__ typename GlobalData::RawDType*>(dstAddrP),
            reinterpret_cast<__ubuf__ typename TileData::DType*>(ubSrcP), 0, nBurst, lenBurstBytes, 0,
            srcGap * BLOCK_BYTE_SIZE, dstGap);
    }
}

template <typename GlobalData, typename TileData, typename UbTile, AtomicType atomicType = AtomicType::AtomicNone>
__tf__ AICORE void TStoreMat(
    typename GlobalData::DType __out__* dst, typename TileData::TileDType __in__ src,
    typename UbTile::TileDType __in__ ubTmp, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4,
    int gStride0, int gStride1, int gStride2, int gStride3, int gStride4, int validRow, int validCol)
{
    __cbuf__ typename TileData::DType* srcAddr = (__cbuf__ typename TileData::DType*)__cce_get_tile_ptr(src);
    __ubuf__ typename TileData::DType* ubAddr = (__ubuf__ typename TileData::DType*)__cce_get_tile_ptr(ubTmp);
    typename GlobalData::DType* dstAddr = dst;

    static_assert(
        !TileData::isRowMajor && (TileData::SFractal == SLayout::RowMajor),
        "TStoreMat: only support NZ fractal layout.");
    TStoreMatViaUb<GlobalData, TileData>(
        dstAddr, srcAddr, ubAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1, gStride2, gStride3,
        gStride4, validRow, validCol);
}

template <typename GlobalData, typename TileData>
PTO_INTERNAL void TStoreVecND(
    typename GlobalData::DType* dstAddr, __ubuf__ typename TileData::DType* srcAddr, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
    int validRow, int validCol)
{
    PTO_ASSERT(validCol == gShape4, "The validCol of TileData must be equal to the 5th dim(Shape4) of ND shape!");
    PTO_ASSERT(
        validRow == gShape0 * gShape1 * gShape2 * gShape3,
        "The validRow of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape3) of ND shape!");
    typename GlobalData::DType* dstGlobalAddr = dstAddr;
    __ubuf__ typename TileData::DType* srcTileAddr = srcAddr;

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
        for (uint32_t i = 0; i < gShape1; i++) {
            for (uint32_t j = 0; j < gShape2; j++) {
                dstGlobalAddr = dstAddr + k * gStride0 + i * gStride1 + j * gStride2;
                srcTileAddr =
                    srcAddr + k * srcStride0 + i * gShape2 * gShape3 * TileData::Cols + j * gShape3 * TileData::Cols;
                TStoreInstr<TileData, GlobalData>(
                    dstGlobalAddr, srcTileAddr, nBurst, lenBurst, burstDstStride, burstSrcStride);
            }
        }
    }
}

template <typename GlobalData, typename TileData>
PTO_INTERNAL void TStoreVecDN(
    typename GlobalData::DType* dstAddr, __ubuf__ typename TileData::DType* srcAddr, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
    int validRow, int validCol)
{
    PTO_ASSERT(validRow == gShape3, "The validRow of TileData must be equal to the 4th dim(Shape3) of DN shape!");
    PTO_ASSERT(
        validCol == gShape0 * gShape1 * gShape2 * gShape4,
        "The validCol of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape4) of DN shape!");
    typename GlobalData::DType* dstGlobalAddr = dstAddr;
    __ubuf__ typename TileData::DType* srcTileAddr = srcAddr;

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
        for (uint32_t i = 0; i < gShape1; i++) {
            for (uint32_t j = 0; j < gShape2; j++) {
                dstGlobalAddr = dstAddr + k * gStride0 + i * gStride1 + j * gStride2;
                srcTileAddr =
                    srcAddr + k * srcStride0 + i * gShape2 * TileData::Rows * gShape4 + j * TileData::Rows * gShape4;
                TStoreInstr<TileData, GlobalData>(
                    dstGlobalAddr, srcTileAddr, nBurst, lenBurst, burstDstStride, burstSrcStride);
            }
        }
    }
}

//
//
//
//
template <typename GlobalData, typename TileData>
__tf__ AICORE void TStoreVecNZFractalSplit(
    typename GlobalData::DType __out__* dst, typename TileData::TileDType __in__ src, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
    int validRow, int validCol)
{
    __ubuf__ typename TileData::DType* srcAddr = (__ubuf__ typename TileData::DType*)__cce_get_tile_ptr(src);
    typename GlobalData::DType* dstAddr = dst;

    constexpr uint32_t c0Size = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    constexpr uint32_t srcInnerCols = TileData::InnerCols;
    constexpr uint32_t subBlocksPerFractal = srcInnerCols / c0Size;
    constexpr uint32_t dstRowBytes = c0Size * sizeof(typename TileData::DType);
    constexpr uint32_t srcRowBytes = srcInnerCols * sizeof(typename TileData::DType);
    constexpr uint32_t srcFractalElems = FRACTAL_NZ_ROW * srcInnerCols;

    //
    //
    //
    //
    //
    int mOuter = gShape2;
    for (int k = 0; k < gShape0; k++) {
        int srcFractal = k / static_cast<int>(subBlocksPerFractal);
        int subBlock = k % static_cast<int>(subBlocksPerFractal);
        for (int m = 0; m < mOuter; m++) {
            __ubuf__ typename TileData::DType* srcP =
                srcAddr + srcFractal * mOuter * static_cast<int>(srcFractalElems) +
                m * static_cast<int>(srcFractalElems) + subBlock * static_cast<int>(c0Size);
            typename GlobalData::DType* dstP = dstAddr + k * gStride0 + m * static_cast<int>(FRACTAL_NZ_ROW * c0Size);
            TStoreInstr<TileData, GlobalData>(
                dstP, srcP, static_cast<uint32_t>(FRACTAL_NZ_ROW), dstRowBytes, static_cast<uint64_t>(dstRowBytes),
                srcRowBytes);
        }
    }
}

template <
    typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
    STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TSTORE_IMPL(GlobalData& dst, TileData& src)
{
    constexpr int dim0 = pto::GlobalTensorDim::DIM_0;
    constexpr int dim1 = pto::GlobalTensorDim::DIM_1;
    constexpr int dim2 = pto::GlobalTensorDim::DIM_2;
    constexpr int dim3 = pto::GlobalTensorDim::DIM_3;
    constexpr int dim4 = pto::GlobalTensorDim::DIM_4;
    static_assert(
        TileData::Loc == pto::TileType::Vec || TileData::Loc == pto::TileType::Acc ||
            TileData::Loc == pto::TileType::Mat,
        "Source TileType only support Vec/Acc/Mat!");
    static_assert(atomicType == AtomicType::AtomicNone, "Fix: AtomicAdd is not supported on Dev0000.");
    if constexpr (TileData::Loc == pto::TileType::Mat) {
        static_assert(sizeof(TileData::DType) == 0, "TSTORE(Mat2GM): use TMOV(Mat->Vec) then TSTORE(Vec2GM) instead.");
    } else if constexpr (TileData::Loc == pto::TileType::Acc) {
        static_assert(sizeof(TileData::DType) == 0, "TSTORE(Acc2GM): use TMOV(Acc->Vec) then TSTORE(Vec2GM) instead.");
    } else if constexpr (TileData::Loc == pto::TileType::Vec) {
        CheckStaticVecCommon<TileData, GlobalData>();

        //
        //
        constexpr uint32_t c0Size = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        if constexpr (
            (!TileData::isRowMajor && (TileData::SFractal == SLayout::RowMajor)) && (TileData::InnerCols > c0Size)) {
            TStoreVecNZFractalSplit<GlobalData, TileData>(
                dst.data(), src.data(), dst.GetShape(dim0), dst.GetShape(dim1), dst.GetShape(dim2), dst.GetShape(dim3),
                dst.GetShape(dim4), dst.GetStride(dim0), dst.GetStride(dim1), dst.GetStride(dim2), dst.GetStride(dim3),
                dst.GetStride(dim4), src.GetValidRow(), src.GetValidCol());
        } else {
            TStore<GlobalData, TileData>(
                dst.data(), src.data(), dst.GetShape(dim0), dst.GetShape(dim1), dst.GetShape(dim2), dst.GetShape(dim3),
                dst.GetShape(dim4), dst.GetStride(dim0), dst.GetStride(dim1), dst.GetStride(dim2), dst.GetStride(dim3),
                dst.GetStride(dim4), src.GetValidRow(), src.GetValidCol());
        }
    }
}

template <
    typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone, ReluPreMode reluPreMode,
    STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TSTORE_IMPL(GlobalData& dst, TileData& src)
{
    static_assert(sizeof(TileData::DType) == 0, "TSTORE(Acc2GM): use TMOV(Acc->Vec) then TSTORE(Vec2GM) instead.");
}

template <
    typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
    ReluPreMode reluPreMode = ReluPreMode::NoRelu, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TSTORE_IMPL(GlobalData& dst, TileData& src, uint64_t preQuantScalar)
{
    static_assert(sizeof(TileData::DType) == 0, "TSTORE(Acc2GM): use TMOV(Acc->Vec) then TSTORE(Vec2GM) instead.");
}

template <
    typename TileData, typename GlobalData, typename FpTileData, AtomicType atomicType = AtomicType::AtomicNone,
    ReluPreMode reluPreMode = ReluPreMode::NoRelu, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TSTORE_IMPL(GlobalData& dst, TileData& src, FpTileData& fp)
{
    static_assert(sizeof(TileData::DType) == 0, "TSTORE(Acc2GM): use TMOV(Acc->Vec) then TSTORE(Vec2GM) instead.");
}
} // namespace pto
#endif
