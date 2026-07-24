/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSTORE_HPP
#define TSTORE_HPP
#include "common.hpp"

namespace pto {

template <typename GlobalData, typename TileData>
PTO_INTERNAL void TStoreUb2gmInstr(
    typename GlobalData::DType* dst, __ubuf__ typename TileData::DType* src, uint16_t nBurst, uint32_t lenBurst,
    uint32_t gmGap, uint32_t ubGap)
{
    if constexpr (sizeof(typename TileData::DType) == 1) {
        copy_ubuf_to_gm_align_b8(dst, src, 0, nBurst, lenBurst, 0, 0, ubGap, gmGap);
    } else if constexpr (sizeof(typename TileData::DType) == 2) {
        copy_ubuf_to_gm_align_b16(dst, src, 0, nBurst, lenBurst, 0, 0, ubGap, gmGap);
    } else if constexpr (sizeof(typename TileData::DType) == 4 || sizeof(typename TileData::DType) == 8) {
        copy_ubuf_to_gm_align_b32(dst, src, 0, nBurst, lenBurst, 0, 0, ubGap, gmGap);
    }
}

template <typename GlobalData, typename TileData>
PTO_INTERNAL void TStoreUb2gmDn2dn(
    typename GlobalData::DType* dstAddr, __ubuf__ typename TileData::DType* srcAddr, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
    int validRow, int validCol)
{
    PTO_ASSERT(validRow == gShape3, "The validCol of TileData must be equal to the 4th dim(Shape3) of DN shape!");
    PTO_ASSERT(
        validCol == gShape0 * gShape1 * gShape2 * gShape4,
        "The validRow of TileData must be equal to (Shape0 * Shape1 * Shape2 * Shape4) of DN shape!");
    PTO_ASSERT(gShape4 < 4096, "The gshape4 (which equals nBurst) must be less than 4096");
    uint16_t nBurst = gShape4;
    uint32_t lenBurst = validRow * sizeof(typename TileData::DType);
    uint32_t gmGap = (gStride4 - gShape3) * sizeof(typename TileData::DType);
    uint32_t ubGap = ((TileData::Rows - gShape3) * sizeof(typename TileData::DType)) >> SHIFT_BLOCK_BYTE;
    typename GlobalData::DType* dstGlobalAddr = dstAddr;
    __ubuf__ typename TileData::DType* srcTileAddr = srcAddr;
    int64_t srcStride2 = TileData::Rows * gShape4;
    int64_t srcStride1 = gShape2 * srcStride2;
    int64_t srcStride0 = gShape1 * srcStride1;

    for (uint32_t i = 0; i < gShape0; i++) {
        int64_t srcAddr0 = i * srcStride0;
        int64_t dstAddr0 = i * gStride0;
        for (uint32_t j = 0; j < gShape1; j++) {
            int64_t srcAddr1 = j * srcStride1;
            int64_t dstAddr1 = j * gStride1;
            for (uint32_t k = 0; k < gShape2; k++) {
                dstGlobalAddr = dstAddr + dstAddr0 + dstAddr1 + k * gStride2;
                srcTileAddr = srcAddr + srcAddr0 + srcAddr1 + k * srcStride2;
                TStoreUb2gmInstr<GlobalData, TileData>(dstGlobalAddr, srcTileAddr, nBurst, lenBurst, gmGap, ubGap);
            }
        }
    }
}

template <typename GlobalData, typename TileData>
PTO_INTERNAL void TStoreUb2gmNz2nz(
    typename GlobalData::DType* dstAddr, __ubuf__ typename TileData::DType* srcAddr, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
    int validRow, int validCol)
{
    static_assert(
        GlobalData::staticShape[3] == FRACTAL_NZ_ROW &&
            GlobalData::staticShape[4] == BLOCK_BYTE_SIZE / sizeof(typename TileData::DType),
        "When TileData is NZ format, the last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");
    PTO_ASSERT(validRow == gShape2 * gShape3, "The validRow of TileData must be equal to Shape2 * Shape3 of NZ shape!");
    PTO_ASSERT(
        validCol == gShape0 * gShape1 * gShape4,
        "The validCol of TileData must be equal to Shape0 * Shape1 * Shape4 of NZ shape!");
    PTO_ASSERT(gShape1 < 4096, "The gshape1 (which equals nBurst) must be less than 4096");
    uint16_t nBurst = gShape1;
    uint32_t lenBurst = validRow * C0_SIZE_BYTE;
    uint32_t gmGap = (gStride1 - gShape2 * gShape3 * gShape4) * sizeof(typename TileData::DType);
    uint32_t ubGap = TileData::Rows - validRow;

    typename GlobalData::DType* dstGlobalAddr = dstAddr;
    __ubuf__ typename TileData::DType* srcTileAddr = srcAddr;

    int64_t tileStride = TileData::Rows * gShape1 * gShape4;
    for (uint32_t i = 0; i < gShape0; i++) {
        dstGlobalAddr = dstAddr + i * gStride0;
        srcTileAddr = srcAddr + i * tileStride;
        TStoreUb2gmInstr<GlobalData, TileData>(dstGlobalAddr, srcTileAddr, nBurst, lenBurst, gmGap, ubGap);
    }
}

template <
    typename GlobalData, typename TileData, QuantMode_t quantizationMode = QuantMode_t::NoQuant,
    ReluPreMode reluPreMode = ReluPreMode::NoRelu, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TStoreAccNz2nz(
    typename GlobalData::DType* dstAddr, __cc__ typename TileData::DType* srcAddr, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
    int validRow, int validCol)
{
    PTO_ASSERT(validRow == gShape2 * gShape3, "The validRow of TileData must be equal to Shape2 * Shape3 of NZ shape!");
    PTO_ASSERT(
        validCol == gShape0 * gShape1 * gShape4,
        "The validCol of TileData must be equal to Shape0 * Shape1 * Shape4 of NZ shape!");
    PTO_ASSERT(
        validRow >= 1 && validRow <= 65535 && validCol % 16 == 0,
        "When GlobalData is NZ format, the range of validRow is [1, 65535] and validCol must be an integer multiple of "
        "16.");

    static_assert(
        GlobalData::staticShape[3] == FRACTAL_NZ_ROW,
        "When GlobalData is NZ format, the second-to-last dimension shall be 16.");
    static_assert(
        (std::is_same_v<typename GlobalData::DType, __gm__ float> &&
         (GlobalData::staticShape[4] == 8 || GlobalData::staticShape[4] == 16)) ||
            (std::is_same_v<typename GlobalData::DType, __gm__ int32_t> && GlobalData::staticShape[4] == 16) ||
            (GlobalData::staticShape[4] == BLOCK_BYTE_SIZE / sizeof(typename GlobalData::DType)),
        "When GlobalData is in NZ format: if DstType is float, the last dimension must be either 8 or 16, "
        "and the dimension value is 8 if and only if Channel Split is enabled; if DstType is int32_t, the "
        "last dimension must be exactly 16. In addition, the last dimension must be static and satisfy 32 / "
        "sizeof(DstType).");

    uint16_t mSize = validRow;
    uint16_t nSize = validCol;

    uint32_t c0Size = sizeof(typename GlobalData::DType) * gShape4;
    uint16_t srcStride = TileData::Rows;
    if constexpr (CompactMode::Normal == TileData::Compact) {
        srcStride = (FRACTAL_NZ_ROW + validRow - 1) / FRACTAL_NZ_ROW * FRACTAL_NZ_ROW;
    }
}

#include "pto/common/arch/memory/tstore_common.hpp"

template <
    typename GlobalData, typename TileData, QuantMode_t quantizationMode = QuantMode_t::NoQuant,
    ReluPreMode reluPreMode = ReluPreMode::NoRelu, STPhase Phase = STPhase::Unspecified>
__tf__ AICORE void TStoreAcc(
    typename GlobalData::DType __out__* dst, typename TileData::TileDType __in__ src, int gShape0, int gShape1,
    int gShape2, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
    int validRow, int validCol)
{
    __cc__ typename TileData::DType* srcAddr = (__cc__ typename TileData::DType*)__cce_get_tile_ptr(src);
    typename GlobalData::DType* dstAddr = dst;

    if constexpr (GlobalData::layout == Layout::ND) {
        TStoreAccNz2nd<GlobalData, TileData, quantizationMode, reluPreMode, Phase>(
            dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1, gStride2, gStride3,
            gStride4, validRow, validCol);
    } else if constexpr (GlobalData::layout == Layout::NZ) {
        TStoreAccNz2nz<GlobalData, TileData, quantizationMode, reluPreMode, Phase>(
            dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1, gStride2, gStride3,
            gStride4, validRow, validCol);
    } else if constexpr (GlobalData::layout == Layout::NC1HWC0) {
        TStoreAccNz2NC1HWC0<GlobalData, TileData, quantizationMode, reluPreMode, Phase>(
            dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride1, gStride3, validRow, validCol);
    } else if constexpr (GlobalData::layout == Layout::NDC1HWC0) {
        TStoreAccNz2NDC1HWC0<GlobalData, TileData, quantizationMode, reluPreMode, Phase>(
            dstAddr, srcAddr, gShape0, gShape1, gShape2, gShape3, gShape4, gStride2, gStride4, validRow, validCol);
    }
}

template <typename GlobalData, typename TileData, typename FpTileData,
          QuantMode_t quantizationMode = QuantMode_t::NoQuant, ReluPreMode reluPreMode = ReluPreMode::NoRelu>
__tf__ AICORE void TStoreAccFp(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
                               typename FpTileData::TileDType __in__ fp, int gShape0, int gShape1, int gShape2,
                               int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3,
                               int gStride4, int validRow, int validCol)
{
    __fbuf__ typename FpTileData::DType *fpDstAddr = (__fbuf__ typename FpTileData::DType *)__cce_get_tile_ptr(fp);
    uint64_t deqTensorAddr = ((uint64_t)fpDstAddr >> static_cast<uint64_t>(7)) << 8;
    set_fpc(deqTensorAddr);
    pipe_barrier(PIPE_FIX);
    if constexpr (GlobalData::layout == Layout::ND) {
        TStoreAccNz2nd<GlobalData, TileData, quantizationMode, reluPreMode>(
            dst, __cce_get_tile_ptr(src), gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1, gStride2,
            gStride3, gStride4, validRow, validCol);
    } else if constexpr (GlobalData::layout == Layout::NZ) {
        TStoreAccNz2nz<GlobalData, TileData, quantizationMode, reluPreMode>(
            dst, __cce_get_tile_ptr(src), gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1, gStride2,
            gStride3, gStride4, validRow, validCol);
    } else if constexpr (GlobalData::layout == Layout::NC1HWC0) {
        TStoreAccNz2NC1HWC0<GlobalData, TileData, quantizationMode, reluPreMode>(
            dst, __cce_get_tile_ptr(src), gShape0, gShape1, gShape2, gShape3, gShape4, gStride1, gStride3, validRow,
            validCol);
    } else if constexpr (GlobalData::layout == Layout::NDC1HWC0) {
        TStoreAccNz2NDC1HWC0<GlobalData, TileData, quantizationMode, reluPreMode>(
            dst, __cce_get_tile_ptr(src), gShape0, gShape1, gShape2, gShape3, gShape4, gStride2, gStride4, validRow,
            validCol);
    }
}

template <typename TileData, typename GlobalData, bool isQuant>
PTO_INTERNAL void CheckAcc2gm(GlobalData& dst, TileData& src)
{
    static_assert(
        (GlobalData::layout == Layout::ND || GlobalData::layout == Layout::NZ ||
         GlobalData::layout == Layout::NC1HWC0 || GlobalData::layout == Layout::NDC1HWC0),
        "The output data layout must be ND, NZ, NC1HWC0 or NDC1HWC0.");
    static_assert(
        std::is_same_v<typename TileData::DType, int32_t> || std::is_same_v<typename TileData::DType, float>,
        "The input data type must be restricted to int32_t/float!");
    if constexpr (!isQuant) {
        static_assert(
            std::is_same_v<typename GlobalData::DType, __gm__ int32_t> ||
                std::is_same_v<typename GlobalData::DType, __gm__ float> ||
                std::is_same_v<typename GlobalData::DType, __gm__ half> ||
                std::is_same_v<typename GlobalData::DType, __gm__ bfloat16_t>,
            "The output data type must be restricted to int32_t/float/half/bfloat16_t!");
    } else if constexpr (isQuant) {
        if constexpr (std::is_same_v<typename TileData::DType, float>) {
            static_assert(
                std::is_same<typename GlobalData::DType, __gm__ int8_t>::value ||
                    std::is_same<typename GlobalData::DType, __gm__ uint8_t>::value,
                "The output data type must be restricted to int8_t/uint8_t when the source data type is float.");
        } else if constexpr (std::is_same_v<typename TileData::DType, int32_t>) {
            static_assert(
                std::is_same<typename GlobalData::DType, __gm__ int8_t>::value ||
                    std::is_same<typename GlobalData::DType, __gm__ uint8_t>::value ||
                    std::is_same<typename GlobalData::DType, __gm__ half>::value,
                "The output data type must be restricted to half/int8_t/uint8_t when the source data type is int32_t.");
        }
    }
    static_assert(TileData::Cols >= 1 && TileData::Cols <= 4095, "The range of Cols is [1, 4095].");
    static_assert(
        (GlobalData::layout == Layout::ND && TileData::Rows >= 1 && TileData::Rows <= 8192) ||
            ((GlobalData::layout == Layout::NZ || GlobalData::layout == Layout::NC1HWC0 ||
              GlobalData::layout == Layout::NDC1HWC0) &&
             TileData::Rows >= 1 && TileData::Rows <= 65535 && TileData::Cols % 16 == 0),
        "When GlobalData is ND format, the range of Rows is [1, 8192]."
        "When GlobalData is NZ or NC1HWC0 or NDC1HWC0 format, the range of Rows is [1, 65535] and Cols "
        "must be an integer multiple of 16.");
    PTO_ASSERT(src.GetValidCol() >= 1 && src.GetValidCol() <= 4095, "The range of validCol is [1, 4095].");
    PTO_ASSERT(
        dst.GetShape(GlobalTensorDim::DIM_0) > 0 && dst.GetShape(GlobalTensorDim::DIM_1) > 0 &&
            dst.GetShape(GlobalTensorDim::DIM_2) > 0 && dst.GetShape(GlobalTensorDim::DIM_3) > 0 &&
            dst.GetShape(GlobalTensorDim::DIM_4) > 0 && src.GetValidRow() > 0 && src.GetValidCol() > 0,
        "The shape of src and dst must be greater than 0!");
}

template <
    typename TileData, typename GlobalData, AtomicType currentAtomicType = AtomicType::AtomicNone,
    STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TSTORE_IMPL(GlobalData& dst, TileData& src)
{
    static_assert(
        TileData::Loc == TileType::Vec || TileData::Loc == TileType::Acc || TileData::Loc == TileType::Mat,
        "Source TileType only support Vec/Acc/Mat!");
    PTO_ASSERT(
        dst.GetShape(GlobalTensorDim::DIM_0) > 0 && dst.GetShape(GlobalTensorDim::DIM_1) > 0 &&
            dst.GetShape(GlobalTensorDim::DIM_2) > 0 && dst.GetShape(GlobalTensorDim::DIM_3) > 0 &&
            dst.GetShape(GlobalTensorDim::DIM_4) > 0 && src.GetValidRow() > 0 && src.GetValidCol() > 0,
        "The shape of src and dst must be greater than 0!");
    if constexpr (currentAtomicType == AtomicType::AtomicAdd) {
        SetAtomicAdd<typename GlobalData::DType>();
    }
    if constexpr (TileData::Loc == TileType::Vec) {
        CheckStaticForVecAndMat<TileData, GlobalData>();
        TStore<GlobalData, TileData>(
            dst.data(), src.data(), dst.GetShape(GlobalTensorDim::DIM_0), dst.GetShape(GlobalTensorDim::DIM_1),
            dst.GetShape(GlobalTensorDim::DIM_2), dst.GetShape(GlobalTensorDim::DIM_3),
            dst.GetShape(GlobalTensorDim::DIM_4), dst.GetStride(GlobalTensorDim::DIM_0),
            dst.GetStride(GlobalTensorDim::DIM_1), dst.GetStride(GlobalTensorDim::DIM_2),
            dst.GetStride(GlobalTensorDim::DIM_3), dst.GetStride(GlobalTensorDim::DIM_4), src.GetValidRow(),
            src.GetValidCol());
    } else if constexpr (TileData::Loc == TileType::Acc) {
        CheckAcc2gm<TileData, GlobalData, false>(dst, src);
        constexpr QuantMode_t quantMode = GetCastPreQuantMode<typename TileData::DType, typename GlobalData::DType>();
        TStoreAcc<GlobalData, TileData, quantMode, ReluPreMode::NoRelu, Phase>(
            dst.data(), src.data(), dst.GetShape(GlobalTensorDim::DIM_0), dst.GetShape(GlobalTensorDim::DIM_1),
            dst.GetShape(GlobalTensorDim::DIM_2), dst.GetShape(GlobalTensorDim::DIM_3),
            dst.GetShape(GlobalTensorDim::DIM_4), dst.GetStride(GlobalTensorDim::DIM_0),
            dst.GetStride(GlobalTensorDim::DIM_1), dst.GetStride(GlobalTensorDim::DIM_2),
            dst.GetStride(GlobalTensorDim::DIM_3), dst.GetStride(GlobalTensorDim::DIM_4), src.GetValidRow(),
            src.GetValidCol());
    } else if constexpr (TileData::Loc == TileType::Mat) {
        CheckStaticForVecAndMat<TileData, GlobalData>();
        TStoreMat<GlobalData, TileData>(
            dst.data(), src.data(), dst.GetShape(GlobalTensorDim::DIM_0), dst.GetShape(GlobalTensorDim::DIM_1),
            dst.GetShape(GlobalTensorDim::DIM_2), dst.GetShape(GlobalTensorDim::DIM_3),
            dst.GetShape(GlobalTensorDim::DIM_4), dst.GetStride(GlobalTensorDim::DIM_0),
            dst.GetStride(GlobalTensorDim::DIM_1), dst.GetStride(GlobalTensorDim::DIM_2),
            dst.GetStride(GlobalTensorDim::DIM_3), dst.GetStride(GlobalTensorDim::DIM_4), src.GetValidRow(),
            src.GetValidCol());
    }
    if constexpr (currentAtomicType == AtomicType::AtomicAdd) {
        SetAtomicNone();
    }
}

template <
    typename TileData, typename GlobalData, AtomicType currentAtomicType = AtomicType::AtomicNone,
    ReluPreMode reluPreMode, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TSTORE_IMPL(GlobalData& dst, TileData& src)
{
    static_assert(TileData::Loc == TileType::Acc, "Source TileType only support Acc!");
    CheckAcc2gm<TileData, GlobalData, false>(dst, src);
    if constexpr (currentAtomicType == AtomicType::AtomicAdd) {
        SetAtomicAdd<typename GlobalData::DType>();
    }
    constexpr QuantMode_t quantMode = GetCastPreQuantMode<typename TileData::DType, typename GlobalData::DType>();
    TStoreAcc<GlobalData, TileData, quantMode, reluPreMode, Phase>(
        dst.data(), src.data(), dst.GetShape(GlobalTensorDim::DIM_0), dst.GetShape(GlobalTensorDim::DIM_1),
        dst.GetShape(GlobalTensorDim::DIM_2), dst.GetShape(GlobalTensorDim::DIM_3),
        dst.GetShape(GlobalTensorDim::DIM_4), dst.GetStride(GlobalTensorDim::DIM_0),
        dst.GetStride(GlobalTensorDim::DIM_1), dst.GetStride(GlobalTensorDim::DIM_2),
        dst.GetStride(GlobalTensorDim::DIM_3), dst.GetStride(GlobalTensorDim::DIM_4), src.GetValidRow(),
        src.GetValidCol());
    if constexpr (currentAtomicType == AtomicType::AtomicAdd) {
        SetAtomicNone();
    }
}

template <
    typename TileData, typename GlobalData, AtomicType currentAtomicType = AtomicType::AtomicNone,
    ReluPreMode reluPreMode = ReluPreMode::NoRelu, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TSTORE_IMPL(GlobalData& dst, TileData& src, uint64_t preQuantScalar)
{
    static_assert(TileData::Loc == TileType::Acc, "Source TileType only support Acc!");
    CheckAcc2gm<TileData, GlobalData, true>(dst, src);
    if constexpr (currentAtomicType == AtomicType::AtomicAdd) {
        SetAtomicAdd<typename GlobalData::DType>();
    }
    set_quant_pre(preQuantScalar);
    constexpr QuantMode_t quantMode = GetScalarPreQuantMode<typename TileData::DType, typename GlobalData::DType>();
    TStoreAcc<GlobalData, TileData, quantMode, reluPreMode, Phase>(
        dst.data(), src.data(), dst.GetShape(GlobalTensorDim::DIM_0), dst.GetShape(GlobalTensorDim::DIM_1),
        dst.GetShape(GlobalTensorDim::DIM_2), dst.GetShape(GlobalTensorDim::DIM_3),
        dst.GetShape(GlobalTensorDim::DIM_4), dst.GetStride(GlobalTensorDim::DIM_0),
        dst.GetStride(GlobalTensorDim::DIM_1), dst.GetStride(GlobalTensorDim::DIM_2),
        dst.GetStride(GlobalTensorDim::DIM_3), dst.GetStride(GlobalTensorDim::DIM_4), src.GetValidRow(),
        src.GetValidCol());
    if constexpr (AtomicType::AtomicAdd == currentAtomicType) {
        SetAtomicNone();
    }
}

template <
    typename TileData, typename GlobalData, typename FpTileData, AtomicType currentAtomicType = AtomicType::AtomicNone,
    ReluPreMode reluPreMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TSTORE_IMPL(GlobalData& dst, TileData& src, FpTileData& fp)
{
    static_assert(TileData::Loc == TileType::Acc, "Source TileType only support Acc!");
    CheckAcc2gm<TileData, GlobalData, true>(dst, src);
    if constexpr (AtomicType::AtomicAdd == currentAtomicType) {
        SetAtomicAdd<typename GlobalData::DType>();
    }
    constexpr QuantMode_t quantMode = GetVectorPreQuantMode<typename TileData::DType, typename GlobalData::DType>();
    TStoreAccFp<GlobalData, TileData, FpTileData, quantMode, reluPreMode>(
        dst.data(), src.data(), fp.data(), dst.GetShape(GlobalTensorDim::DIM_0), dst.GetShape(GlobalTensorDim::DIM_1),
        dst.GetShape(GlobalTensorDim::DIM_2), dst.GetShape(GlobalTensorDim::DIM_3),
        dst.GetShape(GlobalTensorDim::DIM_4), dst.GetStride(GlobalTensorDim::DIM_0),
        dst.GetStride(GlobalTensorDim::DIM_1), dst.GetStride(GlobalTensorDim::DIM_2),
        dst.GetStride(GlobalTensorDim::DIM_3), dst.GetStride(GlobalTensorDim::DIM_4), src.GetValidRow(),
        src.GetValidCol());
    if constexpr (currentAtomicType == AtomicType::AtomicAdd) {
        SetAtomicNone();
    }
}
} // namespace pto
#endif
