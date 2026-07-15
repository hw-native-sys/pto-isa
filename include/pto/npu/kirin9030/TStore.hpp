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
#include "common.hpp"
#include "pto/common/arch/register/tstore_common.hpp"
namespace pto {
template <typename SrcType, typename DstType>
PTO_INTERNAL constexpr QuantMode_t GetCastPreQuantModeGm()
{
    return QuantMode_t::NoQuant;
}

template <typename SrcType, typename DstType>
PTO_INTERNAL constexpr QuantMode_t GetScalarPreQuantModeGm()
{
    QuantMode_t quantPre = QuantMode_t::NoQuant;
    if constexpr (caps::IsSInt32<SrcType>()) {
        if constexpr (caps::IsInt8<DstType>()) {
            quantPre = QuantMode_t::REQ8;
        } else if constexpr (caps::IsFP16<DstType>()) {
            quantPre = QuantMode_t::DEQF16;
        }
    } else if constexpr (caps::IsFP16<SrcType>()) {
        if constexpr (caps::IsInt8<DstType>()) {
            quantPre = QuantMode_t::QF162B8_PRE;
        }
    }
    return quantPre;
}

template <typename SrcType, typename DstType>
PTO_INTERNAL constexpr QuantMode_t GetVectorPreQuantModeGm()
{
    QuantMode_t quantPre = QuantMode_t::NoQuant;
    if constexpr (caps::IsSInt32<SrcType>()) {
        if constexpr (caps::IsInt8<DstType>()) {
            quantPre = QuantMode_t::VREQ8;
        } else if constexpr (caps::IsFP16<DstType>()) {
            quantPre = QuantMode_t::VDEQF16;
        }
    } else if constexpr (caps::IsFP16<SrcType>()) {
        if constexpr (caps::IsInt8<DstType>()) {
            quantPre = QuantMode_t::VQF162B8_PRE;
        }
    }
    return quantPre;
}

template <typename TileData, typename GlobalData, bool isQuant>
PTO_INTERNAL void CheckStaticAcc()
{
    static_assert(
        caps::IsSInt32<typename TileData::DType>() || caps::IsFP16<typename TileData::DType>(),
        "The input data type must be restricted to int32_t/half!");
    static_assert(
        (GlobalData::layout == pto::Layout::ND) || (GlobalData::layout == pto::Layout::NZ),
        "TSTORE(Acc2GM) only support NZ2ND / NZ2NZ.");
    static_assert(TileData::Cols >= 1 && TileData::Cols <= 4095, "The range of Cols is [1, 4095].");
    static_assert(
        (GlobalData::layout == pto::Layout::ND && TileData::Rows >= 1 && TileData::Rows <= 8192) ||
            (GlobalData::layout == pto::Layout::NZ && TileData::Rows >= 1 && TileData::Rows <= 65535 &&
             TileData::Cols % 16 == 0),
        "When GlobalData is ND format, the range of Rows is [1, 8192]."
        "When GlobalData is NZ format, the range of Rows is [1, 65535] and Cols"
        "must be an integer multiple of 16.");
    if constexpr (!isQuant) {
        static_assert(
            caps::IsSInt32<typename GlobalData::RawDType>() || caps::IsFP32<typename GlobalData::RawDType>() ||
                caps::IsFP16<typename GlobalData::RawDType>(),
            "The output data type must be restricted to int32_t/float/half!");
    } else if constexpr (isQuant) {
        if constexpr (caps::IsFP16<typename TileData::DType>()) {
            static_assert(
                caps::IsFP16<typename GlobalData::RawDType>() || caps::IsSInt16<typename GlobalData::RawDType>() ||
                    caps::IsSInt8<typename GlobalData::RawDType>() || caps::IsUInt8<typename GlobalData::RawDType>(),
                "The output data type must be restricted to half/int16_t/int8_t/uint8_t.");
        } else if constexpr (caps::IsSInt32<typename TileData::DType>()) {
            static_assert(
                caps::IsFP16<typename GlobalData::RawDType>() || caps::IsSInt16<typename GlobalData::RawDType>() ||
                    caps::IsSInt8<typename GlobalData::RawDType>() || caps::IsUInt8<typename GlobalData::RawDType>() ||
                    caps::IsSInt32<typename GlobalData::RawDType>(),
                "The output data type must be restricted to half/int16_t/int8_t/uint8_t/int32_t.");
        }
    }
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void CheckStaticVec()
{
    static_assert(
        sizeof(typename TileData::DType) == sizeof(typename GlobalData::RawDType),
        "Source dtype must be same with dst dtype!");
    static_assert(
        caps::IsTypeSupported<typename TileData::DType>(),
        "Data type must be int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/int64_t/uint64_t/half/float!");
    static_assert(
        ((GlobalData::layout == pto::Layout::ND) &&
         (TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox))) ||
            ((GlobalData::layout == pto::Layout::DN) &&
             (!TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox))) ||
            ((GlobalData::layout == pto::Layout::NZ) &&
             (!TileData::isRowMajor && (TileData::SFractal == SLayout::RowMajor))) ||
            (TileData::Rows == 1) || (TileData::Cols == 1),
        "Src and dst layout must be same, only support ND/DN/NZ or the special case of one row/one column!");
    if constexpr (GlobalData::layout == pto::Layout::ND) {
        static_assert(
            (TileData::Cols * sizeof(typename TileData::DType) % BLOCK_BYTE_SIZE == 0) ||
                ((TileData::Cols == 1) && (TileData::Rows * sizeof(typename TileData::DType) % BLOCK_BYTE_SIZE == 0)),
            "Fix: TSTORE For ND layout, Cols * sizeof(DType) must be 32-byte aligned, or Rows * sizeof(DType) must be "
            "32-byte aligned when Cols == 1.");
    } else if constexpr (GlobalData::layout == pto::Layout::DN) {
        static_assert(
            (TileData::Rows * sizeof(typename TileData::DType) % BLOCK_BYTE_SIZE == 0) ||
                ((TileData::Rows == 1) && (TileData::Cols * sizeof(typename TileData::DType) % BLOCK_BYTE_SIZE == 0)),
            "Fix: TSTORE For DN layout, Rows * sizeof(DType) must be 32-byte aligned, or Cols * sizeof(DType) must be "
            "32-byte aligned when Rows == 1.");
    } else {
        static_assert(
            GlobalData::layout == pto::Layout::NZ,
            "Fix: TSTORE Unsupported layout format, only ND/DN/NZ are supported.");
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
        TileData::Loc == pto::TileType::Vec || TileData::Loc == pto::TileType::Acc,
        "Source TileType only suport Vec/Acc!");
    if constexpr (TileData::Loc == pto::TileType::Acc) {
        using L0cT = typename TileData::DType;
        using DstT = typename GlobalData::RawDType;
        CheckStaticAcc<TileData, GlobalData, false>();

        constexpr QuantMode_t quantPre = GetCastPreQuantModeGm<L0cT, DstT>();
        TStoreAcc<GlobalData, TileData, quantPre, ReluPreMode::NoRelu, Phase>(
            dst.data(), src.data(), dst.GetShape(dim0), dst.GetShape(dim1), dst.GetShape(dim2), dst.GetShape(dim3),
            dst.GetShape(dim4), dst.GetStride(dim0), dst.GetStride(dim1), dst.GetStride(dim2), dst.GetStride(dim3),
            dst.GetStride(dim4), src.GetValidRow(), src.GetValidCol());
    } else if constexpr (TileData::Loc == pto::TileType::Vec) {
        CheckStaticVec<TileData, GlobalData>();

        TStore<GlobalData, TileData>(
            dst.data(), src.data(), dst.GetShape(dim0), dst.GetShape(dim1), dst.GetShape(dim2), dst.GetShape(dim3),
            dst.GetShape(dim4), dst.GetStride(dim0), dst.GetStride(dim1), dst.GetStride(dim2), dst.GetStride(dim3),
            dst.GetStride(dim4), src.GetValidRow(), src.GetValidCol());
    }
}

template <
    typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone, ReluPreMode reluPreMode,
    STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TSTORE_IMPL(GlobalData& dst, TileData& src)
{
    static_assert(TileData::Loc == pto::TileType::Acc, "Source TileType only suport Acc!");
    using L0cT = typename TileData::DType;
    using DstT = typename GlobalData::RawDType;
    CheckStaticAcc<TileData, GlobalData, false>();
    constexpr QuantMode_t quantPre = GetCastPreQuantModeGm<L0cT, DstT>();
    constexpr int dim0 = pto::GlobalTensorDim::DIM_0;
    constexpr int dim1 = pto::GlobalTensorDim::DIM_1;
    constexpr int dim2 = pto::GlobalTensorDim::DIM_2;
    constexpr int dim3 = pto::GlobalTensorDim::DIM_3;
    constexpr int dim4 = pto::GlobalTensorDim::DIM_4;
    TStoreAcc<GlobalData, TileData, quantPre, reluPreMode, Phase>(
        dst.data(), src.data(), dst.GetShape(dim0), dst.GetShape(dim1), dst.GetShape(dim2), dst.GetShape(dim3),
        dst.GetShape(dim4), dst.GetStride(dim0), dst.GetStride(dim1), dst.GetStride(dim2), dst.GetStride(dim3),
        dst.GetStride(dim4), src.GetValidRow(), src.GetValidCol());
}

template <
    typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
    ReluPreMode reluPreMode = ReluPreMode::NoRelu, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TSTORE_IMPL(GlobalData& dst, TileData& src, uint64_t preQuantScalar)
{
    static_assert(TileData::Loc == pto::TileType::Acc, "Source TileType only suport Acc!");

    using L0cT = typename TileData::DType;
    using DstT = typename GlobalData::RawDType;
    CheckStaticAcc<TileData, GlobalData, true>();
    constexpr QuantMode_t quantPre = GetScalarPreQuantModeGm<L0cT, DstT>();
    set_quant_pre(preQuantScalar);
    constexpr int dim0 = pto::GlobalTensorDim::DIM_0;
    constexpr int dim1 = pto::GlobalTensorDim::DIM_1;
    constexpr int dim2 = pto::GlobalTensorDim::DIM_2;
    constexpr int dim3 = pto::GlobalTensorDim::DIM_3;
    constexpr int dim4 = pto::GlobalTensorDim::DIM_4;
    TStoreAcc<GlobalData, TileData, quantPre, reluPreMode, Phase>(
        dst.data(), src.data(), dst.GetShape(dim0), dst.GetShape(dim1), dst.GetShape(dim2), dst.GetShape(dim3),
        dst.GetShape(dim4), dst.GetStride(dim0), dst.GetStride(dim1), dst.GetStride(dim2), dst.GetStride(dim3),
        dst.GetStride(dim4), src.GetValidRow(), src.GetValidCol());
}

template <
    typename TileData, typename GlobalData, typename FpTileData, AtomicType atomicType = AtomicType::AtomicNone,
    ReluPreMode reluPreMode = ReluPreMode::NoRelu, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TSTORE_IMPL(GlobalData& dst, TileData& src, FpTileData& fp)
{
    static_assert(TileData::Loc == pto::TileType::Acc, "Source TileType only suport Acc!");
    using DstT = typename GlobalData::RawDType;
    using L0cT = typename TileData::DType;
    CheckStaticAcc<TileData, GlobalData, true>();
    constexpr QuantMode_t quantPre = GetVectorPreQuantModeGm<L0cT, DstT>();
    constexpr int dim0 = pto::GlobalTensorDim::DIM_0;
    constexpr int dim1 = pto::GlobalTensorDim::DIM_1;
    constexpr int dim2 = pto::GlobalTensorDim::DIM_2;
    constexpr int dim3 = pto::GlobalTensorDim::DIM_3;
    constexpr int dim4 = pto::GlobalTensorDim::DIM_4;
    TStoreAccFp<GlobalData, TileData, FpTileData, quantPre, reluPreMode>(
        dst.data(), src.data(), fp.data(), dst.GetShape(dim0), dst.GetShape(dim1), dst.GetShape(dim2),
        dst.GetShape(dim3), dst.GetShape(dim4), dst.GetStride(dim0), dst.GetStride(dim1), dst.GetStride(dim2),
        dst.GetStride(dim3), dst.GetStride(dim4), src.GetValidRow(), src.GetValidCol());
}
} // namespace pto
#endif
