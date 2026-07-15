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
    QuantMode_t quantPre = QuantMode_t::NoQuant;
    if constexpr (caps::IsFP32<SrcType>()) {
        if constexpr (caps::IsBF16<DstType>()) {
            quantPre = QuantMode_t::F322BF16;
        } else if constexpr (caps::IsFP16<DstType>()) {
            quantPre = QuantMode_t::F322F16;
        }
    }
    return quantPre;
}

template <typename SrcType, typename DstType>
PTO_INTERNAL constexpr QuantMode_t GetScalarPreQuantModeGm()
{
    QuantMode_t quantPre = QuantMode_t::NoQuant;
    if constexpr (caps::IsFP32<SrcType>()) {
        if constexpr (caps::IsInt8<DstType>()) {
            quantPre = QuantMode_t::QF322B8_PRE;
        } else if constexpr (caps::IsHF8<DstType>()) {
            quantPre = QuantMode_t::QF322HIF8_PRE;
        } else if constexpr (caps::IsFP16<DstType>()) {
            quantPre = QuantMode_t::QF322F16_PRE;
        } else if constexpr (caps::IsBF16<DstType>()) {
            quantPre = QuantMode_t::QF322BF16_PRE;
        } else if constexpr (caps::IsFP8E4M3<DstType>()) {
            quantPre = QuantMode_t::QF322FP8_PRE;
        } else if constexpr (caps::IsFP32<DstType>()) {
            quantPre = QuantMode_t::QF322F32_PRE;
        }
    } else if constexpr (caps::IsSInt32<SrcType>()) {
        if constexpr (caps::IsInt8<DstType>()) {
            quantPre = QuantMode_t::REQ8;
        } else if constexpr (caps::IsFP16<DstType>()) {
            quantPre = QuantMode_t::DEQF16;
        } else if constexpr (caps::IsBF16<DstType>()) {
            quantPre = QuantMode_t::QS322BF16_PRE;
        }
    }
    return quantPre;
}

template <typename SrcType, typename DstType>
PTO_INTERNAL constexpr QuantMode_t GetVectorPreQuantModeGm()
{
    QuantMode_t quantPre = QuantMode_t::NoQuant;
    if constexpr (caps::IsFP32<SrcType>()) {
        if constexpr (caps::IsInt8<DstType>()) {
            quantPre = QuantMode_t::VQF322B8_PRE;
        } else if constexpr (caps::IsHF8<DstType>()) {
            quantPre = QuantMode_t::VQF322HIF8_PRE;
        } else if constexpr (caps::IsFP16<DstType>()) {
            quantPre = QuantMode_t::VQF322F16_PRE;
        } else if constexpr (caps::IsBF16<DstType>()) {
            quantPre = QuantMode_t::VQF322BF16_PRE;
        } else if constexpr (caps::IsFP8E4M3<DstType>()) {
            quantPre = QuantMode_t::VQF322FP8_PRE;
        } else if constexpr (caps::IsFP32<DstType>()) {
            quantPre = QuantMode_t::VQF322F32_PRE;
        }
    } else if constexpr (caps::IsSInt32<SrcType>()) {
        if constexpr (caps::IsInt8<DstType>()) {
            quantPre = QuantMode_t::VREQ8;
        } else if constexpr (caps::IsFP16<DstType>()) {
            quantPre = QuantMode_t::VDEQF16;
        } else if constexpr (caps::IsBF16<DstType>()) {
            quantPre = QuantMode_t::VQS322BF16_PRE;
        }
    }
    return quantPre;
}

template <typename T>
PTO_INTERNAL void SetAtomicAdd()
{
    static_assert(
        (caps::IsFP16<T>()) || (caps::IsFP32<T>()) || (caps::IsSInt16<T>()) || (caps::IsSInt32<T>()) ||
            (caps::IsSInt8<T>()) || (caps::IsBF16<T>()),
        "Dst and src must be half / float / int16_t / int32_t / int8_t / bfloat16_t.");
    atomic_type_t atomicType = atomic_type_t::ATOMIC_NONE;
    if constexpr (caps::IsFP32<T>()) {
        set_atomic_f32();
    } else if constexpr (caps::IsFP16<T>()) {
        set_atomic_f16();
    } else if constexpr (caps::IsSInt16<T>()) {
        set_atomic_s16();
    } else if constexpr (caps::IsSInt32<T>()) {
        set_atomic_s32();
    } else if constexpr (caps::IsSInt8<T>()) {
        set_atomic_s8();
    } else if constexpr (caps::IsBF16<T>()) {
        set_atomic_bf16();
    }
    set_atomic_add();
}

template <typename TileData, typename GlobalData, bool isQuant>
PTO_INTERNAL void CheckStaticAcc()
{
    static_assert(
        caps::IsSInt32<typename TileData::DType>() || caps::IsFP32<typename TileData::DType>(),
        "The input data type must be restricted to int32_t/float!");
    static_assert(
        (GlobalData::layout == pto::Layout::ND) || (GlobalData::layout == pto::Layout::NZ) ||
            (GlobalData::layout == pto::Layout::NHWC) || (GlobalData::layout == pto::Layout::NCHW) ||
            (GlobalData::layout == pto::Layout::NCDHW),
        "TSTORE(Acc2GM) only support NZ2ND / NZ2NZ / NZ2NHWC / NZ2NCHW / NZ2NCDHW.");
    static_assert(TileData::Cols >= 1 && TileData::Cols <= 4095, "The range of Cols is [1, 4095].");
    static_assert(
        (GlobalData::layout == pto::Layout::ND && TileData::Rows >= 1 && TileData::Rows <= 8192) ||
            ((GlobalData::layout == pto::Layout::NZ || (GlobalData::layout == pto::Layout::NHWC) ||
              (GlobalData::layout == pto::Layout::NCHW) || (GlobalData::layout == pto::Layout::NCDHW)) &&
             TileData::Rows >= 1 && TileData::Rows <= 65535 && TileData::Cols % 16 == 0),
        "When GlobalData is ND format, the range of Rows is [1, 8192]."
        "When GlobalData is NZ/NHWC/NCHW/NCDHW format, the range of Rows is [1, 65535] and Cols"
        "must be an integer multiple of 16.");
    if constexpr (!isQuant) {
        static_assert(
            caps::IsSInt32<typename GlobalData::RawDType>() || caps::IsFP32<typename GlobalData::RawDType>() ||
                caps::IsFP16<typename GlobalData::RawDType>() || caps::IsBF16<typename GlobalData::RawDType>(),
            "The output data type must be restricted to int32_t/float/half/bfloat16_t!");
    } else if constexpr (isQuant) {
        if constexpr (caps::IsFP32<typename TileData::DType>()) {
            static_assert(
                caps::IsInt8<typename GlobalData::RawDType>() || caps::IsBF16<typename GlobalData::RawDType>() ||
                    caps::IsFP16<typename GlobalData::RawDType>() || caps::IsHF8<typename GlobalData::RawDType>() ||
                    caps::IsFP8E4M3<typename GlobalData::RawDType>() || caps::IsFP32<typename GlobalData::RawDType>(),
                "The output data type must be restricted to int8_t/uint8_t/bfloat16_t/half/hifloat8_t/ \
                    float8_e4m3_t/float.");
        } else if constexpr (caps::IsSInt32<typename TileData::DType>()) {
            static_assert(
                caps::IsInt8<typename GlobalData::RawDType>() || caps::IsBF16<typename GlobalData::RawDType>() ||
                    caps::IsFP16<typename GlobalData::RawDType>(),
                "The output data type must be restricted to half/bfloat16_t/int8_t/uint8_t.");
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
        "Data type must be "
        "int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/int64_t/uint64_t/half/bfloat16_t/float/float8_e4m3_t/"
        "float8_e5m2_t/hifloat8_t/float8_e8m0_t/float4_e1m2x2_t/float4_e2m1x2_t!");
    static_assert(
        ((GlobalData::layout == pto::Layout::ND) &&
         (TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox))) ||
            ((GlobalData::layout == pto::Layout::DN) &&
             (!TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox))) ||
            ((GlobalData::layout == pto::Layout::NZ) &&
             (!TileData::isRowMajor && (TileData::SFractal == SLayout::RowMajor))) ||
            (TileData::Rows == 1) || (TileData::Cols == 1),
        "Src and dst layout must be same, only support ND/DN/NZ or the special case of one row/one column!");
    static_assert(
        ((GlobalData::layout == pto::Layout::ND) && (TileData::Cols * sizeof(typename TileData::DType) % 32 == 0)) ||
        ((GlobalData::layout == pto::Layout::DN) && (TileData::Rows * sizeof(typename TileData::DType) % 32 == 0)) ||
        (GlobalData::layout == pto::Layout::NZ) ||
        ((GlobalData::layout == pto::Layout::ND) && (TileData::Rows * sizeof(typename TileData::DType) % 32 == 0) &&
         (TileData::Cols == 1)) ||
        ((GlobalData::layout == pto::Layout::DN) && (TileData::Cols * sizeof(typename TileData::DType) % 32 == 0) &&
         (TileData::Rows == 1)));
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

template <
    typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
    STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TSTORE_IMPL(GlobalData& dst, TileData& src)
{
    static_assert(
        TileData::Loc == pto::TileType::Vec || TileData::Loc == pto::TileType::Acc,
        "Source TileType only suport Vec/Acc!");
    if constexpr (atomicType == AtomicType::AtomicAdd) {
        SetAtomicAdd<typename GlobalData::RawDType>();
    }
    if constexpr (TileData::Loc == pto::TileType::Acc) {
        using L0cT = typename TileData::DType;
        using DstT = typename GlobalData::RawDType;
        CheckStaticAcc<TileData, GlobalData, false>();

        constexpr QuantMode_t quantPre = GetCastPreQuantModeGm<L0cT, DstT>();
        TStoreAcc<GlobalData, TileData, quantPre, ReluPreMode::NoRelu, Phase>(
            dst.data(), src.data(), dst.GetShape(pto::GlobalTensorDim::DIM_0),
            dst.GetShape(pto::GlobalTensorDim::DIM_1), dst.GetShape(pto::GlobalTensorDim::DIM_2),
            dst.GetShape(pto::GlobalTensorDim::DIM_3), dst.GetShape(pto::GlobalTensorDim::DIM_4),
            dst.GetStride(pto::GlobalTensorDim::DIM_0), dst.GetStride(pto::GlobalTensorDim::DIM_1),
            dst.GetStride(pto::GlobalTensorDim::DIM_2), dst.GetStride(pto::GlobalTensorDim::DIM_3),
            dst.GetStride(pto::GlobalTensorDim::DIM_4), src.GetValidRow(), src.GetValidCol());
    } else if constexpr (TileData::Loc == pto::TileType::Vec) {
        CheckStaticVec<TileData, GlobalData>();

        TStore<GlobalData, TileData>(
            dst.data(), src.data(), dst.GetShape(pto::GlobalTensorDim::DIM_0),
            dst.GetShape(pto::GlobalTensorDim::DIM_1), dst.GetShape(pto::GlobalTensorDim::DIM_2),
            dst.GetShape(pto::GlobalTensorDim::DIM_3), dst.GetShape(pto::GlobalTensorDim::DIM_4),
            dst.GetStride(pto::GlobalTensorDim::DIM_0), dst.GetStride(pto::GlobalTensorDim::DIM_1),
            dst.GetStride(pto::GlobalTensorDim::DIM_2), dst.GetStride(pto::GlobalTensorDim::DIM_3),
            dst.GetStride(pto::GlobalTensorDim::DIM_4), src.GetValidRow(), src.GetValidCol());
    }
    if constexpr (atomicType == AtomicType::AtomicAdd) {
        set_atomic_none();
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
    if constexpr (atomicType == AtomicType::AtomicAdd) {
        SetAtomicAdd<DstT>();
    }
    constexpr QuantMode_t quantPre = GetCastPreQuantModeGm<L0cT, DstT>();
    TStoreAcc<GlobalData, TileData, quantPre, reluPreMode, Phase>(
        dst.data(), src.data(), dst.GetShape(pto::GlobalTensorDim::DIM_0), dst.GetShape(pto::GlobalTensorDim::DIM_1),
        dst.GetShape(pto::GlobalTensorDim::DIM_2), dst.GetShape(pto::GlobalTensorDim::DIM_3),
        dst.GetShape(pto::GlobalTensorDim::DIM_4), dst.GetStride(pto::GlobalTensorDim::DIM_0),
        dst.GetStride(pto::GlobalTensorDim::DIM_1), dst.GetStride(pto::GlobalTensorDim::DIM_2),
        dst.GetStride(pto::GlobalTensorDim::DIM_3), dst.GetStride(pto::GlobalTensorDim::DIM_4), src.GetValidRow(),
        src.GetValidCol());
    if constexpr (atomicType == AtomicType::AtomicAdd) {
        set_atomic_none();
    }
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
    if constexpr (atomicType == AtomicType::AtomicAdd) {
        SetAtomicAdd<DstT>();
    }
    constexpr QuantMode_t quantPre = GetScalarPreQuantModeGm<L0cT, DstT>();
    set_quant_pre(preQuantScalar);
    TStoreAcc<GlobalData, TileData, quantPre, reluPreMode, Phase>(
        dst.data(), src.data(), dst.GetShape(pto::GlobalTensorDim::DIM_0), dst.GetShape(pto::GlobalTensorDim::DIM_1),
        dst.GetShape(pto::GlobalTensorDim::DIM_2), dst.GetShape(pto::GlobalTensorDim::DIM_3),
        dst.GetShape(pto::GlobalTensorDim::DIM_4), dst.GetStride(pto::GlobalTensorDim::DIM_0),
        dst.GetStride(pto::GlobalTensorDim::DIM_1), dst.GetStride(pto::GlobalTensorDim::DIM_2),
        dst.GetStride(pto::GlobalTensorDim::DIM_3), dst.GetStride(pto::GlobalTensorDim::DIM_4), src.GetValidRow(),
        src.GetValidCol());
    if constexpr (AtomicType::AtomicAdd == atomicType) {
        set_atomic_none();
    }
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
    if constexpr (AtomicType::AtomicAdd == atomicType) {
        SetAtomicAdd<DstT>();
    }
    constexpr QuantMode_t quantPre = GetVectorPreQuantModeGm<L0cT, DstT>();
    TStoreAccFp<GlobalData, TileData, FpTileData, quantPre, reluPreMode>(
        dst.data(), src.data(), fp.data(), dst.GetShape(pto::GlobalTensorDim::DIM_0),
        dst.GetShape(pto::GlobalTensorDim::DIM_1), dst.GetShape(pto::GlobalTensorDim::DIM_2),
        dst.GetShape(pto::GlobalTensorDim::DIM_3), dst.GetShape(pto::GlobalTensorDim::DIM_4),
        dst.GetStride(pto::GlobalTensorDim::DIM_0), dst.GetStride(pto::GlobalTensorDim::DIM_1),
        dst.GetStride(pto::GlobalTensorDim::DIM_2), dst.GetStride(pto::GlobalTensorDim::DIM_3),
        dst.GetStride(pto::GlobalTensorDim::DIM_4), src.GetValidRow(), src.GetValidCol());
    if constexpr (atomicType == AtomicType::AtomicAdd) {
        set_atomic_none();
    }
}
} // namespace pto
#endif
