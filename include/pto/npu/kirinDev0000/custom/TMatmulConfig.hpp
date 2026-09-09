/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TMATMULCONFIG_HPP
#define TMATMULCONFIG_HPP

#include <cstdint>
#include <type_traits>

namespace pto {

inline namespace TMatmulInternal {
constexpr const int MMAD_MAX_SUPPORT_LENGTH = 4095;
constexpr const int TF32_MODE_BIT = 46;
constexpr const int TF32_TRANS_MODE_BIT = 47;
} // namespace TMatmulInternal

enum class MatmulPreQuant : uint8_t {
    NoQuant = 0,
    ReqS8Vector = 2,
    ReqS8Scalar = 3,
    DeqF16Vector = 4,
    DeqF16Scalar = 5,
    DeqS16Vector = 6,
    DeqS16Scalar = 7,
    DeqS32Vector = 8,
    DeqS32Scalar = 9,
    ReqS4Vector = 10,
    ReqS4Scalar = 11,
    Qs322Fp8E4m3IntVector = 12,
    Qs322Fp8E4m3IntScalar = 13,
};

enum class MatmulPostQuant : uint8_t {
    NoQuant = 0,
    Shift2S16Scalar = 1,
    Shift2S16Vector = 2,
    Shift2S8Scalar = 3,
    Shift2S8Vector = 4,
    Shift2S4Scalar = 5,
    Shift2S4Vector = 6,
    Shift2S32Scalar = 7,
    Shift2S32Vector = 8,
};

enum class MatmulPreRelu : uint8_t {
    NoRelu = 0,
    NormalRelu = 1,
    ScalarRelu = 2,
    VectorRelu = 3,
    LutActivation = 4,
};

enum class MatmulEltwiseOp : uint8_t {
    NoEltwise = 0,
    Add = 1,
    Sub = 2,
    Mul = 3,
    Max = 4,
};

enum class MatmulLsbMask : uint8_t {
    Disable = 0,
    Mask1Lsb = 1,
    Mask2Lsbs = 2,
    Mask3Lsbs = 3,
    Mask4Lsbs = 4,
};

enum class MatmulInstrId : uint8_t {
    Id0 = 0,
    Id1 = 1,
    Id2 = 2,
    Id3 = 3,
};

struct MatmulConfig {
    bool initCtrl = true;
    bool broadcastEn = false;
    MatmulPreQuant preQuant = MatmulPreQuant::NoQuant;
    MatmulPreRelu preRelu = MatmulPreRelu::NoRelu;
    MatmulPostQuant postQuant = MatmulPostQuant::NoQuant;
    bool clipReluEn = false;
    MatmulEltwiseOp eltwiseOp = MatmulEltwiseOp::NoEltwise;
    bool antiqEn = false;
    bool mDBroadcaEn = false;
    MatmulLsbMask lsbMask = MatmulLsbMask::Disable;
    bool gemvCtrl = false;
    bool dependEn = false;
    MatmulInstrId instrId = MatmulInstrId::Id0;
    bool bandwidthCtrl = false;
    bool nbrcBiasCtrl = false;
    uint16_t sizeN = 0;

    constexpr uint64_t Build() const
    {
        return (static_cast<uint64_t>(initCtrl) & 0x1) | ((static_cast<uint64_t>(broadcastEn) & 0x1) << 1) |
               ((static_cast<uint64_t>(preQuant) & 0x1f) << 2) | ((static_cast<uint64_t>(preRelu) & 0x7) << 7) |
               ((static_cast<uint64_t>(postQuant) & 0x1f) << 10) | ((static_cast<uint64_t>(clipReluEn) & 0x1) << 18) |
               ((static_cast<uint64_t>(eltwiseOp) & 0x7) << 20) | ((static_cast<uint64_t>(antiqEn) & 0x1) << 23) |
               ((static_cast<uint64_t>(mDBroadcaEn) & 0x1) << 24) | ((static_cast<uint64_t>(lsbMask) & 0x7) << 28) |
               ((static_cast<uint64_t>(gemvCtrl) & 0x1) << 31) | ((static_cast<uint64_t>(dependEn) & 0x1) << 36) |
               ((static_cast<uint64_t>(instrId) & 0x3) << 37) | ((static_cast<uint64_t>(bandwidthCtrl) & 0x1) << 39) |
               ((static_cast<uint64_t>(nbrcBiasCtrl) & 0x1) << 40) | ((static_cast<uint64_t>(sizeN) & 0xffff) << 48);
    }
};

enum class MatmulReluMode : uint8_t {
    NoRelu = 0,
    NormalRelu = 1,
    ScalarRelu = 2,
    VectorRelu = 3,
};

enum class MatmulQuantMode : uint8_t {
    ScalarQuant = 0,
    VectorQuant = 1,
};

struct MatmulMacroConfig {
    uint64_t preQuantTileAddr = 0;
    uint64_t vectorReluTileAddr = 0;
    float reluScalar = 0.0f;
    float clipReluVal = 0.0f;
    bool gemvCtrl = false;
    MatmulReluMode reluMode = MatmulReluMode::NoRelu;
    MatmulQuantMode quantMode = MatmulQuantMode::VectorQuant;
};

template <typename L0cType, typename OutType>
AICORE constexpr MatmulPreQuant InferPreQuant()
{
    if constexpr (std::is_same_v<OutType, int32_t>) {
        return MatmulPreQuant::NoQuant;
    } else if constexpr (std::is_same_v<OutType, half>) {
        return MatmulPreQuant::DeqF16Vector;
    } else if constexpr (std::is_same_v<OutType, int8_t>) {
        return MatmulPreQuant::ReqS8Vector;
    } else if constexpr (std::is_same_v<OutType, int16_t>) {
        return MatmulPreQuant::DeqS16Vector;
    } else {
        return MatmulPreQuant::NoQuant;
    }
}

template <typename T>
__tf__ PTO_INTERNAL void SetMFpcForFp16()
{
    if constexpr (std::is_same_v<T, half>) {
        uint64_t mFpc = 0;
        mFpc |= ((uint64_t)42) << 40;
        set_m_fpc(mFpc);
    }
}

__tf__ PTO_INTERNAL void SetMReluAlpha(float reluScalar)
{
    uint32_t floatBits = __builtin_bit_cast(uint32_t, reluScalar);
    uint32_t sign = (floatBits >> 31) & 0x1;
    uint32_t exp = (floatBits >> 23) & 0xFF;
    uint32_t mantissa = (floatBits >> 13) & 0x3FF;
    uint32_t m2 = (sign << 18) | (exp << 10) | mantissa;
    uint64_t mReluAlpha = static_cast<uint64_t>(m2) << 13;
    set_m_relu_alpha(mReluAlpha);
}

template <typename OutType>
__tf__ PTO_INTERNAL void SetMQuantPre(float clipReluVal)
{
    uint64_t mQuantPre = 0;
    if constexpr (std::is_same_v<OutType, half> || std::is_same_v<OutType, int16_t>) {
        mQuantPre |= (((uint64_t)(42 - 58 + 127) & 0xff) << 10) << 13;
    }
    if (clipReluVal != 0.0f) {
        uint64_t clipVal = 0;
        if constexpr (std::is_same_v<OutType, half>) {
            half h = static_cast<half>(clipReluVal);
            clipVal = *reinterpret_cast<uint16_t*>(&h);
        } else if constexpr (std::is_same_v<OutType, int8_t>) {
            int16_t v = static_cast<int16_t>(clipReluVal);
            clipVal = static_cast<uint16_t>(v) & 0xFFFF;
        } else if constexpr (std::is_same_v<OutType, int16_t>) {
            int16_t v = static_cast<int16_t>(clipReluVal);
            clipVal = static_cast<uint16_t>(v) & 0xFFFF;
        }
        mQuantPre |= (clipVal & 0xFFFF) << 48;
    }
    set_m_quant_pre(mQuantPre);
}

template <typename TileRes>
__tf__ PTO_INTERNAL void ZeroInitCTile(typename TileRes::TileDType __out__ cData)
{
    using OutType = typename TileRes::DType;
    constexpr uint32_t BLOCK_BYTE = 32;
    __cbuf__ void* c = (__cbuf__ void*)__cce_get_tile_ptr(cData);
    constexpr uint32_t totalBlocks = (TileRes::Rows * TileRes::Cols * sizeof(OutType)) / BLOCK_BYTE;
    constexpr int64_t repeatConfig = ((static_cast<int64_t>(0) & 0x7FFF) << 32) |
                                     ((static_cast<int64_t>(totalBlocks) & 0x7FFF) << 16) |
                                     (static_cast<int64_t>(1) & 0x7FFF);
    if constexpr (std::is_same_v<OutType, half>) {
        pto_create_cbuf_matrix((__cbuf__ half*)c, repeatConfig, (half)0);
    } else {
        pto_create_cbuf_matrix((__cbuf__ uint16_t*)c, repeatConfig, (uint16_t)0);
    }
}

template <typename InputType, MatmulPreQuant preQuant>
__tf__ PTO_INTERNAL void SetMFpcFromConfig(MatmulPreRelu preRelu, const MatmulMacroConfig& cfg)
{
    uint64_t mFpc = 0;
    if constexpr (preQuant != MatmulPreQuant::NoQuant) {
        mFpc |= ((cfg.preQuantTileAddr >> 7) & 0xFF) << 8;
    }
    if (preRelu == MatmulPreRelu::VectorRelu) {
        constexpr uint64_t memBlock1Base = 0x800;
        mFpc |= (((cfg.vectorReluTileAddr - memBlock1Base) >> 6) & 0xFF);
    }
    if constexpr (std::is_same_v<InputType, half>) {
        mFpc |= ((uint64_t)42) << 40;
    }
    set_m_fpc(mFpc);
}

} // namespace pto
#endif
