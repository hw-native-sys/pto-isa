/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_FIXPIPE_HPP
#define PTO_FIXPIPE_HPP
#include <type_traits>

#include <pto/common/arch_macro.hpp>
#include <pto/common/type.hpp>

#ifdef __CPU_SIM
#include <pto/cpu/common.hpp>
#endif

namespace pto {

typedef enum { NOCLIP_RELU = 0, CLIP_RELU = 1 } ClipReluMode_t;

enum class LayoutMode_t : uint8_t {
    NZ2NZ = 0,
    NZ2ND = 1,
    NZ2DN = 2,
};

/*
//     NoQuant,          // 不使能量化功能
//     F322F16,          // float转成成half
//     F322BF16,         // float转成bfloat16_t
//     DEQF16,           // int32_t量化成half, scalar量化
//     VDEQF16,          // int32_t量化成half, tensor量化
//     QF322B8_PRE,      // float量化成int8_t/uint8_t, scalar量化
//     VQF322B8_PRE,     // float量化成int8_t/uint8_t, tensor量化
//     QF162B8_PRE,      // half量化成int8_t/uint8_t, scalar量化
//     VQF162B8_PRE,     // half量化成int8_t/uint8_t, tensor量化
//     REQ8,             // int32_t量化成int8_t/uint8_t, scalar量化
//     VREQ8,            // int32_t量化成int8_t/uint8_t, tensor量化
//     QF162S4_PRE,      // half量化成int4, scalar量化
//     VQF162S4_PRE,     // half量化成int4, tensor量化
//     REQ4,             // int32_t量化成int4, scalar量化
//     VREQ4,            // int32_t量化成int4, tensor量化
//     DEQS16,           // int32_t量化成int16_t, scalar量化
//     VDEQS16,          // int32_t量化成int16_t, tensor量化
//     QF162S16_PRE,     // half量化成int16_t, scalar量化
//     VQF162S16_PRE,    // half量化成int16_t, tensor量化
*/
// A6R2 MaxTrack Mode (Acc->UB). groupmax_addr / rowmax_addr are the UB
// bases from TPipe C2V_GROUPMAX_BUF / C2V_ROWMAX_BUF, not part of this config.
// C2V outputs: groupmax_en → vec+gMax; groupmax_en&&rowmax_en → vec+gMax+rMax;
// otherwise only vec (same as A5/A6). Defaults keep existing call sites unchanged.
template <
    bool absEn = false, bool groupmaxEn = false, bool rowmaxEn = false, bool rowmaxCompareEn = false,
    bool rowmaxOutput = false, uint8_t groupNum = 8>
struct MaxTrack {
    static constexpr bool abs_en = absEn;
    static constexpr bool groupmax_en = groupmaxEn;
    static constexpr bool rowmax_en = rowmaxEn;
    static constexpr bool rowmax_compare_en = rowmaxCompareEn;
    static constexpr bool rowmax_output = rowmaxOutput;
    static constexpr uint8_t group_num = groupNum;
};

// A6R2 MaxTrack Acc→UB only supports float / half / bfloat16_t sideband+result dtypes.
template <typename T>
inline constexpr bool IsMaxTrackDType =
    std::is_same_v<T, float> || std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>;

template <typename MaxTrackCfg>
inline constexpr bool MaxTrackEmitsGMax()
{
    return MaxTrackCfg::groupmax_en;
}

template <typename MaxTrackCfg>
inline constexpr bool MaxTrackEmitsRMax()
{
    return MaxTrackCfg::groupmax_en && MaxTrackCfg::rowmax_en;
}

template <
    LayoutMode_t layoutMode = LayoutMode_t::NZ2ND, QuantMode_t quantMode = QuantMode_t::NoQuant,
    ReluPreMode reluMode = ReluPreMode::NoRelu, STPhase phase = STPhase::Unspecified, uint8_t subBlockId = 0,
    AtomicType atomicT = AtomicType::AtomicNone, ClipReluMode_t clipReluMode = ClipReluMode_t::NOCLIP_RELU,
    bool isChannelSplit = false, typename MaxTrackCfg = MaxTrack<>>
struct FixpipeParams {
    FixpipeParams() = default;
    static constexpr LayoutMode_t LayoutMode = layoutMode;
    static constexpr QuantMode_t QuantPre = quantMode;
    static constexpr ReluPreMode ReluMode = reluMode;
    static constexpr STPhase Phase = phase;
    static constexpr uint8_t SubBlockId = subBlockId;
    static constexpr AtomicType AtomicT = atomicT;
    static constexpr ClipReluMode_t ClipReluMode = clipReluMode;
    static constexpr bool IsChannelSplit = isChannelSplit;
    using MaxTrackT = MaxTrackCfg;
    static constexpr MaxTrackCfg maxTrack{};
    static constexpr bool abs_en = MaxTrackCfg::abs_en;
    static constexpr bool groupmax_en = MaxTrackCfg::groupmax_en;
    static constexpr bool rowmax_en = MaxTrackCfg::rowmax_en;
    static constexpr bool rowmax_compare_en = MaxTrackCfg::rowmax_compare_en;
    static constexpr bool rowmax_output = MaxTrackCfg::rowmax_output;
    static constexpr uint8_t group_num = MaxTrackCfg::group_num;
};

template <QuantMode_t quantPre, typename SrcType>
struct FixpipeConsDType {
    static constexpr bool isHalf = quantPre == QuantMode_t::F322F16 || quantPre == QuantMode_t::QF322F16_PRE ||
                                   quantPre == QuantMode_t::DEQF16 || quantPre == QuantMode_t::VDEQF16;
    static constexpr bool isBfloat16 = quantPre == QuantMode_t::F322BF16 || quantPre == QuantMode_t::QF322BF16_PRE ||
                                       quantPre == QuantMode_t::QS322BF16_PRE ||
                                       quantPre == QuantMode_t::VQS322BF16_PRE;
    static constexpr bool isInt8 = quantPre == QuantMode_t::QF322B8_PRE || quantPre == QuantMode_t::REQ8 ||
                                   quantPre == QuantMode_t::VQF322B8_PRE || quantPre == QuantMode_t::VREQ8;

    using type = std::conditional_t<
        isHalf, half, std::conditional_t<isBfloat16, bfloat16_t, std::conditional_t<isInt8, int8_t, SrcType>>>;
};

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6) || defined(PTO_NPU_ARCH_A6R2)
template <typename SrcType>
struct FixpipeConsDType<QuantMode_t::QF322HIF8_PRE, SrcType> {
    using type = hifloat8_t;
};

template <typename SrcType>
struct FixpipeConsDType<QuantMode_t::QF322FP8_PRE, SrcType> {
    using type = float8_e4m3_t;
};
#endif

template <QuantMode_t quantPre, typename SrcType>
using FixpipeConsDType_t = typename FixpipeConsDType<quantPre, SrcType>::type;

} // end namespace pto

#endif
