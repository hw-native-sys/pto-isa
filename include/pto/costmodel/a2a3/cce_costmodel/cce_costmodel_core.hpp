/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#pragma once

#include <cstddef>
#include <cstdint>

#include <pto/costmodel/arch_config.hpp>
#include <pto/costmodel/trace.hpp>

namespace pto {

enum QuantMode_t
{
    NoQuant = 0,
    F322F16 = 1,
    F322BF16 = 16,
    DEQF16 = 5,
    VDEQF16 = 4,
    QF322B8_PRE = 24,
    QF322HIF8_PRE = 25,
    QF322FP8_PRE = 26,
    QF322F32_PRE = 27,
    QF322F16_PRE = 32,
    QF322BF16_PRE = 34,
    QS322BF16_PRE = 35,
    VQF322B8_PRE = 23,
    VQF322HIF8_PRE = 28,
    VQF322F16_PRE = 33,
    VQF322BF16_PRE = 36,
    VQF322FP8_PRE = 37,
    VQF322F32_PRE = 38,
    REQ8 = 3,
    VREQ8 = 2,
    VQS322BF16_PRE = 39,
    VSHIFTS322S16 = 12,
    SHIFTS322S16 = 13,
};

} // namespace pto

constexpr int DSB_UB = 0;
constexpr int ONLY_VALUE = 0;
constexpr int PIPE_FIX = 0;
constexpr int VA0 = 0;
constexpr int VA1 = 1;
constexpr int VA2 = 2;
constexpr int VA3 = 3;
constexpr int VA4 = 4;
constexpr int VA5 = 5;
constexpr int VA6 = 6;
constexpr int VA7 = 7;

inline int sbitset0(int val, int bit)
{
    return val & ~(1 << bit);
}
inline int sbitset1(int val, int bit)
{
    return val | (1 << bit);
}

inline int get_ctrl(...)
{
    return 0;
}
inline int get_vms4_sr(...)
{
    return 0;
}
inline int get_imm(...)
{
    return 0;
}

inline const ::pto::mocker::evaluator::ArchConfig &CurrentArch()
{
    return ::pto::mocker::evaluator::GetDefaultArchConfig();
}

inline uint64_t EstimateBandwidthCycles(uint64_t bytes, ::pto::mocker::evaluator::PipeKey key)
{
    const auto &arch = CurrentArch();
    const double bandwidth = arch.bandwidth[key];
    if (bandwidth <= 0.0) {
        return 0;
    }
    return static_cast<uint64_t>((static_cast<long double>(bytes) / ::pto::mocker::evaluator::kBytesPerGb) /
                                 static_cast<long double>(bandwidth) * arch.frequency_hz);
}

inline void FlushPipeTail(::pto::mocker::evaluator::PipeKey pipe)
{
    ::pto::mocker::FlushPendingTail(pipe);
}

inline void FlushTailsForPipe(auto pipe)
{
    switch (pipe) {
        case PIPE_V:
            FlushPipeTail(::pto::mocker::evaluator::PipeKey::VECTOR);
            break;
        case PIPE_M:
            FlushPipeTail(::pto::mocker::evaluator::PipeKey::CUBE);
            break;
        case PIPE_MTE1:
            FlushPipeTail(::pto::mocker::evaluator::PipeKey::L1_TO_L0A);
            FlushPipeTail(::pto::mocker::evaluator::PipeKey::L1_TO_L0B);
            FlushPipeTail(::pto::mocker::evaluator::PipeKey::L1_TO_BT);
            FlushPipeTail(::pto::mocker::evaluator::PipeKey::L1_TO_FB);
            FlushPipeTail(::pto::mocker::evaluator::PipeKey::L1_FILL);
            break;
        case PIPE_MTE2:
            FlushPipeTail(::pto::mocker::evaluator::PipeKey::GM_TO_UB);
            FlushPipeTail(::pto::mocker::evaluator::PipeKey::GM_TO_L1);
            break;
        case PIPE_MTE3:
            FlushPipeTail(::pto::mocker::evaluator::PipeKey::UB_TO_GM);
            FlushPipeTail(::pto::mocker::evaluator::PipeKey::L1_TO_GM);
            FlushPipeTail(::pto::mocker::evaluator::PipeKey::L0C_TO_GM);
            FlushPipeTail(::pto::mocker::evaluator::PipeKey::L0C_TO_L1);
            break;
        case PIPE_ALL:
            ::pto::mocker::FlushAllPendingTails();
            break;
        default:
            break;
    }
}

inline uint64_t EstimateLinearCycles(::pto::mocker::evaluator::PipeKey pipe, uint64_t repeat, uint64_t head = 6,
                                     uint64_t slope = 2, uint64_t tail = 0)
{
    uint64_t cycles = slope * repeat;
    if (::pto::mocker::IsPipeQueueEmpty(pipe)) {
        cycles += head;
    }
    ::pto::mocker::SetLastCceTail(pipe, tail);
    return cycles;
}

inline uint64_t EstimateLinearCycles(uint64_t repeat, uint64_t head = 6, uint64_t slope = 2, uint64_t tail = 0)
{
    return EstimateLinearCycles(::pto::mocker::evaluator::PipeKey::VECTOR, repeat, head, slope, tail);
}

inline uint64_t EstimateConstCycles(uint64_t cycles = 1)
{
    return cycles;
}

inline uint64_t CeilDiv(uint64_t x, uint64_t y)
{
    if (y == 0)
        return 0; // 或返回 UINT64_MAX，根据业务逻辑决定
    return (x + y - 1) / y;
}

inline uint64_t ExtractBits(uint64_t value, uint32_t shift, uint64_t mask)
{
    return (value >> shift) & mask;
}

// Temporary common latency model for vconv_*; the detailed behavior is not fully understood yet.
inline uint64_t _EstimateVconvCycles(uint64_t repeat)
{
    return EstimateLinearCycles(repeat, 14, 2, 18);
}

inline void copy_cbuf_to_gm(...)
{}
inline void copy_matrix_cc_to_gm(...)
{}
inline void copy_ubuf_to_gm_align_b16(...)
{}
inline void copy_ubuf_to_gm_align_b32(...)
{}
inline void copy_ubuf_to_gm_align_b8(...)
{}
inline void scatter_vnchwconv_b16(...)
{}
inline void scatter_vnchwconv_b32(...)
{}
inline void scatter_vnchwconv_b8(...)
{}
