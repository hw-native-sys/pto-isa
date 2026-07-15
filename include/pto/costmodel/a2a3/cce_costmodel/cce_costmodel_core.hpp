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

#include <pto/costmodel/arch_config.hpp>
#include <pto/costmodel/trace.hpp>

namespace pto {

enum QuantMode_t {
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
constexpr int VA0 = 0;
constexpr int VA1 = 1;
constexpr int VA2 = 2;
constexpr int VA3 = 3;
constexpr int VA4 = 4;
constexpr int VA5 = 5;
constexpr int VA6 = 6;
constexpr int VA7 = 7;

inline int sbitset0(int val, int bit) { return val & ~(1 << bit); }
inline int sbitset1(int val, int bit) { return val | (1 << bit); }

inline int get_ctrl(...) { return 0; }
inline int get_vms4_sr(...) { return 0; }
inline int get_imm(...) { return 0; }

inline const ::pto::mocker::evaluator::ArchConfig& CurrentArch()
{
    return ::pto::mocker::evaluator::GetDefaultArchConfig();
}

inline uint64_t EstimateBandwidthCycles(uint64_t bytes, ::pto::mocker::evaluator::PipeKey key)
{
    namespace ev = ::pto::mocker::evaluator;
    // Apply Hill-bandwidth env (PTO_BW_MODE) once; default is the legacy flat model
    // (K=0 => hill=peak, totals=0 => no cap), so behaviour is unchanged unless overridden.
    ev::ApplyHillBandwidthFromEnv();
    const double bandwidth = ev::gHillBandwidth.BwEff(key, bytes, ev::gActiveCoreCount);
    if (bandwidth <= 0.0) {
        return 0;
    }
    return static_cast<uint64_t>(
        (static_cast<long double>(bytes) / ev::kBytesPerGb) / static_cast<long double>(bandwidth) *
        CurrentArch().frequency_hz);
}

inline void FlushPipeTail(::pto::mocker::evaluator::PipeKey pipe) { ::pto::mocker::FlushPendingTail(pipe); }

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

inline uint64_t EstimateLinearCycles(
    ::pto::mocker::evaluator::PipeKey pipe, uint64_t repeat, uint64_t head = 6, uint64_t slope = 2, uint64_t tail = 0)
{
    uint64_t cycles = slope * repeat;
    if (pipe == ::pto::mocker::evaluator::PipeKey::VECTOR) {
        // Vector stream model: cycles = lat + (repeat-1)*thru.
        //  - thru (= slope): incremental time per extra repeat; small because it bakes in the overlap
        //    with the predecessor. So the per-instruction output is summable (the inter-instruction
        //    overlap is already in the cost, not the timeline).
        //  - lat (= head+tail): the instruction's startup latency. Paid at stream start (pipe queue
        //    empty == first vec op since the last sync/core boundary); back-to-back vec ops in the
        //    same stream overlap their lat away and pay only thru. (head/tail split is unmeasured:
        //    single-op data fixes only head+tail+slope*repeat, see cce_costmodel_vector_compute.hpp.)
        // No deferred tail for vec (folded into the stream-start charge) => no tail-loss/attribution
        // problem at stream end.
        if (::pto::mocker::IsPipeQueueEmpty(pipe)) {
            cycles += head + tail;
        }
        ::pto::mocker::SetLastCceTail(pipe, 0);
    } else {
        if (::pto::mocker::IsPipeQueueEmpty(pipe)) {
            cycles += head;
        }
        ::pto::mocker::SetLastCceTail(pipe, tail);
    }
    return cycles;
}

inline uint64_t EstimateLinearCycles(uint64_t repeat, uint64_t head = 6, uint64_t slope = 2, uint64_t tail = 0)
{
    return EstimateLinearCycles(::pto::mocker::evaluator::PipeKey::VECTOR, repeat, head, slope, tail);
}

inline uint64_t EstimateConstCycles(uint64_t cycles = 1) { return cycles; }

// Vector ALU ops (vadd/vsub/vmul/vdiv/vmax/vmin/vand/vor) pay a one-time dispatch
// floor when the effective byte count is not aligned to one vector repeat (256B;
// fp32 <=> cols % 64 != 0): the hardware enters count-mask dispatch (~15-19 cyc,
// independent of op/repeat). The floor is orthogonal to the op's slope/head/tail
// and is added only while the mocker has recorded a count-mode mask (see
// set_vector_mask / set_mask_count in cce_costmodel_sync.hpp). Single global
// constant: per-op measured floor_needed medians 12-18 cyc (std < 3) across the
// binary ALU set, so 16 fits all within tolerance.
inline constexpr uint64_t kCountModeFloorCycles = 16;
inline uint64_t EstimateCountModeFloor() { return ::pto::mocker::IsVectorCountMode() ? kCountModeFloorCycles : 0; }

inline uint64_t CeilDiv(uint64_t x, uint64_t y)
{
    if (y == 0)
        return 0; // 或返回 UINT64_MAX，根据业务逻辑决定
    return (x + y - 1) / y;
}

inline uint64_t ExtractBits(uint64_t value, uint32_t shift, uint64_t mask) { return (value >> shift) & mask; }

// Common latency model for vconv_* (fp32->fp16 cast).
// 910B3 标定 (fp32, TCVT, dav-2201): measured = 0.90·repeat + 27 (R²=0.99, rep 1-128).
// Was the uncalibrated placeholder (slope=2, head=14, tail=18) -> ~1.9x over at rep128
// (mock 273 vs real 141). slope 2->1; fixed head+tail kept ~27 (split unmeasured).
inline uint64_t EstimateVconvCycles(uint64_t repeat)
{
    constexpr uint64_t kConvHeadCycles = 14;
    constexpr uint64_t kConvSlope = 1;
    constexpr uint64_t kConvTailCycles = 13;
    return EstimateLinearCycles(repeat, kConvHeadCycles, kConvSlope, kConvTailCycles);
}

inline void copy_cbuf_to_gm(...) {}
inline void copy_matrix_cc_to_gm(...) {}
inline void copy_ubuf_to_gm_align_b16(...) {}
inline void copy_ubuf_to_gm_align_b32(...) {}
inline void copy_ubuf_to_gm_align_b8(...) {}
inline void scatter_vnchwconv_b16(...) {}
inline void scatter_vnchwconv_b32(...) {}
inline void scatter_vnchwconv_b8(...) {}
