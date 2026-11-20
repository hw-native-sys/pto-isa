/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef PTO_MOCKER_A2A3_HYBRID_BASIC_ELEMENTWISE_H_
#define PTO_MOCKER_A2A3_HYBRID_BASIC_ELEMENTWISE_H_

#include <cstddef>
#include <cstdint>
#include <string_view>

#include <pto/costmodel/a2a3/hybrid_costmodel/context.hpp>

namespace pto::mocker::hybrid {

inline CycleEstimate EstimateScalarLoopShift(const PtoCycleContext& ctx)
{
    constexpr uint64_t kLeftOrUint8CyclesPerElement = 39;
    constexpr uint64_t kRightCyclesPerElement = 40;
    constexpr uint64_t kRowControlCycles = 10;
    if ((ctx.opcode != "TSHL" && ctx.opcode != "TSHR") || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        (ctx.src_dtype != "int8" && ctx.src_dtype != "uint8" && ctx.src_dtype != "int16" && ctx.src_dtype != "int32")) {
        return {};
    }
    if (CountCceCalls(*ctx.trace, "vshl") != 0 || CountCceCalls(*ctx.trace, "vshr") != 0) {
        // A future vector lowering must use the native CCE profile instead
        // of this A2/A3 scalar-loop model.
        return {};
    }

    // Current A2/A3 TSHL/TSHR lower to a nested scalar loop.  The loop body
    // occupies 39/40 scalar-pipe cycles per element respectively, plus ten
    // cycles of row-loop control.  These are scalar active-time costs; the
    // native VSHL/VSHR completion profiles describe a different
    // lowering and are deliberately not reused here.
    const uint64_t elements = static_cast<uint64_t>(ctx.rows) * static_cast<uint64_t>(ctx.cols);
    const uint64_t cyclesPerElement =
        (ctx.opcode == "TSHL" || ctx.src_dtype == "uint8") ? kLeftOrUint8CyclesPerElement : kRightCyclesPerElement;
    return {
        cyclesPerElement * elements + kRowControlCycles * static_cast<uint64_t>(ctx.rows),
        ModelKind::PTO_COMPOSITE_ANALYTIC};
}

inline CycleEstimate EstimateIntegerBinaryTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kInt16ElementsPerRepeat = 128;
    constexpr uint64_t kInt32ElementsPerRepeat = 64;
    constexpr uint64_t kRepeatCycles = 2;
    constexpr uint64_t kMulPartialEndpointCycles = 8;
    constexpr uint64_t kAddSubPartialEndpointCycles = 7;
    const bool isAddSub = ctx.opcode == "TADD" || ctx.opcode == "TSUB";
    const bool isMul = ctx.opcode == "TMUL";
    if ((!isAddSub && !isMul) || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        (ctx.src_dtype != "int16" && ctx.src_dtype != "int32")) {
        return {};
    }
    const std::string_view cceName = isMul ? "vmul" : (ctx.opcode == "TADD" ? "vadd" : "vsub");
    if (CountCceCalls(*ctx.trace, cceName) == 0) {
        return {};
    }
    const uint64_t elementsPerRepeat = ctx.src_dtype == "int16" ? kInt16ElementsPerRepeat : kInt32ElementsPerRepeat;
    const uint64_t repeatsPerRow = (static_cast<uint64_t>(ctx.cols) + elementsPerRepeat - 1) / elementsPerRepeat;
    const bool hasPartialRepeat = static_cast<uint64_t>(ctx.cols) % elementsPerRepeat != 0;

    // Native integer VADD/VSUB/VMUL all issue at two cycles per repeat. The
    // benchmark's count-mask path retains one public-call observer endpoint;
    // aligned full-repeat streams have no such endpoint in aiv_vec_time.
    const uint64_t endpoint = hasPartialRepeat ? (isMul ? kMulPartialEndpointCycles : kAddSubPartialEndpointCycles) : 0;
    return {
        static_cast<uint64_t>(ctx.rows) * kRepeatCycles * repeatsPerRow + endpoint,
        ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

struct SerializedLeafProfile {
    std::string_view opcode;
    std::string_view leaf;
    uint64_t integer_startup;
    uint64_t default_startup;
    bool floating_only = false;
};

inline bool IsSerializedIntegerType(std::string_view dtype) { return dtype == "int16" || dtype == "int32"; }

inline bool IsSerializedFloatingType(std::string_view dtype) { return dtype == "fp16" || dtype == "fp32"; }

inline bool TryResolveSerializedLeaf(const PtoCycleContext& ctx, std::string_view& leaf, uint64_t& startup)
{
    constexpr uint64_t kAddSubStartupCycles = 6;
    constexpr uint64_t kIntegerMinMaxStartupCycles = 5;
    constexpr uint64_t kFpMinMaxStartupCycles = 4;
    constexpr uint64_t kMulStartupCycles = 7;
    constexpr uint64_t kIntegerAddScalarStartupCycles = 5;
    constexpr uint64_t kFpAddScalarStartupCycles = 6;
    constexpr uint64_t kIntegerMulScalarStartupCycles = 6;
    constexpr uint64_t kFpMulScalarStartupCycles = 7;
    constexpr uint64_t kLeakyReluStartupCycles = 7;
    constexpr uint64_t kShiftStartupCycles = 6;
    constexpr SerializedLeafProfile profiles[] = {
        {"TADD", "vadd", kAddSubStartupCycles, kAddSubStartupCycles, true},
        {"TPARTADD", "vadd", kAddSubStartupCycles, kAddSubStartupCycles, true},
        {"TSUB", "vsub", kAddSubStartupCycles, kAddSubStartupCycles, true},
        {"TPARTSUB", "vsub", kAddSubStartupCycles, kAddSubStartupCycles, true},
        {"TMAX", "vmax", kIntegerMinMaxStartupCycles, kFpMinMaxStartupCycles},
        {"TPARTMAX", "vmax", kIntegerMinMaxStartupCycles, kFpMinMaxStartupCycles},
        {"TMIN", "vmin", kIntegerMinMaxStartupCycles, kFpMinMaxStartupCycles},
        {"TPARTMIN", "vmin", kIntegerMinMaxStartupCycles, kFpMinMaxStartupCycles},
        {"TMUL", "vmul", kMulStartupCycles, kMulStartupCycles, true},
        {"TPARTMUL", "vmul", kMulStartupCycles, kMulStartupCycles, true},
        {"TADDS", "vadds", kIntegerAddScalarStartupCycles, kFpAddScalarStartupCycles},
        // A2/A3 lowers subtraction by a scalar to VADDS(-x).
        {"TSUBS", "vadds", kIntegerAddScalarStartupCycles, kFpAddScalarStartupCycles},
        {"TMULS", "vmuls", kIntegerMulScalarStartupCycles, kFpMulScalarStartupCycles},
        {"TNEG", "vmuls", kIntegerMulScalarStartupCycles, kFpMulScalarStartupCycles},
        {"TMAXS", "vmaxs", kIntegerMinMaxStartupCycles, kFpMinMaxStartupCycles},
        {"TMINS", "vmins", kIntegerMinMaxStartupCycles, kFpMinMaxStartupCycles},
        {"TLRELU", "vlrelu", kLeakyReluStartupCycles, kLeakyReluStartupCycles},
        {"TSHLS", "vshl", kShiftStartupCycles, kShiftStartupCycles},
        {"TSHRS", "vshr", kShiftStartupCycles, kShiftStartupCycles},
    };
    for (const auto& profile : profiles) {
        if (ctx.opcode == profile.opcode) {
            if (profile.floating_only && !IsSerializedFloatingType(ctx.src_dtype)) {
                return false;
            }
            leaf = profile.leaf;
            startup = IsSerializedIntegerType(ctx.src_dtype) ? profile.integer_startup : profile.default_startup;
            return true;
        }
    }
    return false;
}

inline bool TryEstimateSerializedBankProjection(
    const PtoCycleContext& ctx, std::string_view leaf, uint64_t nativeWidth, CycleEstimate& result)
{
    constexpr uint64_t kAddScalarIntegerBankNumerator = 29;
    constexpr uint64_t kAddScalarFpBankNumerator = 31;
    constexpr uint64_t kAddScalarBankRoundOffset = 8;
    constexpr uint64_t kAddScalarBankDenominator = 16;
    constexpr uint64_t kRepeatedRowBankNumerator = 7;
    constexpr uint64_t kRepeatedRowBankRoundOffset = 2;
    constexpr uint64_t kRepeatedRowBankDenominator = 4;
    constexpr uint64_t kIntegerMulBankNumerator = 15;
    constexpr uint64_t kIntegerMulBankRoundOffset = 4;
    constexpr uint64_t kIntegerMulBankDenominator = 8;
    constexpr uint64_t kIntegerMinMaxBankNumerator = 29;
    constexpr uint64_t kIntegerMinMaxBankRoundOffset = 8;
    constexpr uint64_t kIntegerMinMaxBankDenominator = 16;
    constexpr int kLongRowThreshold = 64;
    if (ctx.cols <= static_cast<int>(nativeWidth)) {
        return false;
    }
    uint64_t cycles;
    if (leaf == "vadds") {
        // Multi-repeat rows revisit the UB bank phase: 31/16 for FP, 29/16 for integers.
        const uint64_t numerator =
            IsSerializedIntegerType(ctx.src_dtype) ? kAddScalarIntegerBankNumerator : kAddScalarFpBankNumerator;
        cycles = (ctx.stub_cycles * numerator + kAddScalarBankRoundOffset) / kAddScalarBankDenominator;
    } else if (
        ctx.rows >= kLongRowThreshold && ((leaf == "vlrelu" && ctx.src_dtype == "fp32") ||
                                          (leaf == "vmuls" && IsSerializedFloatingType(ctx.src_dtype)))) {
        // Long floating-point rows share the same 7/4 repeated-row bank phase.
        cycles =
            (ctx.stub_cycles * kRepeatedRowBankNumerator + kRepeatedRowBankRoundOffset) / kRepeatedRowBankDenominator;
    } else if (leaf == "vmuls" && IsSerializedIntegerType(ctx.src_dtype)) {
        cycles = (ctx.stub_cycles * kIntegerMulBankNumerator + kIntegerMulBankRoundOffset) / kIntegerMulBankDenominator;
    } else if (leaf == "vmaxs" || leaf == "vmins") {
        cycles = (ctx.stub_cycles * kIntegerMinMaxBankNumerator + kIntegerMinMaxBankRoundOffset) /
                 kIntegerMinMaxBankDenominator;
    } else {
        return false;
    }
    result = {cycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
    return true;
}

inline bool IsFullWidthSerializedLeaf(std::string_view leaf)
{
    return leaf == "vadd" || leaf == "vsub" || leaf == "vmax" || leaf == "vmin" || leaf == "vmul" || leaf == "vadds" ||
           leaf == "vmuls" || leaf == "vmaxs" || leaf == "vmins" || leaf == "vshl" || leaf == "vshr";
}

inline CycleEstimate EstimateSerializedVectorLeaf(const PtoCycleContext& ctx)
{
    constexpr uint64_t kFp16ElementsPerRepeat = 128;
    constexpr uint64_t kOtherElementsPerRepeat = 64;
    if (ctx.trace == nullptr || ctx.stub_cycles == 0) {
        return {};
    }
    std::string_view leaf;
    uint64_t startup = 0;
    if (!TryResolveSerializedLeaf(ctx, leaf, startup)) {
        return {};
    }
    const uint64_t nativeWidth =
        (ctx.src_dtype == "fp16" || ctx.src_dtype == "int16") ? kFp16ElementsPerRepeat : kOtherElementsPerRepeat;
    CycleEstimate projection;
    if (TryEstimateSerializedBankProjection(ctx, leaf, nativeWidth, projection)) {
        return projection;
    }
    if (leaf == "vlrelu" && ctx.src_dtype == "fp16" && ctx.cols > static_cast<int>(nativeWidth)) {
        return {};
    }
    if (leaf == "vlrelu" && ctx.src_dtype == "fp32" && ctx.cols == static_cast<int>(nativeWidth)) {
        return {};
    }
    if (IsFullWidthSerializedLeaf(leaf) && ctx.cols > 0 && static_cast<uint64_t>(ctx.cols) % nativeWidth == 0) {
        // Full-width rows form one continuous issue stream without a per-call endpoint.
        return {};
    }
    // Restore one native startup only for a true one-leaf PTO lowering. The stub queue
    // overlaps this boundary across public calls, while the target-pipe measurement retains it.
    if (CountCceCalls(*ctx.trace, leaf) != 1) {
        return {};
    }
    return {ctx.stub_cycles + startup, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateMultiCallShiftTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kBankPhaseNumerator = 15;
    constexpr uint64_t kBankPhaseRoundOffset = 4;
    constexpr uint64_t kBankPhaseDenominator = 8;
    const bool left = ctx.opcode == "TSHLS";
    if ((!left && ctx.opcode != "TSHRS") || ctx.trace == nullptr || ctx.stub_cycles == 0) {
        return {};
    }
    const uint64_t calls = CountCceCalls(*ctx.trace, left ? "vshl" : "vshr");
    if (calls <= 1) {
        return {};
    }
    // Column chunking emits independent native shift leaves whose repeated-row
    // UB bank phase occupies 15/8 of the calibrated contiguous issue window.
    return {
        (ctx.stub_cycles * kBankPhaseNumerator + kBankPhaseRoundOffset) / kBankPhaseDenominator,
        ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateTinyFp16UnaryFloor(const PtoCycleContext& ctx)
{
    constexpr uint64_t kMinimumObservedCycles = 3;
    if (ctx.trace == nullptr || ctx.src_dtype != "fp16" || ctx.stub_cycles == 0 ||
        ctx.stub_cycles >= kMinimumObservedCycles) {
        return {};
    }
    std::string_view leaf;
    if (ctx.opcode == "TABS") {
        leaf = "vabs";
    } else if (ctx.opcode == "TRELU") {
        leaf = "vrelu";
    } else if (ctx.opcode == "TRSQRT") {
        leaf = "vrsqrt";
    } else {
        return {};
    }
    if (CountCceCalls(*ctx.trace, leaf) != 1) {
        return {};
    }
    // Two or fewer FP16 issue beats are below the A3 target-pipe observation
    // granularity. Preserve a three-cycle public-call floor without altering
    // the native CCE leaf startup/slope.
    return {kMinimumObservedCycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

} // namespace pto::mocker::hybrid

#endif
