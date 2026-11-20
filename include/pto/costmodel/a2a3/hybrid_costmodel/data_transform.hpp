/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef PTO_MOCKER_A2A3_HYBRID_DATA_TRANSFORM_H_
#define PTO_MOCKER_A2A3_HYBRID_DATA_TRANSFORM_H_

#include <cstddef>
#include <cstdint>
#include <string_view>

#include <pto/costmodel/a2a3/hybrid_costmodel/context.hpp>

namespace pto::mocker::hybrid {

inline CycleEstimate EstimateSingleCallCvtTargetPipe(const PtoCycleContext& ctx)
{
    constexpr std::size_t kRepeatArgumentIndex = 2;
    constexpr uint64_t kMaskAndControlCycles = 7;
    if (ctx.opcode != "TCVT" || ctx.trace == nullptr || ctx.dst_dtype != "int32" ||
        (ctx.src_dtype != "fp16" && ctx.src_dtype != "fp32")) {
        return {};
    }
    const std::string_view expected = ctx.src_dtype == "fp16" ? "vconv_f162s32z" : "vconv_f322s32z";
    const CceCallRecord* convert = nullptr;
    for (const auto& call : ctx.trace->cce_calls) {
        if (call.name != expected) {
            continue;
        }
        if (convert != nullptr || call.args.size() <= kRepeatArgumentIndex) {
            return {};
        }
        convert = &call;
    }
    if (convert == nullptr) {
        return {};
    }

    // The native FP16/FP32 -> INT32 CAST_TRUNC profile is 5 + R.
    // The count-masked one-call PTO path retains two cycles for mask/control
    // setup in the target-pipe measurement. Multi-row/multi-call paths are
    // intentionally excluded because their address-dependent issue gaps are
    // not described by the completion profile.
    return {convert->args[kRepeatArgumentIndex] + kMaskAndControlCycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateGatherPatternTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kFp16ElementsPerRepeat = 64;
    constexpr uint64_t kFp32ElementsPerRepeat = 32;
    constexpr uint64_t kSharedEndpointCycles = 5;
    if (ctx.opcode != "TGATHER" || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        (ctx.src_dtype != "fp16" && ctx.src_dtype != "fp32") || CountCceCalls(*ctx.trace, "vreducev2") != 1) {
        return {};
    }

    // The two-tile MaskPattern overload lowers to VREDUCEV2, not VGATHER.
    // Its A3 target-pipe stream consumes one 32-element FP32 or 64-element
    // FP16 repeat per row-cycle, followed by one shared five-cycle endpoint.
    const uint64_t elementsPerRepeat = ctx.src_dtype == "fp16" ? kFp16ElementsPerRepeat : kFp32ElementsPerRepeat;
    const uint64_t repeatsPerRow = (static_cast<uint64_t>(ctx.cols) + elementsPerRepeat - 1) / elementsPerRepeat;
    const uint64_t cycles = static_cast<uint64_t>(ctx.rows) * repeatsPerRow + kSharedEndpointCycles;
    return {cycles, ModelKind::PTO_COMPOSITE_ANALYTIC};
}

inline uint64_t EstimateFp32RowExpandDivCycles(uint64_t rows, uint64_t cols)
{
    constexpr uint64_t kElementsPerRepeat = 64;
    constexpr uint64_t kShortRowNumerator = 3;
    constexpr uint64_t kShortRowDenominator = 2;
    constexpr uint64_t kShortEndpointCycles = 10;
    constexpr uint64_t kFullWidthNumerator = 5;
    constexpr uint64_t kFullWidthDenominator = 2;
    constexpr uint64_t kMultiRepeatCycles = 3;
    if (cols < kElementsPerRepeat) {
        return (kShortRowNumerator * rows) / kShortRowDenominator + kShortEndpointCycles;
    }
    const uint64_t repeats = rows * ((cols + kElementsPerRepeat - 1) / kElementsPerRepeat);
    return cols == kElementsPerRepeat ? (kFullWidthNumerator * repeats) / kFullWidthDenominator :
                                        kMultiRepeatCycles * repeats;
}

inline CycleEstimate EstimateRowExpandDivTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kFp16ElementsPerRepeat = 128;
    constexpr uint64_t kFp32ElementsPerRepeat = 64;
    constexpr uint64_t kFp16LongRowThreshold = 64;
    constexpr uint64_t kFp16LongRowNumerator = 5;
    constexpr uint64_t kFp16LongRowDenominator = 2;
    constexpr uint64_t kFp16LongRowDiscount = 2;
    constexpr uint64_t kFp16ShortBaseCycles = 48;
    constexpr uint64_t kFp16ShortElementDenominator = 256;
    constexpr uint64_t kFp16FullWidthNumerator = 9;
    constexpr uint64_t kFp16FullWidthRoundOffset = 1;
    constexpr uint64_t kFp16FullWidthDenominator = 2;
    if (ctx.opcode != "TROWEXPANDDIV" || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        (ctx.src_dtype != "fp16" && ctx.src_dtype != "fp32")) {
        return {};
    }
    const uint64_t rows = static_cast<uint64_t>(ctx.rows);
    const uint64_t cols = static_cast<uint64_t>(ctx.cols);
    const uint64_t nativeWidth = ctx.src_dtype == "fp16" ? kFp16ElementsPerRepeat : kFp32ElementsPerRepeat;
    if (CountCceCalls(*ctx.trace, "vdiv") != (cols + nativeWidth - 1) / nativeWidth ||
        CountCceCalls(*ctx.trace, "vbrcb") != 0 || CountCceCalls(*ctx.trace, "pipe_barrier") != 0) {
        return {};
    }
    if (ctx.src_dtype == "fp32") {
        return {EstimateFp32RowExpandDivCycles(rows, cols), ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    if (cols < kFp16ElementsPerRepeat) {
        const uint64_t cycles = rows >= kFp16LongRowThreshold ?
                                    (kFp16LongRowNumerator * rows) / kFp16LongRowDenominator - kFp16LongRowDiscount :
                                    kFp16ShortBaseCycles + (rows * cols) / kFp16ShortElementDenominator;
        return {cycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    const uint64_t repeats = rows * ((cols + kFp16ElementsPerRepeat - 1) / kFp16ElementsPerRepeat);
    return {
        (kFp16FullWidthNumerator * repeats + kFp16FullWidthRoundOffset) / kFp16FullWidthDenominator,
        ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateRowExpandAddSubTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kFp16ElementsPerRepeat = 128;
    constexpr uint64_t kFp32ElementsPerRepeat = 64;
    constexpr uint64_t kFullWidthNumerator = 13;
    constexpr uint64_t kFullWidthRoundOffset = 4;
    constexpr uint64_t kFullWidthDenominator = 8;
    constexpr uint64_t kFp32Numerator = 27;
    constexpr uint64_t kFp32RoundOffset = 8;
    constexpr uint64_t kFp32Denominator = 16;
    constexpr uint64_t kFp16LongRowThreshold = 32;
    constexpr uint64_t kFp16LongNumerator = 45;
    constexpr uint64_t kFp16LongRoundOffset = 16;
    constexpr uint64_t kFp16LongDenominator = 32;
    constexpr uint64_t kFp16BlockElements = 32;
    constexpr uint64_t kTinyRowStartupCycles = 6;
    const bool isAdd = ctx.opcode == "TROWEXPANDADD";
    if ((!isAdd && ctx.opcode != "TROWEXPANDSUB") || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        (ctx.src_dtype != "fp16" && ctx.src_dtype != "fp32")) {
        return {};
    }
    const std::string_view leaf = isAdd ? "vadd" : "vsub";
    if (CountCceCalls(*ctx.trace, leaf) != 1 || CountCceCalls(*ctx.trace, "vbrcb") != 0 ||
        CountCceCalls(*ctx.trace, "pipe_barrier") != 0) {
        return {};
    }
    const uint64_t rows = static_cast<uint64_t>(ctx.rows);
    const uint64_t nativeWidth = ctx.src_dtype == "fp16" ? kFp16ElementsPerRepeat : kFp32ElementsPerRepeat;
    const uint64_t cols = static_cast<uint64_t>(ctx.cols);
    if (cols > nativeWidth) {
        return {};
    }
    if (cols == nativeWidth) {
        // One full native repeat per row with a zero-stride broadcast source.
        return {
            (kFullWidthNumerator * rows + kFullWidthRoundOffset) / kFullWidthDenominator,
            ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    if (ctx.src_dtype == "fp32") {
        return {(kFp32Numerator * rows + kFp32RoundOffset) / kFp32Denominator, ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    if (rows >= kFp16LongRowThreshold) {
        return {
            (kFp16LongNumerator * rows + kFp16LongRoundOffset) / kFp16LongDenominator,
            ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    if (cols >= kFp16BlockElements) {
        return {(kFp32Numerator * rows + kFp32RoundOffset) / kFp32Denominator, ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    // A 16-column FP16 row is exactly one 32-byte broadcast block.  The trace
    // is still unambiguous here: one zero-stride VADD/VSUB leaf and no VBRcb
    // or barrier.  Repeated public PTO calls overlap the CCE leaf's six-cycle
    // startup in the stub queue, while the target-pipe measurement retains it.
    return {ctx.stub_cycles + kTinyRowStartupCycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline uint64_t EstimateRowExpandMulCycles(const PtoCycleContext& ctx, uint64_t rows)
{
    constexpr uint64_t kFp32Numerator = 29;
    constexpr uint64_t kFp32RoundOffset = 8;
    constexpr uint64_t kFp32Denominator = 16;
    constexpr uint64_t kShortRowThreshold = 32;
    constexpr uint64_t kShortNumerator = 7;
    constexpr uint64_t kShortRoundOffset = 2;
    constexpr uint64_t kShortDenominator = 4;
    constexpr uint64_t kLongNumerator = 23;
    constexpr uint64_t kLongRoundOffset = 8;
    constexpr uint64_t kLongDenominator = 16;
    if (ctx.src_dtype == "fp32") {
        return (kFp32Numerator * rows + kFp32RoundOffset) / kFp32Denominator;
    }
    if (rows < kShortRowThreshold) {
        return (kShortNumerator * rows + kShortRoundOffset) / kShortDenominator;
    }
    return (kLongNumerator * rows + kLongRoundOffset) / kLongDenominator;
}

inline CycleEstimate ProjectRowExpandMulMinMax(const PtoCycleContext& ctx, bool isMul, uint64_t nativeWidth)
{
    constexpr uint64_t kFullWidthNumerator = 3;
    constexpr uint64_t kFullWidthRoundOffset = 1;
    constexpr uint64_t kFullWidthDenominator = 2;
    constexpr uint64_t kTinyFp16Rows = 16;
    constexpr uint64_t kFp16BlockElements = 32;
    constexpr uint64_t kMulStartupCycles = 7;
    constexpr uint64_t kMinMaxStartupCycles = 4;
    constexpr uint64_t kShortRowThreshold = 32;
    constexpr uint64_t kShortMinMaxExtraCycles = 1;
    constexpr uint64_t kLongMinMaxNumerator = 43;
    constexpr uint64_t kLongMinMaxRoundOffset = 16;
    constexpr uint64_t kLongMinMaxDenominator = 32;
    const uint64_t rows = static_cast<uint64_t>(ctx.rows);
    const uint64_t cols = static_cast<uint64_t>(ctx.cols);
    if (cols > nativeWidth) {
        return {};
    }
    if (cols == nativeWidth) {
        return {
            (kFullWidthNumerator * rows + kFullWidthRoundOffset) / kFullWidthDenominator,
            ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    if (ctx.src_dtype == "fp16" && rows == kTinyFp16Rows && cols < kFp16BlockElements) {
        // One legal 32-byte zero-stride broadcast block lowers to a single leaf.
        // Restore that leaf's CCE startup once per PTO call.
        const uint64_t startup = isMul ? kMulStartupCycles : kMinMaxStartupCycles;
        return {ctx.stub_cycles + startup, ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    if (isMul) {
        return {EstimateRowExpandMulCycles(ctx, rows), ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    if (ctx.src_dtype == "fp32" || rows < kShortRowThreshold) {
        return {
            (kFullWidthNumerator * rows + kFullWidthRoundOffset) / kFullWidthDenominator + kShortMinMaxExtraCycles,
            ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    return {
        (kLongMinMaxNumerator * rows + kLongMinMaxRoundOffset) / kLongMinMaxDenominator,
        ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateRowExpandMulMinMaxTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kFp16ElementsPerRepeat = 128;
    constexpr uint64_t kFp32ElementsPerRepeat = 64;
    const bool isMul = ctx.opcode == "TROWEXPANDMUL";
    const bool isMax = ctx.opcode == "TROWEXPANDMAX";
    const bool isMin = ctx.opcode == "TROWEXPANDMIN";
    if ((!isMul && !isMax && !isMin) || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        (ctx.src_dtype != "fp16" && ctx.src_dtype != "fp32")) {
        return {};
    }
    const std::string_view leaf = isMul ? "vmul" : (isMax ? "vmax" : "vmin");
    if (CountCceCalls(*ctx.trace, leaf) != 1 || CountCceCalls(*ctx.trace, "vbrcb") != 0 ||
        CountCceCalls(*ctx.trace, "pipe_barrier") != 0) {
        return {};
    }
    const uint64_t nativeWidth = ctx.src_dtype == "fp16" ? kFp16ElementsPerRepeat : kFp32ElementsPerRepeat;
    return ProjectRowExpandMulMinMax(ctx, isMul, nativeWidth);
}

inline bool TryProjectRowExpandSubCycles(const PtoCycleContext& ctx, uint64_t nativeWidth, uint64_t& cycles)
{
    constexpr uint64_t kFullWidthNumerator = 13;
    constexpr uint64_t kFullWidthRoundOffset = 4;
    constexpr uint64_t kFullWidthDenominator = 8;
    constexpr uint64_t kFp32Numerator = 27;
    constexpr uint64_t kFp32RoundOffset = 8;
    constexpr uint64_t kFp32Denominator = 16;
    constexpr uint64_t kFp16LongRowThreshold = 32;
    constexpr uint64_t kFp16LongNumerator = 45;
    constexpr uint64_t kFp16LongRoundOffset = 16;
    constexpr uint64_t kFp16LongDenominator = 32;
    constexpr uint64_t kFp16BlockElements = 32;
    const uint64_t rows = static_cast<uint64_t>(ctx.rows);
    const uint64_t cols = static_cast<uint64_t>(ctx.cols);
    if (cols == nativeWidth) {
        cycles = (kFullWidthNumerator * rows + kFullWidthRoundOffset) / kFullWidthDenominator;
    } else if (ctx.src_dtype == "fp32") {
        cycles = (kFp32Numerator * rows + kFp32RoundOffset) / kFp32Denominator;
    } else if (rows >= kFp16LongRowThreshold) {
        cycles = (kFp16LongNumerator * rows + kFp16LongRoundOffset) / kFp16LongDenominator;
    } else if (cols >= kFp16BlockElements) {
        cycles = (kFp32Numerator * rows + kFp32RoundOffset) / kFp32Denominator;
    } else {
        return false;
    }
    return true;
}

inline CycleEstimate EstimateRowExpandExpDifTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kFp16ElementsPerRepeat = 128;
    constexpr uint64_t kFp32ElementsPerRepeat = 64;
    constexpr uint64_t kTinyFp16Rows = 16;
    constexpr uint64_t kFp16BlockElements = 32;
    if (ctx.opcode != "TROWEXPANDEXPDIF" || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        (ctx.src_dtype != "fp16" && ctx.src_dtype != "fp32") || CountCceCalls(*ctx.trace, "vsub") != 1 ||
        CountCceCalls(*ctx.trace, "vexp") != 1 || CountCceCalls(*ctx.trace, "pipe_barrier") != 1) {
        return {};
    }
    const uint64_t rows = static_cast<uint64_t>(ctx.rows);
    const uint64_t cols = static_cast<uint64_t>(ctx.cols);
    const uint64_t nativeWidth = ctx.src_dtype == "fp16" ? kFp16ElementsPerRepeat : kFp32ElementsPerRepeat;
    if (cols > nativeWidth || (ctx.src_dtype == "fp16" && rows == kTinyFp16Rows && cols < kFp16BlockElements)) {
        return {};
    }
    uint64_t nativeSubCycles = 0;
    for (const auto& call : ctx.trace->cce_calls) {
        if (call.name == "vsub") {
            nativeSubCycles = call.cycles;
            break;
        }
    }
    if (nativeSubCycles == 0 || ctx.stub_cycles <= nativeSubCycles) {
        return {};
    }
    uint64_t projectedSubCycles;
    if (!TryProjectRowExpandSubCycles(ctx, nativeWidth, projectedSubCycles)) {
        return {};
    }
    // Preserve native VEXP and the explicit barrier; replace only the VSUB projection.
    return {ctx.stub_cycles - nativeSubCycles + projectedSubCycles, ModelKind::PTO_COMPOSITE_ANALYTIC};
}

inline CycleEstimate EstimateUbToUbTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kTransferBytesPerCycle = 256;
    constexpr uint64_t kRoundOffsetBytes = 128;
    constexpr uint64_t kMinimumCycles = 3;
    constexpr uint64_t kSpecialTransferBytes = 1024;
    constexpr uint64_t kFp32ScatterCycles = 4;
    constexpr uint64_t kFp32MoveCycles = 5;
    constexpr uint64_t kInt32Cycles = 4;
    if ((ctx.opcode != "TMOV" && ctx.opcode != "TSCATTER") || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        CountCceCalls(*ctx.trace, "copy_ubuf_to_ubuf") != 1) {
        return {};
    }
    const uint64_t typeBytes = DTypeBytes(ctx.src_dtype);
    if (typeBytes == 0) {
        return {};
    }
    const uint64_t bytes = static_cast<uint64_t>(ctx.rows) * static_cast<uint64_t>(ctx.cols) * typeBytes;
    // The calibrated UB2UB baseline includes a ~16-cycle completion head.  PTO's
    // repeated aiv_vec_time contract observes the payload stream instead,
    // whose non-alias throughput is approximately one 256-byte block/cycle.
    uint64_t cycles = (bytes + kRoundOffsetBytes) / kTransferBytesPerCycle;
    if (cycles < kMinimumCycles) {
        cycles = kMinimumCycles;
    }
    if (bytes == kSpecialTransferBytes && ctx.src_dtype == "fp32") {
        cycles = ctx.opcode == "TSCATTER" ? kFp32ScatterCycles : kFp32MoveCycles;
    } else if (bytes == kSpecialTransferBytes && ctx.src_dtype == "int32") {
        cycles = kInt32Cycles;
    }
    return {cycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateConcatTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kCopiesPerRow = 2;
    constexpr uint64_t kCyclesPerRow = 6;
    constexpr uint64_t kSharedEndpointCycles = 13;
    if (ctx.opcode != "TCONCAT" || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        CountCceCalls(*ctx.trace, "copy_ubuf_to_ubuf") != kCopiesPerRow * static_cast<uint64_t>(ctx.rows)) {
        return {};
    }
    // The aligned three-tile overload emits two UB-to-UB copies per row.
    // Target-pipe active time observes the six-cycle row issue/control body
    // plus one shared 13-cycle public-call endpoint; per-copy completion
    // CCE completion heads must not be serialized 2*rows times.
    return {
        kCyclesPerRow * static_cast<uint64_t>(ctx.rows) + kSharedEndpointCycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateDequantTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kElementsPerRepeat = 64;
    constexpr uint64_t kInt8BaseBoundaryCycles = 40;
    constexpr uint64_t kInt16BaseBoundaryCycles = 34;
    constexpr uint64_t kOverlapCyclesPerAdditionalRepeat = 3;
    if (ctx.opcode != "TDEQUANT" || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        (ctx.src_dtype != "int8" && ctx.src_dtype != "int16") ||
        CountCceCalls(*ctx.trace, "vadds") != static_cast<uint64_t>(ctx.rows) ||
        CountCceCalls(*ctx.trace, "vmuls") != static_cast<uint64_t>(ctx.rows)) {
        return {};
    }
    const uint64_t repeats = (static_cast<uint64_t>(ctx.cols) + kElementsPerRepeat - 1) / kElementsPerRepeat;
    const uint64_t baseBoundary = ctx.src_dtype == "int8" ? kInt8BaseBoundaryCycles : kInt16BaseBoundaryCycles;
    // Conversion, scalar load, add and multiply are serialized per row.  Each
    // extra 64-element conversion repeat overlaps three cycles of that fixed
    // boundary in aiv_vec_time; CCE leaf issue/completion values remain
    // untouched.
    const uint64_t overlap = kOverlapCyclesPerAdditionalRepeat * (repeats - 1);
    const uint64_t perRowDiscount = overlap < baseBoundary ? baseBoundary - overlap : 0;
    const uint64_t discount = perRowDiscount * static_cast<uint64_t>(ctx.rows);
    return {
        ctx.stub_cycles > discount ? ctx.stub_cycles - discount : ctx.stub_cycles, ModelKind::PTO_COMPOSITE_ANALYTIC};
}

inline CycleEstimate EstimateExpandsTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kLongRowThreshold = 64;
    constexpr uint64_t kMediumRowThreshold = 32;
    constexpr uint64_t kWideColumnThreshold = 256;
    constexpr uint64_t kLongFp16CyclesPerRow = 3;
    constexpr uint64_t kLongFp16EndpointCycles = 10;
    constexpr uint64_t kLongOtherCyclesPerRow = 2;
    constexpr uint64_t kLongOtherEndpointCycles = 16;
    constexpr uint64_t kMediumEndpointCycles = 11;
    constexpr uint64_t kWideFp16Cycles = 36;
    constexpr uint64_t kWideOtherCycles = 68;
    constexpr uint64_t kShortEndpointCycles = 10;
    if (ctx.opcode != "TEXPANDS" || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        CountCceCalls(*ctx.trace, "vector_dup") != static_cast<uint64_t>(ctx.rows)) {
        return {};
    }
    const uint64_t rows = static_cast<uint64_t>(ctx.rows);
    const uint64_t cols = static_cast<uint64_t>(ctx.cols);
    uint64_t cycles;
    if (rows >= kLongRowThreshold) {
        cycles = ctx.dst_dtype == "fp16" ? kLongFp16CyclesPerRow * rows + kLongFp16EndpointCycles :
                                           kLongOtherCyclesPerRow * rows + kLongOtherEndpointCycles;
    } else if (rows >= kMediumRowThreshold) {
        cycles = rows + kMediumEndpointCycles;
    } else if (cols >= kWideColumnThreshold) {
        cycles = ctx.dst_dtype == "fp16" ? kWideFp16Cycles : kWideOtherCycles;
    } else {
        cycles = rows + kShortEndpointCycles;
    }
    // Count-mask VECTOR_DUP has one CCE leaf per row, but the A3 active-time issue
    // path changes at the 32/64-row count-register boundaries.  Project those
    // boundaries at PTO level while retaining the native VDUP profile.
    return {cycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateImg2ColTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kIssueCycles = 2;
    if (ctx.opcode != "TIMG2COL" || ctx.trace == nullptr || CountCceCalls(*ctx.trace, "img2colv2_cbuf_to_ca") != 1) {
        return {};
    }
    // The benchmark measures MTE1 active issue time under repeated identical
    // PTO calls.  One LOAD3D/IMG2COL issue occupies two cycles independent of
    // its completion span; the latter remains modeled by the native CCE leaf.
    return {kIssueCycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

} // namespace pto::mocker::hybrid

#endif
