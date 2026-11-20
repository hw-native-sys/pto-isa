/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef PTO_MOCKER_A2A3_HYBRID_COSTMODEL_HPP
#define PTO_MOCKER_A2A3_HYBRID_COSTMODEL_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include <pto/costmodel/a2a3/hybrid_costmodel/context.hpp>
#include <pto/costmodel/a2a3/hybrid_costmodel/basic_elementwise.hpp>
#include <pto/costmodel/a2a3/hybrid_costmodel/comparison_reduction.hpp>
#include <pto/costmodel/a2a3/hybrid_costmodel/data_transform.hpp>

namespace pto::mocker::hybrid {

inline bool HasFmodTopology(const PtoCycleContext& ctx, bool scalar)
{
    constexpr uint64_t kScalarBarrierCountPerRow = 5;
    constexpr uint64_t kTensorBarrierCountPerRow = 4;
    if (ctx.trace == nullptr || ctx.rows <= 0) {
        return false;
    }
    const uint64_t rows = static_cast<uint64_t>(ctx.rows);
    if (CountCceCalls(*ctx.trace, "vdiv") != rows || CountCceCalls(*ctx.trace, "vconv_f322f32z") != rows ||
        CountCceCalls(*ctx.trace, "vsub") != rows) {
        return false;
    }
    if (scalar) {
        return CountCceCalls(*ctx.trace, "vector_dup") == rows && CountCceCalls(*ctx.trace, "vmuls") == rows &&
               CountCceCalls(*ctx.trace, "pipe_barrier") == rows * kScalarBarrierCountPerRow;
    }
    return CountCceCalls(*ctx.trace, "vmul") == rows &&
           CountCceCalls(*ctx.trace, "pipe_barrier") == rows * kTensorBarrierCountPerRow;
}

inline CycleEstimate EstimateFmod(const PtoCycleContext& ctx)
{
    constexpr uint64_t kFp32ElementsPerRepeat = 64;
    constexpr uint64_t kScalarShortRowCycles = 96;
    constexpr uint64_t kScalarRowBaseCycles = 91;
    constexpr uint64_t kScalarRepeatCycles = 6;
    constexpr uint64_t kTensorShortRowCycles = 84;
    constexpr uint64_t kTensorRowBaseCycles = 78;
    constexpr uint64_t kTensorRepeatCycles = 8;
    constexpr uint64_t kSharedEndpointCycles = 4;
    const bool scalar = ctx.opcode == "TFMODS";
    if ((ctx.opcode != "TFMOD" && !scalar) || ctx.dst_dtype != "fp32" || ctx.rows <= 0 || ctx.cols <= 0 ||
        !HasFmodTopology(ctx, scalar)) {
        return {};
    }

    // Both PTO variants process one row at a time under a count mask. The
    // repeat term follows the 256-byte CCE vector width. The fixed row term
    // accounts for the serialized DIV/CONV/MUL/SUB dependency chain while the
    // final four cycles are the shared target-pipe endpoint, charged once per
    // public PTO call rather than once per CCE intrinsic.
    const uint64_t repeats = (static_cast<uint64_t>(ctx.cols) + kFp32ElementsPerRepeat - 1) / kFp32ElementsPerRepeat;
    uint64_t cyclesPerRow;
    if (scalar) {
        cyclesPerRow = ctx.cols < static_cast<int>(kFp32ElementsPerRepeat) ?
                           kScalarShortRowCycles :
                           kScalarRowBaseCycles + kScalarRepeatCycles * repeats;
    } else {
        cyclesPerRow = ctx.cols < static_cast<int>(kFp32ElementsPerRepeat) ?
                           kTensorShortRowCycles :
                           kTensorRowBaseCycles + kTensorRepeatCycles * repeats;
    }
    return {static_cast<uint64_t>(ctx.rows) * cyclesPerRow + kSharedEndpointCycles, ModelKind::PTO_COMPOSITE_ANALYTIC};
}

inline CycleEstimate EstimateMaskedDivTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kFp16ElementsPerRepeat = 128;
    constexpr uint64_t kFp32ElementsPerRepeat = 64;
    constexpr uint64_t kFp16RepeatCycles = 8;
    constexpr uint64_t kFp32RepeatCycles = 4;
    constexpr uint64_t kVectorStartupCycles = 14;
    constexpr uint64_t kRowControlCycles = 2;
    constexpr uint64_t kRowControlEndpointCycles = 16;
    if (ctx.opcode != "TDIV" || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        (ctx.src_dtype != "fp16" && ctx.src_dtype != "fp32") || CountCceCalls(*ctx.trace, "vdiv") != 1) {
        return {};
    }
    const uint64_t elementsPerRepeat = ctx.src_dtype == "fp16" ? kFp16ElementsPerRepeat : kFp32ElementsPerRepeat;
    if (static_cast<uint64_t>(ctx.cols) >= elementsPerRepeat) {
        return {};
    }

    // The small-column PTO path emits one strided VDIV whose repeat count is
    // rows and whose normal mask contains only cols elements.  Charging every
    // repeat as a full 256-byte vector overstates work (most visibly for
    // FP16).  Project the calibrated 8/4-cycle full-repeat slope by effective mask
    // volume, while retaining the measured mask/row-control issue floor.
    const uint64_t totalElements = static_cast<uint64_t>(ctx.rows) * static_cast<uint64_t>(ctx.cols);
    const uint64_t effectiveRepeats = (totalElements + elementsPerRepeat - 1) / elementsPerRepeat;
    const uint64_t slope = ctx.src_dtype == "fp16" ? kFp16RepeatCycles : kFp32RepeatCycles;
    const uint64_t vectorWork = slope * effectiveRepeats + kVectorStartupCycles;
    const uint64_t rowControlFloor = kRowControlCycles * static_cast<uint64_t>(ctx.rows) + kRowControlEndpointCycles;
    return {vectorWork > rowControlFloor ? vectorWork : rowControlFloor, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateMaddTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kFp16ElementsPerRepeat = 128;
    constexpr uint64_t kFp32ElementsPerRepeat = 64;
    constexpr uint64_t kFullRepeatCycles = 4;
    constexpr uint64_t kPartialRepeatCycles = 2;
    constexpr uint64_t kPartialEndpointCycles = 13;
    if (ctx.opcode != "TMADD" || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        (ctx.src_dtype != "fp16" && ctx.src_dtype != "fp32") || CountCceCalls(*ctx.trace, "vmadd") == 0) {
        return {};
    }
    const uint64_t elementsPerRepeat = ctx.src_dtype == "fp16" ? kFp16ElementsPerRepeat : kFp32ElementsPerRepeat;
    const uint64_t fullRepeats = static_cast<uint64_t>(ctx.cols) / elementsPerRepeat;
    const bool hasPartialRepeat = static_cast<uint64_t>(ctx.cols) % elementsPerRepeat != 0;

    // The new native profile establishes a 6-cycle completion slope for
    // separate-operand VMADD and a 4-cycle slope for the accumulator-alias
    // variant. A3 target-pipe active time follows the 4-cycle full-repeat issue
    // rate. Count-masked partial repeats use the measured two-cycle issue body
    // plus one shared source-call observer endpoint.
    uint64_t cyclesPerRow = kFullRepeatCycles * fullRepeats;
    if (hasPartialRepeat) {
        cyclesPerRow += kPartialRepeatCycles;
    }
    const uint64_t endpoint = hasPartialRepeat ? kPartialEndpointCycles : 0;
    return {static_cast<uint64_t>(ctx.rows) * cyclesPerRow + endpoint, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateMulaTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kFp16ElementsPerRepeat = 128;
    constexpr uint64_t kFp32ElementsPerRepeat = 64;
    constexpr uint64_t kFullRepeatCycles = 4;
    constexpr uint64_t kPartialRowCycles = 2;
    constexpr uint64_t kPartialEndpointCycles = 13;
    if (ctx.opcode != "TMULA" || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        (ctx.src_dtype != "fp16" && ctx.src_dtype != "fp32") || CountCceCalls(*ctx.trace, "vmla") != 1) {
        return {};
    }
    const uint64_t elementsPerRepeat = ctx.src_dtype == "fp16" ? kFp16ElementsPerRepeat : kFp32ElementsPerRepeat;
    const uint64_t rows = static_cast<uint64_t>(ctx.rows);
    const uint64_t cols = static_cast<uint64_t>(ctx.cols);
    if (cols >= elementsPerRepeat) {
        // Full-width TMULA is the accumulator-alias VMLA case measured by
        // at a four-cycle issue/completion slope.
        const uint64_t repeats = rows * ((cols + elementsPerRepeat - 1) / elementsPerRepeat);
        return {kFullRepeatCycles * repeats, ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    // The strided sub-vector path issues two target-pipe cycles per row and
    // retains one 13-cycle mask/observer endpoint for the public PTO call.
    return {kPartialRowCycles * rows + kPartialEndpointCycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimatePowTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kElementsPerRepeat = 64;
    constexpr uint64_t kScalarShortBaseCycles = 132;
    constexpr uint64_t kScalarShortRowNumerator = 115;
    constexpr uint64_t kScalarShortRowDenominator = 16;
    constexpr uint64_t kTensorShortBaseCycles = 131;
    constexpr uint64_t kTensorShortRowNumerator = 67;
    constexpr uint64_t kTensorShortRowDenominator = 8;
    constexpr uint64_t kTensorRowBaseCycles = 116;
    constexpr uint64_t kTensorRepeatCycles = 12;
    const bool scalar = ctx.opcode == "TPOWS";
    if ((ctx.opcode != "TPOW" && !scalar) || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        ctx.src_dtype != "fp32" || CountCceCalls(*ctx.trace, "vln") == 0 || CountCceCalls(*ctx.trace, "vexp") == 0) {
        return {};
    }
    const uint64_t rows = static_cast<uint64_t>(ctx.rows);
    const uint64_t cols = static_cast<uint64_t>(ctx.cols);
    if (cols < kElementsPerRepeat) {
        // One strided call per composite leaf.  Project the calibrated CCE leaf
        // completion profiles to the A3 target-pipe stream and retain the
        // shared mask/observer endpoint once per public PTO instruction.
        const uint64_t cycles =
            scalar ? kScalarShortBaseCycles + (kScalarShortRowNumerator * rows) / kScalarShortRowDenominator :
                     kTensorShortBaseCycles + (kTensorShortRowNumerator * rows) / kTensorShortRowDenominator;
        return {cycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    if (!scalar) {
        // Vector exponent TPOW lowers row-by-row.  Each extra native repeat
        // adds the combined 12-cycle issue work of the two LN/MUL/EXP chains;
        // 116 cycles are the per-row dependency and observer boundary.
        const uint64_t repeatsPerRow = (cols + kElementsPerRepeat - 1) / kElementsPerRepeat;
        return {rows * (kTensorRowBaseCycles + kTensorRepeatCycles * repeatsPerRow), ModelKind::PTO_COMPOSITE_ANALYTIC};
    }
    // TPOWS uses the continuous scalar-binary lowering.  Its aligned cases
    // already agree with hardware, so preserve the CCE-composed result.
    return {ctx.stub_cycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateRemainderTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kBarrierCompletionCycles = 16;
    constexpr uint64_t kNonTargetCyclesPerBarrier = 4;
    if ((ctx.opcode != "TREM" && ctx.opcode != "TREMS") || ctx.trace == nullptr || ctx.stub_cycles == 0) {
        return {};
    }
    const uint64_t barriers = CountCceCalls(*ctx.trace, "pipe_barrier", kBarrierCompletionCycles);
    if (barriers == 0 || CountCceCalls(*ctx.trace, "vdiv") == 0 || CountCceCalls(*ctx.trace, "vsub") == 0) {
        return {};
    }
    // The CCE profile uses a 16-cycle completion boundary.  aiv_vec_time observes the
    // target PIPE_V portion; the four non-target observer cycles must not be
    // charged at every internal barrier of this long composite.
    const uint64_t discount = kNonTargetCyclesPerBarrier * barriers;
    return {
        ctx.stub_cycles > discount ? ctx.stub_cycles - discount : ctx.stub_cycles,
        ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateAxpyTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kFp16ElementsPerRepeat = 128;
    constexpr uint64_t kFp32ElementsPerRepeat = 64;
    constexpr uint64_t kPartialRowEndpointCycles = 11;
    if (ctx.opcode != "TAXPY" || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        (ctx.src_dtype != "fp16" && ctx.src_dtype != "fp32") || CountCceCalls(*ctx.trace, "vaxpy") != 1) {
        return {};
    }
    const uint64_t nativeWidth = ctx.src_dtype == "fp16" ? kFp16ElementsPerRepeat : kFp32ElementsPerRepeat;
    if (static_cast<uint64_t>(ctx.cols) >= nativeWidth) {
        return {};
    }
    // Partial-row AXPY keeps eleven cycles of the native 18-cycle completion
    // head in the target-pipe window; the remaining head overlaps mask/control.
    // Full and multi-repeat rows retain the unmodified CCE leaf result.
    return {ctx.stub_cycles + kPartialRowEndpointCycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateRecipTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kFp16ElementsPerRepeat = 128;
    constexpr uint64_t kFp32ElementsPerRepeat = 64;
    constexpr int kSmallColumnThreshold = 32;
    constexpr uint64_t kFp32OrSmallRowCycles = 3;
    constexpr uint64_t kFp32OrSmallEndpointCycles = 29;
    constexpr uint64_t kFp16RowCycles = 5;
    constexpr uint64_t kFp16EndpointCycles = 27;
    const bool scalarDiv = ctx.opcode == "TDIVS";
    if ((ctx.opcode != "TRECIP" && !scalarDiv) || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        (ctx.src_dtype != "fp16" && ctx.src_dtype != "fp32") || CountCceCalls(*ctx.trace, "vector_dup") != 1 ||
        CountCceCalls(*ctx.trace, "vdiv") != 1 || CountCceCalls(*ctx.trace, "pipe_barrier") != 1) {
        return {};
    }
    const uint64_t rows = static_cast<uint64_t>(ctx.rows);
    const uint64_t fullWidth = ctx.src_dtype == "fp16" ? kFp16ElementsPerRepeat : kFp32ElementsPerRepeat;
    if (static_cast<uint64_t>(ctx.cols) >= fullWidth) {
        // For normal finite inputs (and non-alias TDIVS) the native
        // full-repeat stream already matches the calibrated CCE completion closely.
        return {ctx.stub_cycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    if (ctx.src_dtype == "fp32" || ctx.cols <= kSmallColumnThreshold) {
        return {kFp32OrSmallRowCycles * rows + kFp32OrSmallEndpointCycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    return {kFp16RowCycles * rows + kFp16EndpointCycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateGemvTargetPipe(const PtoCycleContext& ctx)
{
    constexpr std::size_t kMadMArgumentIndex = 3;
    constexpr std::size_t kMadKArgumentIndex = 4;
    constexpr std::size_t kMadNArgumentIndex = 5;
    constexpr uint64_t kGemvM = 1;
    constexpr uint64_t kPaddedN = 16;
    constexpr uint64_t kMacElementsPerCycle = 8;
    constexpr uint64_t kRoundOffsetElements = 4;
    constexpr uint64_t kMinimumCycles = 3;
    if ((ctx.opcode != "TGEMV" && ctx.opcode != "TGEMV_ACC" && ctx.opcode != "TGEMV_BIAS") || ctx.trace == nullptr ||
        CountCceCalls(*ctx.trace, "mad") != 1) {
        return {};
    }
    const CceCallRecord* mad = nullptr;
    for (const auto& call : ctx.trace->cce_calls) {
        if (call.name == "mad") {
            mad = &call;
            break;
        }
    }
    if (mad == nullptr || mad->args.size() <= kMadNArgumentIndex || mad->args[kMadMArgumentIndex] != kGemvM ||
        mad->args[kMadNArgumentIndex] != kPaddedN) {
        return {};
    }
    // The A2/A3 GEMV overload lowers to a single MAD with M=1 and padded N=16;
    // only K changes the MAC-pipe active span.  The CCE completion model
    // retains the MAD head (7/10/14 cycles for K=16/64/128), whereas the
    // repeated MAC active-time contract observes a K/8 steady-state slope.
    const uint64_t k = mad->args[kMadKArgumentIndex];
    uint64_t cycles = (k + kRoundOffsetElements) / kMacElementsPerCycle;
    if (cycles < kMinimumCycles) {
        cycles = kMinimumCycles;
    }
    return {cycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimatePreluTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kMinimumBarrierCount = 3;
    constexpr uint64_t kCyclesPerRow = 17;
    if (ctx.opcode != "TPRELU" || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        CountCceCalls(*ctx.trace, "vmul") == 0 || CountCceCalls(*ctx.trace, "vsel") == 0 ||
        CountCceCalls(*ctx.trace, "pipe_barrier") < kMinimumBarrierCount) {
        return {};
    }
    // TPRELU is TMUL -> TCMPS -> TSEL.  Real A3 execution retains a 17-cycle
    // per-row dependency/mask issue interval that is absent when the three
    // independently calibrated CCE streams are merged by the stub timeline.
    return {ctx.stub_cycles + kCyclesPerRow * static_cast<uint64_t>(ctx.rows), ModelKind::PTO_COMPOSITE_ANALYTIC};
}

inline CycleEstimate EstimatePtoCycles(const PtoCycleContext& ctx)
{
    using Estimator = CycleEstimate (*)(const PtoCycleContext&);
    // Preserve the first matching model and the original estimator priority.
    static constexpr Estimator estimators[] = {
        EstimateColumnReduce,
        EstimateFmod,
        EstimateCompareTargetPipe,
        EstimateMaddTargetPipe,
        EstimateMulaTargetPipe,
        EstimatePowTargetPipe,
        EstimateIntegerBinaryTargetPipe,
        EstimateSerializedVectorLeaf,
        EstimateMultiCallShiftTargetPipe,
        EstimateTinyFp16UnaryFloor,
        EstimateAxpyTargetPipe,
        EstimateNotTargetPipe,
        EstimateSingleCallCvtTargetPipe,
        EstimateGatherPatternTargetPipe,
        EstimateScalarLoopShift,
        EstimateMaskedDivTargetPipe,
        EstimateScalarBitwiseComposite,
        EstimateBinaryXorComposite,
        EstimateBinaryBitwiseRowStream,
        EstimateRemainderTargetPipe,
        EstimateIntegerRowMinMaxTargetPipe,
        EstimateRowArgCompositeTargetPipe,
        EstimateRowSumTargetPipe,
        EstimateRowExpandDivTargetPipe,
        EstimateRowExpandAddSubTargetPipe,
        EstimateRowExpandMulMinMaxTargetPipe,
        EstimateRowExpandExpDifTargetPipe,
        EstimateRecipTargetPipe,
        EstimateUbToUbTargetPipe,
        EstimateConcatTargetPipe,
        EstimateGemvTargetPipe,
        EstimateColArgTargetPipe,
        EstimatePartArgTargetPipe,
        EstimatePreluTargetPipe,
        EstimateDequantTargetPipe,
        EstimateSelsTargetPipe,
        EstimateExpandsTargetPipe,
        EstimateImg2ColTargetPipe,
    };
    for (const auto estimate : estimators) {
        if (auto result = estimate(ctx)) {
            return result;
        }
    }
    return {};
}

} // namespace pto::mocker::hybrid

#endif
