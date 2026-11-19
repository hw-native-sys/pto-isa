/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef PTO_MOCKER_A2A3_HYBRID_COMPARISON_REDUCTION_H_
#define PTO_MOCKER_A2A3_HYBRID_COMPARISON_REDUCTION_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>

#include <pto/common/cpu_stub.hpp>

#include <pto/costmodel/a2a3/hybrid_costmodel/context.hpp>
#include <pto/costmodel/perf_sim/costmodel_provider.hpp>

namespace pto::mocker::hybrid {

inline bool IsColumnReduceOpcode(std::string_view opcode)
{
    return opcode == "TCOLSUM" || opcode == "TCOLMAX" || opcode == "TCOLMIN" || opcode == "TCOLPROD";
}

inline CycleEstimate EstimateColumnReduce(const PtoCycleContext& ctx)
{
    if (!IsColumnReduceOpcode(ctx.opcode) || ctx.trace == nullptr || ctx.rows <= 1) {
        return {};
    }

    constexpr uint64_t kCompletionBoundaryCycles = 16;
    const uint64_t barrierCount = CountCceCalls(*ctx.trace, "pipe_barrier", kCompletionBoundaryCycles);
    if (ctx.opcode == "TCOLSUM" && barrierCount < static_cast<uint64_t>(ctx.rows - 1)) {
        // The lightweight formula describes the sequential overload, not the
        // binary-tree overload. Preserve the stub result for the latter.
        return {};
    }

    const std::string formulaOpcode = ctx.opcode == "TCOLMIN" ? "TCOLMAX" : std::string(ctx.opcode);
    const uint64_t formulaCycles = ::pto::perf_sim::EstimateLightweightCycles(
        formulaOpcode, ctx.rows, ctx.cols, ctx.src_dtype.empty() ? "unknown" : std::string(ctx.src_dtype));
    if (formulaCycles != 0) {
        return {formulaCycles, ModelKind::PTO_FORMULA};
    }

    // The calibrated 910B1 PIPE_V profile describes a 16-cycle serialized
    // completion boundary. The PTO accuracy benchmark compares target-pipe
    // active time, where the matching A3 increment is 12 cycles.
    constexpr uint64_t kNonTargetPipeBoundaryCycles = 4;
    const uint64_t discount = barrierCount * kNonTargetPipeBoundaryCycles;
    return {
        ctx.stub_cycles > discount ? ctx.stub_cycles - discount : ctx.stub_cycles,
        ModelKind::CCE_TARGET_PIPE_PROJECTION,
    };
}

inline bool IsMeasuredCompareCall(std::string_view name, bool scalar, std::string_view dtype)
{
    const std::string_view prefix = scalar ? "vcmpvs_" : "vcmpv_";
    if (!name.starts_with(prefix)) {
        return false;
    }
    // The 910B1 native sweep covers all FP16/FP32 compare modes. For INT32,
    // only EQ has a matching CANN overload and measured profile.
    return dtype == "fp16" || dtype == "fp32" || (dtype == "int32" && name.ends_with("_eq"));
}

inline CycleEstimate EstimateCompareTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kFp16ElementsPerRepeat = 128;
    constexpr uint64_t kOtherElementsPerRepeat = 64;
    constexpr uint64_t kFp16PredicateBlockRepeats = 2;
    constexpr uint64_t kOtherPredicateBlockRepeats = 4;
    constexpr uint64_t kStartupCycles = 5;
    constexpr uint64_t kTensorRepeatCycles = 2;
    constexpr std::size_t kRepeatArgumentIndex = 3;
    const bool scalar = ctx.opcode == "TCMPS";
    if ((ctx.opcode != "TCMP" && !scalar) || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0) {
        return {};
    }

    const uint64_t elementsPerRepeat = ctx.src_dtype == "fp16" ? kFp16ElementsPerRepeat : kOtherElementsPerRepeat;
    const uint64_t fallbackRepeat = (static_cast<uint64_t>(ctx.cols) + elementsPerRepeat - 1) / elementsPerRepeat;
    uint64_t callCount = 0;
    uint64_t totalRepeats = 0;
    uint64_t partialPredicateCalls = 0;
    bool hasFullPredicateCall = false;
    const uint64_t predicateBlockRepeats =
        ctx.src_dtype == "fp16" ? kFp16PredicateBlockRepeats : kOtherPredicateBlockRepeats;
    for (const auto& call : ctx.trace->cce_calls) {
        if (!IsMeasuredCompareCall(call.name, scalar, ctx.src_dtype)) {
            continue;
        }
        if (call.args.size() <= kRepeatArgumentIndex) {
            return {};
        }
        ++callCount;
        const uint64_t repeats =
            call.args[kRepeatArgumentIndex] == 0 ? fallbackRepeat : call.args[kRepeatArgumentIndex];
        totalRepeats += repeats;
        if (repeats % predicateBlockRepeats == 0) {
            hasFullPredicateCall = true;
        } else {
            ++partialPredicateCalls;
        }
    }
    if (callCount == 0) {
        return {};
    }

    // The calibrated native completion profiles are 5 + 2R (VCMPV) and 5 + R
    // (VCMPVS), plus a ~6-cycle partial predicate-block tail. The accuracy
    // benchmark measures vector-pipe active time, so retain issue work and one
    // source-call startup while excluding that predicate packing completion.
    // A complete predicate output block forms one overlapped vector stream.
    // A partial block forces the packing/observer boundary on every call.
    const uint64_t startupCount = partialPredicateCalls + (hasFullPredicateCall ? 1 : 0);
    uint64_t cycles = kStartupCycles * startupCount + (scalar ? totalRepeats : kTensorRepeatCycles * totalRepeats);
    if (scalar) {
        // VCMPVS has one extra target-pipe issue cycle for every partial
        // predicate-producing call. Complete predicate blocks share a single
        // five-cycle observer endpoint across the stream.
        cycles += partialPredicateCalls + (hasFullPredicateCall ? kStartupCycles : 0);
    }
    return {cycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateNotTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kSmallTensorElements = 256;
    constexpr uint64_t kSmallTensorCycles = 4;
    constexpr uint64_t kScaleNumerator = 11;
    constexpr uint64_t kScaleDenominator = 10;
    constexpr uint64_t kByteTypeRoundOffset = 5;
    constexpr uint64_t kOtherTypeRoundOffset = 9;
    if (ctx.opcode != "TNOT" || ctx.trace == nullptr || ctx.stub_cycles == 0 ||
        CountCceCalls(*ctx.trace, "vnot") == 0) {
        return {};
    }

    // The native UINT16 VNOT profile is 2 + R completion cycles. PTO's
    // repeated target-pipe stream overlaps the completion head, while the A3
    // active-time counter retains roughly a 10% issue/observer allowance. Do
    // not charge the two-cycle completion startup for every row.
    if (ctx.src_dtype == "int32" && ctx.rows > 0 && ctx.cols > 0 &&
        static_cast<uint64_t>(ctx.rows) * static_cast<uint64_t>(ctx.cols) <= kSmallTensorElements) {
        return {kSmallTensorCycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    const uint64_t numerator = ctx.stub_cycles * kScaleNumerator;
    const bool byteType = ctx.src_dtype == "int8" || ctx.src_dtype == "uint8";
    const uint64_t cycles = byteType ? (numerator + kByteTypeRoundOffset) / kScaleDenominator :
                                       (numerator + kOtherTypeRoundOffset) / kScaleDenominator;
    return {cycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateIntegerRowMinMaxTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kNonTargetCyclesPerBarrier = 4;
    const bool isMax = ctx.opcode == "TROWMAX";
    if ((!isMax && ctx.opcode != "TROWMIN") || ctx.trace == nullptr ||
        (ctx.src_dtype != "int16" && ctx.src_dtype != "int32") || ctx.stub_cycles == 0) {
        return {};
    }
    const std::string_view leaf = isMax ? "vmax" : "vmin";
    const uint64_t leaves = CountCceCalls(*ctx.trace, leaf);
    const uint64_t barriers = CountCceCalls(*ctx.trace, "pipe_barrier");
    if (leaves == 0 || barriers < leaves) {
        return {};
    }
    // The integer lowering is a row-wise chain of native VMAX/VMIN leaves
    // separated by serialized completion barriers.  A3 aiv_vec_time observes
    // twelve of each sixteen barrier cycles on PIPE_V, so remove only the
    // four-cycle observer/non-target part.  In this dependent row chain, one
    // cycle of the integer leaf completion head is also an observer endpoint;
    // The repeat-dependent compute term remains unchanged.
    const uint64_t discount = kNonTargetCyclesPerBarrier * barriers + leaves;
    return {
        ctx.stub_cycles > discount ? ctx.stub_cycles - discount : ctx.stub_cycles,
        ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateRowArgCompositeTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kBarrierCompletionCycles = 16;
    constexpr uint64_t kNonTargetCyclesPerBarrier = 4;
    if ((ctx.opcode != "TROWARGMAX" && ctx.opcode != "TROWARGMIN") || ctx.trace == nullptr || ctx.stub_cycles == 0) {
        return {};
    }
    const uint64_t reduceLeaves = CountCceCalls(*ctx.trace, "vreducev2");
    const std::string_view compareLeaf = ctx.opcode == "TROWARGMAX" ? "vcmax" : "vcmin";
    const uint64_t compareLeaves = CountCceCalls(*ctx.trace, compareLeaf);
    const uint64_t barriers = CountCceCalls(*ctx.trace, "pipe_barrier", kBarrierCompletionCycles);
    if (reduceLeaves == 0 || compareLeaves <= 1 || barriers == 0) {
        // A single VCMAX has a different endpoint contract and is left on the
        // native CCE leaf until that endpoint is measured explicitly.
        return {};
    }
    const uint64_t discount = kNonTargetCyclesPerBarrier * barriers;
    return {
        ctx.stub_cycles > discount ? ctx.stub_cycles - discount : ctx.stub_cycles,
        ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateRowSumTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kBarrierCompletionCycles = 16;
    constexpr uint64_t kNonTargetCyclesPerBarrier = 4;
    constexpr uint64_t kIntegerLeafObserverCycles = 2;
    if (ctx.opcode != "TROWSUM" || ctx.trace == nullptr || ctx.stub_cycles == 0) {
        return {};
    }
    const uint64_t vadds = CountCceCalls(*ctx.trace, "vadd");
    const uint64_t vcadds = CountCceCalls(*ctx.trace, "vcadd");
    const uint64_t barriers = CountCceCalls(*ctx.trace, "pipe_barrier", kBarrierCompletionCycles);
    if ((vadds == 0 && vcadds == 0) || barriers == 0) {
        return {};
    }
    uint64_t discount = kNonTargetCyclesPerBarrier * barriers;
    if (ctx.src_dtype == "int32") {
        // Integer rows lower to many dependent single-repeat VADD leaves.
        // The seven-cycle CCE completion is five target-pipe cycles plus a
        // two-cycle per-call observer head; only the former belongs in the
        // measured PIPE_V active span.
        discount += kIntegerLeafObserverCycles * vadds;
    }
    return {
        ctx.stub_cycles > discount ? ctx.stub_cycles - discount : ctx.stub_cycles,
        ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateBinaryXorComposite(const PtoCycleContext& ctx)
{
    constexpr uint64_t kInt16ElementsPerRepeat = 128;
    constexpr uint64_t kInt32ElementsPerRepeat = 64;
    constexpr uint64_t kAndLeavesPerXor = 2;
    constexpr uint64_t kRowWiseCyclesPerRow = 9;
    constexpr uint64_t kRowWiseEndpointCycles = 10;
    constexpr uint64_t kGeneralCyclesPerRepeat = 7;
    constexpr uint64_t kGeneralEndpointCycles = 33;
    if (ctx.opcode != "TXOR" || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        (ctx.src_dtype != "int16" && ctx.src_dtype != "int32")) {
        return {};
    }
    const uint64_t rows = static_cast<uint64_t>(ctx.rows);
    const uint64_t width = ctx.src_dtype == "int16" ? kInt16ElementsPerRepeat : kInt32ElementsPerRepeat;
    const uint64_t repeatsPerRow = (static_cast<uint64_t>(ctx.cols) + width - 1) / width;
    if (CountCceCalls(*ctx.trace, "vor") == 0 || CountCceCalls(*ctx.trace, "vand") < kAndLeavesPerXor ||
        CountCceCalls(*ctx.trace, "vnot") == 0 || CountCceCalls(*ctx.trace, "pipe_barrier") == 0) {
        return {};
    }
    const bool rowWiseInt32 = ctx.src_dtype == "int32" && ctx.cols == kInt32ElementsPerRepeat &&
                              CountCceCalls(*ctx.trace, "vor") == rows &&
                              CountCceCalls(*ctx.trace, "vand") == kAndLeavesPerXor * rows;
    if (rowWiseInt32) {
        return {kRowWiseCyclesPerRow * rows + kRowWiseEndpointCycles, ModelKind::PTO_COMPOSITE_ANALYTIC};
    }
    // XOR is lowered to the UINT16 bit-view OR/AND/NOT/AND graph.  Use the
    // calibrated native issue rates on the dependency path and charge a single
    // public-call observer endpoint instead of summing four completions.
    return {kGeneralCyclesPerRepeat * rows * repeatsPerRow + kGeneralEndpointCycles, ModelKind::PTO_COMPOSITE_ANALYTIC};
}

inline CycleEstimate EstimateBinaryBitwiseRowStream(const PtoCycleContext& ctx)
{
    constexpr int kInt32RowWidth = 64;
    constexpr int kRowStreamThreshold = 64;
    constexpr uint64_t kCyclesPerRow = 3;
    constexpr uint64_t kEndpointCycles = 9;
    if ((ctx.opcode != "TAND" && ctx.opcode != "TOR") || ctx.trace == nullptr || ctx.src_dtype != "int32" ||
        ctx.cols != kInt32RowWidth || ctx.rows < kRowStreamThreshold) {
        return {};
    }
    const std::string_view leaf = ctx.opcode == "TAND" ? "vand" : "vor";
    if (CountCceCalls(*ctx.trace, leaf) != static_cast<uint64_t>(ctx.rows)) {
        return {};
    }
    // At the 64-call split the UINT16 bit-view lowers one native leaf per row.
    // Project the calibrated two-cycle issue beat through the repeated-row bank
    // phase (one extra beat per row) and retain one shared observer endpoint.
    return {kCyclesPerRow * static_cast<uint64_t>(ctx.rows) + kEndpointCycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateScalarBitwiseComposite(const PtoCycleContext& ctx)
{
    constexpr uint64_t kInt16ElementsPerRepeat = 128;
    constexpr uint64_t kInt32ElementsPerRepeat = 64;
    constexpr uint64_t kAndLeavesPerXor = 2;
    constexpr uint64_t kSimpleCyclesPerRepeat = 3;
    constexpr uint64_t kSimpleEndpointCycles = 18;
    constexpr uint64_t kSimpleRowWiseCyclesPerRow = 6;
    constexpr uint64_t kSimpleRowWiseEndpointCycles = 12;
    constexpr uint64_t kXorCyclesPerRepeat = 9;
    constexpr uint64_t kXorEndpointCycles = 62;
    constexpr uint64_t kXorRowWiseCyclesPerRow = 14;
    constexpr uint64_t kXorRowWiseEndpointCycles = 22;
    constexpr int kInt32RowWidth = 64;
    constexpr int kRowWiseThreshold = 64;
    constexpr uint64_t kXorBarrierCount = 5;
    const bool simple = ctx.opcode == "TANDS" || ctx.opcode == "TORS";
    const bool xorScalar = ctx.opcode == "TXORS";
    if ((!simple && !xorScalar) || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        (ctx.src_dtype != "int16" && ctx.src_dtype != "int32")) {
        return {};
    }
    const uint64_t rows = static_cast<uint64_t>(ctx.rows);
    const uint64_t elementsPerRepeat = ctx.src_dtype == "int16" ? kInt16ElementsPerRepeat : kInt32ElementsPerRepeat;
    const uint64_t repeatsPerRow = (static_cast<uint64_t>(ctx.cols) + elementsPerRepeat - 1) / elementsPerRepeat;

    if (simple) {
        const std::string_view leaf = ctx.opcode == "TANDS" ? "vand" : "vor";
        const uint64_t leafCalls = CountCceCalls(*ctx.trace, leaf);
        const bool rowWiseInt32 = ctx.src_dtype == "int32" && ctx.cols == kInt32RowWidth && leafCalls == rows;
        if (CountCceCalls(*ctx.trace, "vector_dup") != rows || (leafCalls != 1 && !rowWiseInt32) ||
            CountCceCalls(*ctx.trace, "pipe_barrier") != 1) {
            return {};
        }
        // Scalar expansion issues one duplicate per row; the UINT16 VAND/VOR
        // bit-view then consumes two issue slots per 256-byte repeat.  On the
        // A3 target-pipe counter the combined steady stream is three cycles per
        // row-repeat plus one shared 18-cycle observer endpoint.
        uint64_t cycles = kSimpleCyclesPerRepeat * rows * repeatsPerRow + kSimpleEndpointCycles;
        // The INT32 64-column path in the current lowering advances its UINT16
        // repeat by four blocks.  That is half of an INT32 row and creates a
        // serialized overlap once the row stream reaches 64 repeats.
        if (rowWiseInt32 && ctx.rows >= kRowWiseThreshold) {
            cycles = kSimpleRowWiseCyclesPerRow * rows + kSimpleRowWiseEndpointCycles;
        }
        return {cycles, ModelKind::PTO_COMPOSITE_ANALYTIC};
    }

    const bool rowWiseInt32 = ctx.src_dtype == "int32" && ctx.cols == kInt32RowWidth &&
                              CountCceCalls(*ctx.trace, "vor") == rows &&
                              CountCceCalls(*ctx.trace, "vand") == kAndLeavesPerXor * rows;
    if (CountCceCalls(*ctx.trace, "vector_dup") != kAndLeavesPerXor * rows ||
        (CountCceCalls(*ctx.trace, "vor") != 1 && !rowWiseInt32) || CountCceCalls(*ctx.trace, "vnot") != 1 ||
        (CountCceCalls(*ctx.trace, "vand") != kAndLeavesPerXor && !rowWiseInt32) ||
        CountCceCalls(*ctx.trace, "pipe_barrier") != kXorBarrierCount) {
        return {};
    }
    // TXORS is TORS + TANDS + VNOT + final VAND with five serialized
    // boundaries.  Projecting the real dependency graph gives nine issue
    // cycles per row-repeat and one shared endpoint.
    uint64_t cycles = kXorCyclesPerRepeat * rows * repeatsPerRow + kXorEndpointCycles;
    if (rowWiseInt32 && ctx.rows >= kRowWiseThreshold) {
        cycles = kXorRowWiseCyclesPerRow * rows + kXorRowWiseEndpointCycles;
    }
    return {cycles, ModelKind::PTO_COMPOSITE_ANALYTIC};
}

inline CycleEstimate EstimateColArgTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kBoundaryDiscountCycles = 12;
    const bool isMax = ctx.opcode == "TCOLARGMAX";
    const bool isMin = ctx.opcode == "TCOLARGMIN";
    if ((!isMax && !isMin) || ctx.trace == nullptr || ctx.rows <= 1 || ctx.cols <= 0) {
        return {};
    }
    const std::string_view compare = isMax ? "vcmp_ge" : "vcmp_le";
    const uint64_t compareBlocks = CountCceCalls(*ctx.trace, compare);
    if (compareBlocks == 0 || CountCceCalls(*ctx.trace, "vsel") < compareBlocks) {
        return {};
    }
    // The lowering serializes compare -> index select -> value update for
    // every source-row/block pair.  The native model supplies each CCE
    // completion, but aiv_vec_time retains 12 fewer completion cycles at each
    // internal boundary.  Preserve all leaf slopes and remove only that
    // repeated public-PTO-internal boundary charge.
    const uint64_t discount = kBoundaryDiscountCycles * compareBlocks;
    return {
        ctx.stub_cycles > discount ? ctx.stub_cycles - discount : ctx.stub_cycles,
        ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimatePartArgTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kSelectCallsPerBlock = 2;
    constexpr uint64_t kMaxBoundaryDiscountCycles = 19;
    constexpr uint64_t kMinBoundaryDiscountCycles = 20;
    const bool isMax = ctx.opcode == "TPARTARGMAX";
    const bool isMin = ctx.opcode == "TPARTARGMIN";
    if ((!isMax && !isMin) || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0) {
        return {};
    }
    const std::string_view compare = isMax ? "vcmp_ge" : "vcmp_le";
    const uint64_t compareBlocks = CountCceCalls(*ctx.trace, compare);
    if (compareBlocks == 0 || CountCceCalls(*ctx.trace, "vsel") != kSelectCallsPerBlock * compareBlocks) {
        return {};
    }
    // Each block is CMPMASK followed by value and index VSEL.  In the target
    // pipe these three issue streams share their completion boundary; summing
    // isolated CCE completions overcharge 19 cycles for GE and 20 for LE.
    const uint64_t discount = (isMax ? kMaxBoundaryDiscountCycles : kMinBoundaryDiscountCycles) * compareBlocks;
    return {
        ctx.stub_cycles > discount ? ctx.stub_cycles - discount : ctx.stub_cycles, ModelKind::PTO_COMPOSITE_ANALYTIC};
}

inline CycleEstimate EstimateSelsTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kLongRowThreshold = 32;
    constexpr uint64_t kMaskBoundaryElements = 64;
    constexpr uint64_t kB32ElementsPerRepeat = 64;
    constexpr uint64_t kB16ElementsPerRepeat = 128;
    constexpr uint64_t kLongRowCyclesPerRow = 3;
    constexpr uint64_t kLongRowEndpointCycles = 9;
    constexpr uint64_t kLongB32AdditionalRepeatCycles = 11;
    constexpr uint64_t kShortB32SubBoundaryCycles = 36;
    constexpr uint64_t kShortB32BaseCycles = 47;
    constexpr uint64_t kShortB32AdditionalRepeatCycles = 21;
    constexpr uint64_t kShortB16SubBoundaryCycles = 50;
    constexpr uint64_t kShortB16BaseCycles = 41;
    constexpr uint64_t kShortB16AdditionalRepeatCycles = 25;
    if (ctx.opcode != "TSELS" || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        CountCceCalls(*ctx.trace, "vsel") != static_cast<uint64_t>(ctx.rows)) {
        return {};
    }
    const uint64_t dtypeBytes = DTypeBytes(ctx.dst_dtype);
    if (dtypeBytes != sizeof(uint16_t) && dtypeBytes != sizeof(uint32_t)) {
        return {};
    }
    const uint64_t rows = static_cast<uint64_t>(ctx.rows);
    const uint64_t cols = static_cast<uint64_t>(ctx.cols);
    const bool isB32 = dtypeBytes == sizeof(uint32_t);
    const uint64_t elementsPerRepeat = isB32 ? kB32ElementsPerRepeat : kB16ElementsPerRepeat;
    uint64_t cycles;
    if (rows >= kLongRowThreshold) {
        cycles = kLongRowCyclesPerRow * rows + kLongRowEndpointCycles;
        if (isB32 && cols > elementsPerRepeat) {
            cycles += kLongB32AdditionalRepeatCycles * ((cols + elementsPerRepeat - 1) / elementsPerRepeat - 1);
        }
    } else if (isB32) {
        cycles = cols < kMaskBoundaryElements ?
                     kShortB32SubBoundaryCycles :
                     kShortB32BaseCycles +
                         kShortB32AdditionalRepeatCycles * ((cols + elementsPerRepeat - 1) / elementsPerRepeat - 1);
    } else {
        cycles = cols < kMaskBoundaryElements ?
                     kShortB16SubBoundaryCycles :
                     kShortB16BaseCycles +
                         kShortB16AdditionalRepeatCycles * ((cols + elementsPerRepeat - 1) / elementsPerRepeat - 1);
    }
    // TSELS writes the scalar through PIPE_S, transfers it to CMPMASK, then
    // issues one tensor-scalar VSEL per row.  The implementation bit-casts all
    // 32-bit payloads to float and all 16-bit payloads to half, so integer and
    // floating-point types of the same width share the same VSEL path.  The
    // measured sub-64 mask path is slower than the aligned 64-element boundary;
    // preserve that non-monotonic transition rather than treating it as a repeat
    // boundary.  This PTO-level schedule must not change the native VSEL leaf.
    return {cycles, ModelKind::PTO_COMPOSITE_ANALYTIC};
}

} // namespace pto::mocker::hybrid

#endif
