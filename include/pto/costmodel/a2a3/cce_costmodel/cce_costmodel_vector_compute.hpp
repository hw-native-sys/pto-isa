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

#include <pto/costmodel/a2a3/cce_costmodel/cce_costmodel_core.hpp>

inline void scatter_vnchwconv_b16(auto dst, auto src, auto repeat, auto dstStride, auto srcStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "scatter_vnchwconv_b16", cycles, dst, src, repeat, dstStride,
        srcStride);
}
inline void scatter_vnchwconv_b32(auto dst, auto src, auto repeat, auto dstStride, auto srcStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "scatter_vnchwconv_b32", cycles, dst, src, repeat, dstStride,
        srcStride);
}
inline void scatter_vnchwconv_b8(
    auto dst, auto src, auto repeat, auto dstStride, auto srcStride, auto dstHighHalf, auto srcHighHalf)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "scatter_vnchwconv_b8", cycles, dst, src, repeat, dstStride,
        srcStride, dstHighHalf, srcHighHalf);
}
inline void vabs(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 13, 1, 16);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vabs", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
        dstRepeatStride, srcRepeatStride);
}
inline void vadd(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride,
    auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    // 910B3 标定 (fp32, TADD, dav-2201): measured = 2.11·repeat + 24 (R²=1.0).
    // Was (slope=1) -> 2x underestimate at large cols. slope 1->2; fixed 32->24.
    // (head+tail split is an unmeasured assumption: single-op data fixes only the sum.)
    const uint64_t cycles = EstimateLinearCycles(repeat, 10, 2, 14) + EstimateCountModeFloor();
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vadd", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vadds(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride,
    auto src0RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 13, 1, 18);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vadds", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, dstRepeatStride, src0RepeatStride);
}
inline void vand(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride,
    auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    // 910B3 标定 (int16, TAND, dav-2201): measured = 2.03·repeat + 20 (R²=0.999).
    // Bitwise AND: slope 2 kept, fixed 6->20. vor is identical HW cost (TOR fits the same
    // line). (head/tail split is an unmeasured assumption: single-op data fixes only the sum.)
    const uint64_t cycles = EstimateLinearCycles(repeat, 8, 2, 12) + EstimateCountModeFloor();
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vand", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vaxpy(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride,
    auto src0RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vaxpy", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, dstRepeatStride, src0RepeatStride);
}
inline void vbitsort(auto dst, auto src, auto idx, auto repeat)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vbitsort", cycles, dst, src, idx, repeat);
}
inline void vbrcb(auto dst, auto src, auto dstBlockStride, auto dstRepeatStride, auto repeat)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 0, 0, 18);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vbrcb", cycles, dst, src, dstBlockStride, dstRepeatStride, repeat);
}
inline void vcadd(
    auto dst, auto src, auto repeat, auto dstRepeatStride, auto srcBlockStride, auto srcRepeatStride, auto mode)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 14, 7, 32);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vcadd", cycles, dst, src, repeat, dstRepeatStride, srcBlockStride,
        srcRepeatStride, mode);
}
inline void vcgadd(auto dst, auto src, auto repeat, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 14, 1, 24);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vcgadd", cycles, dst, src, repeat, dstRepeatStride,
        src0RepeatStride, src1RepeatStride);
}
inline void vcgmax(auto dst, auto src, auto repeat, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 14, 1, 17);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vcgmax", cycles, dst, src, repeat, dstRepeatStride,
        src0RepeatStride, src1RepeatStride);
}
inline void vcgmin(auto dst, auto src, auto repeat, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 14, 1, 17);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vcgmin", cycles, dst, src, repeat, dstRepeatStride,
        src0RepeatStride, src1RepeatStride);
}
inline void vcmax(
    auto dst, auto src, auto repeat, auto dstRepeatStride, auto srcBlockStride, auto srcRepeatStride, auto mode)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vcmax", cycles, dst, src, repeat, dstRepeatStride, srcBlockStride,
        srcRepeatStride, mode);
}
inline void vcmin(
    auto dst, auto src, auto repeat, auto dstRepeatStride, auto srcBlockStride, auto srcRepeatStride, auto mode)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vcmin", cycles, dst, src, repeat, dstRepeatStride, srcBlockStride,
        srcRepeatStride, mode);
}
inline void vcopy(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 11, 1, 13);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vcopy", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
        dstRepeatStride, srcRepeatStride);
}
inline void vdiv(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride,
    auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    // 910B3 标定 (fp32, TDIV DEFAULT div-mode, dav-2201): measured = 4.03·repeat + 31 (R²=1.0).
    // Was (slope=8) -> 2x overestimate. slope 8->4. NOTE: high-precision div modes would be higher;
    // costmodel has a single vdiv (no mode param), so this models the common DEFAULT path.
    const uint64_t cycles = EstimateLinearCycles(repeat, 11, 4, 19) + EstimateCountModeFloor();
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vdiv", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vector_dup(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 11, 1, 13);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vector_dup", cycles, dst, src, repeat, dstBlockStride,
        srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vexp(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    // 910B3 标定 (fp32, TEXP, dav-2201): measured = 2.00·repeat + 31 (R²=1.0, rep 1-128).
    // Was the uncalibrated placeholder (slope=4, head+tail=37) -> 1.9x over at rep128
    // (mock 545 vs real 287). slope 4->2; head+tail 37->31 (isolated op = slope*rep+head+tail,
    // see EstimateLinearCycles). slope is the real lever; head+tail sum fixed by rep1=33.
    // (head/tail split is an unmeasured assumption: single-op data fixes only head+tail+slope.)
    const uint64_t cycles = EstimateLinearCycles(repeat, 13, 2, 18);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vexp", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
        dstRepeatStride, srcRepeatStride);
}
inline void vgather(auto dst, auto offset, auto srcBaseAddr, auto dstRepeatStride, auto repeat)
{
    // 910B3 标定 (fp32, TGATHER counter-mode, dav-2201): indirect-address gather.
    // TGATHER issues exactly one vgather/row (counter mode, repeat=1); measured
    // per-row marginal = 46.6 cyc/row. Mock per-row = vmuls(1)+barrier(20)+vgather,
    // which matches with vgather head 6->24 (vgather≈26 cyc/call). Slope (per-repeat)
    // stays 2: TGATHER never uses repeat>1, so slope is unverified.
    const uint64_t cycles = EstimateLinearCycles(repeat, /*head=*/24, /*slope=*/2);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vgather", cycles, dst, offset, srcBaseAddr, dstRepeatStride,
        repeat);
}
inline void vgatherb(auto dst, auto offset, auto srcBaseAddr, auto dstRepeatStride, auto dstBlockStride, auto repeat)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vgatherb", cycles, dst, offset, srcBaseAddr, dstRepeatStride,
        dstBlockStride, repeat);
}
inline void vln(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    // 910B3 标定 (fp32, TLOG, dav-2201): measured = 2.00·repeat + 33 (R²=1.0).
    // Natural log (DEFAULT precision -> single vln). slope 2 kept; fixed 6->33 — the
    // high one-time overhead is the transcendental setup/issue latency.
    const uint64_t cycles = EstimateLinearCycles(repeat, 13, 2, 20);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vln", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
        dstRepeatStride, srcRepeatStride);
}
inline void vlrelu(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride,
    auto src0RepeatStride)
{
    // 910B3 标定 (fp32, TLRELU, dav-2201): measured = 1.03·repeat + 26 (R²=1.0).
    // Leaky ReLU: slope 2->1, fixed 6->26.
    const uint64_t cycles = EstimateLinearCycles(repeat, 11, 1, 15);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vlrelu", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, dstRepeatStride, src0RepeatStride);
}
inline void vmax(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride,
    auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    // 910B3 标定 (fp32, TMAX, dav-2201): measured = 2.01·repeat + 23 (R²=1.0).
    // slope 2 confirmed; fixed 30->24.
    const uint64_t cycles = EstimateLinearCycles(repeat, 10, 2, 14) + EstimateCountModeFloor();
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vmax", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vmaxs(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride,
    auto src0RepeatStride)
{
    // 910B3 标定 (fp32, TMAXS, dav-2201): measured = 1.00·repeat + 23 (R²=1.0).
    // Scalar-max: slope 2->1, fixed 6->23. Same cheap class as vrelu (identical fit).
    const uint64_t cycles = EstimateLinearCycles(repeat, 10, 1, 13);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vmaxs", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, dstRepeatStride, src0RepeatStride);
}
inline void vmin(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride,
    auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    // 910B3 标定: vmin is vmax's twin (identical HW cost). Recalibrated to match vmax
    // (slope 2, fixed 24) by symmetry — TMIN unmeasured but shares the vmin instruction.
    const uint64_t cycles = EstimateLinearCycles(repeat, 10, 2, 14) + EstimateCountModeFloor();
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vmin", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vmins(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride,
    auto src0RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 14, 1, 16);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vmins", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, dstRepeatStride, src0RepeatStride);
}
inline void vmrgsort4(auto dst, auto addrArray, auto count, auto config)
{
    // 910B3 标定 (fp32, blockLen=64 single-source merge, dav-2201): 4-way merge.
    // Measured = 78·repeat + 20 -> slope=78, head=20.
    // Was the bare placeholder (slope=2, head=6) -> 39x underestimate.
    // NOTE: only calibrated for single-source (LIST_NUM_1); multi-source paths use
    // repeat=1 and their cost is driven by `count` (data volume), not modeled here.
    const uint64_t cycles = EstimateLinearCycles(ExtractBits(config, 0, 0xffULL), /*head=*/20, /*slope=*/78);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vmrgsort4", cycles, dst, addrArray, count, config);
}
inline void vmul(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride,
    auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    // 910B3 标定 (fp32, TMUL, dav-2201): measured slope 2.3, fixed 25 (clean norm-mode).
    // slope 2 kept; fixed 32->25. This also fixes TMUL clean-shape precision (was floor-blamed).
    const uint64_t cycles = EstimateLinearCycles(repeat, 10, 2, 15) + EstimateCountModeFloor();
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vmul", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vmuls(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride,
    auto src0RepeatStride)
{
    // 910B3 标定 (fp32, TMULS, dav-2201): measured slope 1, fixed 26 (clean norm-mode).
    // slope 1 kept; fixed 33->26. Fixes TMULS clean-shape precision.
    const uint64_t cycles = EstimateLinearCycles(repeat, 11, 1, 15);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vmuls", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, dstRepeatStride, src0RepeatStride);
}
inline void vnot(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    // By symmetry with vand/vor (same bitwise unit): slope 2, fixed ~20.
    // (TNOT kernel did not compile on NPU for direct measurement; bitwise NOT shares
    //  the vector bitwise pipeline cost of AND/OR.)
    const uint64_t cycles = EstimateLinearCycles(repeat, 8, 2, 12);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vnot", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
        dstRepeatStride, srcRepeatStride);
}
inline void vor(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride,
    auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    // 910B3 标定 (int16, TOR, dav-2201): measured = 2.03·repeat + 20 (R²=0.999).
    // Identical to vand (same bitwise unit, cross-validated). slope 2 kept, fixed 6->20.
    const uint64_t cycles = EstimateLinearCycles(repeat, 8, 2, 12) + EstimateCountModeFloor();
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vor", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vreducev2(
    auto dst, auto src0, auto src1, auto repeat, auto src0BlockStride, auto modeOrMaskPattern, auto src0RepeatStride,
    auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 14, 14, 20);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vreducev2", cycles, dst, src0, src1, repeat, src0BlockStride,
        modeOrMaskPattern, src0RepeatStride, src1RepeatStride);
}
inline void vrelu(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    // 910B3 标定 (fp32, TRELU, dav-2201): measured = 1.00·repeat + 23 (R²=1.0).
    // ReLU: slope 2->1, fixed 6->23. Same class as vmaxs (identical fit).
    const uint64_t cycles = EstimateLinearCycles(repeat, 10, 1, 13);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vrelu", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
        dstRepeatStride, srcRepeatStride);
}
inline void vrsqrt(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    // 910B3 标定 (fp32, TRSQRT, dav-2201): measured = 1.07·repeat + 24 (R²=1.0).
    // Reciprocal sqrt: slope 2->1, fixed 6->24.
    const uint64_t cycles = EstimateLinearCycles(repeat, 10, 1, 14);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vrsqrt", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
        dstRepeatStride, srcRepeatStride);
}
inline void vsel(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride,
    auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride, auto mode)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 13, 2, 14);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vsel", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride, mode);
}
inline void vshl(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride,
    auto src0RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vshl", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, dstRepeatStride, src0RepeatStride);
}
inline void vshr(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto dstRepeatStride,
    auto src0RepeatStride, auto isArithmetic = false)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vshr", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, dstRepeatStride, src0RepeatStride, isArithmetic);
}
inline void vsqrt(
    auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
    auto srcRepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 14, 2, 25);
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vsqrt", cycles, dst, src, repeat, dstBlockStride, srcBlockStride,
        dstRepeatStride, srcRepeatStride);
}
inline void vsub(
    auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride, auto src1BlockStride,
    auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    // 910B3 标定 (fp32, TSUB, dav-2201): measured = 2.11·repeat + 24 (R²=1.0).
    // slope 2 confirmed (cross-validated with vadd); fixed 31->24.
    const uint64_t cycles = EstimateLinearCycles(repeat, 10, 2, 14) + EstimateCountModeFloor();
    ::pto::mocker::RecordCceCall(
        ::pto::mocker::evaluator::PipeKey::VECTOR, "vsub", cycles, dst, src0, src1, repeat, dstBlockStride,
        src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
