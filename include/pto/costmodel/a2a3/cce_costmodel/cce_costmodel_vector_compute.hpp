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
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "scatter_vnchwconv_b16", cycles, dst, src,
                                 repeat, dstStride, srcStride);
}
inline void scatter_vnchwconv_b32(auto dst, auto src, auto repeat, auto dstStride, auto srcStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "scatter_vnchwconv_b32", cycles, dst, src,
                                 repeat, dstStride, srcStride);
}
inline void scatter_vnchwconv_b8(auto dst, auto src, auto repeat, auto dstStride, auto srcStride, auto dstHighHalf,
                                 auto srcHighHalf)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "scatter_vnchwconv_b8", cycles, dst, src,
                                 repeat, dstStride, srcStride, dstHighHalf, srcHighHalf);
}
inline void vabs(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
                 auto srcRepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 13, 1, 16);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vabs", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vadd(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride,
                 auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 14, 1, 18);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vadd", cycles, dst, src0, src1, repeat,
                                 dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride,
                                 src1RepeatStride);
}
inline void vadds(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride,
                  auto dstRepeatStride, auto src0RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 13, 1, 18);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vadds", cycles, dst, src0, src1, repeat,
                                 dstBlockStride, src0BlockStride, dstRepeatStride, src0RepeatStride);
}
inline void vand(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride,
                 auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vand", cycles, dst, src0, src1, repeat,
                                 dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride,
                                 src1RepeatStride);
}
inline void vaxpy(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride,
                  auto dstRepeatStride, auto src0RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vaxpy", cycles, dst, src0, src1, repeat,
                                 dstBlockStride, src0BlockStride, dstRepeatStride, src0RepeatStride);
}
inline void vbitsort(auto dst, auto src, auto idx, auto repeat)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vbitsort", cycles, dst, src, idx, repeat);
}
inline void vbrcb(auto dst, auto src, auto dstBlockStride, auto dstRepeatStride, auto repeat)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 0, 0, 18);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vbrcb", cycles, dst, src, dstBlockStride,
                                 dstRepeatStride, repeat);
}
inline void vcadd(auto dst, auto src, auto repeat, auto dstRepeatStride, auto srcBlockStride, auto srcRepeatStride,
                  auto mode)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 14, 7, 32);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vcadd", cycles, dst, src, repeat,
                                 dstRepeatStride, srcBlockStride, srcRepeatStride, mode);
}
inline void vcgadd(auto dst, auto src, auto repeat, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 14, 1, 24);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vcgadd", cycles, dst, src, repeat,
                                 dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vcgmax(auto dst, auto src, auto repeat, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 14, 1, 17);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vcgmax", cycles, dst, src, repeat,
                                 dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vcgmin(auto dst, auto src, auto repeat, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 14, 1, 17);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vcgmin", cycles, dst, src, repeat,
                                 dstRepeatStride, src0RepeatStride, src1RepeatStride);
}
inline void vcmax(auto dst, auto src, auto repeat, auto dstRepeatStride, auto srcBlockStride, auto srcRepeatStride,
                  auto mode)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vcmax", cycles, dst, src, repeat,
                                 dstRepeatStride, srcBlockStride, srcRepeatStride, mode);
}
inline void vcmin(auto dst, auto src, auto repeat, auto dstRepeatStride, auto srcBlockStride, auto srcRepeatStride,
                  auto mode)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vcmin", cycles, dst, src, repeat,
                                 dstRepeatStride, srcBlockStride, srcRepeatStride, mode);
}
inline void vcopy(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
                  auto srcRepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 11, 1, 13);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vcopy", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vdiv(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride,
                 auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 13, 8, 25);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vdiv", cycles, dst, src0, src1, repeat,
                                 dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride,
                                 src1RepeatStride);
}
inline void vector_dup(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
                       auto srcRepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 11, 1, 13);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vector_dup", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vexp(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
                 auto srcRepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 13, 4, 24);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vexp", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vgather(auto dst, auto offset, auto srcBaseAddr, auto dstRepeatStride, auto repeat)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vgather", cycles, dst, offset, srcBaseAddr,
                                 dstRepeatStride, repeat);
}
inline void vgatherb(auto dst, auto offset, auto srcBaseAddr, auto dstRepeatStride, auto dstBlockStride, auto repeat)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vgatherb", cycles, dst, offset,
                                 srcBaseAddr, dstRepeatStride, dstBlockStride, repeat);
}
inline void vln(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
                auto srcRepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vln", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vlrelu(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride,
                   auto dstRepeatStride, auto src0RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vlrelu", cycles, dst, src0, src1, repeat,
                                 dstBlockStride, src0BlockStride, dstRepeatStride, src0RepeatStride);
}
inline void vmax(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride,
                 auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 14, 2, 16);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vmax", cycles, dst, src0, src1, repeat,
                                 dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride,
                                 src1RepeatStride);
}
inline void vmaxs(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride,
                  auto dstRepeatStride, auto src0RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vmaxs", cycles, dst, src0, src1, repeat,
                                 dstBlockStride, src0BlockStride, dstRepeatStride, src0RepeatStride);
}
inline void vmin(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride,
                 auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 14, 2, 16);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vmin", cycles, dst, src0, src1, repeat,
                                 dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride,
                                 src1RepeatStride);
}
inline void vmins(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride,
                  auto dstRepeatStride, auto src0RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 14, 1, 16);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vmins", cycles, dst, src0, src1, repeat,
                                 dstBlockStride, src0BlockStride, dstRepeatStride, src0RepeatStride);
}
inline void vmrgsort4(auto dst, auto addrArray, auto count, auto config)
{
    const uint64_t cycles = EstimateLinearCycles(ExtractBits(config, 0, 0xffULL));
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vmrgsort4", cycles, dst, addrArray, count,
                                 config);
}
inline void vmul(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride,
                 auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 13, 2, 19);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vmul", cycles, dst, src0, src1, repeat,
                                 dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride,
                                 src1RepeatStride);
}
inline void vmuls(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride,
                  auto dstRepeatStride, auto src0RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 14, 1, 19);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vmuls", cycles, dst, src0, src1, repeat,
                                 dstBlockStride, src0BlockStride, dstRepeatStride, src0RepeatStride);
}
inline void vnot(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
                 auto srcRepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vnot", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vor(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride,
                auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vor", cycles, dst, src0, src1, repeat,
                                 dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride,
                                 src1RepeatStride);
}
inline void vreducev2(auto dst, auto src0, auto src1, auto repeat, auto src0BlockStride, auto modeOrMaskPattern,
                      auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 14, 14, 20);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vreducev2", cycles, dst, src0, src1,
                                 repeat, src0BlockStride, modeOrMaskPattern, src0RepeatStride, src1RepeatStride);
}
inline void vrelu(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
                  auto srcRepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vrelu", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vrsqrt(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
                   auto srcRepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vrsqrt", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vsel(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride,
                 auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride, auto mode)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 13, 2, 14);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vsel", cycles, dst, src0, src1, repeat,
                                 dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride,
                                 src1RepeatStride, mode);
}
inline void vshl(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride,
                 auto dstRepeatStride, auto src0RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vshl", cycles, dst, src0, src1, repeat,
                                 dstBlockStride, src0BlockStride, dstRepeatStride, src0RepeatStride);
}
inline void vshr(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride,
                 auto dstRepeatStride, auto src0RepeatStride, auto isArithmetic = false)
{
    const uint64_t cycles = EstimateLinearCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vshr", cycles, dst, src0, src1, repeat,
                                 dstBlockStride, src0BlockStride, dstRepeatStride, src0RepeatStride, isArithmetic);
}
inline void vsqrt(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
                  auto srcRepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 14, 2, 25);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vsqrt", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vsub(auto dst, auto src0, auto src1, auto repeat, auto dstBlockStride, auto src0BlockStride,
                 auto src1BlockStride, auto dstRepeatStride, auto src0RepeatStride, auto src1RepeatStride)
{
    const uint64_t cycles = EstimateLinearCycles(repeat, 13, 2, 18);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vsub", cycles, dst, src0, src1, repeat,
                                 dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride,
                                 src1RepeatStride);
}
