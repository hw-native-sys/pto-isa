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

inline void vconv_f322f32a(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322f32a", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322f32c(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322f32c", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322f32f(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322f32f", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322f32r(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322f32r", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322f32z(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322f32z", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322s16a(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322s16a", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322s16c(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322s16c", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322s16f(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322s16f", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322s16r(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322s16r", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322s16z(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322s16z", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322s32a(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322s32a", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322s32c(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322s32c", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322s32f(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322s32f", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322s32r(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322s32r", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322s32z(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322s32z", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322s64a(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322s64a", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322s64c(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322s64c", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322s64f(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322s64f", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322s64r(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322s64r", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322s64z(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322s64z", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_s162f16(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                          auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s162f16", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_s162f16a(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s162f16a", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_s162f16c(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s162f16c", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_s162f16f(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s162f16f", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_s162f16r(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s162f16r", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_s162f16z(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s162f16z", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_s162f32(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                          auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s162f32", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_s322f32(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                          auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s322f32", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_s322f32a(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s322f32a", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_s322f32c(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s322f32c", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_s322f32f(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s322f32f", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_s322f32r(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s322f32r", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_s322f32z(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s322f32z", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_s322s16(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                          auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s322s16", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_s322s64(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                          auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s322s64", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_s642f32a(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s642f32a", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_s642f32c(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s642f32c", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_s642f32f(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s642f32f", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_s642f32r(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s642f32r", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_s642f32z(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s642f32z", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_s642s32(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                          auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s642s32", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_s82f16(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                         auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s82f16", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_s42f16(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                         auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_s42f16", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_u82f16(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                         auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_u82f16", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
