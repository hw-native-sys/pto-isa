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

inline void vconv_bf162f32(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_bf162f32", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_bf162s32a(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                            auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_bf162s32a", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_bf162s32c(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                            auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_bf162s32c", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_bf162s32f(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                            auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_bf162s32f", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_bf162s32r(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                            auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_bf162s32r", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_bf162s32z(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                            auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_bf162s32z", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_deq(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride, auto dstRepeatStride,
                      auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_deq", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162f32(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                          auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162f32", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s16a(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s16a", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s16c(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s16c", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s16f(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s16f", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s16r(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s16r", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s16z(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s16z", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s32a(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s32a", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s32c(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s32c", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s32f(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s32f", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s32r(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s32r", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s32z(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s32z", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s8a(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                          auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s8a", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s8c(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                          auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s8c", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s8f(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                          auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s8f", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s8r(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                          auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s8r", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s8z(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                          auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s8z", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s4(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                         auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s4", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s4a(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                          auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s4a", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s4c(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                          auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s4c", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s4f(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                          auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s4f", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s4r(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                          auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s4r", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162s4z(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                          auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162s4z", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162u8a(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                          auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162u8a", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162u8c(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                          auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162u8c", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162u8f(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                          auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162u8f", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162u8r(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                          auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162u8r", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f162u8z(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                          auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f162u8z", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322bf16a(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                            auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322bf16a", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322bf16c(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                            auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322bf16c", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322bf16f(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                            auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322bf16f", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322bf16r(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                            auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322bf16r", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322bf16z(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                            auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322bf16z", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322f16(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                          auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322f16", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322f16a(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322f16a", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322f16c(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322f16c", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322f16f(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322f16f", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322f16o(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322f16o", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322f16r(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322f16r", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
inline void vconv_f322f16z(auto dst, auto src, auto repeat, auto dstBlockStride, auto srcBlockStride,
                           auto dstRepeatStride, auto srcRepeatStride)
{
    const uint64_t cycles = _EstimateVconvCycles(repeat);
    ::pto::mocker::RecordCceCall(::pto::mocker::evaluator::PipeKey::VECTOR, "vconv_f322f16z", cycles, dst, src, repeat,
                                 dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}
