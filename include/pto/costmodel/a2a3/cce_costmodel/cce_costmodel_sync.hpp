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

inline void dsb(auto barrierType)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("dsb", cycles, barrierType);
}
inline void ffts_cross_core_sync(auto srcPipe, auto msg)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("ffts_cross_core_sync", cycles, srcPipe, msg);
}
inline void pipe_barrier(auto pipe)
{
    FlushTailsForPipe(pipe);
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("pipe_barrier", cycles, pipe);
}
inline void set_atomic_add()
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_atomic_add", cycles);
}
inline void set_atomic_bf16()
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_atomic_bf16", cycles);
}
inline void set_atomic_f16()
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_atomic_f16", cycles);
}
inline void set_atomic_f32()
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_atomic_f32", cycles);
}
inline void set_atomic_none()
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_atomic_none", cycles);
}
inline void set_atomic_s16()
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_atomic_s16", cycles);
}
inline void set_atomic_s32()
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_atomic_s32", cycles);
}
inline void set_atomic_s8()
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_atomic_s8", cycles);
}
inline void set_cmpmask(auto cmpMaskPtr)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_cmpmask", cycles, cmpMaskPtr);
}
inline void set_ctrl(auto ctrl)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_ctrl", cycles, ctrl);
}
inline void set_deqscale(auto scale)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_deqscale", cycles, scale);
}
inline void set_ffts_base_addr(auto fftsAddr)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_ffts_base_addr", cycles, fftsAddr);
}
inline void set_flag(auto srcPipe, auto dstPipe, auto token)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_flag", cycles, srcPipe, dstPipe, token);
}
inline void set_fmatrix(auto regFmatrix)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_fmatrix", cycles, regFmatrix);
}
inline void set_fmatrix_b(auto regFmatrix)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_fmatrix_b", cycles, regFmatrix);
}
inline void set_fpc(auto deqTensorAddr)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_fpc", cycles, deqTensorAddr);
}
inline void set_mask_count()
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_mask_count", cycles);
}
inline void set_mask_norm()
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_mask_norm", cycles);
}
inline void set_mov_pad_val(auto value)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_mov_pad_val", cycles, value);
}
inline void set_nd_para(auto ndParaSPR)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_nd_para", cycles, ndParaSPR);
}
inline void set_padding(auto paddingValue)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_padding", cycles, paddingValue);
}
inline void set_quant_pre(auto preQuantScalar)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_quant_pre", cycles, preQuantScalar);
}
inline void set_va_reg_sb(auto vaReg, auto addrArray)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_va_reg_sb", cycles, vaReg, addrArray);
}
inline void set_vector_mask(auto mask0, auto mask1)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("set_vector_mask", cycles, mask0, mask1);
}
inline void wait_flag(auto srcPipe, auto dstPipe, auto token)
{
    FlushTailsForPipe(srcPipe);
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("wait_flag", cycles, srcPipe, dstPipe, token);
}
inline void wait_flag_dev(auto flagId)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("wait_flag_dev", cycles, flagId);
}
