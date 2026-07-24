/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/comm/comm_types.hpp>
#include <pto/comm/pto_comm_instr_impl.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

template <int op_type>
AICORE void runTNotify(__gm__ int32_t __out__* out, __gm__ int32_t __in__* src, __gm__ int32_t __in__* signal)
{
    using NotifyOp = pto::comm::NotifyOp;
    pto::comm::Signal counterSignal(src);
    NotifyOp opType = op_type == 1 ? NotifyOp::AtomicAdd : NotifyOp::Set;

    pto::comm::TNOTIFY_IMPL(counterSignal, signal[0], opType);
    out[0] = counterSignal.data()[0];
}

template <int op_type>
void LaunchTNotify(int32_t* out, int32_t* src, int32_t* signal, void* stream)
{
    runTNotify<op_type>(out, src, signal);
}

template void LaunchTNotify<1>(int32_t* out, int32_t* src, int32_t* signal, void* stream);
template void LaunchTNotify<2>(int32_t* out, int32_t* src, int32_t* signal, void* stream);
