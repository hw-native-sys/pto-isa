/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSYNC_HPP
#define TSYNC_HPP
#include <pto/common/type.hpp>
#include <pto/common/event.hpp>

namespace pto {
template <Op OpCode>
PTO_INTERNAL static constexpr pipe_t GetPipeByOp()
{
    if constexpr ((OpCode >= static_cast<Op>(0)) && (OpCode <= Op::OP_COUNT)) {
        return opPipeList[static_cast<int>(OpCode)];
    }
    return PIPE_ALL;
}

// single pipeline wait, only support MTE3 or ALL pipeline
template <Op OpCode>
PTO_INTERNAL void TSYNC_IMPL()
{
#ifndef __PTO_AUTO__
    constexpr pipe_t pipe = GetPipeByOp<OpCode>();
    PTO_STATIC_ASSERT((pipe == PIPE_M) || (pipe == PIPE_MTE1) || (pipe == PIPE_MTE2) || (pipe == PIPE_MTE3) ||
                          (pipe == PIPE_ALL) || (pipe == PIPE_FIX),
                      "Single Op TSYNC only supports MTE2 / MTE3 / ALL pipeline.");
    pipe_barrier((pipe_t)pipe);
#endif
}

template <Op SrcOp, Op DstOp, bool AutoToken = true, event_t EventID = EVENT_ID0>
struct Event {
#ifndef __PTO_AUTO__
    static constexpr Op dstOp = DstOp;
    static constexpr Op srcOp = SrcOp;
    static constexpr pipe_t dstPipe = GetPipeByOp<dstOp>();
    static constexpr pipe_t srcPipe = GetPipeByOp<srcOp>();
    static constexpr bool isSamePipe = (srcPipe == dstPipe);
    static constexpr bool isValidBarrierPipe =
        ((dstPipe == PIPE_M) || (dstPipe == PIPE_MTE1) || (dstPipe == PIPE_MTE2) || (dstPipe == PIPE_MTE3) ||
         (dstPipe == PIPE_ALL) || (dstPipe == PIPE_FIX));

    PTO_STATIC_ASSERT(SrcOp != DstOp, "SrcOp is not allowed to be equal to DstOp.");

#ifdef PTO_FLAG_TEST
    CceEventIdType token = {};
#else
    const event_t token = AutoToken ? EventIdCounter<srcPipe, dstPipe>::GetNextId() : EventID;
#endif
#endif

    PTO_INTERNAL Event &InitAddr(uint64_t fftsAddr)
    {
        return *this;
    }

    template <uint8_t CrossCoreId = 0xff>
    PTO_INTERNAL Event &Wait()
    {
#ifndef __PTO_AUTO__
        if constexpr (isSamePipe) {
            if constexpr (isValidBarrierPipe) {
                pipe_barrier((pipe_t)srcPipe);
            }
        } else {
#ifdef PTO_FLAG_TEST
            __pto_wait_flag((pipe_t)srcPipe, (pipe_t)dstPipe, token);
#else
            wait_flag((pipe_t)srcPipe, (pipe_t)dstPipe, token);
#endif
        }
#endif
        return *this;
    }

    template <uint8_t CrossCoreId = 0xff>
    PTO_INTERNAL Event &Init()
    {
#ifndef __PTO_AUTO__
        if constexpr (!isSamePipe) {
#ifdef PTO_FLAG_TEST
            token = __pto_set_flag((pipe_t)srcPipe, (pipe_t)dstPipe);
#else
            set_flag((pipe_t)srcPipe, (pipe_t)dstPipe, token);
#endif
        }
#endif
        return *this;
    }

    PTO_INTERNAL Event() = default;
    PTO_INTERNAL Event(RecordEvent)
    {
        Init();
    }

    PTO_INTERNAL Event &operator=(RecordEvent)
    {
        return Init();
    }

    template <uint8_t CrossCoreId = 0xff>
    PTO_INTERNAL Event &Record()
    {
        return Init();
    }
};
} // namespace pto
#endif
