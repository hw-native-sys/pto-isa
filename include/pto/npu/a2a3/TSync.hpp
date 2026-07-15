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

#define FFTS_BASE_COUNT_WIDTH 0xf
#define FFTS_MODE_VAL 0x2
#define FFTS_MODE_WIDTH 0x3
#define FFTS_MODE_OFFSET 4
#define FFTS_EVENT_ID_WIDTH 0xf
#define FFTS_EVENT_ID_OFFSET 8
namespace pto {
template <Op op>
PTO_INTERNAL static constexpr pipe_t GetPipeByOpForA3()
{
    return OpPipeEntry<op>::pipe;
}

// single pipeline wait
template <Op OpCode>
PTO_INTERNAL void TSYNC_IMPL()
{
#ifndef __PTO_AUTO__
    constexpr pipe_t pipe = GetPipeByOpForA3<OpCode>();
    PTO_STATIC_ASSERT(
        (pipe == PIPE_S) || (pipe == PIPE_V) || (pipe == PIPE_M) || (pipe == PIPE_MTE1) || (pipe == PIPE_MTE2) ||
            (pipe == PIPE_MTE3) || (pipe == PIPE_FIX) || (pipe == PIPE_ALL),
        "Single Op TSYNC only supports S / V / M / MTE1 / MTE2 / MTE3 / FIX / ALL pipeline.");
    pipe_barrier((pipe_t)pipe);
#endif
}

PTO_INTERNAL uint16_t getFFTSMsg(uint16_t mode, uint16_t eventId, uint16_t baseConst = 0x1)
{
    return (
        (baseConst & FFTS_BASE_COUNT_WIDTH) + ((mode & FFTS_MODE_WIDTH) << FFTS_MODE_OFFSET) +
        ((eventId & FFTS_EVENT_ID_WIDTH) << FFTS_EVENT_ID_OFFSET));
}

template <Op SrcOp, Op DstOp, bool AutoToken = true, event_t EventID = EVENT_ID0>
struct Event : EventBase<Event<SrcOp, DstOp, AutoToken, EventID>, SrcOp, DstOp, AutoToken, EventID> {
    using Base = EventBase<Event, SrcOp, DstOp, AutoToken, EventID>;
    using Base::operator=;

    template <Op op>
    PTO_INTERNAL static constexpr pipe_t GetPipeByOp()
    {
        return GetPipeByOpForA3<op>();
    }
#ifndef __PTO_AUTO__
    static constexpr bool IsCrossCore =
        ((Base::srcOp == Op::TMOV_A2V) && (Base::dstPipe == PIPE_V)) ||
        ((Base::srcOp == Op::TMOV_V2M || Base::srcOp == Op::TEXTRACT_V2M) && (Base::dstPipe == PIPE_MTE1));

    PTO_STATIC_ASSERT((!IsCrossCore) || (!AutoToken), "Cross-core events must manually specify EventID.");
    PTO_STATIC_ASSERT(IsCrossCore || (Base::dstPipe != PIPE_ALL), "DstOp are invalid.");
    PTO_STATIC_ASSERT(IsCrossCore || (Base::srcPipe != PIPE_ALL), "SrcOp are invalid.");
#endif

    PTO_INTERNAL Event& InitAddrImpl(uint64_t fftsAddr)
    {
#ifndef __PTO_AUTO__
        PTO_STATIC_ASSERT(IsCrossCore, "Only cross-core events require setting the initial addr.");
        set_ffts_base_addr(fftsAddr);
#endif
        return *this;
    }

    template <uint8_t CrossCoreId = 0xff>
    PTO_INTERNAL Event& InitImpl()
    {
#ifndef __PTO_AUTO__
        if constexpr (IsCrossCore) {
            PTO_STATIC_ASSERT(
                CrossCoreId != 0xff,
                "Fix: The cross-core id must be assigned by user when the event is a cross-core event.");
            ffts_cross_core_sync(Base::srcPipe, getFFTSMsg(FFTS_MODE_VAL, CrossCoreId));
        } else if constexpr (!Base::isSamePipe) {
#ifdef PTO_FLAG_TEST
            Base::token = __pto_set_flag((pipe_t)Base::srcPipe, (pipe_t)Base::dstPipe);
#else
            set_flag((pipe_t)Base::srcPipe, (pipe_t)Base::dstPipe, Base::token);
#endif
        }
#endif
        return *this;
    }

    template <uint8_t CrossCoreId = 0xff>
    PTO_INTERNAL Event& WaitImpl()
    {
#ifndef __PTO_AUTO__
        if constexpr (IsCrossCore) {
            PTO_STATIC_ASSERT(
                CrossCoreId != 0xff,
                "Fix: The cross-core id must be assigned by user when the event is a cross-core event.");
            wait_flag_dev(CrossCoreId);
        } else if constexpr (Base::isSamePipe) {
            pipe_barrier((pipe_t)Base::srcPipe);
        } else {
#ifdef PTO_FLAG_TEST
            __pto_wait_flag((pipe_t)Base::srcPipe, (pipe_t)Base::dstPipe, Base::token);
#else
            wait_flag((pipe_t)Base::srcPipe, (pipe_t)Base::dstPipe, Base::token);
#endif
        }
#endif
        return *this;
    }

    PTO_INTERNAL Event() = default;
    PTO_INTERNAL Event(RecordEvent re) : Base(re) {}
};

} // namespace pto
#endif
