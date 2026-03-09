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
// opPipeList maps each operation in Op enum to its corresponding pipeline type.
// This array is used to determine which hardware pipeline should be used for each operation.
constexpr pipe_t opPipeList[] = {
    PIPE_MTE2 /* TLOAD */,
    PIPE_MTE3 /* TSTORE_VEC */,
    PIPE_S /* SCALAR */,
    PIPE_S /* TRESHAPE */,
    PIPE_V /* VECTOR */,
    PIPE_V /* TADD */,
    PIPE_V /* TADDS */,
    PIPE_V /* TSUB */,
    PIPE_V /* TMUL */,
    PIPE_V /* TMULS */,
    PIPE_V /* TDIV */,
    PIPE_V /* TDIVS */,
    PIPE_V /* TMIN */,
    PIPE_V /* TMINS */,
    PIPE_V /* TMAX */,
    PIPE_V /* TAND */,
    PIPE_V /* TOR */,
    PIPE_V /* TSEL */,
    PIPE_V /* TSHL */,
    PIPE_V /* TSHR */,
    PIPE_V /* TEXP */,
    PIPE_V /* TSELS */,
    PIPE_V /* TSQRT */,
    PIPE_V /* TRSQRT */,
    PIPE_V /* TEXPANDS */,
    PIPE_V /* TPARTADD */,
    PIPE_V /* TPARTMUL */,
    PIPE_V /* TPARTMAX */,
    PIPE_V /* TPARTMIN */,
    PIPE_V /* TCMPS */,
    PIPE_V /* TMRGSORT */,
    PIPE_V /* TSORT32 */,
    PIPE_S /* TCI */,
    PIPE_V /* TGATHER */,
    PIPE_V /* TGATHERB */,
    PIPE_V /* TCVT */,
    PIPE_V /* TROWSUM */,
    PIPE_V /* TROWPROD */,
    PIPE_V /* TROWMAX */,
    PIPE_V /* TROWMIN */,
    PIPE_V /* TROWEXPAND */,
    PIPE_V /* TCOLSUM */,
    PIPE_V /* TCOLPROD */,
    PIPE_V /* TCOLMAX */,
    PIPE_V /* TCOLMIN */,
    PIPE_V /* TTRANS */,
    PIPE_V /* TTRI */,
    PIPE_V /* TREM */,
    PIPE_V /* TFMOD */,
    PIPE_V /* TREMS */,
    PIPE_V /* TFMODS */,
    PIPE_V /* TSUBS */,
    PIPE_V /* TMAXS */,
    PIPE_V /* TLRELU */,
    PIPE_V /* TPRELU */,
    PIPE_V /* TMOV_V2V */,
    PIPE_FIX /* TMOV_V2M */,
    PIPE_FIX /* TEXTRACT_V2M */,
    PIPE_MTE1 /* TMOV_M2B */,
    PIPE_MTE1 /* TMOV_M2L */,
    PIPE_MTE1 /* TMOV_M2R */,
    PIPE_FIX /* TMOV_M2S */,
    PIPE_FIX /* TMOV_A2V */,
    PIPE_FIX /* TMOV_A2M */,
    PIPE_FIX /* TSTORE_ACC */,
    PIPE_MTE3 /* TSTORE_MAT */,
    PIPE_M /* TMATMUL */,
    PIPE_M /* TGEMV */,
    PIPE_M /* TMATMUL_MX */,
    PIPE_MTE1 /* TEXTRACT_M2LR */,
    PIPE_V /* TANDS */,
    PIPE_V /* TORS */,
    PIPE_V /* TSHLS */,
    PIPE_V /* TSHRS */,
    PIPE_V /* TXOR */,
    PIPE_V /* TXORS */,
    PIPE_FIX /* TEXTRACT_A2M */,
    PIPE_FIX /* TINSERT_A2M */,
    PIPE_MTE1 /* TIMG2COL */,
    PIPE_S /* TSETFMATRIX */,
    PIPE_S /* TSET_IMG2COL_RPT */,
    PIPE_S /* TSET_IMG2COL_PADDING */,
    PIPE_V /* TDEQUANT */,
    PIPE_ALL /* OP_COUNT */,
};

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
    PTO_STATIC_ASSERT(pipe == PIPE_MTE3 || OpCode == Op::OP_COUNT,
                      "Single Op TSYNC only supports MTE3 or ALL pipeline.");
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
        ((srcPipe == PIPE_M) || (dstPipe == PIPE_MTE1) || (dstPipe == PIPE_MTE2) || (dstPipe == PIPE_MTE3) ||
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
