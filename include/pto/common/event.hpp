/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef EVENT_HPP
#define EVENT_HPP

#define EVENT_ID_MAX 8

#include <pto/common/type.hpp>

namespace pto {
enum class Op : uint16_t
{
    TLOAD,      /* GM to Vec/Mat/ */
    TSTORE_VEC, /* Vec to GM */
    SCALAR,
    TRESHAPE,
    VECTOR,
    TADD,
    TADDS,
    TAXPY,
    TSUB,
    TMUL,
    TMULS,
    TDIV,
    TDIVS,
    TMIN,
    TMINS,
    TMAX,
    TAND,
    TOR,
    TSEL,
    TSHL,
    TSHR,
    TEXP,
    TSELS,
    TSQRT,
    TRSQRT,
    TEXPANDS,
    TEXPANDS_MAT,
    TPARTADD,
    TPARTMUL,
    TPARTMAX,
    TPARTMIN,
    TCMPS,
    TMRGSORT,
    TSORT32,
    TCI,
    TGATHER,
    TGATHERB,
    TCVT,
    TROWSUM,
    TROWPROD,
    TROWMAX,
    TROWMIN,
    TROWEXPAND,
    TCOLSUM,
    TCOLPROD,
    TCOLMAX,
    TCOLMIN,
    TTRANS,
    TTRI,
    TREM,
    TFMOD,
    TREMS,
    TFMODS,
    TSUBS,
    TMAXS,
    TLRELU,
    TPRELU,
    TMOV_V2V,     /* Vec to Vec */
    TMOV_V2M,     /* Vec to Mat */
    TEXTRACT_V2M, /* Vec to Mat */
    TMOV_M2B,     /* Mat to Bias */
    TMOV_M2L,     /* Mat to Left */
    TMOV_M2R,     /* Mat to Right */
    TMOV_M2S,     /* Mat to Scaling */
    TMOV_A2V,     /* Acc to Vec */
    TMOV_A2M,     /* Acc to Mat */
    TSTORE_ACC,   /* Acc to GM */
    TSTORE_MAT,   /* Mat to GM */
    TMATMUL,
    TGEMV,
    TMATMUL_MX,
    TEXTRACT_M2LR, /* Mat to Left/Right */
    TANDS,
    TORS,
    TSHLS,
    TSHRS,
    TXOR,
    TXORS,
    TEXTRACT_A2M, /* Acc to Mat */
    TINSERT_A2M,
    TIMG2COL,
    TSETFMATRIX,
    TSET_IMG2COL_RPT,
    TSET_IMG2COL_PADDING,
    TCONCAT,
    TDEQUANT,
    OP_COUNT, // The Total number of operations, please add new operations before OP_COUNT
};

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
    PIPE_V /* TAXPY */,
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
    PIPE_MTE2 /* TEXPANDS_MAT */,
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
    PIPE_V /* TCONCAT */,
    PIPE_V /* TDEQUANT */,
    PIPE_ALL /* OP_COUNT */,
};

struct RecordEvent {
};

template <pipe_t SrcPipe, pipe_t DstPipe>
class EventIdCounter {
public:
    PTO_INTERNAL static event_t GetNextId()
    {
        event_t id = NextId();
        NextId() = (event_t)(((uint8_t)NextId() + 1) % EVENT_ID_MAX);
        return id;
    }
    PTO_INTERNAL static void Reset()
    {
        NextId() = EVENT_ID0;
    }
    PTO_INTERNAL static event_t PeekNextId()
    {
        return NextId();
    }

private:
    static event_t &NextId()
    {
        static event_t id = EVENT_ID0;
        return id;
    }
};

template <typename... WaitEvents>
PTO_INTERNAL void WaitAllEvents(WaitEvents &... events)
{
    (events.Wait(), ...);
}

template <pipe_t SrcPipe, pipe_t DstPipe>
PTO_INTERNAL void PtoSetWaitFlag(event_t SetEventId = EVENT_ID0, event_t WaitEventId = EVENT_ID0)
{
#ifndef __PTO_AUTO__
#ifdef PTO_FLAG_TEST
    CceEventIdType token = __pto_set_flag(SrcPipe, DstPipe);
    __pto_wait_flag(SrcPipe, DstPipe, token);
#else
    set_flag(SrcPipe, DstPipe, SetEventId);
    wait_flag(SrcPipe, DstPipe, WaitEventId);
#endif
#endif
}
} // namespace pto
#endif
