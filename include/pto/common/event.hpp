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

#include <type_traits>
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
    TPOW,
    TPOWS,
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
    TRANDOM,
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
    SETFMATRIX,
    SET_IMG2COL_RPT,
    SET_IMG2COL_PADDING,
    TCONCAT,
    TDEQUANT,
    TADDDEQRELU,
    TABS,
    TNEG,
    TRELU,
    TNOT,
    TLOG,
    TRECIP,
    TCMP,
    TSCATTER,
    TCOLEXPAND,
    TCOLEXPANDDIV,
    TCOLEXPANDMUL,
    TCOLEXPANDADD,
    TCOLEXPANDMAX,
    TCOLEXPANDMIN,
    TCOLEXPANDSUB,
    TCOLEXPANDEXPDIF,
    TROWEXPANDDIV,
    TROWEXPANDMUL,
    TROWEXPANDSUB,
    TROWEXPANDADD,
    TROWEXPANDMAX,
    TROWEXPANDMIN,
    TROWEXPANDEXPDIF,
    TPAIRREDUCESUM,
    TSUBRELUCONV,
    TADDRELUCONV,
    TFUSEDMULADD,
    TMULADDDST,
    TSUBRELU,
    TFUSEDMULADDRELU,
    TPARTARGMAX,
    TPARTARGMIN,
    TCOLARGMAX,
    TCOLARGMIN,
    TROWARGMAX,
    TROWARGMIN,
    TPREFETCH,
    TFILLPAD_VEC,
    TFILLPAD_MAT,
    MGATHER_VEC,
    MGATHER_MAT,
    MSCATTER,
    TQUANT,
    THISTOGRAM,
    TINTERLEAVE,
    TDEINTERLEAVE,
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
    PIPE_V /* TPOW */,
    PIPE_V /* TPOWS */,
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
    PIPE_V /* TRANDOM */,
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

#define PTO_DEFINE_OP_PIPE(Op_, Pipe_)        \
    template <>                               \
    struct OpPipeEntry<Op_> {                 \
        static constexpr pipe_t pipe = Pipe_; \
    }

PTO_DEFINE_OP_PIPE(Op::TLOAD, PIPE_MTE2);
PTO_DEFINE_OP_PIPE(Op::TSTORE_VEC, PIPE_MTE3);
PTO_DEFINE_OP_PIPE(Op::SCALAR, PIPE_S);
PTO_DEFINE_OP_PIPE(Op::TRESHAPE, PIPE_S);
PTO_DEFINE_OP_PIPE(Op::VECTOR, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TADD, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TADDS, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TAXPY, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TSUB, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TMUL, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TMULS, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TDIV, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TDIVS, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TMIN, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TMINS, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TMAX, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TAND, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TOR, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TSEL, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TSHL, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TSHR, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TEXP, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TSELS, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TSQRT, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TRSQRT, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TEXPANDS, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TEXPANDS_MAT, PIPE_MTE2);
PTO_DEFINE_OP_PIPE(Op::TPARTADD, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TPARTMUL, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TPARTMAX, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TPARTMIN, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TPOW, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TPOWS, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TCMPS, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TMRGSORT, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TSORT32, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TCI, PIPE_S);
PTO_DEFINE_OP_PIPE(Op::TGATHER, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TGATHERB, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TCVT, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TROWSUM, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TROWPROD, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TROWMAX, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TROWMIN, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TROWEXPAND, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TRANDOM, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TCOLSUM, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TCOLPROD, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TCOLMAX, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TCOLMIN, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TTRANS, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TTRI, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TREM, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TFMOD, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TREMS, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TFMODS, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TSUBS, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TMAXS, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TLRELU, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TPRELU, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TMOV_V2V, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TMOV_V2M, PIPE_FIX);
PTO_DEFINE_OP_PIPE(Op::TEXTRACT_V2M, PIPE_FIX);
PTO_DEFINE_OP_PIPE(Op::TMOV_M2B, PIPE_MTE1);
PTO_DEFINE_OP_PIPE(Op::TMOV_M2L, PIPE_MTE1);
PTO_DEFINE_OP_PIPE(Op::TMOV_M2R, PIPE_MTE1);
PTO_DEFINE_OP_PIPE(Op::TMOV_M2S, PIPE_FIX);
PTO_DEFINE_OP_PIPE(Op::TMOV_A2V, PIPE_FIX);
PTO_DEFINE_OP_PIPE(Op::TMOV_A2M, PIPE_FIX);
PTO_DEFINE_OP_PIPE(Op::TSTORE_ACC, PIPE_FIX);
PTO_DEFINE_OP_PIPE(Op::TSTORE_MAT, PIPE_MTE3);
PTO_DEFINE_OP_PIPE(Op::TMATMUL, PIPE_M);
PTO_DEFINE_OP_PIPE(Op::TGEMV, PIPE_M);
PTO_DEFINE_OP_PIPE(Op::TEXTRACT_M2LR, PIPE_MTE1);
PTO_DEFINE_OP_PIPE(Op::TANDS, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TORS, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TSHLS, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TSHRS, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TXOR, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TXORS, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TEXTRACT_A2M, PIPE_FIX);
PTO_DEFINE_OP_PIPE(Op::TINSERT_A2M, PIPE_FIX);
PTO_DEFINE_OP_PIPE(Op::TIMG2COL, PIPE_MTE1);
PTO_DEFINE_OP_PIPE(Op::SETFMATRIX, PIPE_S);
PTO_DEFINE_OP_PIPE(Op::SET_IMG2COL_RPT, PIPE_S);
PTO_DEFINE_OP_PIPE(Op::SET_IMG2COL_PADDING, PIPE_S);
PTO_DEFINE_OP_PIPE(Op::TCONCAT, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TDEQUANT, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TADDDEQRELU, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TABS, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TNEG, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TRELU, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TNOT, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TLOG, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TRECIP, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TCMP, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TSCATTER, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TCOLEXPAND, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TCOLEXPANDDIV, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TCOLEXPANDMUL, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TCOLEXPANDADD, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TCOLEXPANDMAX, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TCOLEXPANDMIN, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TCOLEXPANDSUB, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TCOLEXPANDEXPDIF, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TROWEXPANDDIV, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TROWEXPANDMUL, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TROWEXPANDSUB, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TROWEXPANDADD, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TROWEXPANDMAX, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TROWEXPANDMIN, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TROWEXPANDEXPDIF, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TPAIRREDUCESUM, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TSUBRELUCONV, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TADDRELUCONV, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TFUSEDMULADD, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TMULADDDST, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TSUBRELU, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TFUSEDMULADDRELU, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TPARTARGMAX, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TPARTARGMIN, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TCOLARGMAX, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TCOLARGMIN, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TROWARGMAX, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TROWARGMIN, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TPREFETCH, PIPE_MTE2);
PTO_DEFINE_OP_PIPE(Op::TFILLPAD_VEC, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TFILLPAD_MAT, PIPE_MTE2);
PTO_DEFINE_OP_PIPE(Op::MGATHER_VEC, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::MGATHER_MAT, PIPE_MTE2);
PTO_DEFINE_OP_PIPE(Op::MSCATTER, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TQUANT, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::THISTOGRAM, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TINTERLEAVE, PIPE_V);
PTO_DEFINE_OP_PIPE(Op::TDEINTERLEAVE, PIPE_V);

#undef PTO_DEFINE_OP_PIPE

struct RecordEvent {};

template <pipe_t SrcPipe, pipe_t DstPipe>
class EventIdCounter {
public:
    PTO_INTERNAL static event_t GetNextId()
    {
        event_t id = NextId();
#if defined(__CPU_SIM) || defined(__COSTMODEL)
        PTO_CPU_ASSERT(!(OccupiedMask() & (1u << static_cast<uint8_t>(id))),
                       "Event ID still occupied - likely missing Wait()");
        OccupiedMask() |= (1u << static_cast<uint8_t>(id));
#endif
        NextId() = (event_t)(((uint8_t)NextId() + 1) % EVENT_ID_MAX);
        return id;
    }
    PTO_INTERNAL static void Reset()
    {
        NextId() = EVENT_ID0;
#if defined(__CPU_SIM) || defined(__COSTMODEL)
        OccupiedMask() = 0;
#endif
    }
    PTO_INTERNAL static event_t PeekNextId()
    {
        return NextId();
    }
#if defined(__CPU_SIM) || defined(__COSTMODEL)
    PTO_INTERNAL static void MarkFree(event_t id)
    {
        OccupiedMask() &= ~(1u << static_cast<uint8_t>(id));
    }
#endif

private:
    static event_t &NextId()
    {
        static event_t id = EVENT_ID0;
        return id;
    }
#if defined(__CPU_SIM) || defined(__COSTMODEL)
    static uint8_t &OccupiedMask()
    {
        static uint8_t mask = 0;
        return mask;
    }
#endif
};

template <typename... WaitEvents>
PTO_INTERNAL void WaitAllEvents(WaitEvents &...events)
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
struct EventBaseTag {};

template <typename T>
struct is_event : std::is_base_of<EventBaseTag, T> {};

template <typename... Ts>
inline constexpr bool all_events_v = (is_event<Ts>::value && ...);

// CRTP base class for platform-specific Event implementations.
// Derived must provide:
//   template <Op op> static constexpr pipe_t GetPipeByOp()
//   static constexpr bool IsCrossCore
//   template <uint8_t CrossCoreId> PTO_INTERNAL Derived& WaitImpl()
//   template <uint8_t CrossCoreId> PTO_INTERNAL Derived& InitImpl()
//   PTO_INTERNAL Derived& InitAddrImpl(uint64_t fftsAddr)
template <typename Derived, Op SrcOp, Op DstOp, bool AutoToken = true, event_t EventID = EVENT_ID0>
struct EventBase : EventBaseTag {
#ifndef __PTO_AUTO__
    static constexpr Op srcOp = SrcOp;
    static constexpr Op dstOp = DstOp;
    static constexpr pipe_t srcPipe = Derived::template GetPipeByOp<SrcOp>();
    static constexpr pipe_t dstPipe = Derived::template GetPipeByOp<DstOp>();
    static constexpr bool isSamePipe = (srcPipe == dstPipe);

#ifdef PTO_FLAG_TEST
    CceEventIdType token = {};
#else
    const event_t token = AutoToken ? EventIdCounter<srcPipe, dstPipe>::GetNextId() : EventID;
#endif
#endif

    PTO_INTERNAL Derived &InitAddr(uint64_t fftsAddr)
    {
        return self().InitAddrImpl(fftsAddr);
    }

    template <uint8_t CrossCoreId = 0xff>
    PTO_INTERNAL Derived &Wait()
    {
#ifndef __PTO_AUTO__
        return self().template WaitImpl<CrossCoreId>();
#else
        return self();
#endif
    }

    template <uint8_t CrossCoreId = 0xff>
    PTO_INTERNAL Derived &Init()
    {
#ifndef __PTO_AUTO__
        return self().template InitImpl<CrossCoreId>();
#else
        return self();
#endif
    }

    template <uint8_t CrossCoreId = 0xff>
    PTO_INTERNAL Derived &Record()
    {
        return Init<CrossCoreId>();
    }

    PTO_INTERNAL EventBase() = default;
    PTO_INTERNAL EventBase(RecordEvent)
    {
        Init();
    }

    PTO_INTERNAL Derived &operator=(RecordEvent)
    {
        return Init();
    }

private:
    PTO_INTERNAL Derived &self()
    {
        return *static_cast<Derived *>(this);
    }
};

} // namespace pto
#endif
