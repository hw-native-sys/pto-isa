/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_INSTR_HPP
#define PTO_INSTR_HPP

#include "pto/common/debug.h"
#include "pto/common/event.hpp"
#include "pto/common/fifo.hpp"
#ifndef __CPU_SIM
#include "pto/npu/a2a3/grid_intrinsic.hpp"
#endif
#include "pto/common/tassign_check.hpp"

namespace pto {
struct MrgSortExecutedNumList {
    uint16_t mrgSortList0;
    uint16_t mrgSortList1;
    uint16_t mrgSortList2;
    uint16_t mrgSortList3;
};
} // namespace pto

#include "pto/common/pto_instr_impl.hpp"
#ifdef __CPU_SIM
#include "pto/cpu/trace.hpp"
#endif

#define PTO_TEMPLATE_ARGS(...) <__VA_ARGS__>

#ifdef __CPU_SIM
#define PTO_INSTR_SCOPE(API, ...) \
    ::pto::cpu_sim::PtoInstrTraceScope _pto_instr_trace_scope(#API, 1 __VA_OPT__(, ) __VA_ARGS__)
#define PTO_INSTR_SCOPE_OUTS(API, OUT_COUNT, ...) \
    ::pto::cpu_sim::PtoInstrTraceScope _pto_instr_trace_scope(#API, OUT_COUNT __VA_OPT__(, ) __VA_ARGS__)
#define PTO_INSTR_SCOPE_ROLES(API, ROLES, ...) \
    ::pto::cpu_sim::PtoInstrTraceScope _pto_instr_trace_scope(#API, ROLES __VA_OPT__(, ) __VA_ARGS__)
#define MAP_INSTR_IMPL(API, ...)           \
    do {                                   \
        PTO_INSTR_SCOPE(API, __VA_ARGS__); \
        API##_IMPL(__VA_ARGS__);           \
    } while (0)
#define MAP_INSTR_IMPL_OUTS(API, OUT_COUNT, ...)           \
    do {                                                   \
        PTO_INSTR_SCOPE_OUTS(API, OUT_COUNT, __VA_ARGS__); \
        API##_IMPL(__VA_ARGS__);                           \
    } while (0)
#define MAP_INSTR_IMPL_T(API, TEMPLATE_ARGS, ...) \
    do {                                          \
        PTO_INSTR_SCOPE(API, __VA_ARGS__);        \
        API##_IMPL TEMPLATE_ARGS(__VA_ARGS__);    \
    } while (0)
#define MAP_INSTR_IMPL_T_OUTS(API, TEMPLATE_ARGS, OUT_COUNT, ...) \
    do {                                                          \
        PTO_INSTR_SCOPE_OUTS(API, OUT_COUNT, __VA_ARGS__);        \
        API##_IMPL TEMPLATE_ARGS(__VA_ARGS__);                    \
    } while (0)
#define MAP_INSTR_IMPL_ROLES(API, ROLES, ...)           \
    do {                                                \
        PTO_INSTR_SCOPE_ROLES(API, ROLES, __VA_ARGS__); \
        API##_IMPL(__VA_ARGS__);                        \
    } while (0)
#define MAP_INSTR_IMPL_T_ROLES(API, TEMPLATE_ARGS, ROLES, ...) \
    do {                                                       \
        PTO_INSTR_SCOPE_ROLES(API, ROLES, __VA_ARGS__);        \
        API##_IMPL TEMPLATE_ARGS(__VA_ARGS__);                 \
    } while (0)
#else
#define PTO_INSTR_SCOPE(API, ...)
#define PTO_INSTR_SCOPE_OUTS(API, OUT_COUNT, ...)
#define PTO_INSTR_SCOPE_ROLES(API, ROLES, ...)
#define MAP_INSTR_IMPL(API, ...) API##_IMPL(__VA_ARGS__)
#define MAP_INSTR_IMPL_OUTS(API, OUT_COUNT, ...) API##_IMPL(__VA_ARGS__)
#define MAP_INSTR_IMPL_T(API, TEMPLATE_ARGS, ...) API##_IMPL TEMPLATE_ARGS(__VA_ARGS__)
#define MAP_INSTR_IMPL_T_OUTS(API, TEMPLATE_ARGS, OUT_COUNT, ...) API##_IMPL TEMPLATE_ARGS(__VA_ARGS__)
#define MAP_INSTR_IMPL_ROLES(API, ROLES, ...) API##_IMPL(__VA_ARGS__)
#define MAP_INSTR_IMPL_T_ROLES(API, TEMPLATE_ARGS, ROLES, ...) API##_IMPL TEMPLATE_ARGS(__VA_ARGS__)
#endif

#if !defined(__COSTMODEL) && !defined(PTO_COMM_NOT_SUPPORTED)
#include "pto/comm/pto_comm_inst.hpp"
#endif

namespace pto {

template <typename T, typename AddrType>
PTO_INST void TASSIGN(T& obj, AddrType addr)
{
    MAP_INSTR_IMPL(TASSIGN, obj, addr);
}

// Compile-time address overload: TASSIGN<Addr>(tile)
// Performs static bounds and alignment checks when Addr is a compile-time constant.
// Only enabled for Tile / ConvTile types (not GlobalTensor).
template <std::size_t Addr, typename T>
PTO_INST std::enable_if_t<is_tile_data_v<T> || is_conv_tile_v<T>> TASSIGN(T& obj)
{
    // Trigger compile-time checks (static_assert inside tassign_static_check).
    (void)detail::tassign_static_check<std::remove_cv_t<T>, Addr>{};

    // Delegate to the existing runtime TASSIGN path.
    TASSIGN(obj, static_cast<std::size_t>(Addr));
}

template <Op OpCode>
PTO_INST void TSYNC()
{
    MAP_INSTR_IMPL_T_OUTS(TSYNC, PTO_TEMPLATE_ARGS(OpCode), 0);
}

template <SyncCoreType CoreType = SyncCoreType::AIVOnly>
PTO_INST void SYNCALL()
{
#if defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_A5) || defined(__CPU_SIM)
    SYNCALL_IMPL<CoreType>();
#else
    PTO_STATIC_ASSERT(CoreType != CoreType, "SYNCALL is not supported on this backend.");
#endif
}

// Soft SYNCALL: GM shared-counter barrier. CoreType selects AIV-only / AIC-only / MIX.
// Hard Mode with a workspace argument is accepted but ignores gmWorkspace (same as SYNCALL()).
template <
    SyncAllMode Mode, SyncCoreType CoreType = SyncCoreType::AIVOnly, typename GlobalData,
    std::enable_if_t<is_global_data_v<GlobalData>, int> = 0>
PTO_INST void SYNCALL(GlobalData& gmWorkspace, int32_t usedCores = 0)
{
#if defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_A5) || defined(__CPU_SIM)
    if constexpr (Mode == SyncAllMode::Hard) {
        (void)gmWorkspace;
        (void)usedCores;
        SYNCALL_IMPL<CoreType>();
    } else if constexpr (CoreType == SyncCoreType::AIVOnly) {
#ifndef __PTO_AUTO__
        SYNCALL_SOFT_IMPL<CoreType>(gmWorkspace.data(), usedCores);
#endif
    } else if constexpr (CoreType == SyncCoreType::AICOnly) {
#ifndef __PTO_AUTO__
        SYNCALL_SOFT_AIC_IMPL(gmWorkspace.data(), usedCores);
#endif
    } else {
#ifndef __PTO_AUTO__
        SYNCALL_SOFT_MIX_IMPL<CoreType>(gmWorkspace.data(), usedCores);
#endif
    }
#else
    PTO_STATIC_ASSERT(Mode != Mode, "SYNCALL is not supported on this backend.");
#endif
}

template <typename... WaitEvents>
PTO_INST void TSYNC(WaitEvents&... events)
{
    WaitAllEvents(events...);
}

#if defined(_DEBUG) || defined(__CPU_SIM)
template <PrintFormat Format = PrintFormat::Width8_Precision4, typename TileData>
PTO_INST void TPRINT(TileData& src)
{
    MAP_INSTR_IMPL_T_OUTS(TPRINT, PTO_TEMPLATE_ARGS(Format), 0, src);
}

template <PrintFormat Format = PrintFormat::Width8_Precision4, typename TileData, typename GlobalData>
PTO_INST void TPRINT(TileData& src, GlobalData& tmp)
{
    MAP_INSTR_IMPL_T_OUTS(TPRINT, PTO_TEMPLATE_ARGS(Format), 0, src, tmp);
}
#endif

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TADD(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TADD, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename... WaitEvents>
PTO_INST RecordEvent TPAIRREDUCESUM(TileDataDst& dst, TileDataSrc0& src0, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TPAIRREDUCESUM, dst, src0);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TSUBRELUCONV(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSUBRELUCONV, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TADDRELUCONV(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TADDRELUCONV, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TABS(TileDataDst& dst, TileDataSrc& src, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TABS, dst, src);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TAND(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TAND, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TOR(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TOR, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TSUB(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSUB, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TSUBVIEW(TileDataDst& dst, TileDataSrc& src, uint16_t rowIdx, uint16_t colIdx, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSUBVIEW, dst, src, rowIdx, colIdx);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TMUL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMUL, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TMIN(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMIN, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TMAX(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMAX, dst, src0, src1);
    return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TEXPANDS(TileData& dst, typename TileData::DType scalar, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TEXPANDS, dst, scalar);
    return {};
}

template <typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TLOAD(TileData& dst, GlobalData& src, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TLOAD, dst, src);
    return {};
}

template <typename TileData, typename GlobalData>
PTO_INST RecordEvent TPREFETCH(TileData& dst, GlobalData& src)
{
    MAP_INSTR_IMPL(TPREFETCH, dst, src);
    return {};
}

// ============================================================================
// TPREFETCH_ASYNC - L2 cache prefetch via SDMA CMO (opcode = 6).
//
// Stages a contiguous GlobalTensor region into L2 cache so subsequent TLOADs
// hit warm lines. The public compute API takes a compute-side prefetch context;
// the implementation builds the SDMA session in that context, and callers can
// wait on the returned event with evt.Wait(ctx.session).
// ============================================================================
#if (defined(__CCE_AICORE__) || defined(__CPU_SIM)) && !defined(__COSTMODEL) && !defined(PTO_COMM_NOT_SUPPORTED)

template <typename GlobalData, typename... WaitEvents, std::enable_if_t<all_events_v<WaitEvents...>, int> = 0>
PTO_INST comm::AsyncEvent TPREFETCH_ASYNC(GlobalData& srcGlobalData, PrefetchAsyncContext& ctx, WaitEvents&... events)
{
    TSYNC(events...);
    return TPREFETCH_ASYNC_IMPL(srcGlobalData, ctx);
}

#endif // (__CCE_AICORE__ || __CPU_SIM) && !__COSTMODEL && !PTO_COMM_NOT_SUPPORTED

template <
    typename TileDataDst, typename TileDataSrc, typename... WaitEvents,
    std::enable_if_t<all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent
TCMPS(TileDataDst& dst, TileDataSrc& src0, typename TileDataSrc::DType src1, CmpMode mode, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCMPS, dst, src0, src1, mode);
    return {};
}

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents,
    std::enable_if_t<is_tile_data_v<TileDataSrc1> && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent
TCMPS(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, CmpMode mode, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCMPS, dst, src0, src1, mode);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent
TCMP(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, CmpMode cmpMode, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCMP, dst, src0, src1, cmpMode);
    return {};
}

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents,
    std::enable_if_t<all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TCONCAT(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCONCAT, dst, src0, src1);
    return {};
}

template <
    typename DstTile, typename Src0Tile, typename Src1Tile, typename Src0IdxTile, typename Src1IdxTile,
    typename... WaitEvents,
    std::enable_if_t<is_tile_data_v<Src0IdxTile> && is_tile_data_v<Src1IdxTile> && all_events_v<WaitEvents...>, int> =
        0>
PTO_INST RecordEvent
TCONCAT(DstTile& dst, Src0Tile& src0, Src1Tile& src1, Src0IdxTile& src0Idx, Src1IdxTile& src1Idx, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCONCAT, dst, src0, src1, src0Idx, src1Idx);
    return {};
}

template <
    typename DstTile, typename Src0Tile, typename Src1Tile, typename DstIdxTile, typename Src0IdxTile,
    typename Src1IdxTile, typename... WaitEvents,
    std::enable_if_t<is_tile_data_v<Src1IdxTile> && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TCONCAT(
    DstTile& dst, Src0Tile& src0, Src1Tile& src1, DstIdxTile& dstIdx, Src0IdxTile& src0Idx, Src1IdxTile& src1Idx,
    WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCONCAT, dst, src0, src1, dstIdx, src0Idx, src1Idx);
    return {};
}

template <typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, AtomicType::AtomicNone), dst, src);
    return {};
}

// UF-aware overload: allow selecting unit-flag phase while keeping the TSTORE name.
template <STPhase Phase, typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, AtomicType::AtomicNone, Phase), dst, src);
    return {};
}

template <typename TileData, typename GlobalData, AtomicType atomicType, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, atomicType), dst, src);
    return {};
}

template <STPhase Phase, typename TileData, typename GlobalData, AtomicType atomicType, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, atomicType, Phase), dst, src);
    return {};
}

template <
    typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone, ReluPreMode reluPreMode,
    typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, atomicType, reluPreMode), dst, src);
    return {};
}

template <
    STPhase Phase, typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
    ReluPreMode reluPreMode, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, atomicType, reluPreMode, Phase), dst, src);
    return {};
}

template <
    typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
    ReluPreMode reluPreMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, uint64_t preQuantScalar, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(
        TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, atomicType, reluPreMode), dst, src, preQuantScalar);
    return {};
}

template <
    STPhase Phase, typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
    ReluPreMode reluPreMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, uint64_t preQuantScalar, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(
        TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, atomicType, reluPreMode, Phase), dst, src, preQuantScalar);
    return {};
}

template <
    typename TileData, typename GlobalData, typename FpTileData, AtomicType atomicType = AtomicType::AtomicNone,
    ReluPreMode reluPreMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TSTORE_FP(GlobalData& dst, TileData& src, FpTileData& fp, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(
        TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, FpTileData, atomicType, reluPreMode), dst, src, fp);
    return {};
}

template <
    auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1,
    typename... WaitEvents>
PTO_INST RecordEvent TDIV(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TDIV, PTO_TEMPLATE_ARGS(PrecisionType), dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TSHL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSHL, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TSHR(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSHR, dst, src0, src1);
    return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TAND(TileData& dst, TileData& src0, TileData& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TAND, dst, src0, src1);
    return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TOR(TileData& dst, TileData& src0, TileData& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TOR, dst, src0, src1);
    return {};
}

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent
TXOR(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, TileDataTmp& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TXOR, dst, src0, src1, tmp);
    return {};
}

template <
    auto PrecisionType = LogAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TLOG(TileDataDst& dst, TileDataSrc& src, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TLOG, PTO_TEMPLATE_ARGS(PrecisionType), dst, src);
    return {};
}

template <
    auto PrecisionType = RecipAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TRECIP(TileDataDst& dst, TileDataSrc& src, WaitEvents&... events)
{
    TSYNC(events...);
    /*
     * A3's TRECIP instruction does not support setting the source Tile and destination Tile to the same memory.
     */
    MAP_INSTR_IMPL_T(TDIVS, PTO_TEMPLATE_ARGS(static_cast<DivAlgorithm>(PrecisionType)), dst, 1, src);
    return {};
}

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent
TPRELU(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, TileDataTmp& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TPRELU, dst, src0, src1, tmp);
    return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TADDC(TileData& dst, TileData& src0, TileData& src1, TileData& src2, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TADDC, dst, src0, src1, src2);
    return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TSUBC(TileData& dst, TileData& src0, TileData& src1, TileData& src2, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSUBC, dst, src0, src1, src2);
    return {};
}

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6) || defined(__CPU_SIM)
template <
    typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
    typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(
    TileRes& cMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix, TileRightScale& bScaleMatrix,
    WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TGEMV_MX, cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix);
    return {};
}

template <
    AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
    typename TileRightScale, typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(
    TileRes& cMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix, TileRightScale& bScaleMatrix,
    WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TGEMV_MX, PTO_TEMPLATE_ARGS(Phase), cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix);
    return {};
}

template <
    typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
    typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(
    TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix,
    TileRightScale& bScaleMatrix, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TGEMV_MX, cOutMatrix, cInMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix);
    return {};
}

template <
    AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
    typename TileRightScale, typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(
    TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix,
    TileRightScale& bScaleMatrix, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(
        TGEMV_MX, PTO_TEMPLATE_ARGS(Phase), cOutMatrix, cInMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix);
    return {};
}

template <
    typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
    typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(
    TileRes& cMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix, TileRightScale& bScaleMatrix,
    TileBias& biasData, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TGEMV_MX, cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix, biasData);
    return {};
}

template <
    AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
    typename TileRightScale, typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(
    TileRes& cMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix, TileRightScale& bScaleMatrix,
    TileBias& biasData, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(
        TGEMV_MX, PTO_TEMPLATE_ARGS(Phase), cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix, biasData);
    return {};
}

template <
    typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
    typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(
    TileRes& cMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix, TileRightScale& bScaleMatrix,
    WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMATMUL_MX, cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix);
    return {};
}

// UF-aware overload enabling unit-flag selection via AccPhase while retaining the TMATMUL name.
template <
    AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
    typename TileRightScale, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(
    TileRes& cMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix, TileRightScale& bScaleMatrix,
    WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TMATMUL_MX, PTO_TEMPLATE_ARGS(Phase), cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix);
    return {};
}

template <
    typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
    typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(
    TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix,
    TileRightScale& bScaleMatrix, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMATMUL_MX, cOutMatrix, cInMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix);
    return {};
}

template <
    AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
    typename TileRightScale, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(
    TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix,
    TileRightScale& bScaleMatrix, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(
        TMATMUL_MX, PTO_TEMPLATE_ARGS(Phase), cOutMatrix, cInMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix);
    return {};
}

template <
    typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
    typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(
    TileRes& cMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix, TileRightScale& bScaleMatrix,
    TileBias& biasData, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMATMUL_MX, cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix, biasData);
    return {};
}

template <
    AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
    typename TileRightScale, typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(
    TileRes& cMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix, TileRightScale& bScaleMatrix,
    TileBias& biasData, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(
        TMATMUL_MX, PTO_TEMPLATE_ARGS(Phase), cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix, biasData);
    return {};
}

template <uint16_t Rounds = 10, typename DstTile, typename... WaitEvents>
PTO_INST RecordEvent TRANDOM(DstTile& dst, TRandomKey& key, TRandomCounter& counter, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TRANDOM, PTO_TEMPLATE_ARGS(Rounds, DstTile), dst, key, counter);
    return {};
}
#endif

template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMATMUL, cMatrix, aMatrix, bMatrix);
    return {};
}

// UF-aware overload enabling unit-flag selection via AccPhase while retaining the TMATMUL name.
template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TMATMUL, PTO_TEMPLATE_ARGS(Phase), cMatrix, aMatrix, bMatrix);
    return {};
}

template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent
TMATMUL_ACC(TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileRight& bMatrix, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMATMUL_ACC, cOutMatrix, cInMatrix, aMatrix, bMatrix);
    return {};
}

// UF-aware overloads for TMATMUL_ACC: explicit input/output or shared accumulator tile.
template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent
TMATMUL_ACC(TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileRight& bMatrix, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TMATMUL_ACC, PTO_TEMPLATE_ARGS(Phase), cOutMatrix, cInMatrix, aMatrix, bMatrix);
    return {};
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight,
    typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_ACC(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TMATMUL_ACC, PTO_TEMPLATE_ARGS(Phase), cMatrix, aMatrix, bMatrix);
    return {};
}

template <typename TileRes, typename TileLeft, typename TileRight, typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent
TMATMUL_BIAS(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, TileBias& biasData, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMATMUL_BIAS, cMatrix, aMatrix, bMatrix, biasData);
    return {};
}

// UF-aware overload enabling unit-flag selection for bias matmul while keeping the TMATMUL_BIAS name.
template <
    AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent
TMATMUL_BIAS(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, TileBias& biasData, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TMATMUL_BIAS, PTO_TEMPLATE_ARGS(Phase), cMatrix, aMatrix, bMatrix, biasData);
    return {};
}

template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TGEMV(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TGEMV, cMatrix, aMatrix, bMatrix);
    return {};
}

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TGEMV(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TGEMV, PTO_TEMPLATE_ARGS(Phase), cMatrix, aMatrix, bMatrix);
    return {};
}

template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent
TGEMV_ACC(TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileRight& bMatrix, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TGEMV_ACC, cOutMatrix, cInMatrix, aMatrix, bMatrix);
    return {};
}

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent
TGEMV_ACC(TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileRight& bMatrix, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TGEMV_ACC, PTO_TEMPLATE_ARGS(Phase), cOutMatrix, cInMatrix, aMatrix, bMatrix);
    return {};
}

template <typename TileRes, typename TileLeft, typename TileRight, typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent
TGEMV_BIAS(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, TileBias& biasData, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TGEMV_BIAS, cMatrix, aMatrix, bMatrix, biasData);
    return {};
}

template <
    AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent
TGEMV_BIAS(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, TileBias& biasData, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TGEMV_BIAS, PTO_TEMPLATE_ARGS(Phase), cMatrix, aMatrix, bMatrix, biasData);
    return {};
}

template <
    typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData, typename Src2TileData,
    typename Src3TileData, bool exhausted, typename... WaitEvents>
PTO_INST RecordEvent TMRGSORT(
    DstTileData& dst, MrgSortExecutedNumList& executedNumList, TmpTileData& tmp, Src0TileData& src0, Src1TileData& src1,
    Src2TileData& src2, Src3TileData& src3, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(
        TMRGSORT,
        PTO_TEMPLATE_ARGS(DstTileData, TmpTileData, Src0TileData, Src1TileData, Src2TileData, Src3TileData, exhausted),
        dst, executedNumList, tmp, src0, src1, src2, src3);
    return {};
}

template <
    typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData, typename Src2TileData,
    bool exhausted, typename... WaitEvents>
PTO_INST RecordEvent TMRGSORT(
    DstTileData& dst, MrgSortExecutedNumList& executedNumList, TmpTileData& tmp, Src0TileData& src0, Src1TileData& src1,
    Src2TileData& src2, WaitEvents&... events)
{
    TSYNC(events...);
    TMRGSORT_IMPL<DstTileData, TmpTileData, Src0TileData, Src1TileData, Src2TileData, exhausted>(
        dst, executedNumList, tmp, src0, src1, src2);
    return {};
}

template <
    typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData, bool exhausted,
    typename... WaitEvents>
PTO_INST RecordEvent TMRGSORT(
    DstTileData& dst, MrgSortExecutedNumList& executedNumList, TmpTileData& tmp, Src0TileData& src0, Src1TileData& src1,
    WaitEvents&... events)
{
    TSYNC(events...);
    TMRGSORT_IMPL<DstTileData, TmpTileData, Src0TileData, Src1TileData, exhausted>(
        dst, executedNumList, tmp, src0, src1);
    return {};
}

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TMRGSORT(DstTileData& dst, SrcTileData& src, uint32_t blockLen, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMRGSORT, dst, src, blockLen);
    return {};
}

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent
TEXTRACT(DstTileData& dst, SrcTileData& src, uint16_t indexRow = 0, uint16_t indexCol = 0, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TEXTRACT, dst, src, indexRow, indexCol);
    return {};
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode, typename... WaitEvents>
PTO_INST RecordEvent
TEXTRACT(DstTileData& dst, SrcTileData& src, uint16_t indexRow, uint16_t indexCol, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TEXTRACT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, reluMode), dst, src, indexRow, indexCol);
    return {};
}

template <
    typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
    typename... WaitEvents>
PTO_INST RecordEvent
TEXTRACT(DstTileData& dst, SrcTileData& src, uint16_t indexRow, uint16_t indexCol, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(
        TEXTRACT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, mode, reluMode), dst, src, indexRow, indexCol);
    return {};
}

template <
    typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(
    DstTileData& dst, SrcTileData& src, uint64_t preQuantScalar, uint16_t indexRow, uint16_t indexCol,
    WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(
        TEXTRACT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, reluMode), dst, src, preQuantScalar, indexRow, indexCol);
    return {};
}

template <
    typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
    typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(
    DstTileData& dst, SrcTileData& src, uint64_t preQuantScalar, uint16_t indexRow, uint16_t indexCol,
    WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(
        TEXTRACT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, mode, reluMode), dst, src, preQuantScalar, indexRow,
        indexCol);
    return {};
}

template <
    typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
    typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT_FP(
    DstTileData& dst, SrcTileData& src, FpTileData& fp, uint16_t indexRow, uint16_t indexCol, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(
        TEXTRACT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, FpTileData, reluMode), dst, src, fp, indexRow, indexCol);
    return {};
}

template <
    typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode,
    ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(
    DstTileData& dst, SrcTileData& src, FpTileData& fp, uint16_t indexRow, uint16_t indexCol, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(
        TEXTRACT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, FpTileData, mode, reluMode), dst, src, fp, indexRow,
        indexCol);
    return {};
}

template <
    typename TileData, typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL,
    typename... WaitEvents>
PTO_INST RecordEvent
TIMG2COL(TileData& dst, ConvTileData& src, uint16_t posM = 0, uint16_t posK = 0, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TIMG2COL, PTO_TEMPLATE_ARGS(TileData, ConvTileData, FmatrixMode), dst, src, posM, posK);
    return {};
}

template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent SETFMATRIX(ConvTileData& src, WaitEvents&... events)
{
    TSYNC(events...);
    SETFMATRIX_IMPL<ConvTileData, FmatrixMode>(src);
    return {};
}

template <typename OutType, typename... WaitEvents>
PTO_INST RecordEvent SET_QUANT_SCALAR(float preQuantScalar, WaitEvents&... events)
{
    TSYNC(events...);
    SET_QUANT_SCALAR_IMPL<OutType>(preQuantScalar);
    return {};
}

template <typename FpTileData, typename... WaitEvents>
PTO_INST RecordEvent SET_QUANT_VECTOR(FpTileData& fpTile, WaitEvents&... events)
{
    TSYNC(events...);
    SET_QUANT_VECTOR_IMPL<FpTileData>(fpTile);
    return {};
}

#if defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_KIRINX90)
template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent SET_IMG2COL_RPT(ConvTileData& src, WaitEvents&... events)
{
    TSYNC(events...);
    SET_IMG2COL_RPT_IMPL<ConvTileData, FmatrixMode>(src);
    return {};
}

template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent SET_IMG2COL_PADDING(ConvTileData& src, WaitEvents&... events)
{
    TSYNC(events...);
    SET_IMG2COL_PADDING_IMPL<ConvTileData, FmatrixMode>(src);
    return {};
}
#endif
#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRIN9030) || defined(__CPU_SIM)
template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent SET_IMG2COL_RPT(ConvTileData& src, WaitEvents&... events)
{
    TSYNC(events...);
    SET_IMG2COL_RPT_IMPL<ConvTileData, FmatrixMode>(src);
    return {};
}

template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent SET_IMG2COL_PADDING(ConvTileData& src, WaitEvents&... events)
{
    TSYNC(events...);
    SET_IMG2COL_PADDING_IMPL<ConvTileData, FmatrixMode>(src);
    return {};
}
#endif

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode, typename... WaitEvents>
PTO_INST RecordEvent
TINSERT(DstTileData& dst, SrcTileData& src, uint16_t indexRow, uint16_t indexCol, WaitEvents&... events)
{
    TSYNC(events...);
    TINSERT_IMPL<DstTileData, SrcTileData, reluMode>(dst, src, indexRow, indexCol);
    return {};
}

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent
TINSERT(DstTileData& dst, SrcTileData& src, uint16_t indexRow, uint16_t indexCol, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TINSERT, dst, src, indexRow, indexCol);
    return {};
}

template <
    typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
    typename... WaitEvents>
PTO_INST RecordEvent
TINSERT(DstTileData& dst, SrcTileData& src, uint16_t indexRow, uint16_t indexCol, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(
        TINSERT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, mode, reluMode), dst, src, indexRow, indexCol);
    return {};
}

template <
    typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(
    DstTileData& dst, SrcTileData& src, uint64_t preQuantScalar, uint16_t indexRow, uint16_t indexCol,
    WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(
        TINSERT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, reluMode), dst, src, preQuantScalar, indexRow, indexCol);
    return {};
}
template <
    typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
    typename... WaitEvents>
PTO_INST RecordEvent TINSERT(
    DstTileData& dst, SrcTileData& src, uint64_t preQuantScalar, uint16_t indexRow, uint16_t indexCol,
    WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(
        TINSERT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, mode, reluMode), dst, src, preQuantScalar, indexRow,
        indexCol);
    return {};
}
template <
    typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
    typename... WaitEvents>
PTO_INST RecordEvent TINSERT_FP(
    DstTileData& dst, SrcTileData& src, FpTileData& fp, uint16_t indexRow, uint16_t indexCol, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(
        TINSERT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, FpTileData, reluMode), dst, src, fp, indexRow, indexCol);
    return {};
}

template <
    typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode,
    ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent
TINSERT(DstTileData& dst, SrcTileData& src, FpTileData& fp, uint16_t indexRow, uint16_t indexCol, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(
        TINSERT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, FpTileData, mode, reluMode), dst, src, fp, indexRow,
        indexCol);
    return {};
}

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRINX90)
template <TInsertMode mode, typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent
TINSERT(DstTileData& dst, SrcTileData& src, uint16_t indexRow = 0, uint16_t indexCol = 0, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TINSERT, PTO_TEMPLATE_ARGS(mode), dst, src, indexRow, indexCol);
    return {};
}
#endif

template <
    typename TileData, PadValue PadVal = PadValue::Zero, std::enable_if_t<(TileData::Loc == TileType::Mat), int> = 0,
    typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD(TileData& dst, TileData& src, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TFILLPAD, PTO_TEMPLATE_ARGS(TileData, PadVal), dst, src);
    return {};
}

template <
    TFillPadMode mode = TFillPadMode::Normal, typename DstTileData, typename SrcTileData,
    std::enable_if_t<(DstTileData::Loc == TileType::Vec) && (SrcTileData::Loc == TileType::Vec), int> = 0,
    typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD(DstTileData& dst, SrcTileData& src, WaitEvents&... events)
{
    static_assert(
        mode == TFillPadMode::Normal || mode == TFillPadMode::InPlace || mode == TFillPadMode::Expand,
        "TFILLPAD: invalid mode.");
    TSYNC(events...);
    if constexpr (mode == TFillPadMode::Normal) {
        TFILLPAD_IMPL<DstTileData, SrcTileData>(dst, src);
    } else if constexpr (mode == TFillPadMode::InPlace) {
        MAP_INSTR_IMPL(TFILLPAD_INPLACE, dst, src);
    } else if constexpr (mode == TFillPadMode::Expand) {
        MAP_INSTR_IMPL(TFILLPAD_EXPAND, dst, src);
    }
    return {};
}

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD_INPLACE(DstTileData& dst, SrcTileData& src, WaitEvents&... events)
{
    return TFILLPAD<TFillPadMode::InPlace>(dst, src, events...);
}

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD_EXPAND(DstTileData& dst, SrcTileData& src, WaitEvents&... events)
{
    return TFILLPAD<TFillPadMode::Expand>(dst, src, events...);
}

// TSORT32不自动实现wait, 需手动TSYNC(events...)
template <typename DstTileData, typename SrcTileData, typename IdxTileData>
PTO_INST RecordEvent TSORT32(DstTileData& dst, SrcTileData& src, IdxTileData& idx)
{
    MAP_INSTR_IMPL_ROLES(TSORT32, "OIO", dst, src, idx);
    return {};
}

template <typename DstTileData, typename SrcTileData, typename IdxTileData, typename TmpTileData>
PTO_INST RecordEvent TSORT32(DstTileData& dst, SrcTileData& src, IdxTileData& idx, TmpTileData& tmp)
{
    MAP_INSTR_IMPL_ROLES(TSORT32, "OIOI", dst, src, idx, tmp);
    return {};
}

template <typename TileDataD, typename TileDataS0, typename TileDataS1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent
TGATHER(TileDataD& dst, TileDataS0& src0, TileDataS1& src1, TileDataTmp& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TGATHER, dst, src0, src1, tmp);
    return {};
}

template <
    typename TileDataD, typename TileDataS, typename TileDataS1, typename TileDataC, typename TileDataTmp,
    CmpMode cmpMode, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(
    TileDataD& dst, TileDataS& src0, TileDataS1& k_value, TileDataC& cdst, TileDataTmp& tmp, int offset,
    WaitEvents&... events)
{
    TSYNC(events...);
    TGATHER_IMPL<TileDataD, TileDataS, TileDataS1, TileDataC, TileDataTmp, cmpMode>(
        dst, src0, k_value, cdst, tmp, offset);
    return {};
}

template <typename TileData, typename T, int descending, typename... WaitEvents>
PTO_INST RecordEvent TCI(TileData& dst, T start, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TCI, PTO_TEMPLATE_ARGS(TileData, T, descending), dst, start);
    return {};
}

template <typename TileData, typename TileDataTmp, typename T, int descending, typename... WaitEvents>
PTO_INST RecordEvent TCI(TileData& dst, T start, TileDataTmp& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TCI, PTO_TEMPLATE_ARGS(TileData, TileDataTmp, T, descending), dst, start, tmp);
    return {};
}

template <typename TileData, int isUpperOrLower, typename... WaitEvents>
PTO_INST RecordEvent TTRI(TileData& dst, int diagonal, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TTRI, PTO_TEMPLATE_ARGS(TileData, isUpperOrLower), dst, diagonal);
    return {};
}

template <
    typename DstTileData, typename SrcTileData, MaskPattern maskPattern = MaskPattern::P1111,
    auto gatherType = GatherAxis::GATHER_ROW, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(DstTileData& dst, SrcTileData& src, WaitEvents&... events)
{
    TSYNC(events...);
    TGATHER_IMPL<DstTileData, SrcTileData, maskPattern, gatherType>(dst, src);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TPARTADD(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TPARTADD, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TPARTMUL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TPARTMUL, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TPARTMAX(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TPARTMAX, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TPARTMIN(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TPARTMIN, dst, src0, src1);
    return {};
}

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataDstIdx,
    typename TileDataSrc0Idx, typename TileDataSrc1Idx, typename... WaitEvents>
PTO_INST RecordEvent TPARTARGMAX(
    TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, TileDataDstIdx& dstIdx, TileDataSrc0Idx& src0Idx,
    TileDataSrc1Idx& src1Idx, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_ROLES(TPARTARGMAX, "OIIOII", dst, src0, src1, dstIdx, src0Idx, src1Idx);
    return {};
}

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataDstIdx,
    typename TileDataSrc0Idx, typename TileDataSrc1Idx, typename... WaitEvents>
PTO_INST RecordEvent TPARTARGMIN(
    TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, TileDataDstIdx& dstIdx, TileDataSrc0Idx& src0Idx,
    TileDataSrc1Idx& src1Idx, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_ROLES(TPARTARGMIN, "OIIOII", dst, src0, src1, dstIdx, src0Idx, src1Idx);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TFUSEDMULADD(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TFUSEDMULADD, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TMULADDDST(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMULADDDST, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TSUBRELU(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSUBRELU, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TFUSEDMULADDRELU(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TFUSEDMULADDRELU, dst, src0, src1);
    return {};
}

template <typename TileDataD, typename TileDataS, typename TmpTileData>
PTO_INTERNAL void TCVT_IMPL(
    TileDataD& dst, TileDataS& src, TmpTileData& tmp, RoundMode mode, SaturationMode satMode, bool needSetCtrl);
template <typename TileDataD, typename TileDataS, typename TmpTileData>
PTO_INTERNAL void TCVT_IMPL(TileDataD& dst, TileDataS& src, TmpTileData& tmp, RoundMode mode, bool needSetCtrl);
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void TCVT_IMPL(TileDataD& dst, TileDataS& src, RoundMode mode, SaturationMode satMode, bool needSetCtrl);
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void TCVT_IMPL(TileDataD& dst, TileDataS& src, RoundMode mode, bool needSetCtrl);

template <bool NeedSetCtrl = true, typename TileDataD, typename TileDataS, typename TmpTileData, typename... WaitEvents>
PTO_INST RecordEvent
TCVT(TileDataD& dst, TileDataS& src, TmpTileData& tmp, RoundMode mode, SaturationMode satMode, WaitEvents&... events)
{
    TSYNC(events...);
    TCVT_IMPL(dst, src, tmp, mode, satMode, NeedSetCtrl);
    return {};
}

template <bool NeedSetCtrl = true, typename TileDataD, typename TileDataS, typename TmpTileData, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD& dst, TileDataS& src, TmpTileData& tmp, RoundMode mode, WaitEvents&... events)
{
    TSYNC(events...);
    TCVT_IMPL(dst, src, tmp, mode, NeedSetCtrl);
    return {};
}

template <bool NeedSetCtrl = true, typename TileDataD, typename TileDataS, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD& dst, TileDataS& src, RoundMode mode, SaturationMode satMode, WaitEvents&... events)
{
    TSYNC(events...);
    TCVT_IMPL(dst, src, mode, satMode, NeedSetCtrl);
    return {};
}

template <bool NeedSetCtrl = true, typename TileDataD, typename TileDataS, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD& dst, TileDataS& src, RoundMode mode, WaitEvents&... events)
{
    TSYNC(events...);
    TCVT_IMPL(dst, src, mode, NeedSetCtrl);
    return {};
}

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMOV, dst, src);
    return {};
}

template <
    typename DstTileData, typename SrcTileData, typename TmpTileData, typename... WaitEvents,
    std::enable_if_t<is_tile_data_v<TmpTileData>, int> = 0>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, TmpTileData& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMOV, dst, src, tmp);
    return {};
}

// grp_axis-tagged X->ZZ overload (3-arg form). grp_axis=0 selects DN->ZZ on an
// axis-0-grouped (M̂×N) exponent source; grp_axis=1 (default) keeps stock ND->ZZ.
// Only the ZZ transform is parameterised; other TMOV overloads are unchanged.
#if defined(PTO_NPU_ARCH_A5) || defined(__CPU_SIM)
template <
    int grp_axis, typename DstTileData, typename SrcTileData, typename TmpTileData, typename... WaitEvents,
    std::enable_if_t<is_tile_data_v<TmpTileData>, int> = 0>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, TmpTileData& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    TMOV_IMPL<grp_axis, DstTileData, SrcTileData, TmpTileData>(dst, src, tmp);
    return {};
}
#endif

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, WaitEvents&... events)
{
    TSYNC(events...);
    TMOV_IMPL<DstTileData, SrcTileData, reluMode>(dst, src);
    return {};
}

template <STPhase Phase, typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, WaitEvents&... events)
{
    TSYNC(events...);
    TMOV_IMPL<DstTileData, SrcTileData, ReluPreMode::NoRelu, Phase>(dst, src);
    return {};
}

template <STPhase Phase, typename DstTileData, typename SrcTileData, ReluPreMode reluMode, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, WaitEvents&... events)
{
    TSYNC(events...);
    TMOV_IMPL<DstTileData, SrcTileData, reluMode, Phase>(dst, src);
    return {};
}

template <
    typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
    typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, WaitEvents&... events)
{
    TSYNC(events...);
    TMOV_IMPL<DstTileData, SrcTileData, mode, reluMode>(dst, src);
    return {};
}

template <
    STPhase Phase, typename DstTileData, typename SrcTileData, AccToVecMode mode,
    ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, WaitEvents&... events)
{
    TSYNC(events...);
    TMOV_IMPL<DstTileData, SrcTileData, mode, reluMode, Phase>(dst, src);
    return {};
}

template <
    typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
    typename... WaitEvents>
PTO_INST RecordEvent TMOV_FP(DstTileData& dst, SrcTileData& src, FpTileData& fp, WaitEvents&... events)
{
    TSYNC(events...);
    TMOV_IMPL<DstTileData, SrcTileData, FpTileData, reluMode>(dst, src, fp);
    return {};
}

template <
    STPhase Phase, typename DstTileData, typename SrcTileData, typename FpTileData,
    ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TMOV_FP(DstTileData& dst, SrcTileData& src, FpTileData& fp, WaitEvents&... events)
{
    TSYNC(events...);
    TMOV_IMPL<DstTileData, SrcTileData, FpTileData, reluMode, Phase>(dst, src, fp);
    return {};
}

template <
    typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode,
    ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, FpTileData& fp, WaitEvents&... events)
{
    TSYNC(events...);
    TMOV_IMPL<DstTileData, SrcTileData, FpTileData, mode, reluMode>(dst, src, fp);
    return {};
}

template <
    STPhase Phase, typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode,
    ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, FpTileData& fp, WaitEvents&... events)
{
    TSYNC(events...);
    TMOV_IMPL<DstTileData, SrcTileData, FpTileData, mode, reluMode, Phase>(dst, src, fp);
    return {};
}

template <
    typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, uint64_t preQuantScalar, WaitEvents&... events)
{
    TSYNC(events...);
    TMOV_IMPL<DstTileData, SrcTileData, reluMode>(dst, src, preQuantScalar);
    return {};
}

template <
    STPhase Phase, typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
    typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, uint64_t preQuantScalar, WaitEvents&... events)
{
    TSYNC(events...);
    TMOV_IMPL<DstTileData, SrcTileData, reluMode, Phase>(dst, src, preQuantScalar);
    return {};
}

template <
    typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
    typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, uint64_t preQuantScalar, WaitEvents&... events)
{
    TSYNC(events...);
    TMOV_IMPL<DstTileData, SrcTileData, mode, reluMode>(dst, src, preQuantScalar);
    return {};
}

template <
    STPhase Phase, typename DstTileData, typename SrcTileData, AccToVecMode mode,
    ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, uint64_t preQuantScalar, WaitEvents&... events)
{
    TSYNC(events...);
    TMOV_IMPL<DstTileData, SrcTileData, mode, reluMode, Phase>(dst, src, preQuantScalar);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWSUM(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWSUM, dst, src, tmp);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWPROD(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWPROD, dst, src, tmp);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TCOLSUM(TileDataOut& dst, TileDataIn& src, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLSUM, dst, src);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TCOLSUM(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp, bool isBinary, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLSUM, dst, src, tmp, isBinary);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TCOLPROD(TileDataOut& dst, TileDataIn& src, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLPROD, dst, src);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TCOLMAX(TileDataOut& dst, TileDataIn& src, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLMAX, dst, src);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TCOLARGMAX(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLARGMAX, dst, src, tmp);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TCOLARGMIN(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLARGMIN, dst, src, tmp);
    return {};
}

template <
    typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, typename TileDataTmp, typename... WaitEvents,
    std::enable_if_t<is_tile_data_v<TileDataTmp> && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent
TCOLARGMAX(TileDataOutVal& dstVal, TileDataOutIdx& dstIdx, TileDataIn& src, TileDataTmp& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_ROLES(TCOLARGMAX, "OOII", dstVal, dstIdx, src, tmp);
    return {};
}

template <
    typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, typename TileDataTmp, typename... WaitEvents,
    std::enable_if_t<is_tile_data_v<TileDataTmp> && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent
TCOLARGMIN(TileDataOutVal& dstVal, TileDataOutIdx& dstIdx, TileDataIn& src, TileDataTmp& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_ROLES(TCOLARGMIN, "OOII", dstVal, dstIdx, src, tmp);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWMAX(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWMAX, dst, src, tmp);
    return {};
}

template <
    typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents,
    std::enable_if_t<all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TROWARGMAX(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWARGMAX, dst, src, tmp);
    return {};
}

template <
    typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, typename TileDataTmp, typename... WaitEvents,
    std::enable_if_t<is_tile_data_v<TileDataTmp> && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent
TROWARGMAX(TileDataOutVal& dstVal, TileDataOutIdx& dstIdx, TileDataIn& src, TileDataTmp& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWARGMAX, dstVal, dstIdx, src, tmp);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TRESHAPE(TileDataOut& dst, TileDataIn& src, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TRESHAPE, dst, src);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWMIN(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWMIN, dst, src, tmp);
    return {};
}

template <
    typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents,
    std::enable_if_t<all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TROWARGMIN(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWARGMIN, dst, src, tmp);
    return {};
}

template <
    typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, typename TileDataTmp, typename... WaitEvents,
    std::enable_if_t<is_tile_data_v<TileDataTmp> && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent
TROWARGMIN(TileDataOutVal& dstVal, TileDataOutIdx& dstIdx, TileDataIn& src, TileDataTmp& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWARGMIN, dstVal, dstIdx, src, tmp);
    return {};
}

template <
    typename TileDataDst, typename TileDataMask, typename TileDataSrc, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TSELS(
    TileDataDst& dst, TileDataMask& mask, TileDataSrc& src, TileDataTmp& tmp, typename TileDataSrc::DType scalar,
    WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSELS, dst, mask, src, tmp, scalar);
    return {};
}

template <typename TileData, typename MaskTile, typename TmpTile, typename... WaitEvents>
PTO_INST RecordEvent
TSEL(TileData& dst, MaskTile& selMask, TileData& src0, TileData& src1, TmpTile& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSEL, dst, selMask, src0, src1, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TTRANS(TileDataDst& dst, TileDataSrc& src, TileDataTmp& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TTRANS, dst, src, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TMINS(TileDataDst& dst, TileDataSrc& src, typename TileDataSrc::DType scalar, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMINS, dst, src, scalar);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPAND(TileDataDst& dst, TileDataSrc& src, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPAND, dst, src);
    return {};
}

template <
    auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1,
    typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDDIV(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TROWEXPANDDIV, PTO_TEMPLATE_ARGS(PrecisionType), dst, src0, src1);
    return {};
}

template <
    auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1,
    typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent
TROWEXPANDDIV(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, TileDataTmp& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TROWEXPANDDIV, PTO_TEMPLATE_ARGS(PrecisionType), dst, src0, src1, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMUL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDMUL, dst, src0, src1);
    return {};
}

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent
TROWEXPANDMUL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, TileDataTmp& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDMUL, dst, src0, src1, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDSUB(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDSUB, dst, src0, src1);
    return {};
}

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent
TROWEXPANDSUB(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, TileDataTmp& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDSUB, dst, src0, src1, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDADD(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDADD, dst, src0, src1);
    return {};
}

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent
TROWEXPANDADD(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, TileDataTmp& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDADD, dst, src0, src1, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMAX(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDMAX, dst, src0, src1);
    return {};
}

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent
TROWEXPANDMAX(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, TileDataTmp& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDMAX, dst, src0, src1, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMIN(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDMIN, dst, src0, src1);
    return {};
}

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent
TROWEXPANDMIN(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, TileDataTmp& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDMIN, dst, src0, src1, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDEXPDIF(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDEXPDIF, dst, src0, src1);
    return {};
}

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent
TROWEXPANDEXPDIF(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, TileDataTmp& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDEXPDIF, dst, src0, src1, tmp);
    return {};
}

template <
    auto PrecisionType = RsqrtAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc, typename... WaitEvents,
    std::enable_if_t<all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TRSQRT(TileDataDst& dst, TileDataSrc& src, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TRSQRT, PTO_TEMPLATE_ARGS(PrecisionType), dst, src);
    return {};
}

template <
    auto PrecisionType = RsqrtAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc, typename TileDataTmp,
    typename... WaitEvents, std::enable_if_t<is_tile_data_v<TileDataTmp> && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TRSQRT(TileDataDst& dst, TileDataSrc& src, TileDataTmp& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TRSQRT, PTO_TEMPLATE_ARGS(PrecisionType), dst, src, tmp);
    return {};
}

template <
    auto PrecisionType = SqrtAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TSQRT(TileDataDst& dst, TileDataSrc& src, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TSQRT, PTO_TEMPLATE_ARGS(PrecisionType), dst, src);
    return {};
}

template <
    auto PrecisionType = ExpAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TEXP(TileDataDst& dst, TileDataSrc& src, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TEXP, PTO_TEMPLATE_ARGS(PrecisionType), dst, src);
    return {};
}

template <
    auto PrecisionType = PowAlgorithm::DEFAULT, typename DstTile, typename BaseTile, typename ExpTile, typename TmpTile,
    typename... WaitEvents>
PTO_INST RecordEvent TPOW(DstTile& dst, BaseTile& base, ExpTile& exp, TmpTile& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    TPOW_IMPL<PrecisionType>(dst, base, exp, tmp);
    return {};
}

template <
    auto PrecisionType = PowAlgorithm::DEFAULT, typename DstTile, typename BaseTile, typename TmpTile,
    typename... WaitEvents>
PTO_INST RecordEvent
TPOWS(DstTile& dst, BaseTile& base, typename DstTile::DType exp, TmpTile& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    TPOWS_IMPL<PrecisionType>(dst, base, exp, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TNOT(TileDataDst& dst, TileDataSrc& src, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TNOT, dst, src);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TRELU(TileDataDst& dst, TileDataSrc& src, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TRELU, dst, src);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataOffset, typename... WaitEvents>
PTO_INST RecordEvent TGATHERB(TileDataDst& dst, TileDataSrc& src, TileDataOffset& offset, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TGATHERB, dst, src, offset);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TADDS(TileDataDst& dst, TileDataSrc& src0, typename TileDataSrc::DType scalar, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TADDS, dst, src0, scalar);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TAXPY(TileDataDst& dst, TileDataSrc& src0, typename TileDataSrc::DType scalar, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TAXPY, dst, src0, scalar);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TSUBS(TileDataDst& dst, TileDataSrc& src0, typename TileDataSrc::DType scalar, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSUBS, dst, src0, scalar);
    return {};
}

template <
    auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TDIVS(TileDataDst& dst, TileDataSrc& src0, typename TileDataSrc::DType scalar, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TDIVS, PTO_TEMPLATE_ARGS(PrecisionType), dst, src0, scalar);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TMULS(TileDataDst& dst, TileDataSrc& src0, typename TileDataSrc::DType scalar, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMULS, dst, src0, scalar);
    return {};
}

template <
    auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TDIVS(TileDataDst& dst, typename TileDataDst::DType scalar, TileDataSrc& src0, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TDIVS, PTO_TEMPLATE_ARGS(PrecisionType), dst, scalar, src0);
    return {};
}

template <
    auto PrecisionType = FmodSAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TFMODS(TileDataDst& dst, TileDataSrc& src, typename TileDataSrc::DType scalar, WaitEvents&... events)
{
    TSYNC(events...);
    TFMODS_IMPL<PrecisionType>(dst, src, scalar);
    return {};
}

template <
    auto PrecisionType = RemSAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc, typename TileDataTmp,
    typename... WaitEvents>
PTO_INST RecordEvent
TREMS(TileDataDst& dst, TileDataSrc& src, typename TileDataSrc::DType scalar, TileDataTmp& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    TREMS_IMPL<PrecisionType>(dst, src, scalar, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TMAXS(TileDataDst& dst, TileDataSrc& src, typename TileDataSrc::DType scalar, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMAXS, dst, src, scalar);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TANDS(TileDataDst& dst, TileDataSrc& src, typename TileDataDst::DType scalar, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TANDS, dst, src, scalar);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TORS(TileDataDst& dst, TileDataSrc& src, typename TileDataDst::DType scalar, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TORS, dst, src, scalar);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TSHLS(TileDataDst& dst, TileDataSrc& src, typename TileDataDst::DType scalar, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSHLS, dst, src, scalar);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TSHRS(TileDataDst& dst, TileDataSrc& src, typename TileDataDst::DType scalar, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSHRS, dst, src, scalar);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent
TXORS(TileDataDst& dst, TileDataSrc& src0, typename TileDataSrc::DType scalar, TileDataTmp& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TXORS, dst, src0, scalar, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TLRELU(TileDataDst& dst, TileDataSrc& src, typename TileDataSrc::DType scalar, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TLRELU, dst, src, scalar);
    return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent
TADDSC(TileData& dst, TileData& src0, typename TileData::DType scalar, TileData& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TADDSC, dst, src0, scalar, src1);
    return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent
TSUBSC(TileData& dst, TileData& src0, typename TileData::DType scalar, TileData& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSUBSC, dst, src0, scalar, src1);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TCOLMIN(TileDataOut& dst, TileDataIn& src, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLMIN, dst, src);
    return {};
}

template <typename TileDataD, typename TileDataS, typename TileDataI, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(TileDataD& dst, TileDataS& src, TileDataI& indexes, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSCATTER, dst, src, indexes);
    return {};
}

template <
    MaskPattern maskPattern = MaskPattern::P1111, auto ScatterType = ScatterAxis::SCATTER_ROW, typename DstTileData,
    typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(DstTileData& dst, SrcTileData& src, WaitEvents&... events)
{
    TSYNC(events...);
    TSCATTER_IMPL<maskPattern, ScatterType>(dst, src);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPAND(TileDataDst& dst, TileDataSrc& src, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLEXPAND, dst, src);
    return {};
}

template <typename TileDst, typename GlobalData, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MGATHER(TileDst& dst, GlobalData& src, TileInd& indexes, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(MGATHER, dst, src, indexes);
    return {};
}

template <Coalesce CMode, typename TileDst, typename GlobalData, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MGATHER(TileDst& dst, GlobalData& src, TileInd& indexes, WaitEvents&... events)
{
    TSYNC(events...);
    MGATHER_IMPL<CMode>(dst, src, indexes);
    return {};
}

template <
    Coalesce CMode, GatherOOB Mode, typename TileDst, typename GlobalData, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MGATHER(TileDst& dst, GlobalData& src, TileInd& indexes, WaitEvents&... events)
{
    TSYNC(events...);
    MGATHER_IMPL<CMode, Mode>(dst, src, indexes);
    return {};
}

template <Coalesce CMode, typename TileDst, typename GlobalData, typename GlobalIdx, typename GlobalScratch>
PTO_INST RecordEvent MGATHER(TileDst& dst, GlobalData& src, GlobalIdx& indexes, GlobalScratch& scratch)
{
    MGATHER_IMPL<CMode>(dst, src, indexes, scratch);
    return {};
}

template <
    Coalesce CMode, GatherOOB Mode, typename TileDst, typename GlobalData, typename GlobalIdx, typename GlobalScratch>
PTO_INST RecordEvent MGATHER(TileDst& dst, GlobalData& src, GlobalIdx& indexes, GlobalScratch& scratch)
{
    MGATHER_IMPL<CMode, Mode>(dst, src, indexes, scratch);
    return {};
}

#ifdef PTO_NPU_ARCH_A5
template <
    Coalesce CMode, GatherOOB Mode, GatherExec Exec, typename TileDst, typename GlobalData, typename GlobalIdx,
    typename GlobalScratch>
PTO_INST RecordEvent MGATHER(TileDst& dst, GlobalData& src, GlobalIdx& indexes, GlobalScratch& scratch)
{
    MGATHER_IMPL<CMode, Mode, Exec>(dst, src, indexes, scratch);
    return {};
}
#endif

template <typename GlobalData, typename TileSrc, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MSCATTER(GlobalData& dst, TileSrc& src, TileInd& indexes, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(MSCATTER, dst, src, indexes);
    return {};
}

#if defined(PTO_NPU_ARCH_A5) || defined(__CPU_SIM)
template <Coalesce Mode, typename GlobalData, typename TileSrc, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MSCATTER(GlobalData& dst, TileSrc& src, TileInd& indexes, WaitEvents&... events)
{
    TSYNC(events...);
    MSCATTER_IMPL<Mode>(dst, src, indexes);
    return {};
}

template <
    Coalesce Mode, ScatterAtomicOp Atomic, typename GlobalData, typename TileSrc, typename TileInd,
    typename... WaitEvents>
PTO_INST RecordEvent MSCATTER(GlobalData& dst, TileSrc& src, TileInd& indexes, WaitEvents&... events)
{
    TSYNC(events...);
    MSCATTER_IMPL<Mode, Atomic>(dst, src, indexes);
    return {};
}

template <
    Coalesce Mode, ScatterAtomicOp Atomic, ScatterOOB Oob, typename GlobalData, typename TileSrc, typename TileInd,
    typename... WaitEvents>
PTO_INST RecordEvent MSCATTER(GlobalData& dst, TileSrc& src, TileInd& indexes, WaitEvents&... events)
{
    TSYNC(events...);
    MSCATTER_IMPL<Mode, Atomic, Oob>(dst, src, indexes);
    return {};
}

template <
    Coalesce Mode, ScatterAtomicOp Atomic, ScatterOOB Oob, ScatterConflict Conflict, typename GlobalData,
    typename TileSrc, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MSCATTER(GlobalData& dst, TileSrc& src, TileInd& indexes, WaitEvents&... events)
{
    TSYNC(events...);
    MSCATTER_IMPL<Mode, Atomic, Oob, Conflict>(dst, src, indexes);
    return {};
}
#endif

#ifdef PTO_NPU_ARCH_A2A3
template <Coalesce Mode, typename GlobalData, typename TileSrc, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MSCATTER(GlobalData& dst, TileSrc& src, TileInd& indexes, WaitEvents&... events)
{
    TSYNC(events...);
    MSCATTER_IMPL<Mode>(dst, src, indexes);
    return {};
}

template <
    Coalesce Mode, ScatterAtomicOp Atomic, typename GlobalData, typename TileSrc, typename TileInd,
    typename... WaitEvents>
PTO_INST RecordEvent MSCATTER(GlobalData& dst, TileSrc& src, TileInd& indexes, WaitEvents&... events)
{
    TSYNC(events...);
    MSCATTER_IMPL<Mode, Atomic>(dst, src, indexes);
    return {};
}

template <
    Coalesce Mode, ScatterAtomicOp Atomic, ScatterOOB Oob, typename GlobalData, typename TileSrc, typename TileInd,
    typename... WaitEvents>
PTO_INST RecordEvent MSCATTER(GlobalData& dst, TileSrc& src, TileInd& indexes, WaitEvents&... events)
{
    TSYNC(events...);
    MSCATTER_IMPL<Mode, Atomic, Oob>(dst, src, indexes);
    return {};
}
#endif

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TNEG(TileDataDst& dst, TileDataSrc& src, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TNEG, dst, src);
    return {};
}

template <
    auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1,
    typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDDIV(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TCOLEXPANDDIV, PTO_TEMPLATE_ARGS(PrecisionType), dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDMUL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLEXPANDMUL, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDADD(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLEXPANDADD, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDMAX(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLEXPANDMAX, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDMIN(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLEXPANDMIN, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDSUB(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLEXPANDSUB, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDEXPDIF(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLEXPANDEXPDIF, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataPara, typename... WaitEvents>
PTO_INST RecordEvent
TDEQUANT(TileDataDst& dst, TileDataSrc& src, TileDataPara& scale, TileDataPara& offset, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TDEQUANT, dst, src, scale, offset);
    return {};
}

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TADDDEQRELU(
    TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, float deqScale, TileDataTmp& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    TADDDEQRELU_IMPL(dst, src0, src1, deqScale, tmp);
    return {};
}

template <
    auto PrecisionType = RemAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1,
    typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent
TREM(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, TileDataTmp& tmp, WaitEvents&... events)
{
    TSYNC(events...);
    TREM_IMPL<PrecisionType>(dst, src0, src1, tmp);
    return {};
}

template <
    auto PrecisionType = FmodAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1,
    typename... WaitEvents>
PTO_INST RecordEvent TFMOD(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    TSYNC(events...);
    TFMOD_IMPL<PrecisionType>(dst, src0, src1);
    return {};
}

#ifndef PTO_COMM_NOT_SUPPORTED
template <
    typename Pipe, typename TileProd, TileSplitAxis Split, std::enable_if_t<is_tile_data_v<TileProd>, int> = 0,
    typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe& pipe, TileProd& tile, WaitEvents&... events)
{
    TSYNC(events...);
    PTO_INSTR_SCOPE_OUTS(TPUSH, 0, pipe, tile);
    TPUSH_IMPL<Pipe, TileProd, Split>(pipe, tile);
    return {};
}

template <typename Pipe, typename TileProd, TileSplitAxis Split, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe& pipe, TileProd& tile, int32_t subBlockId, WaitEvents&... events)
{
    TSYNC(events...);
    TPUSH_IMPL<Pipe, TileProd, Split>(pipe, tile, subBlockId);
    return {};
}

template <typename TileData, typename Pipe, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(TileData& tile, Pipe& pipe, WaitEvents&... events)
{
    TSYNC(events...);
    TPUSH_IMPL<TileData, Pipe>(tile, pipe);
    return {};
}

template <
    typename Pipe, typename TileCons, TileSplitAxis Split, std::enable_if_t<is_tile_data_v<TileCons>, int> = 0,
    typename... WaitEvents>
PTO_INST RecordEvent TPOP(Pipe& pipe, TileCons& tile, WaitEvents&... events)
{
    TSYNC(events...);
    PTO_INSTR_SCOPE(TPOP, pipe, tile);
    TPOP_IMPL<Pipe, TileCons, Split>(pipe, tile);
    return {};
}

template <typename Pipe, typename TileCons, TileSplitAxis Split, typename... WaitEvents>
PTO_INST RecordEvent TPOP(Pipe& pipe, TileCons& tile, int32_t subBlockId, WaitEvents&... events)
{
    TSYNC(events...);
    TPOP_IMPL<Pipe, TileCons, Split>(pipe, tile, subBlockId);
    return {};
}

template <typename TileData, typename Pipe, typename... WaitEvents>
PTO_INST RecordEvent TPOP(TileData& tile, Pipe& pipe, WaitEvents&... events)
{
    TSYNC(events...);
    PTO_INSTR_SCOPE(TPOP, tile, pipe);
#ifdef __CPU_SIM
    TPOP_REVERSED_IMPL<TileData, Pipe>(tile, pipe);
#else
    TPOP_IMPL(tile, pipe);
#endif
    return {};
}

template <typename Pipe, TileSplitAxis Split, typename... WaitEvents>
PTO_INST RecordEvent TFREE(Pipe& pipe, WaitEvents&... events)
{
    TSYNC(events...);
    PTO_INSTR_SCOPE_OUTS(TFREE, 0, pipe);
    TFREE_IMPL<Pipe, Split>(pipe);
    return {};
}

template <
    typename Pipe, typename GlobalData, TileSplitAxis Split, std::enable_if_t<is_global_data_v<GlobalData>, int> = 0,
    typename... WaitEvents>
PTO_INST RecordEvent TALLOC(Pipe& pipe, GlobalData& gmTensor, WaitEvents&... events)
{
    TSYNC(events...);
    TALLOC_IMPL<Pipe, GlobalData, Split>(pipe, gmTensor);
    return {};
}

template <
    typename Pipe, typename GlobalData, TileSplitAxis Split, std::enable_if_t<is_global_data_v<GlobalData>, int> = 0,
    typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe& pipe, GlobalData& gmTensor, WaitEvents&... events)
{
    TSYNC(events...);
    TPUSH_IMPL<Pipe, GlobalData, Split>(pipe, gmTensor);
    return {};
}

template <typename Pipe, typename TileProd, typename TConfig, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe& pipe, TileProd& tile, WaitEvents&... events)
{
    TSYNC(events...);
    TPUSH_IMPL<Pipe, TileProd, TConfig>(pipe, tile);
    return {};
}

template <
    typename Pipe, typename GlobalData, TileSplitAxis Split, std::enable_if_t<is_global_data_v<GlobalData>, int> = 0,
    typename... WaitEvents>
PTO_INST RecordEvent TPOP(Pipe& pipe, GlobalData& gmTensor, WaitEvents&... events)
{
    TSYNC(events...);
    TPOP_IMPL<Pipe, GlobalData, Split>(pipe, gmTensor);
    return {};
}

template <
    typename Pipe, typename GlobalData, TileSplitAxis Split, std::enable_if_t<is_global_data_v<GlobalData>, int> = 0,
    typename... WaitEvents>
PTO_INST RecordEvent TFREE(Pipe& pipe, GlobalData& gmTensor, WaitEvents&... events)
{
    TSYNC(events...);
    TFREE_IMPL<Pipe, GlobalData, Split>(pipe, gmTensor);
    return {};
}

template <typename Pipe, typename... WaitEvents>
PTO_INST RecordEvent TFREE(Pipe& pipe, WaitEvents&... events)
{
    TSYNC(events...);
    TFREE_IMPL<Pipe>(pipe);
    return {};
}
#endif

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6) || defined(PTO_NPU_ARCH_KIRIN9030) || defined(__CPU_SIM)
template <HistByte byte, typename TileDataDst, typename TileDataSrc, typename TileDataIdx, typename... WaitEvents>
PTO_INST RecordEvent THISTOGRAM(TileDataDst& dst, TileDataSrc& src, TileDataIdx& idx, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(THISTOGRAM, PTO_TEMPLATE_ARGS(byte), dst, src, idx);
    return {};
}

template <
    auto quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataExp, typename TileDataMax,
    typename TileDataScaling, auto scale_alg = QuantScaleAlg::OCP, typename... WaitEvents>
PTO_INST RecordEvent TQUANT(
    TileDataOut& dst, TileDataSrc& src, TileDataExp* exp, TileDataMax* max, TileDataScaling* scaling,
    WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T_ROLES(
        TQUANT,
        PTO_TEMPLATE_ARGS(quant_type, scale_alg, TileDataOut, TileDataSrc, TileDataExp, TileDataMax, TileDataScaling),
        "OIOOO", dst, src, exp, max, scaling);
    return {};
}

template <
    auto quant_type, auto store_mode, typename TileDataOut, typename TileDataSrc, typename TileDataExp,
    typename TileDataMax, typename TileDataScaling, typename... WaitEvents>
PTO_INST RecordEvent TQUANT(
    TileDataOut& dst, TileDataSrc& src, TileDataExp* exp, TileDataMax* max, TileDataScaling* scaling,
    TileDataExp* exp_zz, WaitEvents&... events)
{
    TSYNC(events...);
    TQUANT_IMPL<quant_type, store_mode>(dst, src, exp, max, scaling, exp_zz);
    return {};
}

template <
    int grp_axis, auto mx_alg, typename TileDataOut = void, typename TileDataSrc = void, typename TileDataExp = void,
    typename TileDataMax = void, typename TileDataScaling = void, typename... WaitEvents>
PTO_INST RecordEvent TQUANT(
    TileDataOut& dst, TileDataSrc& src, TileDataExp* exp, TileDataMax* max, TileDataScaling* scaling,
    WaitEvents&... events)
{
    TSYNC(events...);
    TQUANT_IMPL<grp_axis, mx_alg, TileDataOut, TileDataSrc, TileDataExp, TileDataMax, TileDataScaling>(
        dst, src, exp, max, scaling);
    return {};
}

template <
    int grp_axis, auto mx_alg, bool interleave, typename TileDataOut = void, typename TileDataSrc = void,
    typename TileDataExp = void, typename TileDataMax = void, typename TileDataScaling = void, typename... WaitEvents>
PTO_INST RecordEvent TQUANT(
    TileDataOut& dst, TileDataSrc& src, TileDataExp* exp, TileDataMax* max, TileDataScaling* scaling,
    WaitEvents&... events)
{
    TSYNC(events...);
    TQUANT_IMPL<grp_axis, mx_alg, interleave, TileDataOut, TileDataSrc, TileDataExp, TileDataMax, TileDataScaling>(
        dst, src, exp, max, scaling);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TINTERLEAVE(TileDataDst& dst1, TileDataDst& dst0, TileDataSrc& src1, TileDataSrc& src0, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TINTERLEAVE, dst1, dst0, src1, src0);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TDEINTERLEAVE(TileDataDst& dst1, TileDataDst& dst0, TileDataSrc& src1, TileDataSrc& src0, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TDEINTERLEAVE, dst1, dst0, src1, src0);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TDEINTERLEAVE(TileDataDst& dst1, TileDataDst& dst0, TileDataSrc& src, WaitEvents&... events)
{
    TSYNC(events...);
    TDEINTERLEAVE_IMPL(dst1, dst0, src);
    return {};
}
#endif

template <auto quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataPara, typename... WaitEvents>
PTO_INST RecordEvent
TQUANT(TileDataOut& dst, TileDataSrc& src, TileDataPara& scale, TileDataPara* offset = nullptr, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(
        TQUANT, PTO_TEMPLATE_ARGS(quant_type, TileDataOut, TileDataSrc, TileDataPara), dst, src, scale, offset);
    return {};
}

// Tmp-aware overload (A2/A3): the row-wise scale/offset broadcast needs an explicit scratch tile.
template <
    auto quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataPara, typename TileDataTmp,
    typename... WaitEvents>
PTO_INST RecordEvent TQUANT(
    TileDataOut& dst, TileDataSrc& src, TileDataPara& scale, TileDataTmp& tmp, TileDataPara* offset = nullptr,
    WaitEvents&... events)
{
    TSYNC(events...);
    TQUANT_IMPL<quant_type, TileDataOut, TileDataSrc, TileDataPara, TileDataTmp>(dst, src, scale, tmp, offset);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TGET_SCALE_ADDR(TileDataOut& dst, TileDataIn& src, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TGET_SCALE_ADDR, dst, src);
    return {};
}

#ifndef __CPU_SIM
// ---------------------------------------------------------------------------
// GridPipe TPUSH / TPOP overloads (design doc section 4.1, "neighbor-core
// FIFO" form).  These coexist with the cluster-local TPipe overloads above:
// SFINAE on is_grid_pipe_v keeps overload resolution unambiguous.  The
// `Direction` non-type template parameter is constant-folded by the compiler,
// matching design doc section 4.2's requirement that direction be a constant
// at lowering time.
//
// `Direction` is the ONLY channel argument: a grid transfer moves the tile
// exactly one hop, to the ADJACENT cell along it.  There is no distance operand
// to get wrong, and no producer/consumer binding either -- both peers are derived
// from (Direction, coord, shape) at the call site, so the same pipe keeps serving
// the same direction across phases even when the core on the other end changes.
// A longer reach is a relay: one TPUSH per edge, each with its own credit.
// ---------------------------------------------------------------------------

template <
    pto::GridDirection Direction, typename Pipe, typename TileProd, std::enable_if_t<is_grid_pipe_v<Pipe>, int> = 0,
    typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe& pipe, TileProd& tile, WaitEvents&... events)
{
    static_assert(
        Direction != pto::GridDirection::SOURCE, "GridPipe TPUSH<SOURCE> is illegal (design doc section 4.3): "
                                                 "SOURCE is only valid for TPOP.");
    // TPUSH<EAST>(pipe, tile) publishes into the ring of the cell one hop east.
#if defined(PTO_NPU_ARCH_A2A3)
    TSYNC(events...);
    GRID_TPUSH_IMPL<Direction, Pipe, TileProd>(pipe, tile);
#else
    static_assert(
        sizeof(Pipe) == 0, "GridPipe TPUSH not supported on this target profile "
                           "(design doc section 5.4 forbids silent GM fallback).");
#endif
    return {};
}

template <
    pto::GridDirection Direction, typename Pipe, typename TileCons, std::enable_if_t<is_grid_pipe_v<Pipe>, int> = 0,
    typename... WaitEvents>
PTO_INST RecordEvent TPOP(Pipe& pipe, TileCons& tile, WaitEvents&... events)
{
    // Drains this core's own `Direction` ring and returns the free credit to the
    // adjacent upstream cell -- the mirror of the producer's TPUSH<Direction>.
#if defined(PTO_NPU_ARCH_A2A3)
    TSYNC(events...);
    GRID_TPOP_IMPL<Direction, Pipe, TileCons>(pipe, tile);
#else
    static_assert(
        sizeof(Pipe) == 0, "GridPipe TPOP not supported on this target profile "
                           "(design doc section 5.4 forbids silent GM fallback).");
#endif
    return {};
}

// ---------------------------------------------------------------------------
// GridPipe TREDUCE overload: one fused "receive-combine-forward" reduce hop
// along `Direction` (design doc section 5, worked ReduceSum example).  It is the
// AllGather relay hop (TPOP<Dir>+TPUSH<Dir>) with a combine folded in between:
// an interior/sink cell drains the transiting partial from its upstream, folds
// its own `acc` in with `Op` (Sum/Max/Min), and forwards the running reduction
// downstream; a source cell only forwards; a sink cell keeps the complete result
// in `acc` for the caller to store.  Roles are derived from (Direction, coord,
// shape) -- no explicit root flag.  Being built out of TPOP + TPUSH, a reduce hop
// spans exactly one edge like they do; a whole-row reduction is the chain of such
// hops, which is what makes it systolic.  `Op` is the SAME pto::comm::ReduceOp
// the collective TREDUCE uses.  SFINAE on is_grid_pipe_v keeps it distinct from
// the collective pto::comm::TREDUCE (whose first argument is a ParallelGroup).
//
// This is the ISA surface for hardware that can combine ON TRANSIT (随路/过路
// compute): such a fabric lowers the whole hop to one routed reduce-forward and
// `recv` is unused.  On A2/A3 there is no on-transit compute and the adder is
// core-local, so it lowers to the local TPOP + combine + TPUSH sequence in
// GRID_TREDUCE_IMPL; `recv` is the mandatory landing tile for the in-core add.
// ---------------------------------------------------------------------------
template <
    pto::GridDirection Direction, pto::comm::ReduceOp Op, typename Pipe, typename TileAcc, typename TileRecv,
    std::enable_if_t<is_grid_pipe_v<Pipe>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TREDUCE(Pipe& pipe, TileAcc& acc, TileRecv& recv, WaitEvents&... events)
{
    static_assert(
        Direction != pto::GridDirection::SOURCE, "GridPipe TREDUCE<SOURCE> is illegal: SOURCE is only valid for TPOP.");
    // TREDUCE<EAST, Sum>(pipe, acc, recv) folds `acc` into the EAST-flowing
    // reduction and forwards it to the adjacent cell one hop east.
#if defined(PTO_NPU_ARCH_A2A3)
    TSYNC(events...);
    GRID_TREDUCE_IMPL<Direction, Op, Pipe, TileAcc, TileRecv>(pipe, acc, recv);
#else
    static_assert(
        sizeof(Pipe) == 0, "GridPipe TREDUCE not supported on this target profile "
                           "(design doc section 5.4 forbids silent GM fallback).");
#endif
    return {};
}

// ---------------------------------------------------------------------------
// GridGroup TREDUCE overload: the N->1 group fan-in shape of the same reduce.
// EVERY member of the group calls it and compares its own rank-in-group with the
// `sinkBlockId` operand: the block it names collects, the rest contribute.  The sink
// reads every member's resolved contribution out of a uniform-stride arena and
// folds them with `Op` into `acc` -- ONE mov_ubuf_group(op = SUM/MAX/MIN) rather
// than the hop-by-hop chain the relay expands to.  The other members contribute:
// their data is already in the arena, so their half is the handshake alone
// (GridTReduce.hpp section "GRID_TREDUCE_GROUP_IMPL").
//
// `sinkBlockId` is a RUNTIME operand, not the "last member" convention it
// replaces: where the result is collected is a placement decision of the caller
// (it is wherever the value is next needed), and the collecting core is by
// construction the one that issues the gather.  It is a LOGICAL BLOCK ID -- the
// integer get_block_idx() returns, `row * gridCols + col` -- because that is how
// every peer operand in this family names a core; there is no device rank here,
// only blocks of one launch.  Every member must pass the SAME value; the sink is
// the member it points at.  A block outside the group traps as 0x405
// kFaultGroupBadPeer on every member rather than resolving to a stranger.
//
// It DOES take a pipe: the fan-in is notified through the sink's own directional
// scoreboards, the ones TPUSH would use.  Contributors on one side of the sink
// all ring the same scoreboard at different moments; an interior sink therefore
// waits on the two sides separately (empty side skipped), which is the same
// back/forward split TBROADCAST has.  The two TREDUCE
// overloads still cannot be confused: this one's first template argument is a
// pto::GridGroup and the relay's is a pto::GridDirection, so each is discarded
// during substitution into the other.
//
// `scratch` is the in-core combine scratch (one member's worth of UB, required by
// the A3 mock's core-local adder).  `groupSlotBase` / `memberStride` /
// `memberCount` describe the contribution arena (ROW/COL members occupy
// consecutive grid ranks, so uniform stride is always the right model).  Members
// fold in ascending index order, so an SPMD row/col fan-in matches the relay's
// accumulation bit-for-bit.  Contributors
// ignore `acc` / `scratch` / the arena operands, but still pass them so every
// member issues the identical instruction under SPMD.
// ---------------------------------------------------------------------------
template <
    pto::GridGroup Group, pto::comm::ReduceOp Op, typename T, typename Pipe, typename TileAcc, typename TileScratch,
    std::enable_if_t<is_grid_pipe_v<Pipe>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TREDUCE(
    Pipe& pipe, TileAcc& acc, TileScratch& scratch, __gm__ const T* groupSlotBase, uint32_t bytes, uint32_t memberCount,
    int sinkBlockId, uint32_t memberStride = 0, WaitEvents&... events)
{
#if defined(PTO_NPU_ARCH_A2A3)
    TSYNC(events...);
    GRID_TREDUCE_GROUP_IMPL<Group, Op, T, Pipe, TileAcc, TileScratch>(
        pipe, acc, scratch, groupSlotBase, bytes, memberCount, sinkBlockId, memberStride);
#else
    static_assert(
        sizeof(T) == 0, "GridGroup TREDUCE not supported on this target profile "
                        "(design doc section 5.4 forbids silent GM fallback).");
#endif
    return {};
}

// ---------------------------------------------------------------------------
// GridPipe TBROADCAST overload: group broadcast.  The first explicit template
// argument is a GridGroup -- the participant set, a whole row or a whole column
// and nothing else -- which also selects this overload against the unicast
// TPUSH<GridDirection> above (the two scoped enums never interconvert, so
// resolution is unambiguous and folds at compile time).  A group collective
// names its members outright, so there is no hop count here either.
//
// A source writes each receiver's ORDINARY directional ring at its own
// prod_idx % SlotCount and then INCREMENTS that receiver's ready_scb_<dir> by
// one (sync_hscb_add).  There is no group-private ring, no per-source slot
// partition and no agreed count: an add carries no assumption about who else
// writes the scoreboard, which is what lets one scoreboard serve a whole group
// over time -- including a single-source broadcast, where only one member ever
// publishes.
//
// IT RETURNS ONLY ONCE EVERY RECEIVER HAS TPOPed IT.  Consumption is the
// receivers' TPOP and nothing else; this call blocks until all of their TPOPs
// have completed (their free_scb increments are the proof), so on return this
// source's tile occupies no undrained slot anywhere.  The threshold carries no
// SlotCount term: the group contract is stricter than slot reuse, so a source
// gives up cross-round pipelining -- the price of one shared ring.
//
// SOURCES TAKE TURNS, but this instruction does not schedule them.  The group
// shares one ring slot per direction, so at any instant exactly one member may
// be inside TBROADCAST.  The drain wait above is what PROVES the slot is clear
// again; forwarding that verdict to whichever member should publish next is a
// separate instruction, TBNOTIFY<Group>(pipe, dstBlockId), issued right after
// this call, and the member it names is blocked in TBWAIT<Group> until it
// arrives.  The verdict cannot be derived by the next source -- no counter of
// its own moves when someone else's tile is drained, and it cannot read a peer's
// state -- so it is a message, but a free one: ONE increment on the scoreboard
// of the axis the group does not span, no payload and nothing for the receivers
// to drain.  A single-source broadcast has no turn to pass and calls neither.
// Publishing without taking the turn is a CALLER error, not a supported mode,
// and the receive half traps what it can see locally (kFaultGroupOutOfOrder).
//
// Receivers drain a member's shard with the TPOP<GridGroup> overload below, one
// shard per call, naming that member by its logical block id.  See GRID_TBROADCAST_IMPL / GRID_TBPOP_IMPL.
// ---------------------------------------------------------------------------
template <
    pto::GridGroup Group, typename Pipe, typename TileProd, std::enable_if_t<is_grid_pipe_v<Pipe>, int> = 0,
    typename... WaitEvents>
PTO_INST RecordEvent TBROADCAST(Pipe& pipe, TileProd& tile, WaitEvents&... events)
{
#if defined(PTO_NPU_ARCH_A2A3)
    TSYNC(events...);
    GRID_TBROADCAST_IMPL<Group, Pipe, TileProd>(pipe, tile);
#else
    static_assert(
        sizeof(Pipe) == 0, "GridPipe TBROADCAST not supported on this target profile "
                           "(design doc section 5.4 forbids silent GM fallback).");
#endif
    return {};
}

// ---------------------------------------------------------------------------
// GridPipe TBWAIT<GridGroup>: block until this core may write the group's shared
// ring slot -- and write nothing.
//
// It is the back-pressure half of TBROADCAST, on its own: the test that a second
// publish by this core would have to pass, without the payload and without the
// doorbells.  Because the group shares ONE ring slot per direction, that test is
// about the PREVIOUS source's tile, and the previous source is the only core
// that can evaluate it (the receivers credit ITS free_scb when they drain).  So
// it evaluates it inside its own TBROADCAST and forwards the verdict with
// TBNOTIFY<Group> below; TBWAIT consumes one such verdict.
//
// ONE TBWAIT IS ONE TBNOTIFY.  This call consumes exactly one turn token and has
// no exemption for any member, any index or any round: both counts are simply the
// number of calls, which is what lets the two sides stay in step without agreeing
// on an absolute value and without either knowing the schedule.
//
// The whole caller-side obligation of a multi-source group collective is
// therefore two calls around the publish -- take the turn, then pass it on --
// with the two ENDS of the chain left open, since a token must be minted before
// it can be consumed:
//
//     for (int src = 0; src < groupSize; ++src) {
//         if (src == myRank) {
//             if (!isFirstPublisher)                  // nobody notified it
//                 TBWAIT<Group>(pipe);                // wait for the slot, write nothing
//             TBROADCAST<Group>(pipe, myTile);        // publish; returns fully drained
//             if (hasNextPublisher)                   // else the token is stranded
//                 TBNOTIFY<Group>(pipe, nextBlockId); // hand the turn on
//         } else {
//             TPOP<Group>(pipe, recvTile, src);       // the consumption itself
//         }
//     }
//
// Receivers call NOTHING extra -- there is no barrier packet to drain.
//
// SCOPE: a waiter blocks until someone notifies it, and who that is comes
// entirely from the caller's TBNOTIFY, so any publish order works -- the full
// AllGather ring, a subset of the members, or an order picked at runtime.  A
// schedule that keeps circulating over several rounds wraps instead of stopping:
// its last publisher notifies the first, whose next TBWAIT consumes that token.
// The first publisher may also mint its own token with
// TBNOTIFY<Group>(pipe, ownBlockId) and then wait like everyone else.  A
// single-source broadcast omits both halves -- with one publisher nothing else
// can be holding the slot -- and so does a group of one member, which is both
// the first and the last publisher.
// ---------------------------------------------------------------------------
template <pto::GridGroup Group, typename Pipe, std::enable_if_t<is_grid_pipe_v<Pipe>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TBWAIT(Pipe& pipe, WaitEvents&... events)
{
#if defined(PTO_NPU_ARCH_A2A3)
    TSYNC(events...);
    GRID_TBWAIT_IMPL<Group, Pipe>(pipe);
#else
    static_assert(
        sizeof(Pipe) == 0, "GridPipe TBWAIT not supported on this target profile "
                           "(design doc section 5.4 forbids silent GM fallback).");
#endif
    return {};
}

// ---------------------------------------------------------------------------
// GridPipe TBNOTIFY<GridGroup>: hand the publish turn to ONE member of the
// group -- the send half of the TBWAIT handshake, and its only counterpart.
//
// It carries no payload and no ring slot: it adds 1 to the turn scoreboard of
// the block named by `dstBlockId`, on the axis the group does NOT span (NORTH
// for a ROW group, EAST for a COL group), which a group pipe leaves idle.  So
// turn-taking costs no scoreboard, no ring, no window and no packet, and the
// group's own counts are untouched -- the receive half's order check stays
// exact.  There is nothing for the receivers to drain either; only the notified
// member ever reads that word.
//
// WHAT IT MEANS is the verdict TBROADCAST just proved: every receiver has TPOPed
// this source's tile, so the group's shared ring slot is clear.  Hence the one
// ordering rule -- issue it AFTER the TBROADCAST it speaks for.  Issued before,
// it releases a source into a slot that is still occupied.
//
// WHY IT IS SEPARATE from TBROADCAST: the publish establishes a FACT about this
// source's tile, while who publishes next is a SCHEDULE, and the schedule is the
// caller's.  Rank+1 with wrap-around is only the AllGather shape of it; a
// collective where a subset publishes, or where the order is data-dependent, or
// where the turn crosses a phase boundary, names its successor directly instead.
//
// `dstBlockId` names that successor the way every peer operand in this family
// does -- the LOGICAL BLOCK ID `row * gridCols + col`, what get_block_idx()
// returns, not a position within the group; GroupMemberBlockId(Group, coord,
// shape, indexInGroup) is the conversion when a kernel walks the group by
// position.  A block outside the group traps as 0x405 kFaultGroupBadPeer rather
// than incrementing a stranger's word.  Naming THIS core is legal and is how the
// first publisher mints its own token when a caller would rather every member
// run the identical TBWAIT / TBROADCAST / TBNOTIFY sequence; used anywhere else
// it releases the caller's own next TBWAIT, which the instruction cannot tell
// apart from that deliberate self-hand-off.
//
// EVERY TOKEN MUST BE CONSUMED.  TBWAIT has no exemption of any kind, so a
// notification with no matching wait is not merely wasted -- it sits in the
// target's scoreboard and will satisfy the first TBWAIT of a later round, or of
// a later launch that reuses the window.  The last publisher of a finite walk
// therefore does not call this at all; only a schedule that genuinely wraps
// round-to-round has its last publisher notify its first.
// ---------------------------------------------------------------------------
template <pto::GridGroup Group, typename Pipe, std::enable_if_t<is_grid_pipe_v<Pipe>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TBNOTIFY(Pipe& pipe, int dstBlockId, WaitEvents&... events)
{
#if defined(PTO_NPU_ARCH_A2A3)
    TSYNC(events...);
    GRID_TBNOTIFY_IMPL<Group, Pipe>(pipe, dstBlockId);
#else
    static_assert(
        sizeof(Pipe) == 0, "GridPipe TBNOTIFY not supported on this target profile "
                           "(design doc section 5.4 forbids silent GM fallback).");
#endif
    return {};
}

// GridPipe TPOP<GridGroup> overload: drain ONE shard that the source with
// logical block id `srcBlockId` broadcast into this receiver's shared ring (the
// receive half of TBROADCAST).  The GridGroup first template argument selects
// this overload against the unicast TPOP<GridDirection> above.  `srcBlockId`
// names the broadcasting core the same way every peer operand here does -- the
// integer get_block_idx() returns -- and it is used for exactly two things: the
// DIRECTION that source's edge takes, and the address the retire credit goes
// back to.  The wait threshold and the ring slot are this receiver's own
// cons_idx, exactly as TPOP<dir> computes them.  A block outside this group
// traps as 0x405 kFaultGroupBadPeer.  Walking a group by position instead?
// pto::GroupMemberBlockId(Group, coord, shape, index) is the conversion.
template <
    pto::GridGroup Group, typename Pipe, typename TileCons, std::enable_if_t<is_grid_pipe_v<Pipe>, int> = 0,
    typename... WaitEvents>
PTO_INST RecordEvent TPOP(Pipe& pipe, TileCons& tile, int srcBlockId, WaitEvents&... events)
{
#if defined(PTO_NPU_ARCH_A2A3)
    TSYNC(events...);
    GRID_TBPOP_IMPL<Group, Pipe, TileCons>(pipe, tile, srcBlockId);
#else
    static_assert(
        sizeof(Pipe) == 0, "GridPipe TPOP<GridGroup> not supported on this target profile "
                           "(design doc section 5.4 forbids silent GM fallback).");
#endif
    return {};
}
#endif

} // namespace pto
#endif
