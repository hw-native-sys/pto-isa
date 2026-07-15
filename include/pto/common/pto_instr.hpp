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
#include "pto/npu/a2a3/grid_intrinsic.hpp"
#include "pto/common/tassign_check.hpp"
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

template <SyncAllMode Mode, SyncCoreType CoreType = SyncCoreType::AIVOnly, typename GlobalData, typename TileData,
          std::enable_if_t<is_global_data_v<GlobalData> && is_tile_data_v<TileData> && TileData::Loc == TileType::Vec,
                           int> = 0>
PTO_INST void SYNCALL(GlobalData &gmWorkspace, TileData &ubWorkspace, int32_t usedCores = 0)
{
#if defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_A5) || defined(__CPU_SIM)
    if constexpr (Mode == SyncAllMode::Hard) {
        (void)gmWorkspace;
        (void)ubWorkspace;
        (void)usedCores;
        SYNCALL_IMPL<CoreType>();
    } else {
#ifndef __PTO_AUTO__
        SYNCALL_SOFT_IMPL<CoreType>(gmWorkspace.data(), ubWorkspace.data(), usedCores);
#endif
    }
#else
    PTO_STATIC_ASSERT(Mode != Mode, "SYNCALL is not supported on this backend.");
#endif
}

template <SyncAllMode Mode, SyncCoreType CoreType = SyncCoreType::AICOnly, typename GlobalData, typename TileData,
          std::enable_if_t<is_global_data_v<GlobalData> && is_tile_data_v<TileData> && TileData::Loc == TileType::Mat,
                           int> = 0>
PTO_INST void SYNCALL(GlobalData &gmWorkspace, TileData &l1Workspace, int32_t usedCores = 0)
{
#if defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_A5) || defined(__CPU_SIM)
    PTO_STATIC_ASSERT(CoreType == SyncCoreType::AICOnly, "GM+L1 overload is for AIC-only mode.");
    if constexpr (Mode == SyncAllMode::Hard) {
        (void)gmWorkspace;
        (void)l1Workspace;
        (void)usedCores;
        SYNCALL_IMPL<CoreType>();
    } else {
#ifndef __PTO_AUTO__
        SYNCALL_SOFT_AIC_IMPL(gmWorkspace.data(), l1Workspace.data(), usedCores);
#endif
    }
#else
    PTO_STATIC_ASSERT(Mode != Mode, "SYNCALL is not supported on this backend.");
#endif
}

template <
    SyncAllMode Mode, SyncCoreType CoreType = SyncCoreType::Mix, typename GlobalData, typename UbTileData,
    typename L1TileData,
    std::enable_if_t<is_global_data_v<GlobalData> && is_tile_data_v<UbTileData> && UbTileData::Loc == TileType::Vec &&
                         is_tile_data_v<L1TileData> && L1TileData::Loc == TileType::Mat,
                     int> = 0>
PTO_INST void SYNCALL(GlobalData &gmWorkspace, UbTileData &ubWorkspace, L1TileData &l1Workspace, int32_t usedCores = 0)
{
#if defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_A5) || defined(__CPU_SIM)
    if constexpr (Mode == SyncAllMode::Hard) {
        (void)gmWorkspace;
        (void)ubWorkspace;
        (void)l1Workspace;
        (void)usedCores;
        SYNCALL_IMPL<CoreType>();
    } else {
#ifndef __PTO_AUTO__
        SYNCALL_SOFT_MIX_IMPL<CoreType>(gmWorkspace.data(), ubWorkspace.data(), l1Workspace.data(), usedCores);
#endif
    }
#else
    PTO_STATIC_ASSERT(Mode != Mode, "SYNCALL is not supported on this backend.");
#endif
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

template <
    SyncAllMode Mode, SyncCoreType CoreType = SyncCoreType::AIVOnly, typename GlobalData, typename TileData,
    std::enable_if_t<is_global_data_v<GlobalData> && is_tile_data_v<TileData> && TileData::Loc == TileType::Vec, int> =
        0>
PTO_INST void SYNCALL(GlobalData& gmWorkspace, TileData& ubWorkspace, int32_t usedCores = 0)
{
#if defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_A5) || defined(__CPU_SIM)
    if constexpr (Mode == SyncAllMode::Hard) {
        (void)gmWorkspace;
        (void)ubWorkspace;
        (void)usedCores;
        SYNCALL_IMPL<CoreType>();
    } else {
#ifndef __PTO_AUTO__
        SYNCALL_SOFT_IMPL<CoreType>(gmWorkspace.data(), ubWorkspace.data(), usedCores);
#endif
    }
#else
    PTO_STATIC_ASSERT(Mode != Mode, "SYNCALL is not supported on this backend.");
#endif
}

template <
    SyncAllMode Mode, SyncCoreType CoreType = SyncCoreType::AICOnly, typename GlobalData, typename TileData,
    std::enable_if_t<is_global_data_v<GlobalData> && is_tile_data_v<TileData> && TileData::Loc == TileType::Mat, int> =
        0>
PTO_INST void SYNCALL(GlobalData& gmWorkspace, TileData& l1Workspace, int32_t usedCores = 0)
{
#if defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_A5) || defined(__CPU_SIM)
    PTO_STATIC_ASSERT(CoreType == SyncCoreType::AICOnly, "GM+L1 overload is for AIC-only mode.");
    if constexpr (Mode == SyncAllMode::Hard) {
        (void)gmWorkspace;
        (void)l1Workspace;
        (void)usedCores;
        SYNCALL_IMPL<CoreType>();
    } else {
#ifndef __PTO_AUTO__
        SYNCALL_SOFT_AIC_IMPL(gmWorkspace.data(), l1Workspace.data(), usedCores);
#endif
    }
#else
    PTO_STATIC_ASSERT(Mode != Mode, "SYNCALL is not supported on this backend.");
#endif
}

template <
    SyncAllMode Mode, SyncCoreType CoreType = SyncCoreType::Mix, typename GlobalData, typename UbTileData,
    typename L1TileData,
    std::enable_if_t<
        is_global_data_v<GlobalData> && is_tile_data_v<UbTileData> && UbTileData::Loc == TileType::Vec &&
            is_tile_data_v<L1TileData> && L1TileData::Loc == TileType::Mat,
        int> = 0>
PTO_INST void SYNCALL(GlobalData& gmWorkspace, UbTileData& ubWorkspace, L1TileData& l1Workspace, int32_t usedCores = 0)
{
#if defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_A5) || defined(__CPU_SIM)
    if constexpr (Mode == SyncAllMode::Hard) {
        (void)gmWorkspace;
        (void)ubWorkspace;
        (void)l1Workspace;
        (void)usedCores;
        SYNCALL_IMPL<CoreType>();
    } else {
#ifndef __PTO_AUTO__
        SYNCALL_SOFT_MIX_IMPL<CoreType>(gmWorkspace.data(), ubWorkspace.data(), l1Workspace.data(), usedCores);
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
    MAP_INSTR_IMPL(TCONCAT, dst, src0, src1, dstIdx, src0Idx, src1Idx);
    return {};
}

template <typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src, WaitEvents &...events)
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
    MAP_INSTR_IMPL_T(TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, atomicType, reluPreMode), dst, src,
                     preQuantScalar);
    return {};
}

template <
    STPhase Phase, typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
    ReluPreMode reluPreMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, uint64_t preQuantScalar, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, atomicType, reluPreMode, Phase), dst, src,
                     preQuantScalar);
    return {};
}

template <
    typename TileData, typename GlobalData, typename FpTileData, AtomicType atomicType = AtomicType::AtomicNone,
    ReluPreMode reluPreMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TSTORE_FP(GlobalData& dst, TileData& src, FpTileData& fp, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, FpTileData, atomicType, reluPreMode), dst, src,
                     fp);
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

#if defined(PTO_NPU_ARCH_A5) || defined(__CPU_SIM)
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
    MAP_INSTR_IMPL_T(TGEMV_MX, PTO_TEMPLATE_ARGS(Phase), cOutMatrix, cInMatrix, aMatrix, aScaleMatrix, bMatrix,
                     bScaleMatrix);
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
    MAP_INSTR_IMPL_T(TGEMV_MX, PTO_TEMPLATE_ARGS(Phase), cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix,
                     biasData);
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
    MAP_INSTR_IMPL_T(TMATMUL_MX, PTO_TEMPLATE_ARGS(Phase), cOutMatrix, cInMatrix, aMatrix, aScaleMatrix, bMatrix,
                     bScaleMatrix);
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
    MAP_INSTR_IMPL_T(TMATMUL_MX, PTO_TEMPLATE_ARGS(Phase), cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix,
                     biasData);
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
    MAP_INSTR_IMPL_T(TEXTRACT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, mode, reluMode), dst, src, indexRow,
                     indexCol);
    return {};
}

template <
    typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(
    DstTileData& dst, SrcTileData& src, uint64_t preQuantScalar, uint16_t indexRow, uint16_t indexCol,
    WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TEXTRACT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, reluMode), dst, src, preQuantScalar,
                     indexRow, indexCol);
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
    MAP_INSTR_IMPL_T(TEXTRACT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, mode, reluMode), dst, src, preQuantScalar,
                     indexRow, indexCol);
    return {};
}

template <
    typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
    typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT_FP(
    DstTileData& dst, SrcTileData& src, FpTileData& fp, uint16_t indexRow, uint16_t indexCol, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TEXTRACT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, FpTileData, reluMode), dst, src, fp,
                     indexRow, indexCol);
    return {};
}

template <
    typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode,
    ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(
    DstTileData& dst, SrcTileData& src, FpTileData& fp, uint16_t indexRow, uint16_t indexCol, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TEXTRACT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, FpTileData, mode, reluMode), dst, src, fp,
                     indexRow, indexCol);
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
    MAP_INSTR_IMPL_T(TINSERT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, mode, reluMode), dst, src, indexRow,
                     indexCol);
    return {};
}

template <
    typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(
    DstTileData& dst, SrcTileData& src, uint64_t preQuantScalar, uint16_t indexRow, uint16_t indexCol,
    WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TINSERT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, reluMode), dst, src, preQuantScalar, indexRow,
                     indexCol);
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
    MAP_INSTR_IMPL_T(TINSERT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, mode, reluMode), dst, src, preQuantScalar,
                     indexRow, indexCol);
    return {};
}
template <
    typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
    typename... WaitEvents>
PTO_INST RecordEvent TINSERT_FP(
    DstTileData& dst, SrcTileData& src, FpTileData& fp, uint16_t indexRow, uint16_t indexCol, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TINSERT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, FpTileData, reluMode), dst, src, fp, indexRow,
                     indexCol);
    return {};
}

template <
    typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode,
    ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent
TINSERT(DstTileData& dst, SrcTileData& src, FpTileData& fp, uint16_t indexRow, uint16_t indexCol, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TINSERT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, FpTileData, mode, reluMode), dst, src, fp,
                     indexRow, indexCol);
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
    typename DstTileData, typename SrcTileData,
    std::enable_if_t<(DstTileData::Loc == TileType::Vec) && (SrcTileData::Loc == TileType::Vec), int> = 0,
    typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD(DstTileData& dst, SrcTileData& src, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TFILLPAD, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData), dst, src);
    return {};
}

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD_INPLACE(DstTileData& dst, SrcTileData& src, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TFILLPAD_INPLACE, dst, src);
    return {};
}

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD_EXPAND(DstTileData& dst, SrcTileData& src, WaitEvents&... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TFILLPAD_EXPAND, dst, src);
    return {};
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

template <auto PrecisionType = PowAlgorithm::DEFAULT, typename DstTile, typename BaseTile, typename ExpTile,
          typename TmpTile, typename... WaitEvents>
PTO_INST RecordEvent TPOW(DstTile &dst, BaseTile &base, ExpTile &exp, TmpTile &tmp, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TPOW, PTO_TEMPLATE_ARGS(PrecisionType), dst, base, exp, tmp);
    return {};
}

template <auto PrecisionType = PowAlgorithm::DEFAULT, typename DstTile, typename BaseTile, typename TmpTile,
          typename... WaitEvents>
PTO_INST RecordEvent TPOWS(DstTile &dst, BaseTile &base, typename DstTile::DType exp, TmpTile &tmp,
                           WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL_T(TPOWS, PTO_TEMPLATE_ARGS(PrecisionType), dst, base, exp, tmp);
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
    TPUSH_IMPL<Pipe, TileProd, Split>(pipe, tile, subBlockId);
    return {};
}

template <
    typename Pipe, typename TileCons, TileSplitAxis Split, std::enable_if_t<is_tile_data_v<TileCons>, int> = 0,
    typename... WaitEvents>
PTO_INST RecordEvent TPOP(Pipe& pipe, TileCons& tile, WaitEvents&... events)
{
    TSYNC(events...);
    PTO_INSTR_SCOPE_OUTS(TPUSH, 0, tile, pipe);
#ifdef __CPU_SIM
    TPUSH_REVERSED_IMPL<TileData, Pipe>(tile, pipe);
#else
    TPUSH_IMPL(tile, pipe);
#endif
    return {};
}

template <typename Pipe, typename TileCons, TileSplitAxis Split, std::enable_if_t<is_tile_data_v<TileCons>, int> = 0,
          typename... WaitEvents>
PTO_INST RecordEvent TPOP(Pipe &pipe, TileCons &tile, WaitEvents &...events)
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
    PTO_INSTR_SCOPE(TALLOC, pipe, gmTensor);
#ifdef __CPU_SIM
    TALLOC_GLOBAL_IMPL<Pipe, GlobalData, Split>(pipe, gmTensor);
#else
    TALLOC_IMPL(pipe, gmTensor);
#endif
    return {};
}
#endif

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRIN9030) || defined(__CPU_SIM)
template <HistByte byte, typename TileDataDst, typename TileDataSrc, typename TileDataIdx, typename... WaitEvents>
PTO_INST RecordEvent THISTOGRAM(TileDataDst& dst, TileDataSrc& src, TileDataIdx& idx, WaitEvents&... events)
{
    TSYNC(events...);
    PTO_INSTR_SCOPE_OUTS(TPUSH, 0, pipe, gmTensor);
#ifdef __CPU_SIM
    TPUSH_GLOBAL_IMPL<Pipe, GlobalData, Split>(pipe, gmTensor);
#else
    TPUSH_IMPL(pipe, gmTensor);
#endif
    return {};
}

template <typename Pipe, typename TileProd, typename TConfig, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe &pipe, TileProd &tile, WaitEvents &...events)
{
    TSYNC(events...);
    TPUSH_IMPL<Pipe, TileProd, TConfig>(pipe, tile);
    return {};
}

template <typename Pipe, typename GlobalData, TileSplitAxis Split,
          std::enable_if_t<is_global_data_v<GlobalData>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TPOP(Pipe &pipe, GlobalData &gmTensor, WaitEvents &...events)
{
    TSYNC(events...);
    PTO_INSTR_SCOPE(TPOP, pipe, gmTensor);
#ifdef __CPU_SIM
    TPOP_GLOBAL_IMPL<Pipe, GlobalData, Split>(pipe, gmTensor);
#else
    TPOP_IMPL(pipe, gmTensor);
#endif
    return {};
}

template <typename Pipe, typename GlobalData, TileSplitAxis Split,
          std::enable_if_t<is_global_data_v<GlobalData>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TFREE(Pipe &pipe, GlobalData &gmTensor, WaitEvents &...events)
{
    TSYNC(events...);
    PTO_INSTR_SCOPE_OUTS(TFREE, 0, pipe, gmTensor);
#ifdef __CPU_SIM
    TFREE_GLOBAL_IMPL<Pipe, GlobalData, Split>(pipe, gmTensor);
#else
    TFREE_IMPL(pipe, gmTensor);
#endif
    return {};
}

template <typename Pipe, typename... WaitEvents>
PTO_INST RecordEvent TFREE(Pipe &pipe, WaitEvents &...events)
{
    TSYNC(events...);
    PTO_INSTR_SCOPE_OUTS(TFREE, 0, pipe);
    TFREE_IMPL<Pipe>(pipe);
    return {};
}
#endif

#if defined(PTO_NPU_ARCH_A5) || defined(__CPU_SIM)
template <HistByte byte, typename TileDataDst, typename TileDataSrc, typename TileDataIdx, typename... WaitEvents>
PTO_INST RecordEvent THISTOGRAM(TileDataDst &dst, TileDataSrc &src, TileDataIdx &idx, WaitEvents &...events)
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
    MAP_INSTR_IMPL_T(TQUANT, PTO_TEMPLATE_ARGS(quant_type, TileDataOut, TileDataSrc, TileDataPara), dst, src, scale,
                     offset);
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

// ---------------------------------------------------------------------------
// GridPipe TPUSH / TPOP overloads (design doc section 4.1, "neighbor-core
// FIFO" form).  These coexist with the cluster-local TPipe overloads above:
// SFINAE on is_grid_pipe_v keeps overload resolution unambiguous.  The
// `Direction` non-type template parameter is constant-folded by the compiler,
// matching design doc section 4.2's requirement that direction be a constant
// at lowering time.
// ---------------------------------------------------------------------------

template <pto::GridDirection Direction, int Dist = 1, typename Pipe, typename TileProd,
          std::enable_if_t<is_grid_pipe_v<Pipe>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe &pipe, TileProd &tile, WaitEvents &...events)
{
    static_assert(Direction != pto::GridDirection::SOURCE,
                  "GridPipe TPUSH<SOURCE> is illegal (design doc section 4.3): "
                  "SOURCE is only valid for TPOP.");
    // Dist is the routed-unicast hop count; Dist == 1 (default) is the original
    // nearest-neighbor push.  TPUSH<EAST, 2>(pipe, tile) pushes 2 hops east.
#if defined(PTO_NPU_ARCH_A2A3)
    TSYNC(events...);
    GRID_TPUSH_IMPL<Direction, Dist, Pipe, TileProd>(pipe, tile);
#else
    static_assert(sizeof(Pipe) == 0,
                  "GridPipe TPUSH not supported on this target profile "
                  "(design doc section 5.4 forbids silent GM fallback).");
#endif
    return {};
}

template <pto::GridDirection Direction, int Dist = 1, typename Pipe, typename TileCons,
          std::enable_if_t<is_grid_pipe_v<Pipe>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TPOP(Pipe &pipe, TileCons &tile, WaitEvents &...events)
{
    // Dist must match the producer's TPUSH distance for this logical edge so the
    // free-credit doorbell routes back to the K-hop producer; Dist == 1 default.
#if defined(PTO_NPU_ARCH_A2A3)
    TSYNC(events...);
    GRID_TPOP_IMPL<Direction, Dist, Pipe, TileCons>(pipe, tile);
#else
    static_assert(sizeof(Pipe) == 0,
                  "GridPipe TPOP not supported on this target profile "
                  "(design doc section 5.4 forbids silent GM fallback).");
#endif
    return {};
}

// ---------------------------------------------------------------------------
// GridPipe TPUSH broadcast overload: single-source row/column multicast.  This
// is the SAME verb as the unicast TPUSH above, distinguished purely by the type
// of the first explicit template argument -- a GridSpan (ROW/COL) selects this
// overload, a GridDirection selects the unicast one.  The two scoped enums never
// interconvert, so overload resolution is unambiguous and folds at compile time.
//
// One source fans `tile` to every other cell on its row (GridSpan::ROW) or
// column (GridSpan::COL) in a single op -- a true multicast with ONE publish
// fence, not a per-hop TPUSH<Dir, k> loop (see GRID_TRY_TPUSH_BCAST_IMPL).
// Fan-in stays 1, so receivers drain it with the ordinary TPOP<EAST/WEST, dist>
// (ROW) or TPOP<NORTH/SOUTH, dist> (COL).  There is no Dist parameter: a span
// always fans to the grid boundary on both of its arms.
// ---------------------------------------------------------------------------
template <pto::GridSpan Span, typename Pipe, typename TileProd, std::enable_if_t<is_grid_pipe_v<Pipe>, int> = 0,
          typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe &pipe, TileProd &tile, WaitEvents &...events)
{
#if defined(PTO_NPU_ARCH_A2A3)
    TSYNC(events...);
    GRID_TPUSH_BCAST_IMPL<Span, Pipe, TileProd>(pipe, tile);
#else
    static_assert(sizeof(Pipe) == 0,
                  "GridPipe TPUSH<GridSpan> broadcast not supported on this target profile "
                  "(design doc section 5.4 forbids silent GM fallback).");
#endif
    return {};
}

} // namespace pto
#endif
