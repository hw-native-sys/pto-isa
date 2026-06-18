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
#include "pto/common/tassign_check.hpp"
#include "pto/common/pto_instr_impl.hpp"
#if !defined(__COSTMODEL) && !defined(PTO_COMM_NOT_SUPPORTED)
#include "pto/comm/pto_comm_inst.hpp"
#endif

#define MAP_INSTR_IMPL(API, ...) API##_IMPL(__VA_ARGS__)

namespace pto {

template <typename T, typename AddrType>
PTO_INST void TASSIGN(T &obj, AddrType addr)
{
    MAP_INSTR_IMPL(TASSIGN, obj, addr);
}

// Compile-time address overload: TASSIGN<Addr>(tile)
// Performs static bounds and alignment checks when Addr is a compile-time constant.
// Only enabled for Tile / ConvTile types (not GlobalTensor).
template <std::size_t Addr, typename T>
PTO_INST std::enable_if_t<is_tile_data_v<T> || is_conv_tile_v<T>> TASSIGN(T &obj)
{
    // Trigger compile-time checks (static_assert inside tassign_static_check).
    (void)detail::tassign_static_check<std::remove_cv_t<T>, Addr>{};

    // Delegate to the existing runtime TASSIGN path.
    TASSIGN(obj, static_cast<std::size_t>(Addr));
}

template <Op OpCode>
PTO_INST void TSYNC()
{
    TSYNC_IMPL<OpCode>();
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

template <typename... WaitEvents>
PTO_INST void TSYNC(WaitEvents &...events)
{
    WaitAllEvents(events...);
}

#if defined(_DEBUG) || defined(__CPU_SIM)
template <PrintFormat Format = PrintFormat::Width8_Precision4, typename TileData>
PTO_INST void TPRINT(TileData &src)
{
    TPRINT_IMPL<Format>(src);
}

template <PrintFormat Format = PrintFormat::Width8_Precision4, typename TileData, typename GlobalData>
PTO_INST void TPRINT(TileData &src, GlobalData &tmp)
{
    TPRINT_IMPL<Format>(src, tmp);
}
#endif

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TADD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TADD, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename... WaitEvents>
PTO_INST RecordEvent TPAIRREDUCESUM(TileDataDst &dst, TileDataSrc0 &src0, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TPAIRREDUCESUM, dst, src0);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TSUBRELUCONV(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSUBRELUCONV, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TADDRELUCONV(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TADDRELUCONV, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TABS(TileDataDst &dst, TileDataSrc &src, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TABS, dst, src);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TAND(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TAND, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TOR(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TOR, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TSUB(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSUB, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TSUBVIEW(TileDataDst &dst, TileDataSrc &src, uint16_t rowIdx, uint16_t colIdx,
                              WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSUBVIEW, dst, src, rowIdx, colIdx);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TMUL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMUL, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TMIN(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMIN, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TMAX(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMAX, dst, src0, src1);
    return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TEXPANDS(TileData &dst, typename TileData::DType scalar, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TEXPANDS, dst, scalar);
    return {};
}

template <typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TLOAD(TileData &dst, GlobalData &src, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TLOAD, dst, src);
    return {};
}

template <typename TileData, typename GlobalData>
PTO_INST RecordEvent TPREFETCH(TileData &dst, GlobalData &src)
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
PTO_INST comm::AsyncEvent TPREFETCH_ASYNC(GlobalData &srcGlobalData, PrefetchAsyncContext &ctx, WaitEvents &...events)
{
    TSYNC(events...);
    return TPREFETCH_ASYNC_IMPL(srcGlobalData, ctx);
}

#endif // (__CCE_AICORE__ || __CPU_SIM) && !__COSTMODEL && !PTO_COMM_NOT_SUPPORTED

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents,
          std::enable_if_t<all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TCMPS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType src1, CmpMode mode,
                           WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCMPS, dst, src0, src1, mode);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents,
          std::enable_if_t<is_tile_data_v<TileDataSrc1> && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TCMPS(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, CmpMode mode,
                           WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCMPS, dst, src0, src1, mode);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCMP(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, CmpMode cmpMode,
                          WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCMP, dst, src0, src1, cmpMode);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents,
          std::enable_if_t<all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TCONCAT(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCONCAT, dst, src0, src1);
    return {};
}

template <typename DstTile, typename Src0Tile, typename Src1Tile, typename Src0IdxTile, typename Src1IdxTile,
          typename... WaitEvents,
          std::enable_if_t<is_tile_data_v<Src0IdxTile> && is_tile_data_v<Src1IdxTile> && all_events_v<WaitEvents...>,
                           int> = 0>
PTO_INST RecordEvent TCONCAT(DstTile &dst, Src0Tile &src0, Src1Tile &src1, Src0IdxTile &src0Idx, Src1IdxTile &src1Idx,
                             WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCONCAT, dst, src0, src1, src0Idx, src1Idx);
    return {};
}

template <typename DstTile, typename Src0Tile, typename Src1Tile, typename DstIdxTile, typename Src0IdxTile,
          typename Src1IdxTile, typename... WaitEvents,
          std::enable_if_t<is_tile_data_v<Src1IdxTile> && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TCONCAT(DstTile &dst, Src0Tile &src0, Src1Tile &src1, DstIdxTile &dstIdx, Src0IdxTile &src0Idx,
                             Src1IdxTile &src1Idx, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCONCAT, dst, src0, src1, dstIdx, src0Idx, src1Idx);
    return {};
}

template <typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src, WaitEvents &...events)
{
    TSYNC(events...);
    TSTORE_IMPL<TileData, GlobalData, AtomicType::AtomicNone>(dst, src);
    return {};
}

// UF-aware overload: allow selecting unit-flag phase while keeping the TSTORE name.
template <STPhase Phase, typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src, WaitEvents &...events)
{
    TSYNC(events...);
    TSTORE_IMPL<TileData, GlobalData, AtomicType::AtomicNone, Phase>(dst, src);
    return {};
}

template <typename TileData, typename GlobalData, AtomicType atomicType, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src, WaitEvents &...events)
{
    TSYNC(events...);
    TSTORE_IMPL<TileData, GlobalData, atomicType>(dst, src);
    return {};
}

template <STPhase Phase, typename TileData, typename GlobalData, AtomicType atomicType, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src, WaitEvents &...events)
{
    TSYNC(events...);
    TSTORE_IMPL<TileData, GlobalData, atomicType, Phase>(dst, src);
    return {};
}

template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
          ReluPreMode reluPreMode, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src, WaitEvents &...events)
{
    TSYNC(events...);
    TSTORE_IMPL<TileData, GlobalData, atomicType, reluPreMode>(dst, src);
    return {};
}

template <STPhase Phase, typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
          ReluPreMode reluPreMode, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src, WaitEvents &...events)
{
    TSYNC(events...);
    TSTORE_IMPL<TileData, GlobalData, atomicType, reluPreMode, Phase>(dst, src);
    return {};
}

template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src, uint64_t preQuantScalar, WaitEvents &...events)
{
    TSYNC(events...);
    TSTORE_IMPL<TileData, GlobalData, atomicType, reluPreMode>(dst, src, preQuantScalar);
    return {};
}

template <STPhase Phase, typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src, uint64_t preQuantScalar, WaitEvents &...events)
{
    TSYNC(events...);
    TSTORE_IMPL<TileData, GlobalData, atomicType, reluPreMode, Phase>(dst, src, preQuantScalar);
    return {};
}

template <typename TileData, typename GlobalData, typename FpTileData, AtomicType atomicType = AtomicType::AtomicNone,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TSTORE_FP(GlobalData &dst, TileData &src, FpTileData &fp, WaitEvents &...events)
{
    TSYNC(events...);
    TSTORE_IMPL<TileData, GlobalData, FpTileData, atomicType, reluPreMode>(dst, src, fp);
    return {};
}

template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0,
          typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TDIV(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    TDIV_IMPL<PrecisionType>(dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TSHL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSHL, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TSHR(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSHR, dst, src0, src1);
    return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TAND(TileData &dst, TileData &src0, TileData &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TAND, dst, src0, src1);
    return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TOR(TileData &dst, TileData &src0, TileData &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TOR, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TXOR(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp,
                          WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TXOR, dst, src0, src1, tmp);
    return {};
}

template <auto PrecisionType = LogAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc,
          typename... WaitEvents>
PTO_INST RecordEvent TLOG(TileDataDst &dst, TileDataSrc &src, WaitEvents &...events)
{
    TSYNC(events...);
    TLOG_IMPL<PrecisionType>(dst, src);
    return {};
}

template <auto PrecisionType = RecipAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc,
          typename... WaitEvents>
PTO_INST RecordEvent TRECIP(TileDataDst &dst, TileDataSrc &src, WaitEvents &...events)
{
    TSYNC(events...);
    /*
     * A3's TRECIP instruction does not support setting the source Tile and destination Tile to the same memory.
     */
    TDIVS_IMPL<static_cast<DivAlgorithm>(PrecisionType)>(dst, 1, src);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TPRELU(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp,
                            WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TPRELU, dst, src0, src1, tmp);
    return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TADDC(TileData &dst, TileData &src0, TileData &src1, TileData &src2, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TADDC, dst, src0, src1, src2);
    return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TSUBC(TileData &dst, TileData &src0, TileData &src1, TileData &src2, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSUBC, dst, src0, src1, src2);
    return {};
}

#if defined(PTO_NPU_ARCH_A5) || defined(__CPU_SIM)
template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
          typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix,
                              TileRightScale &bScaleMatrix, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TGEMV_MX, cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix);
    return {};
}

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
          typename TileRightScale, typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix,
                              TileRightScale &bScaleMatrix, WaitEvents &...events)
{
    TSYNC(events...);
    TGEMV_MX_IMPL<Phase>(cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix);
    return {};
}

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
          typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix,
                              TileRight &bMatrix, TileRightScale &bScaleMatrix, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TGEMV_MX, cOutMatrix, cInMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix);
    return {};
}

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
          typename TileRightScale, typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix,
                              TileRight &bMatrix, TileRightScale &bScaleMatrix, WaitEvents &...events)
{
    TSYNC(events...);
    TGEMV_MX_IMPL<Phase>(cOutMatrix, cInMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix);
    return {};
}

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
          typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix,
                              TileRightScale &bScaleMatrix, TileBias &biasData, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TGEMV_MX, cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix, biasData);
    return {};
}

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
          typename TileRightScale, typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix,
                              TileRightScale &bScaleMatrix, TileBias &biasData, WaitEvents &...events)
{
    TSYNC(events...);
    TGEMV_MX_IMPL<Phase>(cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix, biasData);
    return {};
}

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
          typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix,
                                TileRightScale &bScaleMatrix, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMATMUL_MX, cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix);
    return {};
}

// UF-aware overload enabling unit-flag selection via AccPhase while retaining the TMATMUL name.
template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
          typename TileRightScale, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix,
                                TileRightScale &bScaleMatrix, WaitEvents &...events)
{
    TSYNC(events...);
    TMATMUL_MX_IMPL<Phase>(cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix);
    return {};
}

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
          typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix,
                                TileRight &bMatrix, TileRightScale &bScaleMatrix, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMATMUL_MX, cOutMatrix, cInMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix);
    return {};
}

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
          typename TileRightScale, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix,
                                TileRight &bMatrix, TileRightScale &bScaleMatrix, WaitEvents &...events)
{
    TSYNC(events...);
    TMATMUL_MX_IMPL<Phase>(cOutMatrix, cInMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix);
    return {};
}

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
          typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix,
                                TileRightScale &bScaleMatrix, TileBias &biasData, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMATMUL_MX, cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix, biasData);
    return {};
}

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
          typename TileRightScale, typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix,
                                TileRightScale &bScaleMatrix, TileBias &biasData, WaitEvents &...events)
{
    TSYNC(events...);
    TMATMUL_MX_IMPL<Phase>(cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix, biasData);
    return {};
}

template <uint16_t Rounds = 10, typename DstTile, typename... WaitEvents>
PTO_INST RecordEvent TRANDOM(DstTile &dst, TRandomKey &key, TRandomCounter &counter, WaitEvents &...events)
{
    TSYNC(events...);
    TRANDOM_IMPL<Rounds, DstTile>(dst, key, counter);
    return {};
}
#endif

template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMATMUL, cMatrix, aMatrix, bMatrix);
    return {};
}

// UF-aware overload enabling unit-flag selection via AccPhase while retaining the TMATMUL name.
template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, WaitEvents &...events)
{
    TSYNC(events...);
    TMATMUL_IMPL<Phase>(cMatrix, aMatrix, bMatrix);
    return {};
}

template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_ACC(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix,
                                 WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMATMUL_ACC, cOutMatrix, cInMatrix, aMatrix, bMatrix);
    return {};
}

// UF-aware overloads for TMATMUL_ACC: explicit input/output or shared accumulator tile.
template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_ACC(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix,
                                 WaitEvents &...events)
{
    TSYNC(events...);
    TMATMUL_ACC_IMPL<Phase>(cOutMatrix, cInMatrix, aMatrix, bMatrix);
    return {};
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight,
          typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_ACC(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, WaitEvents &...events)
{
    TSYNC(events...);
    TMATMUL_ACC_IMPL<Phase>(cMatrix, aMatrix, bMatrix);
    return {};
}

template <typename TileRes, typename TileLeft, typename TileRight, typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_BIAS(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData,
                                  WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMATMUL_BIAS, cMatrix, aMatrix, bMatrix, biasData);
    return {};
}

// UF-aware overload enabling unit-flag selection for bias matmul while keeping the TMATMUL_BIAS name.
template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename TileBias,
          typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_BIAS(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData,
                                  WaitEvents &...events)
{
    TSYNC(events...);
    TMATMUL_BIAS_IMPL<Phase>(cMatrix, aMatrix, bMatrix, biasData);
    return {};
}

template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TGEMV(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TGEMV, cMatrix, aMatrix, bMatrix);
    return {};
}

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TGEMV(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, WaitEvents &...events)
{
    TSYNC(events...);
    TGEMV_IMPL<Phase>(cMatrix, aMatrix, bMatrix);
    return {};
}

template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TGEMV_ACC(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix,
                               WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TGEMV_ACC, cOutMatrix, cInMatrix, aMatrix, bMatrix);
    return {};
}

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TGEMV_ACC(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix,
                               WaitEvents &...events)
{
    TSYNC(events...);
    TGEMV_ACC_IMPL<Phase>(cOutMatrix, cInMatrix, aMatrix, bMatrix);
    return {};
}

template <typename TileRes, typename TileLeft, typename TileRight, typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TGEMV_BIAS(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData,
                                WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TGEMV_BIAS, cMatrix, aMatrix, bMatrix, biasData);
    return {};
}

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename TileBias,
          typename... WaitEvents>
PTO_INST RecordEvent TGEMV_BIAS(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData,
                                WaitEvents &...events)
{
    TSYNC(events...);
    TGEMV_BIAS_IMPL<Phase>(cMatrix, aMatrix, bMatrix, biasData);
    return {};
}

template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData,
          typename Src2TileData, typename Src3TileData, bool exhausted, typename... WaitEvents>
PTO_INST RecordEvent TMRGSORT(DstTileData &dst, MrgSortExecutedNumList &executedNumList, TmpTileData &tmp,
                              Src0TileData &src0, Src1TileData &src1, Src2TileData &src2, Src3TileData &src3,
                              WaitEvents &...events)
{
    TSYNC(events...);
    TMRGSORT_IMPL<DstTileData, TmpTileData, Src0TileData, Src1TileData, Src2TileData, Src3TileData, exhausted>(
        dst, executedNumList, tmp, src0, src1, src2, src3);
    return {};
}

template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData,
          typename Src2TileData, bool exhausted, typename... WaitEvents>
PTO_INST RecordEvent TMRGSORT(DstTileData &dst, MrgSortExecutedNumList &executedNumList, TmpTileData &tmp,
                              Src0TileData &src0, Src1TileData &src1, Src2TileData &src2, WaitEvents &...events)
{
    TSYNC(events...);
    TMRGSORT_IMPL<DstTileData, TmpTileData, Src0TileData, Src1TileData, Src2TileData, exhausted>(dst, executedNumList,
                                                                                                 tmp, src0, src1, src2);
    return {};
}

template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData, bool exhausted,
          typename... WaitEvents>
PTO_INST RecordEvent TMRGSORT(DstTileData &dst, MrgSortExecutedNumList &executedNumList, TmpTileData &tmp,
                              Src0TileData &src0, Src1TileData &src1, WaitEvents &...events)
{
    TSYNC(events...);
    TMRGSORT_IMPL<DstTileData, TmpTileData, Src0TileData, Src1TileData, exhausted>(dst, executedNumList, tmp, src0,
                                                                                   src1);
    return {};
}

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TMRGSORT(DstTileData &dst, SrcTileData &src, uint32_t blockLen, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMRGSORT, dst, src, blockLen);
    return {};
}

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0,
                              WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TEXTRACT, dst, src, indexRow, indexCol);
    return {};
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol,
                              WaitEvents &...events)
{
    TSYNC(events...);
    TEXTRACT_IMPL<DstTileData, SrcTileData, reluMode>(dst, src, indexRow, indexCol);
    return {};
}

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol,
                              WaitEvents &...events)
{
    TSYNC(events...);
    TEXTRACT_IMPL<DstTileData, SrcTileData, mode, reluMode>(dst, src, indexRow, indexCol);
    return {};
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint16_t indexRow,
                              uint16_t indexCol, WaitEvents &...events)
{
    TSYNC(events...);
    TEXTRACT_IMPL<DstTileData, SrcTileData, reluMode>(dst, src, preQuantScalar, indexRow, indexCol);
    return {};
}

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint16_t indexRow,
                              uint16_t indexCol, WaitEvents &...events)
{
    TSYNC(events...);
    TEXTRACT_IMPL<DstTileData, SrcTileData, mode, reluMode>(dst, src, preQuantScalar, indexRow, indexCol);
    return {};
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow,
                                 uint16_t indexCol, WaitEvents &...events)
{
    TSYNC(events...);
    TEXTRACT_IMPL<DstTileData, SrcTileData, FpTileData, reluMode>(dst, src, fp, indexRow, indexCol);
    return {};
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow, uint16_t indexCol,
                              WaitEvents &...events)
{
    TSYNC(events...);
    TEXTRACT_IMPL<DstTileData, SrcTileData, FpTileData, mode, reluMode>(dst, src, fp, indexRow, indexCol);
    return {};
}

template <typename TileData, typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL,
          typename... WaitEvents>
PTO_INST RecordEvent TIMG2COL(TileData &dst, ConvTileData &src, uint16_t posM = 0, uint16_t posK = 0,
                              WaitEvents &...events)
{
    TSYNC(events...);
    TIMG2COL_IMPL<TileData, ConvTileData, FmatrixMode>(dst, src, posM, posK);
    return {};
}

template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent SETFMATRIX(ConvTileData &src, WaitEvents &...events)
{
    TSYNC(events...);
    SETFMATRIX_IMPL<ConvTileData, FmatrixMode>(src);
    return {};
}

#ifdef PTO_NPU_ARCH_A2A3
template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent SET_IMG2COL_RPT(ConvTileData &src, WaitEvents &...events)
{
    TSYNC(events...);
    SET_IMG2COL_RPT_IMPL<ConvTileData, FmatrixMode>(src);
    return {};
}

template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent SET_IMG2COL_PADDING(ConvTileData &src, WaitEvents &...events)
{
    TSYNC(events...);
    SET_IMG2COL_PADDING_IMPL<ConvTileData, FmatrixMode>(src);
    return {};
}
#endif
#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRIN9030) || defined(__CPU_SIM)
template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent SET_IMG2COL_RPT(ConvTileData &src, WaitEvents &...events)
{
    TSYNC(events...);
    SET_IMG2COL_RPT_IMPL<ConvTileData, FmatrixMode>(src);
    return {};
}

template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent SET_IMG2COL_PADDING(ConvTileData &src, WaitEvents &...events)
{
    TSYNC(events...);
    SET_IMG2COL_PADDING_IMPL<ConvTileData, FmatrixMode>(src);
    return {};
}
#endif

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol,
                             WaitEvents &...events)
{
    TSYNC(events...);
    TINSERT_IMPL<DstTileData, SrcTileData, reluMode>(dst, src, indexRow, indexCol);
    return {};
}

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol,
                             WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TINSERT, dst, src, indexRow, indexCol);
    return {};
}

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol,
                             WaitEvents &...events)
{
    TSYNC(events...);
    TINSERT_IMPL<DstTileData, SrcTileData, mode, reluMode>(dst, src, indexRow, indexCol);
    return {};
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint16_t indexRow,
                             uint16_t indexCol, WaitEvents &...events)
{
    TSYNC(events...);
    TINSERT_IMPL<DstTileData, SrcTileData, reluMode>(dst, src, preQuantScalar, indexRow, indexCol);
    return {};
}
template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint16_t indexRow,
                             uint16_t indexCol, WaitEvents &...events)
{
    TSYNC(events...);
    TINSERT_IMPL<DstTileData, SrcTileData, mode, reluMode>(dst, src, preQuantScalar, indexRow, indexCol);
    return {};
}
template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TINSERT_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow,
                                uint16_t indexCol, WaitEvents &...events)
{
    TSYNC(events...);
    TINSERT_IMPL<DstTileData, SrcTileData, FpTileData, reluMode>(dst, src, fp, indexRow, indexCol);
    return {};
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow, uint16_t indexCol,
                             WaitEvents &...events)
{
    TSYNC(events...);
    TINSERT_IMPL<DstTileData, SrcTileData, FpTileData, mode, reluMode>(dst, src, fp, indexRow, indexCol);
    return {};
}

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRIN9030)
template <TInsertMode mode, typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0,
                             WaitEvents &...events)
{
    TSYNC(events...);
    TINSERT_IMPL<mode>(dst, src, indexRow, indexCol);
    return {};
}
#endif

template <typename TileData, PadValue PadVal = PadValue::Zero,
          std::enable_if_t<(TileData::Loc == TileType::Mat), int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD(TileData &dst, TileData &src, WaitEvents &...events)
{
    TSYNC(events...);
    TFILLPAD_IMPL<TileData, PadVal>(dst, src);
    return {};
}

template <typename DstTileData, typename SrcTileData,
          std::enable_if_t<(DstTileData::Loc == TileType::Vec) && (SrcTileData::Loc == TileType::Vec), int> = 0,
          typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD(DstTileData &dst, SrcTileData &src, WaitEvents &...events)
{
    TSYNC(events...);
    TFILLPAD_IMPL<DstTileData, SrcTileData>(dst, src);
    return {};
}

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD_INPLACE(DstTileData &dst, SrcTileData &src, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TFILLPAD_INPLACE, dst, src);
    return {};
}

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD_EXPAND(DstTileData &dst, SrcTileData &src, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TFILLPAD_EXPAND, dst, src);
    return {};
}

// TSORT32不自动实现wait, 需手动TSYNC(events...)
template <typename DstTileData, typename SrcTileData, typename IdxTileData>
PTO_INST RecordEvent TSORT32(DstTileData &dst, SrcTileData &src, IdxTileData &idx)
{
    MAP_INSTR_IMPL(TSORT32, dst, src, idx);
    return {};
}

template <typename DstTileData, typename SrcTileData, typename IdxTileData, typename TmpTileData>
PTO_INST RecordEvent TSORT32(DstTileData &dst, SrcTileData &src, IdxTileData &idx, TmpTileData &tmp)
{
    MAP_INSTR_IMPL(TSORT32, dst, src, idx, tmp);
    return {};
}

template <typename TileDataD, typename TileDataS0, typename TileDataS1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(TileDataD &dst, TileDataS0 &src0, TileDataS1 &src1, TileDataTmp &tmp,
                             WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TGATHER, dst, src0, src1, tmp);
    return {};
}

template <typename TileDataD, typename TileDataS, typename TileDataS1, typename TileDataC, typename TileDataTmp,
          CmpMode cmpMode, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(TileDataD &dst, TileDataS &src0, TileDataS1 &k_value, TileDataC &cdst, TileDataTmp &tmp,
                             int offset, WaitEvents &...events)
{
    TSYNC(events...);
    TGATHER_IMPL<TileDataD, TileDataS, TileDataS1, TileDataC, TileDataTmp, cmpMode>(dst, src0, k_value, cdst, tmp,
                                                                                    offset);
    return {};
}

template <typename TileData, typename T, int descending, typename... WaitEvents>
PTO_INST RecordEvent TCI(TileData &dst, T start, WaitEvents &...events)
{
    TSYNC(events...);
    TCI_IMPL<TileData, T, descending>(dst, start);
    return {};
}

template <typename TileData, typename TileDataTmp, typename T, int descending, typename... WaitEvents>
PTO_INST RecordEvent TCI(TileData &dst, T start, TileDataTmp &tmp, WaitEvents &...events)
{
    TSYNC(events...);
    TCI_IMPL<TileData, TileDataTmp, T, descending>(dst, start, tmp);
    return {};
}

template <typename TileData, int isUpperOrLower, typename... WaitEvents>
PTO_INST RecordEvent TTRI(TileData &dst, int diagonal, WaitEvents &...events)
{
    TSYNC(events...);
    TTRI_IMPL<TileData, isUpperOrLower>(dst, diagonal);
    return {};
}

template <typename DstTileData, typename SrcTileData, MaskPattern maskPattern = MaskPattern::P1111,
          auto gatherType = GatherAxis::GATHER_ROW, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(DstTileData &dst, SrcTileData &src, WaitEvents &...events)
{
    TSYNC(events...);
    TGATHER_IMPL<DstTileData, SrcTileData, maskPattern, gatherType>(dst, src);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TPARTADD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TPARTADD, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TPARTMUL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TPARTMUL, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TPARTMAX(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TPARTMAX, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TPARTMIN(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TPARTMIN, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataDstIdx,
          typename TileDataSrc0Idx, typename TileDataSrc1Idx, typename... WaitEvents>
PTO_INST RecordEvent TPARTARGMAX(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataDstIdx &dstIdx,
                                 TileDataSrc0Idx &src0Idx, TileDataSrc1Idx &src1Idx, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TPARTARGMAX, dst, src0, src1, dstIdx, src0Idx, src1Idx);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataDstIdx,
          typename TileDataSrc0Idx, typename TileDataSrc1Idx, typename... WaitEvents>
PTO_INST RecordEvent TPARTARGMIN(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataDstIdx &dstIdx,
                                 TileDataSrc0Idx &src0Idx, TileDataSrc1Idx &src1Idx, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TPARTARGMIN, dst, src0, src1, dstIdx, src0Idx, src1Idx);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TFUSEDMULADD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TFUSEDMULADD, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TMULADDDST(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMULADDDST, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TSUBRELU(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSUBRELU, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TFUSEDMULADDRELU(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TFUSEDMULADDRELU, dst, src0, src1);
    return {};
}

template <typename TileDataD, typename TileDataS, typename TmpTileData>
PTO_INTERNAL void TCVT_IMPL(TileDataD &dst, TileDataS &src, TmpTileData &tmp, RoundMode mode, SaturationMode satMode,
                            bool needSetCtrl);
template <typename TileDataD, typename TileDataS, typename TmpTileData>
PTO_INTERNAL void TCVT_IMPL(TileDataD &dst, TileDataS &src, TmpTileData &tmp, RoundMode mode, bool needSetCtrl);
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void TCVT_IMPL(TileDataD &dst, TileDataS &src, RoundMode mode, SaturationMode satMode, bool needSetCtrl);
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void TCVT_IMPL(TileDataD &dst, TileDataS &src, RoundMode mode, bool needSetCtrl);

template <bool NeedSetCtrl = true, typename TileDataD, typename TileDataS, typename TmpTileData, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, TmpTileData &tmp, RoundMode mode, SaturationMode satMode,
                          WaitEvents &...events)
{
    TSYNC(events...);
    TCVT_IMPL(dst, src, tmp, mode, satMode, NeedSetCtrl);
    return {};
}

template <bool NeedSetCtrl = true, typename TileDataD, typename TileDataS, typename TmpTileData, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, TmpTileData &tmp, RoundMode mode, WaitEvents &...events)
{
    TSYNC(events...);
    TCVT_IMPL(dst, src, tmp, mode, NeedSetCtrl);
    return {};
}

template <bool NeedSetCtrl = true, typename TileDataD, typename TileDataS, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, RoundMode mode, SaturationMode satMode, WaitEvents &...events)
{
    TSYNC(events...);
    TCVT_IMPL(dst, src, mode, satMode, NeedSetCtrl);
    return {};
}

template <bool NeedSetCtrl = true, typename TileDataD, typename TileDataS, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, RoundMode mode, WaitEvents &...events)
{
    TSYNC(events...);
    TCVT_IMPL(dst, src, mode, NeedSetCtrl);
    return {};
}

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMOV, dst, src);
    return {};
}

template <typename DstTileData, typename SrcTileData, typename TmpTileData, typename... WaitEvents,
          std::enable_if_t<is_tile_data_v<TmpTileData>, int> = 0>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, TmpTileData &tmp, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMOV, dst, src, tmp);
    return {};
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, WaitEvents &...events)
{
    TSYNC(events...);
    TMOV_IMPL<DstTileData, SrcTileData, reluMode>(dst, src);
    return {};
}

template <STPhase Phase, typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, WaitEvents &...events)
{
    TSYNC(events...);
    TMOV_IMPL<DstTileData, SrcTileData, ReluPreMode::NoRelu, Phase>(dst, src);
    return {};
}

template <STPhase Phase, typename DstTileData, typename SrcTileData, ReluPreMode reluMode, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, WaitEvents &...events)
{
    TSYNC(events...);
    TMOV_IMPL<DstTileData, SrcTileData, reluMode, Phase>(dst, src);
    return {};
}

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, WaitEvents &...events)
{
    TSYNC(events...);
    TMOV_IMPL<DstTileData, SrcTileData, mode, reluMode>(dst, src);
    return {};
}

template <STPhase Phase, typename DstTileData, typename SrcTileData, AccToVecMode mode,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, WaitEvents &...events)
{
    TSYNC(events...);
    TMOV_IMPL<DstTileData, SrcTileData, mode, reluMode, Phase>(dst, src);
    return {};
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TMOV_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp, WaitEvents &...events)
{
    TSYNC(events...);
    TMOV_IMPL<DstTileData, SrcTileData, FpTileData, reluMode>(dst, src, fp);
    return {};
}

template <STPhase Phase, typename DstTileData, typename SrcTileData, typename FpTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TMOV_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp, WaitEvents &...events)
{
    TSYNC(events...);
    TMOV_IMPL<DstTileData, SrcTileData, FpTileData, reluMode, Phase>(dst, src, fp);
    return {};
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, FpTileData &fp, WaitEvents &...events)
{
    TSYNC(events...);
    TMOV_IMPL<DstTileData, SrcTileData, FpTileData, mode, reluMode>(dst, src, fp);
    return {};
}

template <STPhase Phase, typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, FpTileData &fp, WaitEvents &...events)
{
    TSYNC(events...);
    TMOV_IMPL<DstTileData, SrcTileData, FpTileData, mode, reluMode, Phase>(dst, src, fp);
    return {};
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, WaitEvents &...events)
{
    TSYNC(events...);
    TMOV_IMPL<DstTileData, SrcTileData, reluMode>(dst, src, preQuantScalar);
    return {};
}

template <STPhase Phase, typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, WaitEvents &...events)
{
    TSYNC(events...);
    TMOV_IMPL<DstTileData, SrcTileData, reluMode, Phase>(dst, src, preQuantScalar);
    return {};
}

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, WaitEvents &...events)
{
    TSYNC(events...);
    TMOV_IMPL<DstTileData, SrcTileData, mode, reluMode>(dst, src, preQuantScalar);
    return {};
}

template <STPhase Phase, typename DstTileData, typename SrcTileData, AccToVecMode mode,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, WaitEvents &...events)
{
    TSYNC(events...);
    TMOV_IMPL<DstTileData, SrcTileData, mode, reluMode, Phase>(dst, src, preQuantScalar);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWSUM(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWSUM, dst, src, tmp);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWPROD(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWPROD, dst, src, tmp);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TCOLSUM(TileDataOut &dst, TileDataIn &src, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLSUM, dst, src);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TCOLSUM(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, bool isBinary, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLSUM, dst, src, tmp, isBinary);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TCOLPROD(TileDataOut &dst, TileDataIn &src, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLPROD, dst, src);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TCOLMAX(TileDataOut &dst, TileDataIn &src, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLMAX, dst, src);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TCOLARGMAX(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLARGMAX, dst, src, tmp);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TCOLARGMIN(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLARGMIN, dst, src, tmp);
    return {};
}

template <typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, typename TileDataTmp,
          typename... WaitEvents, std::enable_if_t<is_tile_data_v<TileDataTmp> && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TCOLARGMAX(TileDataOutVal &dstVal, TileDataOutIdx &dstIdx, TileDataIn &src, TileDataTmp &tmp,
                                WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLARGMAX, dstVal, dstIdx, src, tmp);
    return {};
}

template <typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, typename TileDataTmp,
          typename... WaitEvents, std::enable_if_t<is_tile_data_v<TileDataTmp> && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TCOLARGMIN(TileDataOutVal &dstVal, TileDataOutIdx &dstIdx, TileDataIn &src, TileDataTmp &tmp,
                                WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLARGMIN, dstVal, dstIdx, src, tmp);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWMAX(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWMAX, dst, src, tmp);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents,
          std::enable_if_t<all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TROWARGMAX(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWARGMAX, dst, src, tmp);
    return {};
}

template <typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, typename TileDataTmp,
          typename... WaitEvents, std::enable_if_t<is_tile_data_v<TileDataTmp> && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TROWARGMAX(TileDataOutVal &dstVal, TileDataOutIdx &dstIdx, TileDataIn &src, TileDataTmp &tmp,
                                WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWARGMAX, dstVal, dstIdx, src, tmp);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TRESHAPE(TileDataOut &dst, TileDataIn &src, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TRESHAPE, dst, src);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWMIN(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWMIN, dst, src, tmp);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents,
          std::enable_if_t<all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TROWARGMIN(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWARGMIN, dst, src, tmp);
    return {};
}

template <typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, typename TileDataTmp,
          typename... WaitEvents, std::enable_if_t<is_tile_data_v<TileDataTmp> && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TROWARGMIN(TileDataOutVal &dstVal, TileDataOutIdx &dstIdx, TileDataIn &src, TileDataTmp &tmp,
                                WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWARGMIN, dstVal, dstIdx, src, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataMask, typename TileDataSrc, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TSELS(TileDataDst &dst, TileDataMask &mask, TileDataSrc &src, TileDataTmp &tmp,
                           typename TileDataSrc::DType scalar, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSELS, dst, mask, src, tmp, scalar);
    return {};
}

template <typename TileData, typename MaskTile, typename TmpTile, typename... WaitEvents>
PTO_INST RecordEvent TSEL(TileData &dst, MaskTile &selMask, TileData &src0, TileData &src1, TmpTile &tmp,
                          WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSEL, dst, selMask, src0, src1, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TTRANS(TileDataDst &dst, TileDataSrc &src, TileDataTmp &tmp, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TTRANS, dst, src, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TMINS(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar,
                           WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMINS, dst, src, scalar);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPAND(TileDataDst &dst, TileDataSrc &src, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPAND, dst, src);
    return {};
}

template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0,
          typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDDIV(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    TROWEXPANDDIV_IMPL<PrecisionType>(dst, src0, src1);
    return {};
}

template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0,
          typename TileDataSrc1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDDIV(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp,
                                   WaitEvents &...events)
{
    TSYNC(events...);
    TROWEXPANDDIV_IMPL<PrecisionType>(dst, src0, src1, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMUL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDMUL, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMUL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp,
                                   WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDMUL, dst, src0, src1, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDSUB(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDSUB, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDSUB(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp,
                                   WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDSUB, dst, src0, src1, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDADD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDADD, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDADD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp,
                                   WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDADD, dst, src0, src1, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMAX(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDMAX, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMAX(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp,
                                   WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDMAX, dst, src0, src1, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMIN(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDMIN, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMIN(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp,
                                   WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDMIN, dst, src0, src1, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDEXPDIF(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDEXPDIF, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDEXPDIF(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp,
                                      WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDEXPDIF, dst, src0, src1, tmp);
    return {};
}

template <auto PrecisionType = RsqrtAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc,
          typename... WaitEvents, std::enable_if_t<all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TRSQRT(TileDataDst &dst, TileDataSrc &src, WaitEvents &...events)
{
    TSYNC(events...);
    TRSQRT_IMPL<PrecisionType>(dst, src);
    return {};
}

template <auto PrecisionType = RsqrtAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc,
          typename TileDataTmp, typename... WaitEvents,
          std::enable_if_t<is_tile_data_v<TileDataTmp> && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TRSQRT(TileDataDst &dst, TileDataSrc &src, TileDataTmp &tmp, WaitEvents &...events)
{
    TSYNC(events...);
    TRSQRT_IMPL<PrecisionType>(dst, src, tmp);
    return {};
}

template <auto PrecisionType = SqrtAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc,
          typename... WaitEvents>
PTO_INST RecordEvent TSQRT(TileDataDst &dst, TileDataSrc &src, WaitEvents &...events)
{
    TSYNC(events...);
    TSQRT_IMPL<PrecisionType>(dst, src);
    return {};
}

template <auto PrecisionType = ExpAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc,
          typename... WaitEvents>
PTO_INST RecordEvent TEXP(TileDataDst &dst, TileDataSrc &src, WaitEvents &...events)
{
    TSYNC(events...);
    TEXP_IMPL<PrecisionType>(dst, src);
    return {};
}

template <auto PrecisionType = PowAlgorithm::DEFAULT, typename DstTile, typename BaseTile, typename ExpTile,
          typename TmpTile, typename... WaitEvents>
PTO_INTERNAL RecordEvent TPOW(DstTile &dst, BaseTile &base, ExpTile &exp, TmpTile &tmp, WaitEvents &...events)
{
    TSYNC(events...);
    TPOW_IMPL<PrecisionType>(dst, base, exp, tmp);
    return {};
}

template <auto PrecisionType = PowAlgorithm::DEFAULT, typename DstTile, typename BaseTile, typename TmpTile,
          typename... WaitEvents>
PTO_INTERNAL RecordEvent TPOWS(DstTile &dst, BaseTile &base, typename DstTile::DType exp, TmpTile &tmp,
                               WaitEvents &...events)
{
    TSYNC(events...);
    TPOWS_IMPL<PrecisionType>(dst, base, exp, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TNOT(TileDataDst &dst, TileDataSrc &src, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TNOT, dst, src);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TRELU(TileDataDst &dst, TileDataSrc &src, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TRELU, dst, src);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataOffset, typename... WaitEvents>
PTO_INST RecordEvent TGATHERB(TileDataDst &dst, TileDataSrc &src, TileDataOffset &offset, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TGATHERB, dst, src, offset);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TADDS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar,
                           WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TADDS, dst, src0, scalar);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TAXPY(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar,
                           WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TAXPY, dst, src0, scalar);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TSUBS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar,
                           WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSUBS, dst, src0, scalar);
    return {};
}

template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc,
          typename... WaitEvents>
PTO_INST RecordEvent TDIVS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar,
                           WaitEvents &...events)
{
    TSYNC(events...);
    TDIVS_IMPL<PrecisionType>(dst, src0, scalar);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TMULS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar,
                           WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMULS, dst, src0, scalar);
    return {};
}

template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc,
          typename... WaitEvents>
PTO_INST RecordEvent TDIVS(TileDataDst &dst, typename TileDataDst::DType scalar, TileDataSrc &src0,
                           WaitEvents &...events)
{
    TSYNC(events...);
    TDIVS_IMPL<PrecisionType>(dst, scalar, src0);
    return {};
}

template <auto PrecisionType = FmodSAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc,
          typename... WaitEvents>
PTO_INST RecordEvent TFMODS(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar,
                            WaitEvents &...events)
{
    TSYNC(events...);
    TFMODS_IMPL<PrecisionType>(dst, src, scalar);
    return {};
}

template <auto PrecisionType = RemSAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TREMS(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar, TileDataTmp &tmp,
                           WaitEvents &...events)
{
    TSYNC(events...);
    TREMS_IMPL<PrecisionType>(dst, src, scalar, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TMAXS(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar,
                           WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMAXS, dst, src, scalar);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TANDS(TileDataDst &dst, TileDataSrc &src, typename TileDataDst::DType scalar,
                           WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TANDS, dst, src, scalar);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TORS(TileDataDst &dst, TileDataSrc &src, typename TileDataDst::DType scalar, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TORS, dst, src, scalar);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TSHLS(TileDataDst &dst, TileDataSrc &src, typename TileDataDst::DType scalar,
                           WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSHLS, dst, src, scalar);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TSHRS(TileDataDst &dst, TileDataSrc &src, typename TileDataDst::DType scalar,
                           WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSHRS, dst, src, scalar);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TXORS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar, TileDataTmp &tmp,
                           WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TXORS, dst, src0, scalar, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TLRELU(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar,
                            WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TLRELU, dst, src, scalar);
    return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TADDSC(TileData &dst, TileData &src0, typename TileData::DType scalar, TileData &src1,
                            WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TADDSC, dst, src0, scalar, src1);
    return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TSUBSC(TileData &dst, TileData &src0, typename TileData::DType scalar, TileData &src1,
                            WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSUBSC, dst, src0, scalar, src1);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TCOLMIN(TileDataOut &dst, TileDataIn &src, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLMIN, dst, src);
    return {};
}

template <typename TileDataD, typename TileDataS, typename TileDataI, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(TileDataD &dst, TileDataS &src, TileDataI &indexes, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSCATTER, dst, src, indexes);
    return {};
}

template <MaskPattern maskPattern = MaskPattern::P1111, auto ScatterType = ScatterAxis::SCATTER_ROW,
          typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(DstTileData &dst, SrcTileData &src, WaitEvents &...events)
{
    TSYNC(events...);
    TSCATTER_IMPL<maskPattern, ScatterType>(dst, src);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPAND(TileDataDst &dst, TileDataSrc &src, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLEXPAND, dst, src);
    return {};
}

template <typename TileDst, typename GlobalData, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MGATHER(TileDst &dst, GlobalData &src, TileInd &indexes, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(MGATHER, dst, src, indexes);
    return {};
}

template <Coalesce CMode, typename TileDst, typename GlobalData, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MGATHER(TileDst &dst, GlobalData &src, TileInd &indexes, WaitEvents &...events)
{
    TSYNC(events...);
    MGATHER_IMPL<CMode>(dst, src, indexes);
    return {};
}

template <Coalesce CMode, GatherOOB Mode, typename TileDst, typename GlobalData, typename TileInd,
          typename... WaitEvents>
PTO_INST RecordEvent MGATHER(TileDst &dst, GlobalData &src, TileInd &indexes, WaitEvents &...events)
{
    TSYNC(events...);
    MGATHER_IMPL<CMode, Mode>(dst, src, indexes);
    return {};
}

template <Coalesce CMode, typename TileDst, typename GlobalData, typename GlobalIdx, typename GlobalScratch>
PTO_INST RecordEvent MGATHER(TileDst &dst, GlobalData &src, GlobalIdx &indexes, GlobalScratch &scratch)
{
    MGATHER_IMPL<CMode>(dst, src, indexes, scratch);
    return {};
}

template <Coalesce CMode, GatherOOB Mode, typename TileDst, typename GlobalData, typename GlobalIdx,
          typename GlobalScratch>
PTO_INST RecordEvent MGATHER(TileDst &dst, GlobalData &src, GlobalIdx &indexes, GlobalScratch &scratch)
{
    MGATHER_IMPL<CMode, Mode>(dst, src, indexes, scratch);
    return {};
}

#ifdef PTO_NPU_ARCH_A5
template <Coalesce CMode, GatherOOB Mode, GatherExec Exec, typename TileDst, typename GlobalData, typename GlobalIdx,
          typename GlobalScratch>
PTO_INST RecordEvent MGATHER(TileDst &dst, GlobalData &src, GlobalIdx &indexes, GlobalScratch &scratch)
{
    MGATHER_IMPL<CMode, Mode, Exec>(dst, src, indexes, scratch);
    return {};
}
#endif

template <typename GlobalData, typename TileSrc, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MSCATTER(GlobalData &dst, TileSrc &src, TileInd &indexes, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(MSCATTER, dst, src, indexes);
    return {};
}

#if defined(PTO_NPU_ARCH_A5) || defined(__CPU_SIM)
template <Coalesce Mode, typename GlobalData, typename TileSrc, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MSCATTER(GlobalData &dst, TileSrc &src, TileInd &indexes, WaitEvents &...events)
{
    TSYNC(events...);
    MSCATTER_IMPL<Mode>(dst, src, indexes);
    return {};
}

template <Coalesce Mode, ScatterAtomicOp Atomic, typename GlobalData, typename TileSrc, typename TileInd,
          typename... WaitEvents>
PTO_INST RecordEvent MSCATTER(GlobalData &dst, TileSrc &src, TileInd &indexes, WaitEvents &...events)
{
    TSYNC(events...);
    MSCATTER_IMPL<Mode, Atomic>(dst, src, indexes);
    return {};
}

template <Coalesce Mode, ScatterAtomicOp Atomic, ScatterOOB Oob, typename GlobalData, typename TileSrc,
          typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MSCATTER(GlobalData &dst, TileSrc &src, TileInd &indexes, WaitEvents &...events)
{
    TSYNC(events...);
    MSCATTER_IMPL<Mode, Atomic, Oob>(dst, src, indexes);
    return {};
}

template <Coalesce Mode, ScatterAtomicOp Atomic, ScatterOOB Oob, ScatterConflict Conflict, typename GlobalData,
          typename TileSrc, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MSCATTER(GlobalData &dst, TileSrc &src, TileInd &indexes, WaitEvents &...events)
{
    TSYNC(events...);
    MSCATTER_IMPL<Mode, Atomic, Oob, Conflict>(dst, src, indexes);
    return {};
}
#endif

#ifdef PTO_NPU_ARCH_A2A3
template <Coalesce Mode, typename GlobalData, typename TileSrc, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MSCATTER(GlobalData &dst, TileSrc &src, TileInd &indexes, WaitEvents &...events)
{
    TSYNC(events...);
    MSCATTER_IMPL<Mode>(dst, src, indexes);
    return {};
}

template <Coalesce Mode, ScatterAtomicOp Atomic, typename GlobalData, typename TileSrc, typename TileInd,
          typename... WaitEvents>
PTO_INST RecordEvent MSCATTER(GlobalData &dst, TileSrc &src, TileInd &indexes, WaitEvents &...events)
{
    TSYNC(events...);
    MSCATTER_IMPL<Mode, Atomic>(dst, src, indexes);
    return {};
}

template <Coalesce Mode, ScatterAtomicOp Atomic, ScatterOOB Oob, typename GlobalData, typename TileSrc,
          typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MSCATTER(GlobalData &dst, TileSrc &src, TileInd &indexes, WaitEvents &...events)
{
    TSYNC(events...);
    MSCATTER_IMPL<Mode, Atomic, Oob>(dst, src, indexes);
    return {};
}
#endif

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TNEG(TileDataDst &dst, TileDataSrc &src, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TNEG, dst, src);
    return {};
}

template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0,
          typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDDIV(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    TCOLEXPANDDIV_IMPL<PrecisionType>(dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDMUL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLEXPANDMUL, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDADD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLEXPANDADD, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDMAX(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLEXPANDMAX, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDMIN(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLEXPANDMIN, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDSUB(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLEXPANDSUB, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDEXPDIF(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLEXPANDEXPDIF, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataPara, typename... WaitEvents>
PTO_INST RecordEvent TDEQUANT(TileDataDst &dst, TileDataSrc &src, TileDataPara &scale, TileDataPara &offset,
                              WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TDEQUANT, dst, src, scale, offset);
    return {};
}

template <auto PrecisionType = RemAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0,
          typename TileDataSrc1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TREM(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp,
                          WaitEvents &...events)
{
    TSYNC(events...);
    TREM_IMPL<PrecisionType>(dst, src0, src1, tmp);
    return {};
}

template <auto PrecisionType = FmodAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0,
          typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TFMOD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &...events)
{
    TSYNC(events...);
    TFMOD_IMPL<PrecisionType>(dst, src0, src1);
    return {};
}

template <typename Pipe, typename TileProd, TileSplitAxis Split, std::enable_if_t<is_tile_data_v<TileProd>, int> = 0,
          typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe &pipe, TileProd &tile, WaitEvents &...events)
{
    TSYNC(events...);
    TPUSH_IMPL<Pipe, TileProd, Split>(pipe, tile);
    return {};
}

template <typename TileData, typename Pipe, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(TileData &tile, Pipe &pipe, WaitEvents &...events)
{
    TSYNC(events...);
    TPUSH_IMPL<TileData, Pipe>(tile, pipe);
    return {};
}

template <typename Pipe, typename TileCons, TileSplitAxis Split, std::enable_if_t<is_tile_data_v<TileCons>, int> = 0,
          typename... WaitEvents>
PTO_INST RecordEvent TPOP(Pipe &pipe, TileCons &tile, WaitEvents &...events)
{
    TSYNC(events...);
    TPOP_IMPL<Pipe, TileCons, Split>(pipe, tile);
    return {};
}

template <typename TileData, typename Pipe, typename... WaitEvents>
PTO_INST RecordEvent TPOP(TileData &tile, Pipe &pipe, WaitEvents &...events)
{
    TSYNC(events...);
    TPOP_IMPL<TileData, Pipe>(tile, pipe);
    return {};
}

template <typename Pipe, TileSplitAxis Split, typename... WaitEvents>
PTO_INST RecordEvent TFREE(Pipe &pipe, WaitEvents &...events)
{
    TSYNC(events...);
    TFREE_IMPL<Pipe, Split>(pipe);
    return {};
}

template <typename Pipe, typename GlobalData, TileSplitAxis Split,
          std::enable_if_t<is_global_data_v<GlobalData>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TALLOC(Pipe &pipe, GlobalData &gmTensor, WaitEvents &...events)
{
    TSYNC(events...);
    TALLOC_IMPL<Pipe, GlobalData, Split>(pipe, gmTensor);
    return {};
}

template <typename Pipe, typename GlobalData, TileSplitAxis Split,
          std::enable_if_t<is_global_data_v<GlobalData>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe &pipe, GlobalData &gmTensor, WaitEvents &...events)
{
    TSYNC(events...);
    TPUSH_IMPL<Pipe, GlobalData, Split>(pipe, gmTensor);
    return {};
}

template <typename Pipe, typename GlobalData, TileSplitAxis Split,
          std::enable_if_t<is_global_data_v<GlobalData>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TPOP(Pipe &pipe, GlobalData &gmTensor, WaitEvents &...events)
{
    TSYNC(events...);
    TPOP_IMPL<Pipe, GlobalData, Split>(pipe, gmTensor);
    return {};
}

template <typename Pipe, typename GlobalData, TileSplitAxis Split,
          std::enable_if_t<is_global_data_v<GlobalData>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TFREE(Pipe &pipe, GlobalData &gmTensor, WaitEvents &...events)
{
    TSYNC(events...);
    TFREE_IMPL<Pipe, GlobalData, Split>(pipe, gmTensor);
    return {};
}

template <typename Pipe, typename... WaitEvents>
PTO_INST RecordEvent TFREE(Pipe &pipe, WaitEvents &...events)
{
    TSYNC(events...);
    TFREE_IMPL<Pipe>(pipe);
    return {};
}
#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRIN9030) || defined(__CPU_SIM)
template <HistByte byte, typename TileDataDst, typename TileDataSrc, typename TileDataIdx, typename... WaitEvents>
PTO_INST RecordEvent THISTOGRAM(TileDataDst &dst, TileDataSrc &src, TileDataIdx &idx, WaitEvents &...events)
{
    TSYNC(events...);
    THISTOGRAM_IMPL<byte>(dst, src, idx);
    return {};
}

template <auto quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataExp, typename TileDataMax,
          typename TileDataScaling, auto scale_alg = QuantScaleAlg::OCP, typename... WaitEvents>
PTO_INST RecordEvent TQUANT(TileDataOut &dst, TileDataSrc &src, TileDataExp *exp, TileDataMax *max,
                            TileDataScaling *scaling, WaitEvents &...events)
{
    TSYNC(events...);
    TQUANT_IMPL<quant_type, scale_alg, TileDataOut, TileDataSrc, TileDataExp, TileDataMax, TileDataScaling>(
        dst, src, exp, max, scaling);
    return {};
}

template <auto quant_type, auto store_mode, typename TileDataOut, typename TileDataSrc, typename TileDataExp,
          typename TileDataMax, typename TileDataScaling, typename... WaitEvents>
PTO_INST RecordEvent TQUANT(TileDataOut &dst, TileDataSrc &src, TileDataExp *exp, TileDataMax *max,
                            TileDataScaling *scaling, TileDataExp *exp_zz, WaitEvents &...events)
{
    TSYNC(events...);
    TQUANT_IMPL<quant_type, store_mode>(dst, src, exp, max, scaling, exp_zz);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TInterleave(TileDataDst &dst1, TileDataDst &dst0, TileDataSrc &src1, TileDataSrc &src0,
                                 WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TINTERLEAVE, dst1, dst0, src1, src0);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TDeInterleave(TileDataDst &dst1, TileDataDst &dst0, TileDataSrc &src1, TileDataSrc &src0,
                                   WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TDEINTERLEAVE, dst1, dst0, src1, src0);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TDeInterleave(TileDataDst &dst1, TileDataDst &dst0, TileDataSrc &src, WaitEvents &...events)
{
    TSYNC(events...);
    TDEINTERLEAVE_IMPL(dst1, dst0, src);
    return {};
}
#endif

template <auto quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataPara, typename... WaitEvents>
PTO_INST RecordEvent TQUANT(TileDataOut &dst, TileDataSrc &src, TileDataPara &scale, TileDataPara *offset = nullptr,
                            WaitEvents &...events)
{
    TSYNC(events...);
    TQUANT_IMPL<quant_type, TileDataOut, TileDataSrc, TileDataPara>(dst, src, scale, offset);
    return {};
}

// Tmp-aware overload (A2/A3): the row-wise scale/offset broadcast needs an explicit scratch tile.
template <auto quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataPara, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TQUANT(TileDataOut &dst, TileDataSrc &src, TileDataPara &scale, TileDataTmp &tmp,
                            TileDataPara *offset = nullptr, WaitEvents &...events)
{
    TSYNC(events...);
    TQUANT_IMPL<quant_type, TileDataOut, TileDataSrc, TileDataPara, TileDataTmp>(dst, src, scale, tmp, offset);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TGET_SCALE_ADDR(TileDataOut &dst, TileDataIn &src, WaitEvents &...events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TGET_SCALE_ADDR, dst, src);
    return {};
}

} // namespace pto
#endif
