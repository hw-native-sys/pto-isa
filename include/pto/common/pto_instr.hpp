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

template <typename... WaitEvents>
PTO_INST void TSYNC(WaitEvents &... events)
{
    WaitAllEvents(events...);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TADD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TADD, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TABS(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TABS, dst, src);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TAND(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TAND, dst, src0, src1);
    return {};
}

#ifdef __PTO_AUTO__
// temp hack: needed by auto mode to support aliasing right now
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TALIAS(TileDataDst &original, TileDataSrc &alias, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TALIAS, original, alias);
    return {};
}
#endif

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TOR(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TOR, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TSUB(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSUB, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TSUBVIEW(TileDataDst &dst, TileDataSrc &src, uint16_t rowIdx, uint16_t colIdx,
                              WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSUBVIEW, dst, src, rowIdx, colIdx);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TMUL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMUL, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TMIN(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMIN, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TMAX(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMAX, dst, src0, src1);
    return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TEXPANDS(TileData &dst, typename TileData::DType scalar, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TEXPANDS, dst, scalar);
    return {};
}

template <typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TLOAD(TileData &dst, GlobalData &src, WaitEvents &... events)
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

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TCMPS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType src1, CmpMode mode,
                           WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCMPS, dst, src0, src1, mode);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1,
          typename = std::void_t<typename TileDataSrc1::DType>, typename... WaitEvents>
PTO_INST RecordEvent TCMPS(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, CmpMode mode,
                           WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCMPS, dst, src0, src1, mode);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TCMP(TileDataDst &dst, TileDataSrc &src0, TileDataSrc &src1, CmpMode cmpMode,
                          WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCMP, dst, src0, src1, cmpMode);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCONCAT(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCONCAT, dst, src0, src1);
    return {};
}

template <typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src, WaitEvents &... events)
{
    TSYNC(events...);
    TSTORE_IMPL<TileData, GlobalData, AtomicType::AtomicNone>(dst, src);
    return {};
}

// UF-aware overload: allow selecting unit-flag phase while keeping the TSTORE name.
template <STPhase Phase, typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src, WaitEvents &... events)
{
    TSYNC(events...);
    TSTORE_IMPL<TileData, GlobalData, AtomicType::AtomicNone, Phase>(dst, src);
    return {};
}

template <typename TileData, typename GlobalData, AtomicType atomicType, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src, WaitEvents &... events)
{
    TSYNC(events...);
    TSTORE_IMPL<TileData, GlobalData, atomicType>(dst, src);
    return {};
}

template <STPhase Phase, typename TileData, typename GlobalData, AtomicType atomicType, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src, WaitEvents &... events)
{
    TSYNC(events...);
    TSTORE_IMPL<TileData, GlobalData, atomicType, Phase>(dst, src);
    return {};
}

template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
          ReluPreMode reluPreMode, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src, WaitEvents &... events)
{
    TSYNC(events...);
    TSTORE_IMPL<TileData, GlobalData, atomicType, reluPreMode>(dst, src);
    return {};
}

template <STPhase Phase, typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
          ReluPreMode reluPreMode, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src, WaitEvents &... events)
{
    TSYNC(events...);
    TSTORE_IMPL<TileData, GlobalData, atomicType, reluPreMode, Phase>(dst, src);
    return {};
}

template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src, uint64_t preQuantScalar, WaitEvents &... events)
{
    TSYNC(events...);
    TSTORE_IMPL<TileData, GlobalData, atomicType, reluPreMode>(dst, src, preQuantScalar);
    return {};
}

template <STPhase Phase, typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src, uint64_t preQuantScalar, WaitEvents &... events)
{
    TSYNC(events...);
    TSTORE_IMPL<TileData, GlobalData, atomicType, reluPreMode, Phase>(dst, src, preQuantScalar);
    return {};
}

template <typename TileData, typename GlobalData, typename FpTileData, AtomicType atomicType = AtomicType::AtomicNone,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TSTORE_FP(GlobalData &dst, TileData &src, FpTileData &fp, WaitEvents &... events)
{
    TSYNC(events...);
    TSTORE_IMPL<TileData, GlobalData, FpTileData, atomicType, reluPreMode>(dst, src, fp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TDIV(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TDIV, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TSHL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSHL, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TSHR(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSHR, dst, src0, src1);
    return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TAND(TileData &dst, TileData &src0, TileData &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TAND, dst, src0, src1);
    return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TOR(TileData &dst, TileData &src0, TileData &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TOR, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TXOR(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp,
                          WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TXOR, dst, src0, src1, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TLOG(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TLOG, dst, src);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TRECIP(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events)
{
    TSYNC(events...);
    /*
     * A3's TRECIP instruction does not support setting the source Tile and destination Tile to the same memory.
     */
    MAP_INSTR_IMPL(TDIVS, dst, 1, src);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TPRELU(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp,
                            WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TPRELU, dst, src0, src1, tmp);
    return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TPRINT(TileData &src, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TPRINT, src);
    return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TADDC(TileData &dst, TileData &src0, TileData &src1, TileData &src2, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TADDC, dst, src0, src1, src2);
    return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TSUBC(TileData &dst, TileData &src0, TileData &src1, TileData &src2, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSUBC, dst, src0, src1, src2);
    return {};
}

#ifdef PTO_NPU_ARCH_A5
template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
          typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix,
                              TileRightScale &bScaleMatrix, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TGEMV_MX, cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix);
    return {};
}

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
          typename TileRightScale, typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix,
                              TileRightScale &bScaleMatrix, WaitEvents &... events)
{
    TSYNC(events...);
    TGEMV_MX_IMPL<Phase>(cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix);
    return {};
}

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
          typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix,
                              TileRight &bMatrix, TileRightScale &bScaleMatrix, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TGEMV_MX, cOutMatrix, cInMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix);
    return {};
}

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
          typename TileRightScale, typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix,
                              TileRight &bMatrix, TileRightScale &bScaleMatrix, WaitEvents &... events)
{
    TSYNC(events...);
    TGEMV_MX_IMPL<Phase>(cOutMatrix, cInMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix);
    return {};
}

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
          typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix,
                              TileRightScale &bScaleMatrix, TileBias &biasData, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TGEMV_MX, cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix, biasData);
    return {};
}

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
          typename TileRightScale, typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix,
                              TileRightScale &bScaleMatrix, TileBias &biasData, WaitEvents &... events)
{
    TSYNC(events...);
    TGEMV_MX_IMPL<Phase>(cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix, biasData);
    return {};
}

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
          typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix,
                                TileRightScale &bScaleMatrix, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMATMUL_MX, cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix);
    return {};
}

// UF-aware overload enabling unit-flag selection via AccPhase while retaining the TMATMUL name.
template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
          typename TileRightScale, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix,
                                TileRightScale &bScaleMatrix, WaitEvents &... events)
{
    TSYNC(events...);
    TMATMUL_MX_IMPL<Phase>(cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix);
    return {};
}

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
          typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix,
                                TileRight &bMatrix, TileRightScale &bScaleMatrix, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMATMUL_MX, cOutMatrix, cInMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix);
    return {};
}

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
          typename TileRightScale, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix,
                                TileRight &bMatrix, TileRightScale &bScaleMatrix, WaitEvents &... events)
{
    TSYNC(events...);
    TMATMUL_MX_IMPL<Phase>(cOutMatrix, cInMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix);
    return {};
}

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
          typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix,
                                TileRightScale &bScaleMatrix, TileBias &biasData, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMATMUL_MX, cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix, biasData);
    return {};
}

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
          typename TileRightScale, typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(TileRes &cMatrix, TileLeft &aMatrix, TileLeftScale &aScaleMatrix, TileRight &bMatrix,
                                TileRightScale &bScaleMatrix, TileBias &biasData, WaitEvents &... events)
{
    TSYNC(events...);
    TMATMUL_MX_IMPL<Phase>(cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix, biasData);
    return {};
}

#endif

template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMATMUL, cMatrix, aMatrix, bMatrix);
    return {};
}

// UF-aware overload enabling unit-flag selection via AccPhase while retaining the TMATMUL name.
template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, WaitEvents &... events)
{
    TSYNC(events...);
    TMATMUL_IMPL<Phase>(cMatrix, aMatrix, bMatrix);
    return {};
}

template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_ACC(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix,
                                 WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMATMUL_ACC, cOutMatrix, cInMatrix, aMatrix, bMatrix);
    return {};
}

// UF-aware overloads for TMATMUL_ACC: explicit input/output or shared accumulator tile.
template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_ACC(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix,
                                 WaitEvents &... events)
{
    TSYNC(events...);
    TMATMUL_ACC_IMPL<Phase>(cOutMatrix, cInMatrix, aMatrix, bMatrix);
    return {};
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight,
          typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_ACC(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, WaitEvents &... events)
{
    TSYNC(events...);
    TMATMUL_ACC_IMPL<Phase>(cMatrix, aMatrix, bMatrix);
    return {};
}

template <typename TileRes, typename TileLeft, typename TileRight, typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_BIAS(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData,
                                  WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMATMUL_BIAS, cMatrix, aMatrix, bMatrix, biasData);
    return {};
}

// UF-aware overload enabling unit-flag selection for bias matmul while keeping the TMATMUL_BIAS name.
template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename TileBias,
          typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_BIAS(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData,
                                  WaitEvents &... events)
{
    TSYNC(events...);
    TMATMUL_BIAS_IMPL<Phase>(cMatrix, aMatrix, bMatrix, biasData);
    return {};
}

template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TGEMV(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TGEMV, cMatrix, aMatrix, bMatrix);
    return {};
}

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TGEMV(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, WaitEvents &... events)
{
    TSYNC(events...);
    TGEMV_IMPL<Phase>(cMatrix, aMatrix, bMatrix);
    return {};
}

template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TGEMV_ACC(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix,
                               WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TGEMV_ACC, cOutMatrix, cInMatrix, aMatrix, bMatrix);
    return {};
}

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TGEMV_ACC(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix,
                               WaitEvents &... events)
{
    TSYNC(events...);
    TGEMV_ACC_IMPL<Phase>(cOutMatrix, cInMatrix, aMatrix, bMatrix);
    return {};
}

template <typename TileRes, typename TileLeft, typename TileRight, typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TGEMV_BIAS(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData,
                                WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TGEMV_BIAS, cMatrix, aMatrix, bMatrix, biasData);
    return {};
}

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename TileBias,
          typename... WaitEvents>
PTO_INST RecordEvent TGEMV_BIAS(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData,
                                WaitEvents &... events)
{
    TSYNC(events...);
    TGEMV_BIAS_IMPL<Phase>(cMatrix, aMatrix, bMatrix, biasData);
    return {};
}

template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData,
          typename Src2TileData, typename Src3TileData, bool exhausted, typename... WaitEvents>
PTO_INST RecordEvent TMRGSORT(DstTileData &dst, MrgSortExecutedNumList &executedNumList, TmpTileData &tmp,
                              Src0TileData &src0, Src1TileData &src1, Src2TileData &src2, Src3TileData &src3,
                              WaitEvents &... events)
{
    TSYNC(events...);
    TMRGSORT_IMPL<DstTileData, TmpTileData, Src0TileData, Src1TileData, Src2TileData, Src3TileData, exhausted>(
        dst, executedNumList, tmp, src0, src1, src2, src3);
    return {};
}

template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData,
          typename Src2TileData, bool exhausted, typename... WaitEvents>
PTO_INST RecordEvent TMRGSORT(DstTileData &dst, MrgSortExecutedNumList &executedNumList, TmpTileData &tmp,
                              Src0TileData &src0, Src1TileData &src1, Src2TileData &src2, WaitEvents &... events)
{
    TSYNC(events...);
    TMRGSORT_IMPL<DstTileData, TmpTileData, Src0TileData, Src1TileData, Src2TileData, exhausted>(dst, executedNumList,
                                                                                                 tmp, src0, src1, src2);
    return {};
}

template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData, bool exhausted,
          typename... WaitEvents>
PTO_INST RecordEvent TMRGSORT(DstTileData &dst, MrgSortExecutedNumList &executedNumList, TmpTileData &tmp,
                              Src0TileData &src0, Src1TileData &src1, WaitEvents &... events)
{
    TSYNC(events...);
    TMRGSORT_IMPL<DstTileData, TmpTileData, Src0TileData, Src1TileData, exhausted>(dst, executedNumList, tmp, src0,
                                                                                   src1);
    return {};
}

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TMRGSORT(DstTileData &dst, SrcTileData &src, uint32_t blockLen, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMRGSORT, dst, src, blockLen);
    return {};
}

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0,
                              WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TEXTRACT, dst, src, indexRow, indexCol);
    return {};
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol,
                              WaitEvents &... events)
{
    TSYNC(events...);
    TEXTRACT_IMPL<DstTileData, SrcTileData, reluMode>(dst, src, indexRow, indexCol);
    return {};
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint16_t indexRow,
                              uint16_t indexCol, WaitEvents &... events)
{
    TSYNC(events...);
    TEXTRACT_IMPL<DstTileData, SrcTileData, reluMode>(dst, src, preQuantScalar, indexRow, indexCol);
    return {};
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow,
                                 uint16_t indexCol, WaitEvents &... events)
{
    TSYNC(events...);
    TEXTRACT_IMPL<DstTileData, SrcTileData, FpTileData, reluMode>(dst, src, fp, indexRow, indexCol);
    return {};
}

template <typename TileData, typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL,
          typename... WaitEvents>
PTO_INST RecordEvent TIMG2COL(TileData &dst, ConvTileData &src, uint16_t posM = 0, uint16_t posK = 0,
                              WaitEvents &... events)
{
    TSYNC(events...);
    TIMG2COL_IMPL<TileData, ConvTileData, FmatrixMode>(dst, src, posM, posK);
    return {};
}

template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent TSETFMATRIX(ConvTileData &src, WaitEvents &... events)
{
    TSYNC(events...);
    TSETFMATRIX_IMPL<ConvTileData, FmatrixMode>(src);
    return {};
}

#ifdef PTO_NPU_ARCH_A2A3
template <typename ConvTileData, typename... WaitEvents>
PTO_INST RecordEvent TSET_IMG2COL_RPT(ConvTileData &src, WaitEvents &... events)
{
    TSYNC(events...);
    TSET_IMG2COL_RPT_IMPL<ConvTileData>(src);
    return {};
}

template <typename ConvTileData, typename... WaitEvents>
PTO_INST RecordEvent TSET_IMG2COL_PADDING(ConvTileData &src, WaitEvents &... events)
{
    TSYNC(events...);
    TSET_IMG2COL_PADDING_IMPL<ConvTileData>(src);
    return {};
}
#endif
#if defined PTO_NPU_ARCH_A5
template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent TSET_IMG2COL_RPT(ConvTileData &src, WaitEvents &... events)
{
    TSYNC(events...);
    TSET_IMG2COL_RPT_IMPL<ConvTileData, FmatrixMode>(src);
    return {};
}

template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent TSET_IMG2COL_PADDING(ConvTileData &src, WaitEvents &... events)
{
    TSYNC(events...);
    TSET_IMG2COL_PADDING_IMPL<ConvTileData, FmatrixMode>(src);
    return {};
}
#endif

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol,
                             WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TINSERT, dst, src, indexRow, indexCol);
    return {};
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol,
                             WaitEvents &... events)
{
    TSYNC(events...);
    TINSERT_IMPL<DstTileData, SrcTileData, reluMode>(dst, src, indexRow, indexCol);
    return {};
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint16_t indexRow,
                             uint16_t indexCol, WaitEvents &... events)
{
    TSYNC(events...);
    TINSERT_IMPL<DstTileData, SrcTileData, reluMode>(dst, src, preQuantScalar, indexRow, indexCol);
    return {};
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TINSERT_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow,
                                uint16_t indexCol, WaitEvents &... events)
{
    TSYNC(events...);
    TINSERT_IMPL<DstTileData, SrcTileData, FpTileData, reluMode>(dst, src, fp, indexRow, indexCol);
    return {};
}

#ifdef PTO_NPU_ARCH_A5
template <TInsertMode mode, typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0,
                             WaitEvents &... events)
{
    TSYNC(events...);
    TINSERT_IMPL<mode>(dst, src, indexRow, indexCol);
    return {};
}
#endif

template <typename TileData, PadValue PadVal = PadValue::Zero,
          std::enable_if_t<(TileData::Loc == TileType::Mat), int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD(TileData &dst, TileData &src, WaitEvents &... events)
{
    TSYNC(events...);
    TFILLPAD_IMPL<TileData, PadVal>(dst, src);
    return {};
}

template <typename DstTileData, typename SrcTileData,
          std::enable_if_t<(DstTileData::Loc == TileType::Vec) && (SrcTileData::Loc == TileType::Vec), int> = 0,
          typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD(DstTileData &dst, SrcTileData &src, WaitEvents &... events)
{
    TSYNC(events...);
    TFILLPAD_IMPL<DstTileData, SrcTileData>(dst, src);
    return {};
}

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD_INPLACE(DstTileData &dst, SrcTileData &src, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TFILLPAD_INPLACE, dst, src);
    return {};
}

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD_EXPAND(DstTileData &dst, SrcTileData &src, WaitEvents &... events)
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
                             WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TGATHER, dst, src0, src1, tmp);
    return {};
}

template <typename TileDataD, typename TileDataS, typename TileDataC, typename TileDataTmp, CmpMode cmpMode, int offset,
          typename... WaitEvents>
PTO_INST RecordEvent TGATHER(TileDataD &dst, TileDataS &src0, typename TileDataS::DType k_value, TileDataC &cdst,
                             TileDataTmp &tmp, WaitEvents &... events)
{
    TSYNC(events...);
    TGATHER_IMPL<TileDataD, TileDataS, TileDataC, TileDataTmp, cmpMode, offset>(dst, src0, k_value, cdst, tmp);
    return {};
}

template <typename TileData, typename T, int descending, typename... WaitEvents>
PTO_INST RecordEvent TCI(TileData &dst, T start, WaitEvents &... events)
{
    TSYNC(events...);
    TCI_IMPL<TileData, T, descending>(dst, start);
    return {};
}

template <typename TileData, int isUpperOrLower, typename... WaitEvents>
PTO_INST RecordEvent TTRI(TileData &dst, int diagonal, WaitEvents &... events)
{
    TSYNC(events...);
    TTRI_IMPL<TileData, isUpperOrLower>(dst, diagonal);
    return {};
}

template <typename DstTileData, typename SrcTileData, MaskPattern maskPattern, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(DstTileData &dst, SrcTileData &src, WaitEvents &... events)
{
    TSYNC(events...);
    TGATHER_IMPL<DstTileData, SrcTileData, maskPattern>(dst, src);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TPARTADD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TPARTADD, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TPARTMUL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TPARTMUL, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TPARTMAX(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TPARTMAX, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TPARTMIN(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TPARTMIN, dst, src0, src1);
    return {};
}

template <typename TileDataD, typename TileDataS, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, RoundMode mode, SaturationMode satMode,
                          WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCVT, dst, src, mode, satMode);
    return {};
}

template <typename TileDataD, typename TileDataS, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD &dst, TileDataS &src, RoundMode mode, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCVT, dst, src, mode);
    return {};
}

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMOV, dst, src);
    return {};
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, WaitEvents &... events)
{
    TSYNC(events...);
    TMOV_IMPL<DstTileData, SrcTileData, reluMode>(dst, src);
    return {};
}

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, WaitEvents &... events)
{
    TSYNC(events...);
    TMOV_IMPL<DstTileData, SrcTileData, mode, reluMode>(dst, src);
    return {};
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TMOV_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp, WaitEvents &... events)
{
    TSYNC(events...);
    TMOV_IMPL<DstTileData, SrcTileData, FpTileData, reluMode>(dst, src, fp);
    return {};
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, FpTileData &fp, WaitEvents &... events)
{
    TSYNC(events...);
    TMOV_IMPL<DstTileData, SrcTileData, FpTileData, mode, reluMode>(dst, src, fp);
    return {};
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, WaitEvents &... events)
{
    TSYNC(events...);
    TMOV_IMPL<DstTileData, SrcTileData, reluMode>(dst, src, preQuantScalar);
    return {};
}

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, WaitEvents &... events)
{
    TSYNC(events...);
    TMOV_IMPL<DstTileData, SrcTileData, mode, reluMode>(dst, src, preQuantScalar);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWSUM(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWSUM, dst, src, tmp);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWPROD(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWPROD, dst, src, tmp);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TCOLSUM(TileDataOut &dst, TileDataIn &src, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLSUM, dst, src);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TCOLSUM(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, bool isBinary, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLSUM, dst, src, tmp, isBinary);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TCOLPROD(TileDataOut &dst, TileDataIn &src, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLPROD, dst, src);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TCOLMAX(TileDataOut &dst, TileDataIn &src, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLMAX, dst, src);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWMAX(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWMAX, dst, src, tmp);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TRESHAPE(TileDataOut &dst, TileDataIn &src, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TRESHAPE, dst, src);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWMIN(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWMIN, dst, src, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataMask, typename TileDataSrc, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TSELS(TileDataDst &dst, TileDataMask &mask, TileDataSrc &src, TileDataTmp &tmp,
                           typename TileDataSrc::DType scalar, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSELS, dst, mask, src, tmp, scalar);
    return {};
}

template <typename TileData, typename MaskTile, typename TmpTile, typename... WaitEvents>
PTO_INST RecordEvent TSEL(TileData &dst, MaskTile &selMask, TileData &src0, TileData &src1, TmpTile &tmp,
                          WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSEL, dst, selMask, src0, src1, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TTRANS(TileDataDst &dst, TileDataSrc &src, TileDataTmp &tmp, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TTRANS, dst, src, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TMINS(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar,
                           WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMINS, dst, src, scalar);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPAND(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPAND, dst, src);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDDIV(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDDIV, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDDIV(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp,
                                   WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDDIV, dst, src0, src1, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMUL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDMUL, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMUL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp,
                                   WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDMUL, dst, src0, src1, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDSUB(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDSUB, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDSUB(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp,
                                   WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDSUB, dst, src0, src1, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDADD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDADD, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDADD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp,
                                   WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDADD, dst, src0, src1, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMAX(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDMAX, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMAX(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp,
                                   WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDMAX, dst, src0, src1, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMIN(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDMIN, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMIN(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp,
                                   WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDMIN, dst, src0, src1, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDEXPDIF(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDEXPDIF, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDEXPDIF(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp,
                                      WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TROWEXPANDEXPDIF, dst, src0, src1, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TRSQRT(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TRSQRT, dst, src);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, typename... WaitEvents,
          typename = std::void_t<typename TileDataTmp::DType>>
PTO_INST RecordEvent TRSQRT(TileDataDst &dst, TileDataSrc &src, TileDataTmp &tmp, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TRSQRT, dst, src, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TSQRT(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSQRT, dst, src);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TEXP(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TEXP, dst, src);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TNOT(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TNOT, dst, src);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TRELU(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TRELU, dst, src);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataOffset, typename... WaitEvents>
PTO_INST RecordEvent TGATHERB(TileDataDst &dst, TileDataSrc &src, TileDataOffset &offset, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TGATHERB, dst, src, offset);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TADDS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar,
                           WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TADDS, dst, src0, scalar);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TAXPY(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar,
                           WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TAXPY, dst, src0, scalar);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TSUBS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar,
                           WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSUBS, dst, src0, scalar);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TDIVS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar,
                           WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TDIVS, dst, src0, scalar);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TMULS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar,
                           WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMULS, dst, src0, scalar);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TDIVS(TileDataDst &dst, typename TileDataDst::DType scalar, TileDataSrc &src0,
                           WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TDIVS, dst, scalar, src0);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TFMODS(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar,
                            WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TFMODS, dst, src, scalar);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TREMS(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar,
                           WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TREMS, dst, src, scalar);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TMAXS(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar,
                           WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TMAXS, dst, src, scalar);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TANDS(TileDataDst &dst, TileDataSrc &src, typename TileDataDst::DType scalar,
                           WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TANDS, dst, src, scalar);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TORS(TileDataDst &dst, TileDataSrc &src, typename TileDataDst::DType scalar,
                          WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TORS, dst, src, scalar);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TSHLS(TileDataDst &dst, TileDataSrc &src, typename TileDataDst::DType scalar,
                           WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSHLS, dst, src, scalar);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TSHRS(TileDataDst &dst, TileDataSrc &src, typename TileDataDst::DType scalar,
                           WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSHRS, dst, src, scalar);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TXORS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar, TileDataTmp &tmp,
                           WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TXORS, dst, src0, scalar, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TLRELU(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar,
                            WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TLRELU, dst, src, scalar);
    return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TADDSC(TileData &dst, TileData &src0, typename TileData::DType scalar, TileData &src1,
                            WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TADDSC, dst, src0, scalar, src1);
    return {};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TSUBSC(TileData &dst, TileData &src0, typename TileData::DType scalar, TileData &src1,
                            WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSUBSC, dst, src0, scalar, src1);
    return {};
}

template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TCOLMIN(TileDataOut &dst, TileDataIn &src, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLMIN, dst, src);
    return {};
}

template <typename TileDataD, typename TileDataS, typename TileDataI, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(TileDataD &dst, TileDataS &src, TileDataI &indexes, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TSCATTER, dst, src, indexes);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPAND(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLEXPAND, dst, src);
    return {};
}

template <typename TileDst, typename GlobalData, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MGATHER(TileDst &dst, GlobalData &src, TileInd &indexes, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(MGATHER, dst, src, indexes);
    return {};
}

#ifdef PTO_NPU_ARCH_A5
template <GatherOOB Mode, typename TileDst, typename GlobalData, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MGATHER(TileDst &dst, GlobalData &src, TileInd &indexes, WaitEvents &... events)
{
    TSYNC(events...);
    MGATHER_IMPL<Mode>(dst, src, indexes);
    return {};
}
#endif

template <typename GlobalData, typename TileSrc, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MSCATTER(GlobalData &dst, TileSrc &src, TileInd &indexes, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(MSCATTER, dst, src, indexes);
    return {};
}

#ifdef PTO_NPU_ARCH_A5
template <ScatterAtomicOp Atomic, typename GlobalData, typename TileSrc, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MSCATTER(GlobalData &dst, TileSrc &src, TileInd &indexes, WaitEvents &... events)
{
    TSYNC(events...);
    MSCATTER_IMPL<Atomic>(dst, src, indexes);
    return {};
}

template <ScatterAtomicOp Atomic, ScatterOOB Mode, typename GlobalData, typename TileSrc, typename TileInd,
          typename... WaitEvents>
PTO_INST RecordEvent MSCATTER(GlobalData &dst, TileSrc &src, TileInd &indexes, WaitEvents &... events)
{
    TSYNC(events...);
    MSCATTER_IMPL<Atomic, Mode>(dst, src, indexes);
    return {};
}
#endif

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TNEG(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TNEG, dst, src);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDDIV(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLEXPANDDIV, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDMUL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLEXPANDMUL, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDADD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLEXPANDADD, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDMAX(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLEXPANDMAX, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDMIN(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLEXPANDMIN, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDSUB(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLEXPANDSUB, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDEXPDIF(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TCOLEXPANDEXPDIF, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataPara, typename... WaitEvents>
PTO_INST RecordEvent TDEQUANT(TileDataDst &dst, TileDataSrc &src, TileDataPara &scale, TileDataPara &offset,
                              WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TDEQUANT, dst, src, scale, offset);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TREM(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TREM, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TFMOD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TFMOD, dst, src0, src1);
    return {};
}

template <typename Pipe, typename TileProd, TileSplitAxis Split, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe &pipe, TileProd &tile, WaitEvents &... events)
{
    TSYNC(events...);
    TPUSH_IMPL<Pipe, TileProd, Split>(pipe, tile);
    return {};
}

template <typename TileData, typename Pipe, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(TileData &tile, Pipe &pipe, WaitEvents &... events)
{
    TSYNC(events...);
    TPUSH_IMPL<TileData, Pipe>(tile, pipe);
    return {};
}

template <typename Pipe, typename TileCons, TileSplitAxis Split, typename... WaitEvents>
PTO_INST RecordEvent TPOP(Pipe &pipe, TileCons &tile, WaitEvents &... events)
{
    TSYNC(events...);
    TPOP_IMPL<Pipe, TileCons, Split>(pipe, tile);
    return {};
}

template <typename TileData, typename Pipe, typename... WaitEvents>
PTO_INST RecordEvent TPOP(TileData &tile, Pipe &pipe, WaitEvents &... events)
{
    TSYNC(events...);
    TPOP_IMPL<TileData, Pipe>(tile, pipe);
    return {};
}

template <typename Pipe, TileSplitAxis Split, typename... WaitEvents>
PTO_INST RecordEvent TFREE(Pipe &pipe, WaitEvents &... events)
{
    TSYNC(events...);
    TFREE_IMPL<Pipe, Split>(pipe);
    return {};
}

template <typename Pipe, typename... WaitEvents>
PTO_INST RecordEvent TFREE(Pipe &pipe, WaitEvents &... events)
{
    TSYNC(events...);
    TFREE_IMPL<Pipe>(pipe);
    return {};
}

#ifdef PTO_NPU_ARCH_A5
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TPACK(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events)
{
    TSYNC(events...);
    MAP_INSTR_IMPL(TPACK, dst, src);
    return {};
}

template <bool MSBorLSB = true, typename TileDataDst, typename TileDataSrc, typename TileDataIdx,
          typename... WaitEvents>
PTO_INST RecordEvent THISTOGRAM(TileDataDst &dst, TileDataSrc &src, TileDataIdx &idx, WaitEvents &... events)
{
    TSYNC(events...);
    THISTOGRAM_IMPL<MSBorLSB>(dst, src, idx);
    return {};
}

template <auto quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataExp, typename TileDataMax,
          typename... WaitEvents>
PTO_INST RecordEvent TQUANT(TileDataOut &dst, TileDataSrc &src, TileDataExp *exp, TileDataMax *max,
                            TileDataSrc *scaling, WaitEvents &... events)
{
    TSYNC(events...);
    TQUANT_IMPL<quant_type, TileDataOut, TileDataSrc, TileDataExp, TileDataMax>(dst, src, exp, max, scaling);
    return {};
}

template <auto quant_type, auto store_mode, typename TileDataOut, typename TileDataSrc, typename TileDataExp,
          typename TileDataMax, typename TileDataIdx, typename... WaitEvents>
PTO_INST RecordEvent TQUANT(TileDataOut &dst, TileDataSrc &src, TileDataExp *exp, TileDataMax *max,
                            TileDataSrc *scaling, TileDataExp *exp_zz, TileDataIdx *vgather_idx, WaitEvents &... events)
{
    TSYNC(events...);
    TQUANT_IMPL<quant_type, store_mode, TileDataOut, TileDataSrc, TileDataExp, TileDataMax, TileDataIdx>(
        dst, src, exp, max, scaling, exp_zz, vgather_idx);
    return {};
}
#endif

template <auto quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataPara, typename... WaitEvents>
PTO_INST RecordEvent TQUANT(TileDataOut &dst, TileDataSrc &src, TileDataPara &scale, TileDataPara *offset = nullptr,
                            WaitEvents &... events)
{
    TSYNC(events...);
    TQUANT_IMPL<quant_type, TileDataOut, TileDataSrc, TileDataPara>(dst, src, scale, offset);
    return {};
}
template <typename TileDataDst, typename TileDataSrc>
__tf__ PTO_INTERNAL OP_NAME(TGET_SCALE_ADDR)
    OP_TYPE(element_wise) void TGetScaleAddr(typename TileDataDst::TileDType __out__ dst,
                                             typename TileDataSrc::TileDType __in__ src)
{
    return;
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TGET_SCALE_ADDR(TileDataDst &dst, TileDataSrc &src)
{
    TGetScaleAddr<TileDataDst, TileDataSrc>(dst.data(), src.data());
}

} // namespace pto
#endif
