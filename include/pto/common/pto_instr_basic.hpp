/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_INSTR_BASIC_HPP
#define PTO_INSTR_BASIC_HPP
// Shared public wrappers. Included inside the active PTO target namespace.
// The backend supplies MAP_INSTR_IMPL and detail::PtoWaitEvents.

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TADD(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TADD, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TABS(TileDataDst& dst, TileDataSrc& src, WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TABS, dst, src);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TAND(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TAND, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TOR(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TOR, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TSUB(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TSUB, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TMUL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TMUL, dst, src0, src1);
    return {};
}

template <
    auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1,
    typename... WaitEvents>
PTO_INST RecordEvent TDIV(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TDIV, PTO_TEMPLATE_ARGS(PrecisionType), dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TMAX(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TMAX, dst, src0, src1);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TMIN(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TMIN, dst, src0, src1);
    return {};
}

template <
    auto PrecisionType = SqrtAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TSQRT(TileDataDst& dst, TileDataSrc& src, WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TSQRT, PTO_TEMPLATE_ARGS(PrecisionType), dst, src);
    return {};
}

template <
    auto PrecisionType = ExpAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TEXP(TileDataDst& dst, TileDataSrc& src, WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TEXP, PTO_TEMPLATE_ARGS(PrecisionType), dst, src);
    return {};
}

template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TMATMUL, cMatrix, aMatrix, bMatrix);
    return {};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent
TCMP(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, CmpMode cmpMode, WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TCMP, dst, src0, src1, cmpMode);
    return {};
}

template <typename TileData, typename MaskTile, typename TmpTile, typename... WaitEvents>
PTO_INST RecordEvent
TSEL(TileData& dst, MaskTile& selMask, TileData& src0, TileData& src1, TmpTile& tmp, WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TSEL, dst, selMask, src0, src1, tmp);
    return {};
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TTRANS(TileDataDst& dst, TileDataSrc& src, TileDataTmp& tmp, WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TTRANS, dst, src, tmp);
    return {};
}

template <typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TLOAD(TileData& dst, GlobalData& src, WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
#if PTO_FORWARD_L2HINT_TO_IMPL
    MAP_INSTR_IMPL_T(TLOAD, PTO_TEMPLATE_ARGS(TLoadL2Hint::NormalFirstVictim), dst, src);
#else
    // Kirin IMPLs keep pre-L2-hint signatures.
    MAP_INSTR_IMPL(TLOAD, dst, src);
#endif
    return {};
}

#endif
