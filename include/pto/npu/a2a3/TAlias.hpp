/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

/**
 * This file defines some dummy TAlias functions needed by auto mode.
 * These APIs only serve as a short-term, temporary hack to support aliasing
 * in auto mode. They should be removed once aliasing is properly supported in
 * auto mode.
 */

#ifdef __PTO_AUTO__
#ifndef TALIAS_A2A3_HPP
#define TALIAS_A2A3_HPP
#include <pto/common/type.hpp>

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void checkAlias()
{
    using namespace pto;
    static_assert(is_tile_data_v<TileDataSrc>, "input must be a Tile instance.");
    static_assert(is_tile_data_v<TileDataDst>, "output must be a Tile instance.");

    using DType = typename TileDataSrc::DType;
    using NewElement = typename TileDataDst::DType;

    constexpr auto Loc = TileDataSrc::Loc;
    constexpr auto NewLoc = TileDataDst::Loc;

    constexpr int Numel = TileDataSrc::Numel;
    constexpr int NewNumel = TileDataDst::Numel;

    constexpr auto SFractal = TileDataSrc::SFractal;
    constexpr auto NewSFractal = TileDataDst::SFractal;

    // 1. TileType must match
    static_assert(Loc == NewLoc, "TRESHAPE: Source and target TileType must be identical.");

    // 2. Byte size must match
    static_assert(sizeof(DType) * Numel == sizeof(NewElement) * NewNumel, "TRESHAPE: Total byte size must match.");

    // 3. reshape between non-boxed and boxed tile is not allowed.
    static_assert((SFractal == SLayout::NoneBox && NewSFractal == SLayout::NoneBox) ||
                      (SFractal != SLayout::NoneBox && NewSFractal != SLayout::NoneBox),
                  "TRESHAPE: Cannot reshape between boxed and non-boxed layouts.");
}

template <typename TileDataDst, typename TileDataSrc>
__tf__ PTO_INTERNAL void TAlias(typename TileDataDst::TileDType __out__ original,
                                typename TileDataSrc::TileDType __in__ alias)
{
    return;
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TALIAS_IMPL(TileDataDst &original, TileDataSrc &alias)
{
    checkAlias<TileDataDst, TileDataSrc>();
    TAlias<TileDataDst, TileDataSrc>(original.data(), alias.data());
}

#endif // TALIAS_A2A3_HPP
#endif // __PTO_AUTO__
