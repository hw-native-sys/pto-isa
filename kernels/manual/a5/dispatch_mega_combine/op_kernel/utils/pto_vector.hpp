/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_PTO_VECTOR_HPP
#define DISPATCH_MEGA_COMBINE_PTO_VECTOR_HPP

#include <cstdint>

#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

#include "kernel_operator.h"

template <typename Element, int TileElems = 1024>
using PtoVecTile = pto::Tile<pto::TileType::Vec, Element, 1, TileElems, pto::BLayout::RowMajor, -1, -1>;

__tf__ AICORE void PtoGatherMaskCountInt32(
    uint64_t srcUbOffset, uint64_t maskUbOffset, uint64_t dstUbOffset, uint64_t countUbOffset, uint32_t validRouteCount)
{
    constexpr uint32_t kRoutesPerRepeat = CCE_VL / sizeof(int32_t);
    __ubuf__ uint8_t* ubBase = reinterpret_cast<__ubuf__ uint8_t*>(0);
    __ubuf__ int32_t* srcPtr = reinterpret_cast<__ubuf__ int32_t*>(ubBase + srcUbOffset);
    __ubuf__ uint32_t* maskPtr = reinterpret_cast<__ubuf__ uint32_t*>(ubBase + maskUbOffset);
    __ubuf__ int32_t* dstPtr = reinterpret_cast<__ubuf__ int32_t*>(ubBase + dstUbOffset);
    __ubuf__ uint32_t* countPtr = reinterpret_cast<__ubuf__ uint32_t*>(ubBase + countUbOffset);
    const uint16_t repeatTimes = static_cast<uint16_t>((validRouteCount + kRoutesPerRepeat - 1U) / kRoutesPerRepeat);

    __VEC_SCOPE__
    {
        vector_s32 routeIds;
        vector_s32 selectedRouteIds;
        vector_u32 packedMask;
        vector_bool routeMask;
        vector_bool validMask;
        vector_bool executeMask;
        vector_align dstAlign;
        vector_align maskAlign;
        uint32_t remainingRoutes = validRouteCount;

        sprclr(SPR_AR);
        for (uint16_t repeat = 0U; repeat < repeatTimes; ++repeat) {
            vlds(routeIds, srcPtr, static_cast<uint32_t>(repeat) * kRoutesPerRepeat, NORM);
            __ubuf__ uint32_t* currentMask = maskPtr + static_cast<uint32_t>(repeat) * 2U;
            vldas(maskAlign, currentMask);
            vldus(packedMask, maskAlign, currentMask);
            movvp(routeMask, packedMask, 0);
            validMask = plt_b32(remainingRoutes, POST_UPDATE);
            pand(executeMask, routeMask, validMask, validMask);
            vsqz(selectedRouteIds, routeIds, executeMask, MODE_STORED);
            vstur(dstAlign, (vector_u32)selectedRouteIds, reinterpret_cast<__ubuf__ uint32_t*>(dstPtr), POST_UPDATE);
        }
        vstar(dstAlign, reinterpret_cast<__ubuf__ uint32_t*>(dstPtr));
        sprsts(SPR_AR, countPtr, 0);
    }
}

template <typename Element, int TileElems>
constexpr int kPtoVectorBlockRows = (static_cast<uint64_t>(TileElems) * sizeof(Element) <= 32U * 1024U) ? 2 : 1;

template <typename Element, int TileElems = 1024>
using PtoVecBlockTile = pto::Tile<
    pto::TileType::Vec, Element, kPtoVectorBlockRows<Element, TileElems>, TileElems, pto::BLayout::RowMajor, -1, -1>;

template <typename Element>
using PtoVectorShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;

template <typename Element>
using PtoVectorStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;

template <typename Element>
using PtoVectorGlobal = pto::GlobalTensor<Element, PtoVectorShape<Element>, PtoVectorStride<Element>, pto::Layout::ND>;

template <typename Element>
using PtoVectorBlockShape = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;

template <typename Element>
using PtoVectorBlockStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;

template <typename Element>
using PtoVectorBlockGlobal =
    pto::GlobalTensor<Element, PtoVectorBlockShape<Element>, PtoVectorBlockStride<Element>, pto::Layout::ND>;

template <typename Element>
AICORE inline uint64_t PtoElemOffsetBytes(uint32_t offset)
{
    return static_cast<uint64_t>(offset) * sizeof(Element);
}

template <typename Tile, typename Element>
AICORE inline void PtoAssignUbTile(Tile& tile, uint64_t baseOffsetBytes, uint32_t elemOffset)
{
    pto::TASSIGN(tile, baseOffsetBytes + PtoElemOffsetBytes<Element>(elemOffset));
}

template <typename Element, int TileElems = 1024>
AICORE inline void PtoLoadVector(uint64_t dstUbOffsetBytes, __gm__ Element* src, uint32_t elemNum)
{
    using Tile = PtoVecTile<Element, TileElems>;
    using BlockTile = PtoVecBlockTile<Element, TileElems>;
    constexpr uint32_t blockRows = static_cast<uint32_t>(kPtoVectorBlockRows<Element, TileElems>);

    uint32_t offset = 0;
    while ((elemNum - offset) / TileElems >= 2U) {
        const uint32_t fullChunks = (elemNum - offset) / TileElems;
        const uint32_t rowNum = fullChunks > blockRows ? blockRows : fullChunks;
        BlockTile tile(rowNum, TileElems);
        PtoAssignUbTile<BlockTile, Element>(tile, dstUbOffsetBytes, offset);
        PtoVectorBlockShape<Element> srcShape(rowNum, TileElems);
        PtoVectorBlockStride<Element> srcStride(
            static_cast<int64_t>(rowNum) * TileElems, static_cast<int64_t>(rowNum) * TileElems,
            static_cast<int64_t>(rowNum) * TileElems, TileElems);
        PtoVectorBlockGlobal<Element> srcGlobal(src + offset, srcShape, srcStride);
        pto::TLOAD(tile, srcGlobal);
        offset += rowNum * TileElems;
    }
    while (offset < elemNum) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        PtoVectorShape<Element> srcShape(cur);
        PtoVectorStride<Element> srcStride(cur, cur, cur, cur);
        PtoVectorGlobal<Element> srcGlobal(src + offset, srcShape, srcStride);
        Tile tile(1, cur);
        PtoAssignUbTile<Tile, Element>(tile, dstUbOffsetBytes, offset);
        pto::TLOAD(tile, srcGlobal);
        offset += cur;
    }
}

template <bool AtomicAdd, typename Element, int TileElems = 1024>
AICORE inline void PtoStoreVectorImpl(__gm__ Element* dst, uint64_t srcUbOffsetBytes, uint32_t elemNum)
{
    using Tile = PtoVecTile<Element, TileElems>;
    using BlockTile = PtoVecBlockTile<Element, TileElems>;
    constexpr uint32_t blockRows = static_cast<uint32_t>(kPtoVectorBlockRows<Element, TileElems>);

    uint32_t offset = 0;
    while ((elemNum - offset) / TileElems >= 2U) {
        const uint32_t fullChunks = (elemNum - offset) / TileElems;
        const uint32_t rowNum = fullChunks > blockRows ? blockRows : fullChunks;
        PtoVectorBlockShape<Element> dstShape(rowNum, TileElems);
        PtoVectorBlockStride<Element> dstStride(
            static_cast<int64_t>(rowNum) * TileElems, static_cast<int64_t>(rowNum) * TileElems,
            static_cast<int64_t>(rowNum) * TileElems, TileElems);
        PtoVectorBlockGlobal<Element> dstGlobal(dst + offset, dstShape, dstStride);
        BlockTile tile(rowNum, TileElems);
        PtoAssignUbTile<BlockTile, Element>(tile, srcUbOffsetBytes, offset);
        if constexpr (AtomicAdd) {
            pto::TSTORE<BlockTile, decltype(dstGlobal), pto::AtomicType::AtomicAdd>(dstGlobal, tile);
        } else {
            pto::TSTORE(dstGlobal, tile);
        }
        offset += rowNum * TileElems;
    }
    while (offset < elemNum) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        PtoVectorShape<Element> dstShape(cur);
        PtoVectorStride<Element> dstStride(cur, cur, cur, cur);
        PtoVectorGlobal<Element> dstGlobal(dst + offset, dstShape, dstStride);
        Tile tile(1, cur);
        PtoAssignUbTile<Tile, Element>(tile, srcUbOffsetBytes, offset);
        if constexpr (AtomicAdd) {
            pto::TSTORE<Tile, decltype(dstGlobal), pto::AtomicType::AtomicAdd>(dstGlobal, tile);
        } else {
            pto::TSTORE(dstGlobal, tile);
        }
        offset += cur;
    }
}

template <typename Element, int TileElems = 1024>
AICORE inline void PtoStoreVector(__gm__ Element* dst, uint64_t srcUbOffsetBytes, uint32_t elemNum)
{
    PtoStoreVectorImpl<false, Element, TileElems>(dst, srcUbOffsetBytes, elemNum);
}

template <typename Element, int TileElems = 1024>
AICORE inline void PtoStoreAtomicAddVector(__gm__ Element* dst, uint64_t srcUbOffsetBytes, uint32_t elemNum)
{
    PtoStoreVectorImpl<true, Element, TileElems>(dst, srcUbOffsetBytes, elemNum);
}

template <typename DstElement, typename SrcElement, int TileElems = 1024>
AICORE inline void PtoCastUb(
    uint64_t dstUbOffsetBytes, uint64_t srcUbOffsetBytes, uint32_t elemNum, pto::RoundMode mode)
{
    using DstTile = PtoVecTile<DstElement, TileElems>;
    using SrcTile = PtoVecTile<SrcElement, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        DstTile dstTile(1, cur);
        SrcTile srcTile(1, cur);
        PtoAssignUbTile<DstTile, DstElement>(dstTile, dstUbOffsetBytes, offset);
        PtoAssignUbTile<SrcTile, SrcElement>(srcTile, srcUbOffsetBytes, offset);
        pto::TCVT(dstTile, srcTile, mode);
    }
}

template <typename Element, int TileElems = 1024>
AICORE inline void PtoFillUb(uint64_t dstUbOffsetBytes, Element scalar, uint32_t elemNum)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        Tile tile(1, cur);
        PtoAssignUbTile<Tile, Element>(tile, dstUbOffsetBytes, offset);
        pto::TEXPANDS(tile, scalar);
    }
}

template <typename Element, int TileElems = 1024>
AICORE inline void PtoMoveUb(uint64_t dstUbOffsetBytes, uint64_t srcUbOffsetBytes, uint32_t elemNum)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        Tile dstTile(1, cur);
        Tile srcTile(1, cur);
        PtoAssignUbTile<Tile, Element>(dstTile, dstUbOffsetBytes, offset);
        PtoAssignUbTile<Tile, Element>(srcTile, srcUbOffsetBytes, offset);
        pto::TMOV(dstTile, srcTile);
    }
}

template <typename Element, int TileElems = 1024>
AICORE inline void PtoAddScalarUb(
    uint64_t dstUbOffsetBytes, uint64_t srcUbOffsetBytes, uint32_t elemNum, Element scalar)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        Tile dstTile(1, cur);
        Tile srcTile(1, cur);
        PtoAssignUbTile<Tile, Element>(dstTile, dstUbOffsetBytes, offset);
        PtoAssignUbTile<Tile, Element>(srcTile, srcUbOffsetBytes, offset);
        pto::TADDS(dstTile, srcTile, scalar);
    }
}

template <typename Element, int TileElems = 1024>
AICORE inline void PtoMulScalarUb(
    uint64_t dstUbOffsetBytes, uint64_t srcUbOffsetBytes, uint32_t elemNum, Element scalar)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        Tile dstTile(1, cur);
        Tile srcTile(1, cur);
        PtoAssignUbTile<Tile, Element>(dstTile, dstUbOffsetBytes, offset);
        PtoAssignUbTile<Tile, Element>(srcTile, srcUbOffsetBytes, offset);
        pto::TMULS(dstTile, srcTile, scalar);
    }
}

template <typename Element, int TileElems = 1024>
AICORE inline Element PtoGetValue(uint64_t ubOffsetBytes, uint32_t elemOffset)
{
    using Tile = PtoVecTile<Element, TileElems>;
    uint64_t tileOffsetBytes =
        ubOffsetBytes + static_cast<uint64_t>(elemOffset / TileElems) * TileElems * sizeof(Element);
    Tile tile(1, TileElems);
    pto::TASSIGN(tile, tileOffsetBytes);
    return tile.GetValue(elemOffset % TileElems);
}

template <typename Element, int TileElems = 1024>
AICORE inline void PtoSetValue(uint64_t ubOffsetBytes, uint32_t elemOffset, Element value)
{
    using Tile = PtoVecTile<Element, TileElems>;
    uint64_t tileOffsetBytes =
        ubOffsetBytes + static_cast<uint64_t>(elemOffset / TileElems) * TileElems * sizeof(Element);
    Tile tile(1, TileElems);
    pto::TASSIGN(tile, tileOffsetBytes);
    tile.SetValue(elemOffset % TileElems, value);
}

// Use register intrinsics inside the SIMD function; RegTensor is not available here.
__simd_callee__ inline void PtoInclusivePrefixSumInt32(__ubuf__ int32_t* values, uint32_t elementCount)
{
    constexpr uint32_t kElementsPerVector = CCE_VL / sizeof(int32_t);
    vector_s32 prefixReg;
    vector_s32 shiftedReg;
    vector_s32 carryReg;
    vector_s32 laneIndexReg;
    vector_s32 shiftedIndexReg;
    vector_bool fullMask = pset_b32(PAT_ALL);

    vci(laneIndexReg, 0, INC_ORDER);
    vbr(carryReg, 0);
    for (uint32_t vectorOffset = 0U; vectorOffset < elementCount; vectorOffset += kElementsPerVector) {
        uint32_t validCount = elementCount - vectorOffset;
        validCount = validCount < kElementsPerVector ? validCount : kElementsPerVector;
        uint32_t maskCount = validCount;
        vector_bool activeMask = plt_b32(maskCount, POST_UPDATE);
        vlds(prefixReg, values, vectorOffset, NORM);

        for (uint32_t shift = 1U; shift < validCount; shift <<= 1U) {
            vadds(shiftedIndexReg, laneIndexReg, -static_cast<int32_t>(shift), fullMask, MODE_ZEROING);
            vmaxs(shiftedIndexReg, shiftedIndexReg, 0, fullMask, MODE_ZEROING);
            vselr((vector_u32&)shiftedReg, (vector_u32&)prefixReg, (vector_u32&)shiftedIndexReg);
            vector_bool shiftedLaneMask;
            vcmps_ge(shiftedLaneMask, laneIndexReg, static_cast<int32_t>(shift), activeMask);
            vadd(prefixReg, prefixReg, shiftedReg, shiftedLaneMask, MODE_MERGING);
        }

        vadd(prefixReg, prefixReg, carryReg, activeMask, MODE_MERGING);
        vsts(prefixReg, values, vectorOffset, NORM_B32, activeMask);
        vbr(shiftedIndexReg, static_cast<int32_t>(validCount - 1U));
        vselr((vector_u32&)carryReg, (vector_u32&)prefixReg, (vector_u32&)shiftedIndexReg);
    }
}

__simd_vf__ inline void PtoComputeExpertCountTablesVF(
    __ubuf__ int32_t* count, __ubuf__ int32_t* expertCounts, uint32_t elementCount, uint32_t expertCount,
    uint32_t worldSize)
{
    PtoInclusivePrefixSumInt32(count, elementCount);
    mem_bar(VST_VLD);

    constexpr uint32_t kElementsPerVector = CCE_VL / sizeof(int32_t);
    vector_s32 expertEndPrefixReg;
    vector_s32 previousEndPrefixReg;
    vector_s32 expertCountReg;
    vector_s32 previousVectorEndReg;
    vector_s32 laneIndexReg;
    vector_s32 gatherIndexReg;
    vector_s32 previousLaneIndexReg;
    vector_bool fullMask = pset_b32(PAT_ALL);
    vector_bool firstLaneMask = pset_b32(PAT_VL1);

    vci(laneIndexReg, 0, INC_ORDER);
    vbr(previousVectorEndReg, 0);
    for (uint32_t expertOffset = 0U; expertOffset < expertCount; expertOffset += kElementsPerVector) {
        uint32_t validCount = expertCount - expertOffset;
        validCount = validCount < kElementsPerVector ? validCount : kElementsPerVector;
        uint32_t maskCount = validCount;
        vector_bool activeMask = plt_b32(maskCount, POST_UPDATE);

        vadds(gatherIndexReg, laneIndexReg, static_cast<int32_t>(expertOffset), fullMask, MODE_ZEROING);
        vmuls(gatherIndexReg, gatherIndexReg, static_cast<int32_t>(worldSize), fullMask, MODE_ZEROING);
        vadds(gatherIndexReg, gatherIndexReg, static_cast<int32_t>(worldSize - 1U), fullMask, MODE_ZEROING);
        vgather2(expertEndPrefixReg, count, (vector_u32&)gatherIndexReg, activeMask);

        vadds(previousLaneIndexReg, laneIndexReg, -1, fullMask, MODE_ZEROING);
        vmaxs(previousLaneIndexReg, previousLaneIndexReg, 0, fullMask, MODE_ZEROING);
        vselr((vector_u32&)previousEndPrefixReg, (vector_u32&)expertEndPrefixReg, (vector_u32&)previousLaneIndexReg);
        vsel(previousEndPrefixReg, previousVectorEndReg, previousEndPrefixReg, firstLaneMask);
        vsub(expertCountReg, expertEndPrefixReg, previousEndPrefixReg, activeMask, MODE_ZEROING);
        vsts(expertCountReg, expertCounts, expertOffset, NORM_B32, activeMask);

        vbr(previousLaneIndexReg, static_cast<int32_t>(validCount - 1U));
        vselr((vector_u32&)previousVectorEndReg, (vector_u32&)expertEndPrefixReg, (vector_u32&)previousLaneIndexReg);
    }
}

AICORE inline void PtoBuildExpertMajorCountTablesInt32(
    uint64_t cumsumUbOffset, uint64_t expertCountUbOffset, uint32_t expertCount, uint32_t worldSize)
{
    __ubuf__ uint8_t* ubBase = reinterpret_cast<__ubuf__ uint8_t*>(0);
    __ubuf__ int32_t* cumsum = reinterpret_cast<__ubuf__ int32_t*>(ubBase + cumsumUbOffset);
    __ubuf__ int32_t* expertCounts = reinterpret_cast<__ubuf__ int32_t*>(ubBase + expertCountUbOffset);
    asc_vf_call<PtoComputeExpertCountTablesVF>(cumsum, expertCounts, expertCount * worldSize, expertCount, worldSize);
    pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();
}

AICORE inline void PtoFillArithProgressionInt32(uint64_t dstUb, int32_t firstValue, int32_t diffValue, uint32_t count)
{
    using Tile = PtoVecTile<int32_t>;
    int32_t value = firstValue;
    for (uint32_t offset = 0; offset < count; offset += 1024) {
        const uint32_t cur = (count - offset > 1024) ? 1024 : (count - offset);
        Tile tile(1, cur);
        PtoAssignUbTile<Tile, int32_t>(tile, dstUb, offset);
        for (uint32_t idx = 0; idx < cur; ++idx) {
            tile.SetValue(idx, value);
            value += diffValue;
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
}

#endif // DISPATCH_MEGA_COMBINE_PTO_VECTOR_HPP
