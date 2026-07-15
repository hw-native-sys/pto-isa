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
#include <type_traits>

#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

#include "kernel_operator.h"

template <typename Element, int TileElems = 1024>
using PtoVecTile = pto::Tile<pto::TileType::Vec, Element, 1, TileElems, pto::BLayout::RowMajor, -1, -1>;

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
AICORE inline void PtoAddUb(
    uint64_t dstUbOffsetBytes, uint64_t src0UbOffsetBytes, uint64_t src1UbOffsetBytes, uint32_t elemNum)
{
    using Tile = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        Tile dstTile(1, cur);
        Tile src0Tile(1, cur);
        Tile src1Tile(1, cur);
        PtoAssignUbTile<Tile, Element>(dstTile, dstUbOffsetBytes, offset);
        PtoAssignUbTile<Tile, Element>(src0Tile, src0UbOffsetBytes, offset);
        PtoAssignUbTile<Tile, Element>(src1Tile, src1UbOffsetBytes, offset);
        pto::TADD(dstTile, src0Tile, src1Tile);
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
AICORE inline void CastDynamicQuantInputToFp32(uint64_t fp32Ub, uint64_t rawInputUb, uint32_t elemNum)
{
    if constexpr (!std::is_same_v<Element, float>) {
        using DstTile = PtoVecTile<float, TileElems>;
        using SrcTile = PtoVecTile<Element, TileElems>;
        for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
            const uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
            DstTile dstTile(1, cur);
            SrcTile srcTile(1, cur);
            pto::TASSIGN(dstTile, fp32Ub + static_cast<uint64_t>(offset) * sizeof(float));
            pto::TASSIGN(srcTile, rawInputUb + static_cast<uint64_t>(offset) * sizeof(Element));
            pto::TCVT(dstTile, srcTile, pto::RoundMode::CAST_NONE);
        }
    }
}

AICORE inline void BuildDynamicQuantAbs(uint64_t absUb, uint64_t inputUb, uint32_t elemNum)
{
    using Tile = PtoVecTile<float>;
    for (uint32_t offset = 0; offset < elemNum; offset += 1024) {
        const uint32_t cur = (elemNum - offset > 1024) ? 1024 : (elemNum - offset);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        Tile dstTile(1, cur);
        Tile srcTile(1, cur);
        pto::TASSIGN(dstTile, absUb + elemOffset);
        pto::TASSIGN(srcTile, inputUb + elemOffset);
        pto::TABS(dstTile, srcTile);
    }
}

AICORE inline void ReduceDynamicQuantAbsMax(uint64_t maxUb, uint64_t absUb, uint32_t elemNum)
{
    using Tile = PtoVecTile<float>;
    using RowMaxTile = pto::Tile<pto::TileType::Vec, float, 8, 1, pto::BLayout::ColMajor, -1, 1>;
    using ScalarTile = pto::Tile<pto::TileType::Vec, float, 1, 8, pto::BLayout::RowMajor, -1, -1>;
    bool firstChunk = true;
    for (uint32_t offset = 0; offset < elemNum; offset += 1024) {
        const uint32_t cur = (elemNum - offset > 1024) ? 1024 : (elemNum - offset);
        Tile srcTile(1, cur);
        Tile tmpTile(1, cur);
        RowMaxTile rowMaxTile(1);
        pto::TASSIGN(srcTile, absUb + static_cast<uint64_t>(offset) * sizeof(float));
        pto::TASSIGN(tmpTile, absUb + static_cast<uint64_t>(offset) * sizeof(float));
        pto::TASSIGN(rowMaxTile, firstChunk ? maxUb : absUb);
        pto::TROWMAX(rowMaxTile, srcTile, tmpTile);
        pto::TSYNC<pto::Op::TROWMAX>();
        if (!firstChunk) {
            ScalarTile accTile(1, 1);
            ScalarTile newTile(1, 1);
            ScalarTile dstTile(1, 1);
            pto::TASSIGN(accTile, maxUb);
            pto::TASSIGN(newTile, absUb);
            pto::TASSIGN(dstTile, maxUb);
            pto::TMAX(dstTile, accTile, newTile);
            pto::TSYNC<pto::Op::TMAX>();
        }
        firstChunk = false;
    }
}

AICORE inline void DivideDynamicQuantInputByScale(uint64_t dstUb, uint64_t inputUb, uint64_t scaleUb, uint32_t elemNum)
{
    using Tile = PtoVecTile<float>;
    for (uint32_t offset = 0; offset < elemNum; offset += 1024) {
        const uint32_t cur = (elemNum - offset > 1024) ? 1024 : (elemNum - offset);
        const uint64_t elemOffset = static_cast<uint64_t>(offset) * sizeof(float);
        Tile dstTile(1, cur);
        Tile numeratorTile(1, cur);
        Tile denominatorTile(1, cur);
        pto::TASSIGN(dstTile, dstUb + elemOffset);
        pto::TASSIGN(numeratorTile, inputUb + elemOffset);
        pto::TASSIGN(denominatorTile, scaleUb + elemOffset);
        pto::TDIV(dstTile, numeratorTile, denominatorTile);
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
