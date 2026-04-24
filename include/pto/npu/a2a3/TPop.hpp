/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPOP_HPP
#define TPOP_HPP

#include <pto/common/fifo.hpp>
#include <pto/npu/a2a3/TStore.hpp>
#include <pto/npu/a2a3/TPush.hpp>

namespace pto {
/**
 * TPOP: Pop Tile from FIFO
 * * Flow:
 * 1. [Wait]    Wait for data ready (Cross-Core)
 * 2. [Load]    Load data from GM
 * 3. [Free]    Release GM space (Cross-Core)
 */
template <typename Pipe, typename TileCons, TileSplitAxis Split>
PTO_INTERNAL void TPOP_IMPL(Pipe &pipe, TileCons &tile)
{
    // 1. Cross-Core: Wait for Data
    bool isWait = pipe.cons.getWaitStatus();
    if (isWait) {
        pipe.cons.wait();
    }

    // 2. Address Calculation & Load
    pipe.cons.template pop<TileCons, Split>(pipe.fifo, tile);
    pipe.cons.tileIndex++;

    // 3. Cross-Core: Free Space
    bool isFree = pipe.cons.getFreeStatus();
    if (isFree) {
        pipe.cons.free();
    }
}

template <typename Pipe, typename TileCons, TileSplitAxis Split>
PTO_INTERNAL void TPOP_IMPL(Pipe &pipe, TileCons &tile, PipeChunk chunk, int entryOffset)
{
    const bool doWait = (chunk == PipeChunk::Single || chunk == PipeChunk::First);
    const bool doFree = (chunk == PipeChunk::Single || chunk == PipeChunk::Last);
    const bool advanceSlot = (chunk == PipeChunk::Single || chunk == PipeChunk::Last);

    if (doWait) {
        pipe.cons.wait();
    }

    auto consumer = pipe.cons;
    consumer.entryOffset = entryOffset;
    consumer.template pop<TileCons, Split>(pipe.fifo, tile);

    if (advanceSlot) {
        pipe.cons.tileIndex++;
    }

    if (doFree) {
        consumer.free();
    }
}

template <typename Pipe, TileSplitAxis Split>
PTO_INTERNAL void TFREE_IMPL(Pipe &pipe)
{
    return;
}

//--------------------------------------------
template <typename TileData, typename Pipe>
PTO_INTERNAL void TPOP_IMPL(TileData &tile, Pipe &pipe)
{
    // 1. Cross-Core: Wait for Data
    bool isWait = pipe.cons.getWaitStatus();
    if (isWait) {
        pipe.cons.wait();
    }

    // 2. Address Calculation & Load
    pipe.cons.pop(pipe.fifo, tile);
    pipe.cons.tile_id++;

    // 3. Cross-Core: Free Space
    bool isFree = pipe.cons.getFreeStatus();
    if (isFree) {
        pipe.cons.free();
    }
}

template <typename Pipe>
PTO_INTERNAL void TFREE_IMPL(Pipe &pipe)
{
    return;
}

} // namespace pto

#endif
