/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <limits>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace {
constexpr uint32_t kHashMul1 = 0x7FEB352D;
constexpr uint32_t kHashMul2 = 0x846CA68B;
constexpr int32_t kEmptyKey = std::numeric_limits<int32_t>::min();
constexpr int32_t kNotFound = -1;
} // namespace

template <int kTileRows, int kTileCols, int kCap, int kMaxProbe>
AICORE void runHashFind(
    __gm__ int32_t __out__* out, __gm__ int32_t __in__* table_keys, __gm__ int32_t __in__* table_vals,
    __gm__ int32_t __in__* queries)
{
    static_assert((kCap & (kCap - 1)) == 0, "hashfind: capacity must be power-of-two");

    using TileI32 = Tile<TileType::Vec, int32_t, kTileRows, kTileCols, BLayout::RowMajor, -1, -1>;
    using TileU32 = Tile<TileType::Vec, uint32_t, kTileRows, kTileCols, BLayout::RowMajor, -1, -1>;

    using TableShape = Shape<1, 1, 1, 1, kCap>;
    using TableStride = Stride<1, 1, 1, kCap, 1>;
    using TableGT = GlobalTensor<int32_t, TableShape, TableStride>;

    using TileShape = Shape<1, 1, 1, kTileRows, kTileCols>;
    using TileStride = Stride<1, 1, 1, kTileCols, 1>;
    using TileGT = GlobalTensor<int32_t, TileShape, TileStride>;

    TableGT keysGlobal(table_keys);
    TableGT valsGlobal(table_vals);
    TileGT queryGlobal(queries);
    TileGT outGlobal(out);

    TileI32 qTile(kTileRows, kTileCols);
    TileU32 qU32Tile(kTileRows, kTileCols);
    TileI32 outTile(kTileRows, kTileCols);
    TileI32 doneTile(kTileRows, kTileCols);
    TileU32 hTile(kTileRows, kTileCols);
    TileU32 idxTile(kTileRows, kTileCols);
    TileI32 keyTile(kTileRows, kTileCols);
    TileI32 valTile(kTileRows, kTileCols);

    // Scratch tiles for vector-tile operations.
    TileU32 shift16(kTileRows, kTileCols);
    TileU32 shift15(kTileRows, kTileCols);
    TileU32 tmpU32(kTileRows, kTileCols);
    TileU32 xorTmp(kTileRows, kTileCols);
    TileI32 matchTile(kTileRows, kTileCols);
    TileI32 emptyTile(kTileRows, kTileCols);
    TileI32 pendingTile(kTileRows, kTileCols);
    TileI32 emptyKeyTile(kTileRows, kTileCols);
    TileI32 zeroTile(kTileRows, kTileCols);
    TileI32 shouldWrite(kTileRows, kTileCols);
    TileI32 delta(kTileRows, kTileCols);
    TileI32 update(kTileRows, kTileCols);

    // No direct Tile memory assignment is made (via TASSIGN)
    // So, __PTO_AUTO__ macro should be enabled in compiler definitions for auto memory assignment

    TLOAD(qTile, queryGlobal);
    TCVT(qU32Tile, qTile, RoundMode::CAST_NONE);

    // Initialize output and per-element done flags.
    TEXPANDS(outTile, kNotFound);
    TEXPANDS(doneTile, static_cast<int32_t>(0));
    TEXPANDS(emptyKeyTile, kEmptyKey);
    TEXPANDS(zeroTile, static_cast<int32_t>(0));

    constexpr uint32_t mask = static_cast<uint32_t>(kCap - 1);

    // Compute the base hash indices for all query elements using PTO vector-tile instructions.
    // h = hash_u32(q) & (cap - 1)
    TMOV(hTile, qU32Tile);
    TEXPANDS(shift16, static_cast<uint32_t>(16));
    TEXPANDS(shift15, static_cast<uint32_t>(15));

    TSHR(tmpU32, hTile, shift16);
    TXOR(hTile, hTile, tmpU32, xorTmp);
    TMULS(hTile, hTile, kHashMul1);

    TSHR(tmpU32, hTile, shift15);
    TXOR(hTile, hTile, tmpU32, xorTmp);
    TMULS(hTile, hTile, kHashMul2);

    TSHR(tmpU32, hTile, shift16);
    TXOR(hTile, hTile, tmpU32, xorTmp);
    TANDS(hTile, hTile, mask);

    for (int probe = 0; probe < kMaxProbe; ++probe) {
        // idx = (h + probe) & mask
        TMOV(idxTile, hTile);
        TADDS(idxTile, idxTile, static_cast<uint32_t>(probe));
        TANDS(idxTile, idxTile, mask);

        MGATHER<Coalesce::Elem>(keyTile, keysGlobal, idxTile);
        MGATHER<Coalesce::Elem>(valTile, valsGlobal, idxTile);

        // match = (key == query), empty = (key == empty_key)
        TCMP(matchTile, keyTile, qTile, CmpMode::EQ);
        TCMP(emptyTile, keyTile, emptyKeyTile, CmpMode::EQ);

        // pending = (done == 0)
        TCMP(pendingTile, doneTile, zeroTile, CmpMode::EQ);

        // shouldWrite = match & pending
        TAND(shouldWrite, matchTile, pendingTile);

        // out = out + shouldWrite * (val - out)
        TSUB(delta, valTile, outTile);
        TMUL(update, delta, shouldWrite);
        TADD(outTile, outTile, update);

        // done |= (match | empty)
        TOR(shouldWrite, matchTile, emptyTile);
        TOR(doneTile, doneTile, shouldWrite);

        // Early-exit: if all lanes are done, stop probing.
        bool any_pending = false;
        for (int lane = 0; lane < kTileRows * kTileCols; ++lane) {
            if (doneTile.data()[lane] == 0) {
                any_pending = true;
                break;
            }
        }
        if (!any_pending) {
            break;
        }
    }

    TSTORE(outGlobal, outTile);
    out = outGlobal.data();
}

template <int kTileRows, int kTileCols, int kCap, int kMaxProbe>
void LaunchHashFind(int32_t* out, int32_t* table_keys, int32_t* table_vals, int32_t* queries, void* stream)
{
    (void)stream;
    runHashFind<kTileRows, kTileCols, kCap, kMaxProbe>(out, table_keys, table_vals, queries);
}

template void LaunchHashFind<16, 16, 512, 64>(
    int32_t* out, int32_t* table_keys, int32_t* table_vals, int32_t* queries, void* stream);
