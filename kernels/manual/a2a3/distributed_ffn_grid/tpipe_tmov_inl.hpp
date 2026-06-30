/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Cube <-> Vector tile movement expressed as a single TMOV interface.
//
// The mixed Cube/Vec kernels exchange intermediate FFN tiles over the regular
// A2/A3 cluster-local TPipe FIFO (C2V / V2C).  Rather than spelling out the
// producer-side TPUSH and the consumer-side TPOP at every call site, the kernel
// body issues a directional TMOV and lets this header lower it to the correct
// cross-core primitive:
//
//   TMOV(pipe, tile)   producer side  -> TPUSH tile into the C2V/V2C FIFO
//   TMOV(tile, pipe)   consumer side  -> TPOP  the next slot into tile
//
// The push/pop split, the C2V vs V2C direction, and which physical core writes
// vs reads all stay encoded in the TPipe type and its __DAV_CUBE__/__DAV_VEC__
// guards, so the kernel no longer references TPUSH/TPOP for Cube<->Vector
// traffic.  This mirrors how a real WSE fabric move hides the producer/consumer
// handshake behind one tile-move op.
//
// Both overloads default to TILE_NO_SPLIT and forward to the existing public
// TPUSH/TPOP, reusing the established cross-core sync/record machinery verbatim.
// They take exactly two tile/pipe arguments (no trailing wait-event pack), which
// makes them strictly more specialized than the generic
// TMOV(dst, src, events...) overload: overload resolution therefore prefers
// these for any (TPipe, Tile) / (Tile, TPipe) pair and still falls back to the
// generic tile-to-tile TMOV for everything else.  The GridDirection-based
// GridPipe TPUSH/TPOP (neighbor-cell reduce/gather) are unaffected: GridPipe is
// not a cluster-local TPipe and is excluded by is_cube_vec_pipe_v below.

#ifndef DISTRIBUTED_FFN_GRID_TPIPE_TMOV_INL_HPP
#define DISTRIBUTED_FFN_GRID_TPIPE_TMOV_INL_HPP

#include <type_traits>

#include <pto/pto-inst.hpp>

// The PTO ISA surface (TPipe, is_tile_data_v, RecordEvent, TPUSH/TPOP/TMOV) is
// only declared in the device / sim / costmodel passes, matching the guard in
// pto/pto-inst.hpp.  Keep these overloads behind the same condition so the host
// preprocessing pass of the -xcce compile does not see undeclared identifiers.
#if defined(__CPU_SIM) || defined(__CCE_AICORE__) || defined(__COSTMODEL)

namespace pto {

// SFINAE marker for the cluster-local Cube<->Vector TPipe (C2V / V2C / Both /
// V2C_CTRL).  GridPipe and plain tiles are intentionally excluded so the TMOV
// overloads below never shadow the generic tile-to-tile move or the GridPipe
// neighbor TPUSH/TPOP.
template <typename T>
struct is_cube_vec_pipe : std::false_type {};

template <uint8_t FlagID, uint8_t DirType, uint32_t SlotSize, uint32_t SlotNum, uint32_t LocalSlotNum, bool IsNoSplit,
          bool EN_UNIT_FLAG>
struct is_cube_vec_pipe<TPipe<FlagID, DirType, SlotSize, SlotNum, LocalSlotNum, IsNoSplit, EN_UNIT_FLAG>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_cube_vec_pipe_v = is_cube_vec_pipe<std::remove_cv_t<std::remove_reference_t<T>>>::value;

// TMOV(pipe, tile): producer side.  Implicitly TPUSH tile into the C2V/V2C FIFO.
template <TileSplitAxis Split = TileSplitAxis::TILE_NO_SPLIT, typename Pipe, typename TileProd,
          std::enable_if_t<is_cube_vec_pipe_v<Pipe> && is_tile_data_v<TileProd>, int> = 0>
PTO_INST RecordEvent TMOV(Pipe &pipe, TileProd &tile)
{
    return TPUSH<Pipe, TileProd, Split>(pipe, tile);
}

// TMOV(tile, pipe): consumer side.  Implicitly TPOP the next slot into tile.
template <TileSplitAxis Split = TileSplitAxis::TILE_NO_SPLIT, typename TileCons, typename Pipe,
          std::enable_if_t<is_tile_data_v<TileCons> && is_cube_vec_pipe_v<Pipe>, int> = 0>
PTO_INST RecordEvent TMOV(TileCons &tile, Pipe &pipe)
{
    return TPOP<Pipe, TileCons, Split>(pipe, tile);
}

} // namespace pto

#endif // __CPU_SIM || __CCE_AICORE__ || __COSTMODEL

#endif // DISTRIBUTED_FFN_GRID_TPIPE_TMOV_INL_HPP
