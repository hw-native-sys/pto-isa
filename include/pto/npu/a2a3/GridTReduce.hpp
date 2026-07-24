/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// A2/A3 backend for GridPipe TREDUCE<Direction, Op, Dist>: a fused
// "receive-combine-forward" reduce hop along a mesh direction.  Builds on
// GridTPush.hpp / GridTPop.hpp -- see the V7 design spec section 5 (worked
// ReduceSum example), which frames the row reduce as the SAME single-hop SPSC
// handshake as AllGather, differing ONLY in the per-hop middle operation:
// AllGather relays the tile, ReduceSum folds it in with a combine before
// forwarding.  TREDUCE is that fused hop.
//
// Semantics per cell (all roles derived from (Dir, Dist, coord, shape); no
// explicit "am I root" flag needed -- the mesh boundary defines source/sink):
//   * interior/sink (has an upstream Dist hops back along Dir):
//         recv  <- TPOP<Dir,Dist>     (drain the transiting partial from upstream)
//         acc   <- combine(acc, recv) (fold in this cell's local contribution)
//   * source/interior (has a downstream Dist hops on along Dir):
//         TPUSH<Dir,Dist>(acc)        (forward the running reduction one hop)
//   * sink (no downstream): acc holds the COMPLETE reduction; the caller stores it.
//
// Along-the-path / on-transit compute (随路/过路计算): a fabric that can combine
// in the router collapses this whole receive-combine-forward body into ONE
// routed reduce-forward instruction -- `recv` and the in-core combine disappear
// into the transfer.  On A3 there is no such fabric and the adder is core-local,
// so TREDUCE lowers to exactly the local TPOP + combine + TPUSH sequence below.
// Nothing in the (Dir, Op, Dist, pipe, acc, recv) signature changes between the
// two lowerings; only the body does.

#ifndef PTO_A2A3_GRID_TREDUCE_HPP
#define PTO_A2A3_GRID_TREDUCE_HPP

#include <cstdint>

#include <pto/comm/comm_types.hpp>         // pto::comm::ReduceOp (Sum/Max/Min) -- shared with the collective TREDUCE
#include <pto/npu/a2a3/GridTPop.hpp>       // GRID_TPOP_IMPL (receive half)
#include <pto/npu/a2a3/GridTPush.hpp>      // GRID_TPUSH_IMPL (forward half) + payload hooks
#include <pto/npu/a2a3/grid_intrinsic.hpp> // CanPopK / CanPushK topology
#include <pto/npu/a2a3/grid_pipe_runtime.hpp>

namespace pto {

// Per-hop combine: fold the transiting partial `recv` into the accumulator `acc`
// with the reduce operator.  On A3 the adder lives inside the core, so this is an
// ordinary in-UB Vec op (TADD/TMAX/TMIN, resolved via ADL on the tile type); a
// future along-the-path-compute lowering performs this in the fabric during the
// transfer and drops the call entirely.  `Op` is a compile-time constant, so the
// branch folds away and only the selected instruction is instantiated.
template <pto::comm::ReduceOp Op, typename TileAcc, typename TileRecv>
AICORE inline void GridReduceCombine(TileAcc &acc, TileRecv &recv)
{
    if constexpr (Op == pto::comm::ReduceOp::Sum) {
        TADD(acc, acc, recv);
    } else if constexpr (Op == pto::comm::ReduceOp::Max) {
        TMAX(acc, acc, recv);
    } else {
        static_assert(Op == pto::comm::ReduceOp::Min, "GridPipe TREDUCE supports ReduceOp Sum/Max/Min only");
        TMIN(acc, acc, recv);
    }
}

// GridPipe TREDUCE<Dir, Op, Dist>: one fused reduce hop (see the file header).
// `acc` is in/out -- on entry the cell's local contribution, on return the
// running reduction up to and including this cell (at the sink, the complete
// result).  `recv` is the landing tile for the transiting partial (mandatory on
// A3's in-core adder; unused by a fabric that combines on transit).  Both tiles
// must share the reduce dtype/shape.
//
// Fences use the same conservative pipe_barrier(PIPE_ALL) + dsb(DSB_DDR) publish
// form as GridTPush.hpp (parse-safe on every target profile).  A full barrier
// subsumes the fine-grained MTE2->V->MTE3 crossings the hand-written kernel used:
//   * after the pop, before the combine: drain the MTE2 slot->recv copy (and the
//     caller's MTE2 producer of acc) so the Vec combine reads settled UB;
//   * after the combine / before the push: drain the V combine so the MTE3
//     payload copy reads the settled accumulator.
// The push half additionally carries its own data-before-ready publish fence
// inside GRID_TPUSH_IMPL.  The pop / push are gated on CanPopK / CanPushK so a
// boundary cell never enters GRID_TPUSH_IMPL's out-of-mesh fault path.
template <pto::GridDirection Dir, pto::comm::ReduceOp Op, int Dist, typename Pipe, typename TileAcc, typename TileRecv>
AICORE void GRID_TREDUCE_IMPL(Pipe &pipe, TileAcc &acc, TileRecv &recv)
{
    static_assert(Dir != pto::GridDirection::SOURCE, "GridPipe TREDUCE<SOURCE> is illegal (SOURCE is TPOP-only)");
    static_assert(Dist >= 1, "GridPipe TREDUCE distance must be >= 1 (routed K-hop reduce)");

    // Receive-and-combine half.  A source cell (no upstream along Dir) has nothing
    // to drain and forwards its own contribution unchanged.
    const bool didCombine = CanPopK(Dir, pipe.coord, pipe.shape, Dist);
    if (didCombine) {
        GRID_TPOP_IMPL<Dir, Dist, Pipe, TileRecv>(pipe, recv);
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);
        GridReduceCombine<Op, TileAcc, TileRecv>(acc, recv);
    }

    // Forward half.  A sink cell (no downstream along Dir) keeps the complete
    // reduction in `acc` for the caller to store.
    if (CanPushK(Dir, pipe.coord, pipe.shape, Dist)) {
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);
        GRID_TPUSH_IMPL<Dir, Dist, Pipe, TileAcc>(pipe, acc);
    }
}

// Forward declaration: TileUbPtr is provided by the demo's
// gridpipe_payload_inl.hpp (same pluggable payload-hook contract as the other
// a2a3_grid_payload helpers).  Kept out-of-line so this group-reduce facade
// stays tile-agnostic (it hands the reduce_group_to_ubuf intrinsic raw UB ptrs).
namespace a2a3_grid_payload {
template <typename TileT>
__tf__ AICORE __ubuf__ void *TileUbPtr(TileT &tile);
} // namespace a2a3_grid_payload

// ===========================================================================
// GRID_TREDUCE_GROUP_IMPL: N->1 group fan-in reduce.  The TARGET core (caller
// == sink) reads every member's resolved contribution slot and folds them
// element-wise with Op into `acc`, via the reduce_group_to_ubuf intrinsic (the
// new reduce-semantics CCE instruction, grid_cce_intrinsic.hpp).  This is the
// hardware-accelerated single-instruction form of an N->1 fan-in -- a DIFFERENT
// collective shape from the directional GRID_TREDUCE_IMPL relay above (§7.1):
// every member's contribution is read directly by the sink, not relayed hop by
// hop.  The directional relay above stays available for nearest-neighbor chains.
//
// `scratch` is the in-core combine scratch (one member's worth of UB; required
// by the A3 mock's in-core Vec combine).  `groupSlotBase` / `memberStride` /
// `memberCount` describe the resolved per-member contribution arena -- e.g. the
// contributors' partial buffer laid out uniform-stride.  The combine folds
// members in ascending index order (member 0 seeds acc), so an SPMD row/col
// fan-in reproduces the relay's left-to-right accumulation bit-for-bit (FP add
// is commutative).
// ===========================================================================
template <pto::GridGroup Group, pto::comm::ReduceOp Op, typename T, typename TileAcc, typename TileScratch>
AICORE void GRID_TREDUCE_GROUP_IMPL(TileAcc &acc, TileScratch &scratch, __gm__ const T *groupSlotBase,
                                    uint32_t bytes, uint32_t memberCount, pto::GridRect rect = {},
                                    uint32_t memberStride = 0)
{
    __ubuf__ T *dst = reinterpret_cast<__ubuf__ T *>(a2a3_grid_payload::TileUbPtr<TileAcc>(acc));
    __ubuf__ T *scr = reinterpret_cast<__ubuf__ T *>(a2a3_grid_payload::TileUbPtr<TileScratch>(scratch));
    reduce_group_to_ubuf<Group, Op, T>(dst, groupSlotBase, bytes, memberCount, rect, memberStride, scr);
}

} // namespace pto

#endif // PTO_A2A3_GRID_TREDUCE_HPP
