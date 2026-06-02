/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// CCE-intrinsic-style API for neighbor-core monotonic counters.
//
// These two functions intentionally sit at the same layer as hardware-adapter
// intrinsics such as dcci: callers pass the concrete backend operand, while the
// function body is the only place that knows whether the target is native
// hardware or today's GM-counter mock.
//
// When hardware support is available, define PTO_GRID_COUNTER_NATIVE_INTRINSIC
// and provide compiler builtins with the same contract.  Call sites do not need
// to change.

#ifndef PTO_GRID_COUNTER_INTRINSIC_HPP
#define PTO_GRID_COUNTER_INTRINSIC_HPP

#include <cstdint>

#include <pto/common/type.hpp>
#if !defined(PTO_GRID_COUNTER_NATIVE_INTRINSIC)
#include <pto/common/grid_pipe_mock_spr.hpp>
#endif

namespace pto {

enum class NeighborCounterKind : uint8_t
{
    Ready = 0,
    Free = 1,
};

// Backend operand for the neighbor-counter intrinsic.
//
// Native hardware: `addr` is ignored and the compiler lowers (kind, dir, value)
// to SPR/WFE instructions.
// Current mock: `addr` points to the GM counter that represents either the
// peer-visible counter for set or the local mirror counter for wait.
struct NeighborCounterOperand {
    __gm__ uint32_t *addr = nullptr;
};

// Set a neighbor-visible monotonic counter `dist` hops away along `dir`.
//
// `dist` is the routed-unicast hop count; dist == 1 is the original adjacent
// doorbell (fully backward compatible).  For dist > 1 the counter write is
// routed `dist` hops in the kind-implied direction -- ready flows downstream
// along `dir`, free flows upstream against `dir` -- so a producer can ring a
// consumer K hops away (and vice versa for the free credit).
//
// Hardware contract (dist == 1):
//   mtspr_neighbor_counter(Ready, EAST, 1, value) ~= mtspr SPR_RDY_EAST, value
//   mtspr_neighbor_counter(Free,  EAST, 1, value) ~= mtspr SPR_FREE_EAST, value
// Native lowering for dist > 1 must provide a routed remote-notify that lands on
// the target core's same-named SPR; the direction-only SPR doorbell alone is
// adjacency-scoped.
//
// Memory ordering: release.  Earlier payload writes must become visible before
// the peer can observe this counter update.
AICORE inline void mtspr_neighbor_counter(NeighborCounterKind kind, uint32_t dir, uint32_t dist, uint32_t value,
                                          NeighborCounterOperand operand = {})
{
#if defined(PTO_GRID_COUNTER_NATIVE_INTRINSIC)
    (void)operand;
    __builtin_pto_mtspr_neighbor_counter(static_cast<uint32_t>(kind), dir, dist, value);
#else
    // Mock target is fully encoded in operand.addr (the caller resolves the
    // K-hop rank via RemoteCounterPtr), so the mock store needs neither dir nor
    // dist -- they exist only to drive the native routed-notify above.
    (void)dir;
    (void)dist;
    if (kind == NeighborCounterKind::Ready) {
        grid_mock::MockMtsprReady(operand.addr, value);
    } else {
        grid_mock::MockMtsprFree(operand.addr, value);
    }
#endif
}

// Wait until a neighbor-produced counter mirror reaches threshold.
//
// Hardware contract:
//   wfe_neighbor_counter(Ready, EAST, n) ~= wfe SPR_RDY_EAST, n
//   wfe_neighbor_counter(Free,  EAST, n) ~= wfe SPR_FREE_EAST, n
//
// Memory ordering: acquire.  Operations after the wait must not be reordered
// before the counter condition has been satisfied.
AICORE inline bool wfe_neighbor_counter(NeighborCounterKind kind, uint32_t dir, uint32_t threshold,
                                        NeighborCounterOperand operand = {}, uint32_t maxSpins = 0)
{
#if defined(PTO_GRID_COUNTER_NATIVE_INTRINSIC)
    (void)operand;
    (void)maxSpins;
    __builtin_pto_wfe_neighbor_counter(static_cast<uint32_t>(kind), dir, threshold);
    return true;
#else
    (void)dir;
    if (kind == NeighborCounterKind::Ready) {
        return grid_mock::MockTryWfeReady(operand.addr, threshold, maxSpins);
    }
    return grid_mock::MockTryWfeFree(operand.addr, threshold, maxSpins);
#endif
}

} // namespace pto

#endif // PTO_GRID_COUNTER_INTRINSIC_HPP
