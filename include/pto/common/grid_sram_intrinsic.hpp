/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// CCE-intrinsic-style API for neighbor-core SRAM addressing and transfer.
// Grid TPUSH/TPOP calls this layer so the same protocol can use either native
// __builtin_pto_* instructions or the current A2/A3 mock lowering.
//
// Grid TPUSH/TPOP payload movement targets neighbor-visible on-chip storage,
// not a global-memory object in the hardware contract. Hardware exposes this
// path as local SRAM <-> neighbor SRAM without distinguishing UB and CBUF in
// the cross-core intrinsic names. Current A2/A3 demos emulate those neighbor
// windows with local GM windows; the mock address operand is therefore kept as
// backend context while the public API exposes address-register style values.

#ifndef PTO_GRID_SRAM_INTRINSIC_HPP
#define PTO_GRID_SRAM_INTRINSIC_HPP

#include <cstdint>

#include <pto/common/type.hpp>

namespace pto {

// Address-register model for a neighbor-visible SRAM location.
// Native hardware: value is the encoded SRAM address register payload.
// Current mock: value is the integer form of the backing fake-window pointer.
struct neighbor_sram_addr {
    uint64_t value = 0;
};

// Backend operand used only by the mock lowering. Native hardware ignores it
// and derives the neighbor SRAM mapping from dir/peer configuration registers.
struct NeighborSramOperand {
    __gm__ void *runtimeCtx = nullptr;
};

namespace grid_sram_mock {

AICORE uint64_t MockGetNeighborSramAddr(__gm__ void *runtimeCtx, uint64_t localSramAddr, int32_t peerRank);
AICORE void MockCopySramToNeighborSram(neighbor_sram_addr dst, neighbor_sram_addr src, uint32_t bytes, uint64_t config);
AICORE void MockCopyNeighborSramToSram(neighbor_sram_addr dst, neighbor_sram_addr src, uint32_t bytes, uint64_t config);

} // namespace grid_sram_mock

// Resolve a local SRAM slot address to the same slot in a neighbor core.
//
// Parameter order follows CCE data/address instructions: output register first,
// source address register next, then topology/control operands.
AICORE inline void get_neighbor_sram_addr(neighbor_sram_addr &dst, neighbor_sram_addr src, uint32_t dir,
                                          int32_t peerRank, NeighborSramOperand operand = {})
{
#if defined(PTO_GRID_SRAM_NATIVE_INTRINSIC)
    (void)operand;
    dst.value = __builtin_pto_get_neighbor_sram_addr(src.value, dir, peerRank);
#else
    (void)dir;
    dst.value = grid_sram_mock::MockGetNeighborSramAddr(operand.runtimeCtx, src.value, peerRank);
#endif
}

// Cross-core payload writes. Source and destination are SRAM address-register
// values; the intrinsic name intentionally does not expose UB/CBUF variants.
AICORE inline void copy_sram_to_neighbor_sram(neighbor_sram_addr dst, neighbor_sram_addr src, uint32_t bytes,
                                              uint64_t config)
{
#if defined(PTO_GRID_SRAM_NATIVE_INTRINSIC)
    __builtin_pto_copy_sram_to_neighbor_sram(dst.value, src.value, bytes, config);
#else
    // A2/A3 validation keeps the intrinsic call shape and only changes the
    // final lowering to a GM-backed fake window.
    grid_sram_mock::MockCopySramToNeighborSram(dst, src, bytes, config);
#endif
}

// Cross-core payload read interface. The native lowering is intentionally left
// as an interface placeholder until hardware/compiler support is available.
AICORE inline void copy_neighbor_sram_to_sram(neighbor_sram_addr dst, neighbor_sram_addr src, uint32_t bytes,
                                              uint64_t config)
{
#if defined(PTO_GRID_SRAM_NATIVE_INTRINSIC)
    (void)dst;
    (void)src;
    (void)bytes;
    (void)config;
#else
    // Keep TPOP visible as intrinsic-style SRAM transfer in the A2/A3 mock.
    grid_sram_mock::MockCopyNeighborSramToSram(dst, src, bytes, config);
#endif
}

} // namespace pto

#endif // PTO_GRID_SRAM_INTRINSIC_HPP
