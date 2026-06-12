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

// ---------------------------------------------------------------------------
// GmSramArena: explicit GM address-segment model of future-hardware per-core
// SRAM.
//
// Real silicon gives every core a private on-chip SRAM that the NoC fabric can
// only *write* into from a neighbor (TPUSH = cross-hop write), never *read* out
// of remotely (TPOP only drains the local core's own SRAM).  Until that
// hardware exists we model the SRAM as a contiguous GM arena cut into equal,
// per-core address segments.  Core `c` owns segment `c`:
//
//   [base + c*segBytes, base + (c+1)*segBytes)
//
// The NoC contract this encodes:
//   * a core may WRITE across segments  (TPUSH pushes into a neighbor segment),
//   * a core may only READ its own segment (TPOP pops from local SRAM).
//
// The arena is the single source of truth for "which core owns this address",
// so the mock TPOP path can reject a cross-segment read instead of silently
// servicing it through the GM-backed fake window (which physically *can* read
// any address, unlike the fabric it stands in for).
struct GmSramArena {
    uint64_t base = 0;     // segment 0 base == contiguous arena base
    uint64_t segBytes = 0; // bytes per per-core segment (== HCCL winSize in the demo)
    uint32_t numSegs = 0;  // number of cores / segments

    AICORE constexpr uint64_t SegmentBase(int seg) const
    {
        return base + static_cast<uint64_t>(seg) * segBytes;
    }

    // Index of the segment that owns `addr`, or -1 if `addr` is outside the arena.
    AICORE constexpr int SegmentOf(uint64_t addr) const
    {
        if (numSegs == 0 || segBytes == 0 || addr < base) {
            return -1;
        }
        uint64_t idx = (addr - base) / segBytes;
        return idx < numSegs ? static_cast<int>(idx) : -1;
    }

    // True iff [addr, addr+bytes) lies entirely within segment `seg`.  This is
    // exactly the "may core `seg` read this slot?" test used by the TPOP guard.
    AICORE constexpr bool InSegment(int seg, uint64_t addr, uint64_t bytes) const
    {
        if (seg < 0 || static_cast<uint32_t>(seg) >= numSegs) {
            return false;
        }
        uint64_t lo = SegmentBase(seg);
        uint64_t hi = lo + segBytes;
        return addr >= lo && (addr + bytes) <= hi && (addr + bytes) >= addr; // last term traps wrap-around
    }
};

// Compile-time self-test of the segment classifier.  It is built into every
// A2/A3 kernel that pulls in this header (GridTPush.hpp -> pto_instr_impl.hpp),
// so a regression in the segment math fails the build rather than silently
// mis-routing a TPOP.  It also doubles as executable documentation of the rule.
AICORE constexpr bool GmSramArenaSelfCheck()
{
    GmSramArena arena{0x1000, 0x100, 4};          // 4 cores, 0x100-byte segments, based at 0x1000
    bool ok = true;
    ok = ok && (arena.SegmentOf(0x1000) == 0);    // first byte of core 0
    ok = ok && (arena.SegmentOf(0x11FF) == 1);    // last byte of core 1
    ok = ok && (arena.SegmentOf(0x1200) == 2);    // first byte of core 2
    ok = ok && (arena.SegmentOf(0x0FFF) == -1);   // below the arena
    ok = ok && (arena.SegmentOf(0x1400) == -1);   // past the arena ([0x1000,0x1400))
    ok = ok && arena.InSegment(1, 0x1100, 0x40);  // wholly inside core 1 -> local
    ok = ok && !arena.InSegment(1, 0x11F0, 0x40); // spills past core 1 -> not local
    ok = ok && !arena.InSegment(1, 0x1200, 0x10); // core 1 reading core 2 -> not local
    return ok;
}
static_assert(GmSramArenaSelfCheck(), "GmSramArena segment classifier self-test failed");

namespace grid_sram_mock {

AICORE uint64_t MockGetNeighborSramAddr(__gm__ void *runtimeCtx, uint64_t localSramAddr, int32_t peerRank);
AICORE void MockCopySramToNeighborSram(neighbor_sram_addr dst, neighbor_sram_addr src, uint32_t bytes, uint64_t config);
AICORE void MockCopyNeighborSramToSram(neighbor_sram_addr dst, neighbor_sram_addr src, uint32_t bytes, uint64_t config);
// Read-locality predicate for the TPOP guard.  Returns true iff
// [slotAddr, slotAddr+bytes) lies inside callerRank's own GmSramArena segment.
// The demo owns the definition because it knows the runtime window layout.
AICORE bool MockSramPopIsLocal(__gm__ void *runtimeCtx, uint64_t slotAddr, uint32_t bytes, int32_t callerRank);

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

// TPOP read-locality guard.  Returns true iff `slot` is the calling core's own
// SRAM segment, i.e. the read is legal under the NoC "TPOP only pops local
// SRAM" rule.
//
// Native hardware physically cannot read a neighbor's SRAM (the fabric has no
// remote-read path), so the read address is always local by construction and
// the native lowering is a no-op that returns true.  The A2/A3 mock backs SRAM
// with a GM-mapped fake window that *can* be read at any address, so we must
// check explicitly: the mock validates `slot` against callerRank's GmSramArena
// segment and reports a cross-segment read instead of silently servicing it.
AICORE inline bool sram_pop_is_local(neighbor_sram_addr slot, uint32_t bytes, int32_t callerRank,
                                     NeighborSramOperand operand = {})
{
#if defined(PTO_GRID_SRAM_NATIVE_INTRINSIC)
    (void)slot;
    (void)bytes;
    (void)callerRank;
    (void)operand;
    return true;
#else
    return grid_sram_mock::MockSramPopIsLocal(operand.runtimeCtx, slot.value, bytes, callerRank);
#endif
}

} // namespace pto

#endif // PTO_GRID_SRAM_INTRINSIC_HPP
