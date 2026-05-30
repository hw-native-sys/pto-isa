/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_PERF_SIM_TILE_DEP_TRACKER_HPP
#define PTO_PERF_SIM_TILE_DEP_TRACKER_HPP

#include "recorder.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace pto::perf_sim {

// ── Tile signal-event tracker ──
// For each instruction:
//   1. Checks input tiles (remaining args) against prior writes → cross-pipe waits
//   2. Registers the output tile (first arg) address → signal_event + writer PipeStage
//
// Design (see design.md):
//   - Cross-pipe deps: generate wait on the latest writer's signal_event
//   - Same-pipe deps: NO wait (FIFO queue ordering guarantees it)
//   - Inputs checked BEFORE output registration (handles in-place ops correctly)

class TileDepTracker {
public:
    struct DepResult {
        event_t signal_event = -1;
        static constexpr int MAX_WAIT = 8;
        event_t wait_events[MAX_WAIT] = {-1, -1, -1, -1, -1, -1, -1, -1};
        int wait_count = 0;
    };

    // Track data dependencies for one instruction.
    // First arg = output tile, Rest args = input tiles.
    // Generates cross-pipe waits only (same-pipe handled by FIFO).
    //
    // NOTE: Scalar-stage instructions (TASSIGN, TRESHAPE, TALLOC, TFREE, etc.)
    // are memory-setup operations, NOT data producers. They must NOT be
    // registered as "last writers" because multiple loop iterations reuse the
    // same stack addresses for temporaries. Registering them would cause
    // iteration N+1's compute op to wait on iteration N's signal — which
    // has already been consumed — creating deadlock.
    template <typename First, typename... Rest>
    static DepResult TrackByAddr(const char *opcode, PipeStage current_stage, First &first, Rest &...rest)
    {
        DepResult result;

        if (current_stage == PipeStage::Scalar) {
            // Scalar instructions: track latest event for fallback deps.
            // Allocate event for all subblocks so the fallback event number
            // is valid.  Only subblock 0 records the instruction (PtoRecorder
            // filter), so only subblock 0's Scalar signals in simulation.
            // VecCore1 fallback works because both subblocks start Counter
            // from 0 → same channel → resolved by subblock 0's signal.
            uint32_t sub = current_subblock_id;
            event_t ev = Counter()++;
            LatestScalarEvent(sub) = ev;
            if (sub == 0) {
                result.signal_event = ev;
            }
            DebugLog("SCALAR_SET", opcode, current_stage, AddrOf(first), ev);
            return result;
        }

        // Step 1: Check input deps BEFORE registering output.
        CheckInputs(opcode, current_stage, result, rest...);

        // Scalar fallback: if no cross-pipe dep found, wait on latest Scalar event.
        // Works for all subblocks because both start Counter from 0, so the
        // Scalar event channel is signaled by subblock 0's Scalar instruction.
        if (result.wait_count == 0) {
            uint32_t sub = current_subblock_id;
            event_t scalar_ev = LatestScalarEvent(sub);
            if (scalar_ev >= 0 && result.wait_count < DepResult::MAX_WAIT) {
                result.wait_events[result.wait_count++] = scalar_ev;
                DebugLog("SCALAR_FB", opcode, current_stage, AddrOf(first), -1, scalar_ev);
            }
        }

        // Step 2: Register output tile as writer.
        uintptr_t out_addr = AddrOf(first);
        event_t ev = Counter()++;
        result.signal_event = ev;
        AddrMap()[out_addr] = {ev, current_stage};

        return result;
    }

    // Save current core's event count, reset for next core
    static void FinishCore()
    {
        CoreCounts().push_back(Counter());
        AddrMap().clear();
        Counter() = 0;
        LatestScalarEvent(0) = -1;
        LatestScalarEvent(1) = -1;
    }

    // Full reset (before first core)
    static void Clear()
    {
        AddrMap().clear();
        Counter() = 0;
        CoreCounts().clear();
        LatestScalarEvent(0) = -1;
        LatestScalarEvent(1) = -1;
    }

    // Max events per core (for channel sizing after all cores recorded)
    static int MaxEventsPerCore()
    {
        auto &c = CoreCounts();
        if (c.empty())
            return 64;
        return *std::max_element(c.begin(), c.end()) + 1;
    }

private:
    struct AddrInfo {
        event_t signal_event;
        PipeStage writer_stage;
    };

    // ── Scalar fallback tracking ──

    static event_t &LatestScalarEvent(uint32_t subblock_id)
    {
        static thread_local event_t events[2] = {-1, -1};
        return events[subblock_id & 1];
    }

    static bool DebugDepEnabled()
    {
        static bool enabled = (std::getenv("PTO_DEBUG_DEP") != nullptr);
        return enabled;
    }

    static void DebugLog(const char *tag, const char *opcode, PipeStage stage, uintptr_t addr, event_t ev = -1,
                         event_t wait_ev = -1)
    {
        if (!DebugDepEnabled())
            return;
        fprintf(stderr, "[DEP] %-14s %-12s stage=%-8s addr=0x%08lx ev=%d wait=%d\n", tag, opcode, PipeStageName(stage),
                (unsigned long)addr, ev, wait_ev);
    }

    // ── Input dependency checking ──

    // Base case: no more inputs to check
    static void CheckInputs(const char * /*opcode*/, PipeStage /*current_stage*/, DepResult & /*result*/)
    {}

    // Recursive: check one input arg, then recurse
    // NOTE: Scalar TASSIGN/TALLOC/etc are memory-setup (not data-producing).
    // Skipping them as "last writer" avoids cross-iteration pollution.
    template <typename T, typename... Ts>
    static void CheckInputs(const char *opcode, PipeStage current_stage, DepResult &result, T &arg, Ts &...rest)
    {
        uintptr_t addr = AddrOf(arg);
        auto it = AddrMap().find(addr);
        if (it != AddrMap().end() && it->second.writer_stage != current_stage &&
            it->second.writer_stage != PipeStage::Scalar) {
            if (result.wait_count < DepResult::MAX_WAIT) {
                result.wait_events[result.wait_count++] = it->second.signal_event;
            }
        } else if (DebugDepEnabled()) {
            const char *reason = (it == AddrMap().end())                    ? "NO_WRITER" :
                                 (it->second.writer_stage == current_stage) ? "SAME_PIPE" :
                                                                              "SCALAR_WRITER";
            fprintf(stderr, "[DEP] %-14s %-12s addr=0x%08lx writer=%-8s current=%-8s %s\n", "INPUT_MISS", opcode,
                    (unsigned long)addr, it != AddrMap().end() ? PipeStageName(it->second.writer_stage) : "NONE",
                    PipeStageName(current_stage), reason);
        }
        CheckInputs(opcode, current_stage, result, rest...);
    }

    // Resolve tile address to the actual data buffer pointer, not the C++
    // stack wrapper address or the address of an internal pointer member.
    //
    // For GlobalTensor: data() returns DType* → use pointer value directly
    // For Tile (UB/L1/L0): data() returns TileDType& where TileDType is a
    //   pointer type (e.g. float*) → auto d = arg.data() copies the pointer,
    //   is_pointer_v detects it, and we use the pointer value as the address.
    // This correctly returns the heap address of the actual data buffer,
    // not the stack address of the data_ member variable.

    template <typename T>
    static uintptr_t AddrOf(T &arg)
    {
        using RawT = std::remove_reference_t<T>;
        using MutableT = std::remove_const_t<RawT>;
        if constexpr (requires(MutableT &value) { value.data(); }) {
            auto &mutable_arg = const_cast<MutableT &>(arg);
            auto d = mutable_arg.data();
            if constexpr (std::is_pointer_v<decltype(d)>) {
                // data() returns a pointer (GlobalTensor) or copies a pointer
                // member (Tile where TileDType = float*)
                return reinterpret_cast<uintptr_t>(d);
            } else {
                // data() returns a non-pointer reference — take address
                return reinterpret_cast<uintptr_t>(&(mutable_arg.data()));
            }
        } else {
            return reinterpret_cast<uintptr_t>(&arg);
        }
    }

    static std::unordered_map<uintptr_t, AddrInfo> &AddrMap()
    {
        static thread_local std::unordered_map<uintptr_t, AddrInfo> m;
        return m;
    }

    static int &Counter()
    {
        static thread_local int c = 0;
        return c;
    }

    static std::vector<int> &CoreCounts()
    {
        static thread_local std::vector<int> v;
        return v;
    }
};

} // namespace pto::perf_sim

#endif
