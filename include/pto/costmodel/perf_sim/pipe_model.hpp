/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_PERF_SIM_PIPE_MODEL_HPP
#define PTO_PERF_SIM_PIPE_MODEL_HPP

#include "recorder.hpp"
#include "tile_dep_tracker.hpp"
#include "cache_model.hpp"
#include <algorithm>
#include <array>
#include <deque>
#include <list>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace pto::perf_sim {

// ── PipeEntry: one operation queued on a pipe ──

struct PipeEntry {
    enum State
    {
        IDLE,
        RUNNING,
        WAITING,
        DONE
    };
    State state = IDLE;
    std::string name;
    uint64_t duration = 0;
    event_t signal_event = -1;
    static constexpr int MAX_WAIT = 8; // expanded for merged sync waits
    event_t wait_events[MAX_WAIT] = {-1, -1, -1, -1, -1, -1, -1, -1};
    int wait_count = 0;
    static constexpr int MAX_SIGNALS = 8;
    event_t extra_signals[MAX_SIGNALS] = {-1, -1, -1, -1, -1, -1, -1, -1};
    int extra_signal_count = 0;
    uint64_t start_cycle = 0;
    uint64_t end_cycle = 0;
    // Cache info for MTE operations
    uint64_t addr = 0;
    uint64_t size = 0;
    bool is_mte = false;
    bool is_sync_wait = false; // true for explicit WAIT entries (pipe-level block)
};

// ── PipeQueue: one per pipeline stage ──

struct PipeQueue {
    std::string label;
    std::list<PipeEntry> entries;
    PipeEntry *current = nullptr;
    uint64_t next_free_cycle = 0;

    bool Empty() const
    {
        return entries.empty() && current == nullptr;
    }
    bool CanAccept() const
    {
        return current == nullptr;
    }
    void Push(PipeEntry &&e)
    {
        entries.push_back(std::move(e));
    }
};

// ── EventChannel: cross-pipe synchronization (counter-based) ──
// Uses a counter instead of a boolean so multiple SIGNALs for the same
// (dst_pipe, event_id) pair accumulate correctly across iterations.

struct EventChannel {
    int count = 0; // number of pending signals
    uint64_t signal_cycle = 0;
    bool signaled() const
    {
        return count > 0;
    }
};

// ── Channel layout ──
// Two independent spaces to avoid collision:
//   [0,  SYNC_BASE)                        → auto-dep (TileDepTracker, flat event_id)
//   [SYNC_BASE, SYNC_BASE + HW_PIPES*EPP)  → sync (set_flag/wait_flag, per hw dst_pipe)
//
// Key: use hardware pipe IDs (0-7), NOT PipeStage, because PIPE_MTE2 maps to
// two PipeStages (MTE2_AIC/MTE2_AIV) but is ONE hardware pipe for sync purposes.

constexpr int EVENTS_PER_PIPE = 64; // max event IDs per pipe (hardware has ~8)
constexpr int HW_PIPE_COUNT = 8;    // PIPE_S=0..PIPE_FIX=7
constexpr int MIN_SYNC_CHANNEL_BASE = 340;
constexpr int VEC_SYNC_CHANNEL_SPACES = 2;

inline int SyncChannelBase()
{
    // Must be called after all recording is done (TileDepTracker has counts)
    // Enforce a minimum so EncodeSyncKey(PIPE_V=1, 0) stays
    // above the auto-dep range [0, 340). Without this, small MaxEventsPerCore
    // values (e.g. 220) cause EncodeSyncKey results (e.g. 284) to fall in the
    // auto-dep gap, making explicit sync waits misclassified as auto-dep.
    return std::max(MIN_SYNC_CHANNEL_BASE, TileDepTracker::MaxEventsPerCore());
}

inline int TotalChannels()
{
    return SyncChannelBase() + HW_PIPE_COUNT * EVENTS_PER_PIPE * VEC_SYNC_CHANNEL_SPACES;
}

// Encode a sync channel key using hardware pipe ID (not PipeStage)
inline event_t EncodeSyncKey(int hw_dst_pipe, event_t event_id, uint32_t subblock_id = 0)
{
    event_t base = SyncChannelBase() + hw_dst_pipe * EVENTS_PER_PIPE;
    if (subblock_id > 0) {
        base += HW_PIPE_COUNT * EVENTS_PER_PIPE; // VecCore1 offset
    }
    return base + event_id;
}

// ── PipeTimeline: simulation output ──

struct PipeEvent {
    std::string name;
    std::string pipe_label;
    uint64_t start_cycle;
    uint64_t end_cycle;
    // Dependency info for flow-event visualization
    event_t signal_event = -1;
    event_t wait_events[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    int wait_count = 0;
    event_t extra_signals[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    int extra_signal_count = 0;
    bool is_sync = false;  // true for SIGNAL/WAIT entries
    bool stuck = false;    // true for entries that never completed
    uint64_t duration = 0; // original duration (for stuck entries)
};

struct PipeTimeline {
    std::vector<PipeEvent> events;
    uint64_t total_cycles = 0;
};

#include "pipe_model_impl.inl"

} // namespace pto::perf_sim

#endif
