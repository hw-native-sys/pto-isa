/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

inline void SignalChannel(std::vector<EventChannel> &channels, event_t event, uint64_t now)
{
    if (event < 0 || static_cast<size_t>(event) >= channels.size())
        return;
    auto &channel = channels[event];
    channel.count++;
    channel.signal_cycle = now;
}

inline bool AnyRunning(const std::vector<PipeQueue> &queues)
{
    for (auto &q : queues) {
        if (q.current && q.current->state == PipeEntry::RUNNING)
            return true;
    }
    return false;
}

struct StepProgress {
    bool any_active = false;
    bool dispatched = false;
    uint64_t next_event = UINT64_MAX;
};

template <typename SignalFn>
inline void CompleteRunningEntryCommon(PipeQueue &queue, PipeTimeline &timeline, uint64_t now, StepProgress &progress,
                                       SignalFn signal_fn)
{
    if (!queue.current || queue.current->state != PipeEntry::RUNNING)
        return;
    progress.any_active = true;
    if (now < queue.current->end_cycle) {
        progress.next_event = std::min(progress.next_event, queue.current->end_cycle);
        return;
    }
    timeline.events.push_back(BuildPipeEvent(*queue.current, queue.label));
    signal_fn(queue.current->signal_event);
    for (int si = 0; si < queue.current->extra_signal_count; ++si)
        signal_fn(queue.current->extra_signals[si]);
    queue.current->state = PipeEntry::DONE;
    queue.current = nullptr;
    queue.next_free_cycle = now;
}

inline void CompleteRunningEntry(PipeQueue &queue, PipeTimeline &timeline, std::vector<EventChannel> &channels,
                                 uint64_t now, StepProgress &progress)
{
    CompleteRunningEntryCommon(queue, timeline, now, progress,
                               [&](event_t event) { SignalChannel(channels, event, now); });
}

inline void PopDoneEntries(std::vector<PipeQueue> &queues)
{
    for (auto &queue : queues) {
        while (!queue.entries.empty() && queue.entries.front().state == PipeEntry::DONE)
            queue.entries.pop_front();
    }
}

inline bool SingleCoreWaitsReady(const PipeEntry &entry, const std::vector<EventChannel> &channels)
{
    for (int i = 0; i < entry.wait_count; ++i) {
        event_t wait = entry.wait_events[i];
        if (wait >= 0 && (static_cast<size_t>(wait) >= channels.size() || !channels[wait].signaled()))
            return false;
    }
    return true;
}

inline void ConsumeSingleCoreSyncWaits(const PipeEntry &entry, std::vector<EventChannel> &channels)
{
    int sync_base = SyncChannelBase();
    for (int i = 0; i < entry.wait_count; ++i) {
        event_t wait = entry.wait_events[i];
        if (wait >= sync_base && static_cast<size_t>(wait) < channels.size())
            channels[wait].count--;
    }
}

inline void StartEntry(PipeQueue &queue, PipeEntry &entry, uint64_t now, L2CacheModel &cache, StepProgress &progress)
{
    uint64_t duration = entry.duration;
    if (entry.is_mte && entry.size > 0) {
        auto result_cache = cache.Access(entry.addr, entry.size, duration);
        duration = result_cache.adjusted_duration;
    }
    entry.state = PipeEntry::RUNNING;
    entry.start_cycle = std::max(queue.next_free_cycle, now);
    entry.end_cycle = entry.start_cycle + duration;
    queue.current = &entry;
    progress.any_active = true;
    progress.dispatched = true;
}

inline void TryDispatchSingleCore(PipeQueue &queue, std::vector<EventChannel> &channels, L2CacheModel &cache,
                                  uint64_t now, StepProgress &progress)
{
    if (!queue.CanAccept() || queue.entries.empty())
        return;
    auto &entry = queue.entries.front();
    if (entry.name == "BARRIER") {
        StartEntry(queue, entry, now, cache, progress);
        return;
    }
    if (entry.wait_count > 0 && entry.state == PipeEntry::WAITING) {
        if (!SingleCoreWaitsReady(entry, channels)) {
            progress.any_active = true;
            return;
        }
        ConsumeSingleCoreSyncWaits(entry, channels);
        entry.state = PipeEntry::IDLE;
    }
    if (entry.state == PipeEntry::IDLE)
        StartEntry(queue, entry, now, cache, progress);
}

inline void CollectStuckEntries(const std::vector<PipeQueue> &queues, PipeTimeline &timeline)
{
    for (auto &queue : queues) {
        for (auto &entry : queue.entries)
            timeline.events.push_back(BuildPipeEvent(entry, queue.label, true));
    }
}

constexpr int EVENTS_PER_CORE = 8192;        // max channels per core (auto-dep + sync)
constexpr int CROSS_CHANNEL_OFFSET = 262144; // must be > max_cores * EVENTS_PER_CORE

inline event_t EncodeMergedSyncEvent(const SyncRecord &sync, bool cross_core_enabled)
{
    if (cross_core_enabled && sync.cross_core) {
        return CROSS_CHANNEL_OFFSET + sync.event_id;
    }
    return EncodeSyncKey(sync.dst_pipe, sync.event_id, sync.subblock_id);
}

inline void PushBarrierEntry(std::vector<PipeQueue> &queues)
{
    PipeEntry entry;
    entry.name = "BARRIER";
    entry.duration = 0;
    queues[static_cast<int>(PipeStage::Scalar)].Push(std::move(entry));
}

inline void PushSignalEntry(std::vector<PipeQueue> &queues, const MergedEntry &entry, PipeStage last_mte2,
                            bool include_seq, bool cross_core_enabled)
{
    const auto &sync = entry.sync;
    PipeEntry pipe_entry;
    pipe_entry.name = "SIGNAL(" + std::to_string(sync.event_id) + ")";
    if (include_seq) {
        pipe_entry.name += ":" + std::to_string(entry.seq);
    }
    pipe_entry.duration = 0;
    pipe_entry.signal_event = EncodeMergedSyncEvent(sync, cross_core_enabled);
    if (!sync.cross_core && sync.subblock_id == 0 && IsCubeSidePipe(sync.src_pipe) && !IsCubeSidePipe(sync.dst_pipe)) {
        AppendExtraSignal(pipe_entry, EncodeSyncKey(sync.dst_pipe, sync.event_id, 1));
    }
    queues[QueueIndex(SyncPipeToStage(sync.src_pipe, last_mte2), sync.subblock_id)].Push(std::move(pipe_entry));
}

inline void PushWaitEntry(std::vector<PipeQueue> &queues, const MergedEntry &entry, PipeStage last_mte2,
                          bool include_seq, bool cross_core_enabled)
{
    const auto &sync = entry.sync;
    PipeEntry pipe_entry;
    pipe_entry.name = "WAIT(" + std::to_string(sync.event_id) + ")";
    if (include_seq) {
        pipe_entry.name += ":" + std::to_string(entry.seq);
    }
    pipe_entry.duration = 0;
    pipe_entry.is_sync_wait = true;
    AppendWait(pipe_entry, EncodeMergedSyncEvent(sync, cross_core_enabled));
    pipe_entry.state = PipeEntry::WAITING;
    queues[QueueIndex(SyncPipeToStage(sync.dst_pipe, last_mte2), sync.subblock_id)].Push(std::move(pipe_entry));
}

inline void PushInstrEntry(std::vector<PipeQueue> &queues, const MergedEntry &entry, PipeStage &last_mte2,
                           bool include_seq, CvDependencyState &cv_state)
{
    int idx = QueueIndex(entry.instr.stage, entry.instr.subblock_id);
    if (idx < 0 || idx >= static_cast<int>(queues.size())) {
        return;
    }
    if (IsMTE2(entry.instr.stage)) {
        last_mte2 = entry.instr.stage;
    }
    cv_state.ObserveProducer(entry.instr);
    queues[idx].Push(BuildInstrEntry(entry.instr, entry.seq, include_seq, cv_state));
}

inline void PushMergedEntry(std::vector<PipeQueue> &queues, const MergedEntry &entry, PipeStage &last_mte2,
                            bool include_seq, bool cross_core_enabled, CvDependencyState &cv_state)
{
    if (!entry.is_sync) {
        PushInstrEntry(queues, entry, last_mte2, include_seq, cv_state);
        return;
    }
    switch (entry.sync.kind) {
        case SyncKind::Barrier:
            PushBarrierEntry(queues);
            break;
        case SyncKind::Signal:
            PushSignalEntry(queues, entry, last_mte2, include_seq, cross_core_enabled);
            break;
        case SyncKind::Wait:
            PushWaitEntry(queues, entry, last_mte2, include_seq, cross_core_enabled);
            break;
    }
}

inline void FillQueuesFromMergedImpl(std::vector<PipeQueue> &queues, const std::vector<MergedEntry> &merged,
                                     bool include_seq, bool cross_core_enabled)
{
    PipeStage last_mte2 = PipeStage::MTE2_AIV;
    CvDependencyState cv_state;
    for (auto &entry : merged) {
        if (entry.sync_consumed) {
            continue;
        }
        PushMergedEntry(queues, entry, last_mte2, include_seq, cross_core_enabled, cv_state);
    }
}

inline void FillQueuesFromMerged(std::vector<PipeQueue> &queues, std::vector<EventChannel> &channels,
                                 const std::vector<MergedEntry> &merged)
{
    (void)channels;
    FillQueuesFromMergedImpl(queues, merged, true, false);
}

// ── Event-driven Step simulation with cache ──

inline PipeTimeline StepSimulate(std::vector<PipeQueue> &queues, std::vector<EventChannel> &channels,
                                 L2CacheModel &cache, uint32_t max_steps = 500000)
{
    PipeTimeline result;
    uint64_t now = 0;

    for (uint32_t step = 0; step < max_steps; ++step) {
        StepProgress progress;
        for (auto &queue : queues)
            CompleteRunningEntry(queue, result, channels, now, progress);

        PopDoneEntries(queues);

        for (auto &queue : queues)
            TryDispatchSingleCore(queue, channels, cache, now, progress);

        if (!progress.any_active)
            break;
        if (!progress.dispatched) {
            if (!AnyRunning(queues))
                break;
            if (progress.next_event != UINT64_MAX && progress.next_event > now)
                now = progress.next_event;
        }
    }

    CollectStuckEntries(queues, result);

    for (auto &ev : result.events)
        result.total_cycles = std::max(result.total_cycles, ev.end_cycle);
    return result;
}

// ── RunPipeline: main simulation entry point ──

inline PipeTimeline RunPipeline()
{
    auto merged = MergeRecords();
    InlineSyncIntoMerged(merged);
    auto queues = MakePipeQueues();
    std::vector<EventChannel> channels(TotalChannels());
    FillQueuesFromMerged(queues, channels, merged);

    L2CacheModel cache;
    return StepSimulate(queues, channels, cache);
}

// ── Multi-core simulation data structures ──

// Helper: lookup EventChannel for multi-core simulation
inline EventChannel *LookupChannel(std::vector<EventChannel> &intra, std::vector<EventChannel> &cross, uint32_t core_id,
                                   event_t event_id)
{
    if (event_id < 0)
        return nullptr;
    if (event_id >= CROSS_CHANNEL_OFFSET) {
        int idx = event_id - CROSS_CHANNEL_OFFSET;
        if (idx < static_cast<int>(cross.size()))
            return &cross[idx];
        return nullptr;
    }
    int idx = core_id * EVENTS_PER_CORE + event_id;
    if (idx < static_cast<int>(intra.size()))
        return &intra[idx];
    return nullptr;
}

inline void SignalChannel(EventChannel *channel, uint64_t now)
{
    if (!channel)
        return;
    channel->count++;
    channel->signal_cycle = now;
}

struct CorePipeline {
    uint32_t core_id;
    std::vector<PipeQueue> queues; // 7 pipes per core
};

struct MultiCoreTimeline {
    std::vector<PipeTimeline> per_core;
    uint64_t total_cycles = 0;
};

inline bool AnyRunning(const std::vector<CorePipeline> &core_pipelines)
{
    for (auto &core : core_pipelines) {
        if (AnyRunning(core.queues))
            return true;
    }
    return false;
}

inline void CompleteRunningEntry(CorePipeline &core, PipeQueue &queue, PipeTimeline &timeline,
                                 std::vector<EventChannel> &intra_channels, std::vector<EventChannel> &cross_channels,
                                 uint64_t now, StepProgress &progress)
{
    CompleteRunningEntryCommon(queue, timeline, now, progress, [&](event_t event) {
        SignalChannel(LookupChannel(intra_channels, cross_channels, core.core_id, event), now);
    });
}

inline bool MultiCoreWaitsReady(const PipeEntry &entry, uint32_t core_id, std::vector<EventChannel> &intra_channels,
                                std::vector<EventChannel> &cross_channels)
{
    for (int i = 0; i < entry.wait_count; ++i) {
        auto *channel = LookupChannel(intra_channels, cross_channels, core_id, entry.wait_events[i]);
        if (channel && !channel->signaled())
            return false;
    }
    return true;
}

inline void ConsumeMultiCoreSyncWaits(const PipeEntry &entry, uint32_t core_id,
                                      std::vector<EventChannel> &intra_channels,
                                      std::vector<EventChannel> &cross_channels)
{
    int sync_base = SyncChannelBase();
    for (int i = 0; i < entry.wait_count; ++i) {
        if (entry.wait_events[i] < sync_base)
            continue;
        auto *channel = LookupChannel(intra_channels, cross_channels, core_id, entry.wait_events[i]);
        if (channel)
            channel->count--;
    }
}

inline void TryDispatchMultiCore(CorePipeline &core, PipeQueue &queue, std::vector<EventChannel> &intra_channels,
                                 std::vector<EventChannel> &cross_channels, L2CacheModel &cache, uint64_t now,
                                 StepProgress &progress)
{
    if (!queue.CanAccept() || queue.entries.empty())
        return;
    auto &entry = queue.entries.front();
    if (entry.name == "BARRIER") {
        StartEntry(queue, entry, now, cache, progress);
        return;
    }
    if (entry.wait_count > 0 && entry.state == PipeEntry::WAITING) {
        if (!MultiCoreWaitsReady(entry, core.core_id, intra_channels, cross_channels)) {
            progress.any_active = true;
            return;
        }
        ConsumeMultiCoreSyncWaits(entry, core.core_id, intra_channels, cross_channels);
        entry.state = PipeEntry::IDLE;
    }
    if (entry.state == PipeEntry::IDLE)
        StartEntry(queue, entry, now, cache, progress);
}

inline void CollectStuckEntries(const std::vector<CorePipeline> &core_pipelines, MultiCoreTimeline &timeline)
{
    for (size_t c = 0; c < core_pipelines.size(); ++c) {
        for (auto &queue : core_pipelines[c].queues) {
            for (auto &entry : queue.entries)
                timeline.per_core[c].events.push_back(BuildPipeEvent(entry, queue.label, true));
        }
    }
}

// ── Per-physical-core merge (combines subblock logical cores) ──

inline std::vector<std::vector<MergedEntry>> MergeRecordsPerCore(uint32_t num_phys_cores)
{
    std::vector<std::vector<MergedEntry>> result(num_phys_cores);
    for (uint32_t c = 0; c < num_phys_cores; ++c) {
        result[c] = MergeRecordsForPhysicalCore(c);
    }
    return result;
}

// ── Multi-core FillQueuesFromMerged (handles cross_core offset) ──

inline void FillQueuesFromMerged(std::vector<PipeQueue> &queues, std::vector<EventChannel> & /*intra_channels*/,
                                 std::vector<EventChannel> & /*cross_channels*/, const std::vector<MergedEntry> &merged)
{
    FillQueuesFromMergedImpl(queues, merged, false, true);
}

// ── Event-driven multi-core step simulation ──

inline MultiCoreTimeline StepSimulateMultiCore(std::vector<CorePipeline> &core_pipelines,
                                               std::vector<EventChannel> &intra_channels,
                                               std::vector<EventChannel> &cross_channels, L2CacheModel &cache,
                                               uint32_t max_steps = 500000)
{
    MultiCoreTimeline result;
    result.per_core.resize(core_pipelines.size());
    uint64_t now = 0;

    for (uint32_t step = 0; step < max_steps; ++step) {
        StepProgress progress;
        for (auto &cp : core_pipelines) {
            for (auto &queue : cp.queues)
                CompleteRunningEntry(cp, queue, result.per_core[cp.core_id], intra_channels, cross_channels, now,
                                     progress);
        }

        for (auto &cp : core_pipelines)
            PopDoneEntries(cp.queues);

        for (auto &cp : core_pipelines) {
            for (auto &queue : cp.queues)
                TryDispatchMultiCore(cp, queue, intra_channels, cross_channels, cache, now, progress);
        }

        if (!progress.any_active)
            break;
        if (!progress.dispatched) {
            if (!AnyRunning(core_pipelines))
                break; // true deadlock
        }
        if (progress.next_event != UINT64_MAX && progress.next_event > now)
            now = progress.next_event;
    }

    CollectStuckEntries(core_pipelines, result);

    for (auto &tl : result.per_core) {
        for (auto &ev : tl.events)
            tl.total_cycles = std::max(tl.total_cycles, ev.end_cycle);
        result.total_cycles = std::max(result.total_cycles, tl.total_cycles);
    }
    return result;
}
