/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

using LabelMap = std::unordered_map<int, std::unordered_map<std::string, std::string>>;

struct FlowSrc {
    int pid;
    std::string tid;
    std::string name;
    uint64_t ts;
};

using SignalMap = std::unordered_map<int64_t, std::vector<FlowSrc>>;

static int64_t JsonFlowKey(int pid, event_t ev)
{
    return static_cast<int64_t>(pid) * 1000000LL + ev;
}

static LabelMap BuildLabelMaps(uint32_t num_cores)
{
    LabelMap label_maps;
    for (uint32_t c = 0; c < num_cores; ++c) {
        auto &labels = label_maps[static_cast<int>(c)];
        for (auto &t : GetTracksForCore(c)) {
            labels[t.tid] = t.display_name;
        }
    }
    return label_maps;
}

static std::string DisplayTid(const LabelMap &label_maps, int pid, const std::string &label)
{
    auto core_it = label_maps.find(pid);
    if (core_it == label_maps.end()) {
        return label;
    }
    auto label_it = core_it->second.find(label);
    return label_it == core_it->second.end() ? label : label_it->second;
}

static void CollectJsonSignals(const LabelMap &label_maps, SignalMap &signal_map, int pid,
                               const std::vector<PipeEvent> &events)
{
    auto add_signal = [&](event_t signal, const PipeEvent &ev) {
        if (signal >= 0) {
            signal_map[JsonFlowKey(pid, signal)].push_back(
                {pid, DisplayTid(label_maps, pid, ev.pipe_label), ev.name, ev.end_cycle});
        }
    };
    for (auto &ev : events) {
        add_signal(ev.signal_event, ev);
        for (int i = 0; i < ev.extra_signal_count; ++i) {
            add_signal(ev.extra_signals[i], ev);
        }
    }
}

static SignalMap BuildJsonSignalMap(const SimReport &report, const LabelMap &label_maps)
{
    SignalMap signal_map;
    if (report.num_cores == 1) {
        CollectJsonSignals(label_maps, signal_map, 0, report.timeline.events);
        return signal_map;
    }
    for (uint32_t c = 0; c < report.num_cores; ++c) {
        CollectJsonSignals(label_maps, signal_map, static_cast<int>(c), report.multi_timeline.per_core[c].events);
    }
    return signal_map;
}

static std::unordered_map<std::string, int> BuildJsonSortMap(int pid)
{
    std::unordered_map<std::string, int> sort_map;
    auto tracks = GetTracksForCore(static_cast<uint32_t>(pid));
    for (int i = 0; i < static_cast<int>(tracks.size()); ++i) {
        sort_map[tracks[i].tid] = i;
    }
    return sort_map;
}

static const FlowSrc *FindJsonFlowSource(const SignalMap &signal_map, int pid, event_t wait, uint64_t wait_start)
{
    auto it = signal_map.find(JsonFlowKey(pid, wait));
    if (it == signal_map.end()) {
        return nullptr;
    }
    const FlowSrc *best = nullptr;
    for (const auto &src : it->second) {
        bool scalar_meta = src.name.find("TASSIGN(") == 0 || src.name.find("TRESHAPE(") == 0 ||
                           src.name.find("TALLOC(") == 0 || src.name.find("TFREE(") == 0;
        if (src.ts <= wait_start && !scalar_meta && (best == nullptr || src.ts > best->ts)) {
            best = &src;
        }
    }
    return best;
}

static void WriteJsonFlow(std::ostream &out, const FlowSrc &src, int pid, const std::string &dst_tid,
                          const PipeEvent &ev, int64_t flow_id)
{
    out << ",\n";
    out << "  {\"name\": \"dep\", \"cat\": \"flow\", \"ph\": \"s\","
        << "\"id\": " << flow_id << ","
        << "\"pid\": " << src.pid << ","
        << "\"tid\": \"" << src.tid << "\","
        << "\"ts\": " << src.ts << ","
        << "\"args\":{\"src\":\"" << src.name << "\"}}";
    out << ",\n";
    out << "  {\"name\": \"dep\", \"cat\": \"flow\", \"ph\": \"f\","
        << "\"id\": " << flow_id << ","
        << "\"pid\": " << pid << ","
        << "\"tid\": \"" << dst_tid << "\","
        << "\"ts\": " << ev.start_cycle << ","
        << "\"args\":{\"dst\":\"" << ev.name << "\"}}";
}

static void WriteJsonEvent(std::ostream &out, const LabelMap &label_maps, const SignalMap &signal_map, int pid,
                           const PipeEvent &ev, bool &first, int64_t &flow_id)
{
    if ((ev.end_cycle == ev.start_cycle && ev.is_sync) ||
        (ev.end_cycle == ev.start_cycle && ev.pipe_label == "Scalar")) {
        return;
    }
    std::string dst_tid = DisplayTid(label_maps, pid, ev.pipe_label);
    if (!first) {
        out << ",\n";
    }
    first = false;
    out << "  {\"name\": \"" << ev.name << "\","
        << "\"cat\": \"" << dst_tid << "\","
        << "\"ph\": \"X\","
        << "\"ts\": " << ev.start_cycle << ","
        << "\"dur\": " << (ev.stuck ? ev.duration : (ev.end_cycle - ev.start_cycle)) << ","
        << "\"pid\": " << pid << ","
        << "\"tid\": \"" << dst_tid << "\","
        << "\"args\":{\"signal_event\":" << ev.signal_event << ",\"wait_events\":\"";
    for (int wi = 0; wi < ev.wait_count; ++wi) {
        out << (wi > 0 ? ";" : "") << ev.wait_events[wi];
    }
    out << "\"}}";
    for (int i = 0; i < ev.wait_count; ++i) {
        auto *src = FindJsonFlowSource(signal_map, pid, ev.wait_events[i], ev.start_cycle);
        if (src != nullptr && src->tid != dst_tid) {
            WriteJsonFlow(out, *src, pid, dst_tid, ev, flow_id++);
        }
    }
}

static void WriteSortedJsonEvents(std::ostream &out, const LabelMap &label_maps, const SignalMap &signal_map, int pid,
                                  const std::vector<PipeEvent> &events, bool &first, int64_t &flow_id)
{
    auto sort_map = BuildJsonSortMap(pid);
    std::vector<size_t> idx(events.size());
    for (size_t i = 0; i < idx.size(); ++i) {
        idx[i] = i;
    }
    std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        if (events[a].start_cycle != events[b].start_cycle) {
            return events[a].start_cycle < events[b].start_cycle;
        }
        int sa = sort_map.count(events[a].pipe_label) ? sort_map[events[a].pipe_label] : 99;
        int sb = sort_map.count(events[b].pipe_label) ? sort_map[events[b].pipe_label] : 99;
        return sa < sb;
    });
    for (auto i : idx) {
        WriteJsonEvent(out, label_maps, signal_map, pid, events[i], first, flow_id);
    }
}

static void WriteJsonEvents(std::ostream &out, const SimReport &report, const LabelMap &label_maps,
                            const SignalMap &signal_map)
{
    bool first = false;
    int64_t flow_id = 1;
    if (report.num_cores == 1) {
        WriteSortedJsonEvents(out, label_maps, signal_map, 0, report.timeline.events, first, flow_id);
        return;
    }
    for (uint32_t c = 0; c < report.num_cores; ++c) {
        WriteSortedJsonEvents(out, label_maps, signal_map, static_cast<int>(c),
                              report.multi_timeline.per_core[c].events, first, flow_id);
    }
}

static void WriteJsonTrackMeta(std::ostream &out, int pid)
{
    int sort_idx = 0;
    for (auto &t : GetTracksForCore(static_cast<uint32_t>(pid))) {
        out << ",\n";
        out << "  {\"ph\":\"M\",\"pid\":" << pid << ",\"tid\":\"" << t.display_name << "\",\"name\":\"thread_name\""
            << ",\"args\":{\"name\":\"" << t.display_name << "\"}}";
        out << ",\n";
        out << "  {\"ph\":\"M\",\"pid\":" << pid << ",\"tid\":\"" << t.display_name
            << "\",\"name\":\"thread_sort_index\""
            << ",\"args\":{\"sort_index\":" << sort_idx++ << "}}";
    }
}

static void WriteJsonProcessMeta(std::ostream &out, uint32_t core_id, bool emit_leading_comma)
{
    if (emit_leading_comma) {
        out << ",";
    }
    out << "\n  {\"ph\":\"M\",\"pid\":" << core_id << ",\"tid\":0,\"name\":\"process_name\""
        << ",\"args\":{\"name\":\"Core " << core_id << "\"}}";
    out << ",\n  {\"ph\":\"M\",\"pid\":" << core_id << ",\"tid\":0,\"name\":\"process_sort_index\""
        << ",\"args\":{\"sort_index\":" << core_id << "}}";
    WriteJsonTrackMeta(out, static_cast<int>(core_id));
}

static void WriteJsonMetadata(std::ostream &out, const SimReport &report)
{
    if (report.num_cores == 1) {
        out << "  {\"ph\":\"M\",\"pid\":0,\"tid\":0,"
            << "\"name\":\"process_name\",\"args\":{\"name\":\"Core 0\"}}";
        out << ",\n  {\"ph\":\"M\",\"pid\":0,\"tid\":0,"
            << "\"name\":\"process_sort_index\",\"args\":{\"sort_index\":0}}";
        WriteJsonTrackMeta(out, 0);
        return;
    }
    for (uint32_t c = 0; c < report.num_cores; ++c) {
        WriteJsonProcessMeta(out, c, c > 0);
    }
}

static void WriteSwimlaneJson(const std::string &path, const SimReport &report)
{
    if (!EnsureParentDir(path)) {
        return;
    }

    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "[perf_sim] Cannot write JSON to: " << path << "\n";
        return;
    }

    out << "[\n";
    auto label_maps = BuildLabelMaps(report.num_cores);
    auto signal_map = BuildJsonSignalMap(report, label_maps);
    WriteJsonMetadata(out, report);
    WriteJsonEvents(out, report, label_maps, signal_map);
    out << "\n]\n";
    out.close();
}
