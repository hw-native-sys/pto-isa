/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

static void WritePipelineSummaryCSV(const std::string &path, const SimReport &report)
{
    if (!EnsureParentDir(path)) {
        return;
    }

    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "[perf_sim] Cannot write CSV to: " << path << "\n";
        return;
    }

    out << "op_name,core_id,unit,total_cycles,active_start_cycle,active_end_cycle,active_cycles,busy_cycles,"
        << "scalar_cycles,mte2_aic_cycles,mte2_aiv_cycles,mte1_cycles,cube_cycles,fixp_cycles,vec_cycles,"
        << "mte3_cycles\n";

    for (const auto &row : BuildPipelineSummary(report)) {
        out << report.op_name << "," << row.core_id << "," << row.unit << "," << row.total_cycles << ","
            << row.active_start_cycle << "," << row.active_end_cycle << "," << row.active_cycles << ","
            << row.busy_cycles << "," << row.scalar_cycles << "," << row.mte2_aic_cycles << "," << row.mte2_aiv_cycles
            << "," << row.mte1_cycles << "," << row.cube_cycles << "," << row.fixp_cycles << "," << row.vec_cycles
            << "," << row.mte3_cycles << "\n";
    }
    out.close();
}

struct DotNodeInfo {
    int dot_id;
    std::string label;
    std::string color;
    std::string pipe;
    event_t wait_events[8];
    int wait_count;
    bool stuck;
    bool is_sync;
};

static const char *DotPipeColor(const std::string &pipe)
{
    if (pipe == "Scalar")
        return "\"#D5D5D5\"";
    if (pipe == "MTE2_AIV")
        return "\"#ADD8E6\"";
    if (pipe == "MTE2_AIC")
        return "\"#87CEEB\"";
    if (pipe == "MTE1")
        return "\"#00CED1\"";
    if (pipe == "MTE3")
        return "\"#FFA500\"";
    if (pipe == "VEC")
        return "\"#90EE90\"";
    if (pipe == "CUBE")
        return "\"#FFD700\"";
    if (pipe == "FIXP")
        return "\"#FFB6C1\"";
    return "white";
}

static const char *DotGroupColor(const std::string &pipe)
{
    if (pipe == "Scalar")
        return "#EEEEEE";
    if (pipe == "MTE2_AIV")
        return "#E8F4F8";
    if (pipe == "MTE2_AIC")
        return "#E0F0F8";
    if (pipe == "MTE1")
        return "#E0FFFF";
    if (pipe == "MTE3")
        return "#FFF0E0";
    if (pipe == "VEC")
        return "#E8FFE8";
    if (pipe == "CUBE")
        return "#FFFFF0";
    if (pipe == "FIXP")
        return "#FFF0F5";
    return "#F0F0F0";
}

static void AddDotEvents(const SimReport &report, int pid, const std::vector<PipeEvent> &events,
                         std::vector<DotNodeInfo> &nodes, std::unordered_map<event_t, int> &signal_to_node, int &dot_id)
{
    for (auto &ev : events) {
        if (ev.end_cycle == ev.start_cycle && ev.pipe_label == "Scalar" && !ev.is_sync)
            continue;

        int idx = static_cast<int>(nodes.size());
        DotNodeInfo node;
        node.dot_id = dot_id++;
        std::string core_prefix = (report.num_cores > 1) ? "C" + std::to_string(pid) + " " : "";
        node.label =
            core_prefix + ev.name + "\\n" + std::to_string(ev.start_cycle) + "-" + std::to_string(ev.end_cycle);
        node.color = ev.stuck ? "\"#FF4444\"" : DotPipeColor(ev.pipe_label);
        node.pipe = ev.pipe_label;
        node.wait_count = ev.wait_count;
        for (int i = 0; i < ev.wait_count; ++i)
            node.wait_events[i] = ev.wait_events[i];
        node.stuck = ev.stuck;
        node.is_sync = ev.is_sync;

        nodes.push_back(std::move(node));
        if (ev.signal_event >= 0)
            signal_to_node[ev.signal_event] = idx;
    }
}

static std::vector<DotNodeInfo> BuildDotNodes(const SimReport &report, std::unordered_map<event_t, int> &signal_to_node)
{
    std::vector<DotNodeInfo> nodes;
    int dot_id = 0;
    if (report.num_cores == 1) {
        AddDotEvents(report, 0, report.timeline.events, nodes, signal_to_node, dot_id);
        return nodes;
    }
    for (uint32_t c = 0; c < report.num_cores; ++c) {
        AddDotEvents(report, static_cast<int>(c), report.multi_timeline.per_core[c].events, nodes, signal_to_node,
                     dot_id);
    }
    return nodes;
}

static std::unordered_map<std::string, std::vector<int>> BuildDotPipeGroups(const std::vector<DotNodeInfo> &nodes)
{
    std::unordered_map<std::string, std::vector<int>> pipe_groups;
    for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
        pipe_groups[nodes[i].pipe].push_back(i);
    }
    return pipe_groups;
}

static void WriteDotNode(std::ostream &out, const DotNodeInfo &node)
{
    out << "    n" << node.dot_id << " [label=\"" << node.label << "\", fillcolor=" << node.color;
    if (node.is_sync)
        out << ", shape=ellipse, width=0.6, height=0.3";
    if (node.stuck)
        out << ", fontcolor=\"white\", penwidth=2";
    out << "];\n";
}

static void WriteDotGroups(std::ostream &out, const std::vector<DotNodeInfo> &nodes)
{
    for (auto &[pipe_name, indices] : BuildDotPipeGroups(nodes)) {
        out << "  subgraph \"cluster_" << pipe_name << "\" {\n";
        out << "    label=\"" << pipe_name << "\";\n";
        out << "    style=filled;\n";
        out << "    color=\"" << DotGroupColor(pipe_name) << "\";\n";
        for (int idx : indices) {
            WriteDotNode(out, nodes[idx]);
        }
        out << "  }\n\n";
    }
}

static void WriteDotEdges(std::ostream &out, const std::vector<DotNodeInfo> &nodes,
                          const std::unordered_map<event_t, int> &signal_to_node)
{
    for (auto &node : nodes) {
        for (int i = 0; i < node.wait_count; ++i) {
            auto it = signal_to_node.find(node.wait_events[i]);
            if (it != signal_to_node.end()) {
                out << "  n" << nodes[it->second].dot_id << " -> n" << node.dot_id << ";\n";
            }
        }
    }
}

static void WriteDotLegend(std::ostream &out)
{
    out << "\n  subgraph cluster_legend {\n";
    out << "    label=\"Legend\";\n";
    out << "    style=filled;\n";
    out << "    color=\"white\";\n";
    out << "    node [shape=plaintext, style=\"\"];\n";
    out << "    leg [label=<<TABLE BORDER=\"0\" CELLBORDER=\"0\">"
        << "<TR><TD BGCOLOR=\"#90EE90\">VEC</TD>"
        << "<TD BGCOLOR=\"#FFD700\">CUBE</TD>"
        << "<TD BGCOLOR=\"#ADD8E6\">MTE2_AIV</TD>"
        << "<TD BGCOLOR=\"#87CEEB\">MTE2_AIC</TD>"
        << "<TD BGCOLOR=\"#00CED1\">MTE1</TD>"
        << "<TD BGCOLOR=\"#FFA500\">MTE3</TD>"
        << "<TD BGCOLOR=\"#FFB6C1\">FIXP</TD>"
        << "<TD BGCOLOR=\"#FF4444\"><FONT COLOR=\"white\">STUCK</FONT></TD>"
        << "</TR></TABLE>>];\n";
    out << "  }\n";
}

static void WriteDependencyDOT(const std::string &path, const SimReport &report)
{
    if (!EnsureParentDir(path)) {
        return;
    }

    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "[perf_sim] Cannot write DOT to: " << path << "\n";
        return;
    }

    out << "digraph pipeline {\n";
    out << "  rankdir=LR;\n";
    out << "  node [shape=box, style=filled, fontsize=8, "
        << "fontname=\"Courier\"];\n";
    out << "  edge [color=\"#888888\", arrowsize=0.6];\n\n";

    std::unordered_map<event_t, int> signal_to_node;
    auto nodes = BuildDotNodes(report, signal_to_node);
    WriteDotGroups(out, nodes);
    WriteDotEdges(out, nodes, signal_to_node);
    WriteDotLegend(out);

    out << "}\n";
    out.close();
}

// ── Sync Edge Audit: count SIGNAL/WAIT per channel, check balance ──

static const char *HwPipeLabel(int hw_pipe)
{
    static const char *names[] = {"S", "VEC", "MTE1", "MTE2", "MTE3", "CUBE", "ALL", "FIXP"};
    return (hw_pipe >= 0 && hw_pipe < 8) ? names[hw_pipe] : "?";
}

struct EdgeStat {
    std::unordered_map<std::string, int> src_counts; // src_pipe -> signal count
    int hw_dst = -1;
    int evt_id = -1;
    int sig = 0;
    int wait = 0;
};

static void DecodeSyncChannel(event_t channel, int sync_base, EdgeStat &stat)
{
    int off = channel - sync_base;
    stat.hw_dst = off / EVENTS_PER_PIPE;
    stat.evt_id = off % EVENTS_PER_PIPE;
}

static void AddSyncSignal(std::unordered_map<event_t, EdgeStat> &stats, event_t channel, int sync_base,
                          const std::string &pipe_label)
{
    if (channel < sync_base) {
        return;
    }
    auto &stat = stats[channel];
    stat.src_counts[pipe_label]++;
    DecodeSyncChannel(channel, sync_base, stat);
    stat.sig++;
}

static void AddSyncWait(std::unordered_map<event_t, EdgeStat> &stats, event_t channel, int sync_base)
{
    if (channel < sync_base) {
        return;
    }
    auto &stat = stats[channel];
    DecodeSyncChannel(channel, sync_base, stat);
    stat.wait++;
}

static std::string BuildSourceSummary(const std::unordered_map<std::string, int> &src_counts)
{
    std::string src_str;
    std::vector<std::pair<std::string, int>> src_sorted(src_counts.begin(), src_counts.end());
    std::sort(src_sorted.begin(), src_sorted.end(), [](const auto &a, const auto &b) { return a.second > b.second; });
    for (size_t i = 0; i < src_sorted.size(); ++i) {
        if (i > 0) {
            src_str += " + ";
        }
        src_str += src_sorted[i].first + ":" + std::to_string(src_sorted[i].second);
    }
    return src_str;
}

static void PrintSyncEdgeAudit(const std::vector<PipeEvent> &events, std::ostream &os)
{
    int sync_base = SyncChannelBase();
    std::unordered_map<event_t, EdgeStat> stats;

    for (auto &ev : events) {
        bool is_sig = (ev.name.find("SIGNAL(") == 0);
        bool is_wait_ev = (ev.name.find("WAIT(") == 0);

        // Standalone SIGNAL
        if (is_sig && ev.signal_event >= sync_base) {
            AddSyncSignal(stats, ev.signal_event, sync_base, ev.pipe_label);
        }
        // Standalone WAIT
        if (is_wait_ev) {
            for (int i = 0; i < ev.wait_count; ++i) {
                AddSyncWait(stats, ev.wait_events[i], sync_base);
            }
        }
        // Inlined extra_signals (SIGNALs merged into instructions)
        for (int i = 0; i < ev.extra_signal_count; ++i) {
            AddSyncSignal(stats, ev.extra_signals[i], sync_base, ev.pipe_label);
        }
        // Sync waits in regular instructions
        if (!is_sig && !is_wait_ev) {
            for (int i = 0; i < ev.wait_count; ++i) {
                AddSyncWait(stats, ev.wait_events[i], sync_base);
            }
        }
    }

    // Sort by channel key
    std::vector<std::pair<event_t, EdgeStat>> sorted(stats.begin(), stats.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) { return a.first < b.first; });

    os << "\n-- Sync Edge Audit --\n";
    os << "  #   Ch    -> Dst   Flag  SIG  WAIT  OK  Sources\n";
    int n = 0;
    int tot_sig = 0, tot_wait = 0, unbal = 0;
    for (auto &[ch, st] : sorted) {
        bool ok = (st.sig == st.wait);
        if (!ok)
            unbal++;
        tot_sig += st.sig;
        tot_wait += st.wait;

        os << "  " << std::setw(2) << ++n << "  " << std::setw(3) << ch << "   -> " << std::left << std::setw(4)
           << HwPipeLabel(st.hw_dst) << std::right << "   " << st.evt_id << "    " << std::setw(3) << st.sig << "  "
           << std::setw(4) << st.wait << "   " << (ok ? "OK" : "!!") << "   " << BuildSourceSummary(st.src_counts)
           << "\n";
    }
    os << "  Total: " << tot_sig << " SIGNAL, " << tot_wait << " WAIT, " << unbal << " unbalanced\n";
}

private:
static void PrintCoreDetail(const PipeTimeline &timeline, uint64_t total_cycles, uint32_t core_id, std::ostream &os)
{
    // Collect pipe stats by label
    std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> pipe_stats;
    for (auto &ev : timeline.events) {
        auto &[busy, count] = pipe_stats[ev.pipe_label];
        busy += ev.stuck ? ev.duration : (ev.end_cycle - ev.start_cycle);
        count++;
    }

    // Print in fixed track order (AIC → AIV-0 → AIV-1)
    auto tracks = GetTracksForCore(core_id);
    os << "Pipe utilization:\n";
    for (auto &t : tracks) {
        auto it = pipe_stats.find(t.tid);
        if (it == pipe_stats.end())
            continue;
        auto &[busy, count] = it->second;
        double util = total_cycles > 0 ? 100.0 * busy / total_cycles : 0.0;
        os << "  " << t.display_name << " : " << count << " ops, busy " << busy << " cycles (" << util << "%)\n";
    }
    // Print any pipes not in tracks (e.g., Scalar)
    for (auto &[name, stats] : pipe_stats) {
        bool found = false;
        for (auto &t : tracks)
            if (t.tid == name) {
                found = true;
                break;
            }
        if (found)
            continue;
        auto &[busy, count] = stats;
        double util = total_cycles > 0 ? 100.0 * busy / total_cycles : 0.0;
        os << "  " << name << " : " << count << " ops, busy " << busy << " cycles (" << util << "%)\n";
    }

    os << "\nTimeline:\n";
    for (auto &ev : timeline.events) {
        os << "  [" << ev.pipe_label << "] " << ev.name << "  " << ev.start_cycle << " - " << ev.end_cycle << " ("
           << (ev.stuck ? ev.duration : (ev.end_cycle - ev.start_cycle)) << " cycles)\n";
    }
    os << "\n";
}
}
;
