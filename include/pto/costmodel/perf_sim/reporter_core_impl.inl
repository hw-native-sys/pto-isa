/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// ── PerfSimReporter: run simulation, print text, write JSON ──

class PerfSimReporter {
public:
    // Fixed track layout for Chrome Trace (1AIC + 2AIV per core)
    struct TrackInfo {
        const char *tid;          // queue label (must match PipeQueue label)
        std::string display_name; // [AIC-{core_id}] PIPE or [AIV-{sub}] PIPE
        const char *group;        // AIC / AIV-0 / AIV-1
    };

    static std::vector<TrackInfo> GetTracksForCore(uint32_t core_id)
    {
        std::string aic = "[AIC-" + std::to_string(core_id) + "] ";
        std::string aiv0 = "[AIC-" + std::to_string(core_id) + "/AIV-0] ";
        std::string aiv1 = "[AIC-" + std::to_string(core_id) + "/AIV-1] ";
        return {
            {"Scalar", aic + "Scalar", "AIC"},      {"MTE2_AIC", aic + "MTE2", "AIC"},
            {"MTE1", aic + "MTE1", "AIC"},          {"CUBE", aic + "CUBE", "AIC"},
            {"FIXP", aic + "FIXP", "AIC"},          {"MTE2_AIV", aiv0 + "MTE2", "AIV-0"},
            {"VEC", aiv0 + "VEC", "AIV-0"},         {"MTE3", aiv0 + "MTE3", "AIV-0"},
            {"MTE2_AIV_1", aiv1 + "MTE2", "AIV-1"}, {"VEC_1", aiv1 + "VEC", "AIV-1"},
            {"MTE3_1", aiv1 + "MTE3", "AIV-1"},
        };
    }

    struct PipelineSummaryRow {
        uint32_t core_id = 0;
        std::string unit;
        uint64_t total_cycles = 0;
        uint64_t active_start_cycle = 0;
        uint64_t active_end_cycle = 0;
        uint64_t active_cycles = 0;
        uint64_t busy_cycles = 0;
        uint64_t scalar_cycles = 0;
        uint64_t mte2_aic_cycles = 0;
        uint64_t mte2_aiv_cycles = 0;
        uint64_t mte1_cycles = 0;
        uint64_t cube_cycles = 0;
        uint64_t fixp_cycles = 0;
        uint64_t vec_cycles = 0;
        uint64_t mte3_cycles = 0;
    };

    static uint64_t EventBusyCycles(const PipeEvent &event)
    {
        return event.stuck ? event.duration : (event.end_cycle - event.start_cycle);
    }

    static bool EnsureParentDir(const std::string &path)
    {
        std::filesystem::path output_path(path);
        auto parent = output_path.parent_path();
        if (parent.empty()) {
            return true;
        }

        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            std::cerr << "[perf_sim] Cannot create output directory: " << parent.string() << "\n";
            return false;
        }
        return true;
    }

    static PipelineSummaryRow BuildPipelineSummaryRow(const PipeTimeline &timeline, uint32_t core_id,
                                                      const std::string &unit, const std::vector<std::string> &pipes)
    {
        std::unordered_map<std::string, uint64_t> busy_cycles;
        uint64_t active_start = std::numeric_limits<uint64_t>::max();
        uint64_t active_end = 0;
        for (auto &ev : timeline.events) {
            if (std::find(pipes.begin(), pipes.end(), ev.pipe_label) == pipes.end())
                continue;
            uint64_t busy = EventBusyCycles(ev);
            busy_cycles[ev.pipe_label] += busy;
            if (busy > 0) {
                active_start = std::min(active_start, ev.start_cycle);
                active_end = std::max(active_end, ev.end_cycle);
            }
        }

        auto get = [&](const char *pipe) -> uint64_t {
            auto it = busy_cycles.find(pipe);
            return it == busy_cycles.end() ? 0 : it->second;
        };

        PipelineSummaryRow row;
        row.core_id = core_id;
        row.unit = unit;
        row.total_cycles = timeline.total_cycles;
        row.active_start_cycle = active_start == std::numeric_limits<uint64_t>::max() ? 0 : active_start;
        row.active_end_cycle = active_end;
        row.active_cycles = active_end > row.active_start_cycle ? active_end - row.active_start_cycle : 0;
        row.scalar_cycles = get("Scalar");
        row.mte2_aic_cycles = get("MTE2_AIC");
        row.mte2_aiv_cycles = get("MTE2_AIV") + get("MTE2_AIV_1");
        row.mte1_cycles = get("MTE1");
        row.cube_cycles = get("CUBE");
        row.fixp_cycles = get("FIXP");
        row.vec_cycles = get("VEC") + get("VEC_1");
        row.mte3_cycles = get("MTE3") + get("MTE3_1");
        row.busy_cycles = row.scalar_cycles + row.mte2_aic_cycles + row.mte2_aiv_cycles + row.mte1_cycles +
                          row.cube_cycles + row.fixp_cycles + row.vec_cycles + row.mte3_cycles;
        return row;
    }

    static std::vector<PipelineSummaryRow> BuildPipelineSummary(const SimReport &report)
    {
        std::vector<PipelineSummaryRow> rows;
        rows.reserve(report.num_cores * 3);
        auto append_core = [&](const PipeTimeline &timeline, uint32_t core_id) {
            rows.push_back(
                BuildPipelineSummaryRow(timeline, core_id, "AIC", {"Scalar", "MTE2_AIC", "MTE1", "CUBE", "FIXP"}));
            rows.push_back(BuildPipelineSummaryRow(timeline, core_id, "AIV0", {"MTE2_AIV", "VEC", "MTE3"}));
            rows.push_back(BuildPipelineSummaryRow(timeline, core_id, "AIV1", {"MTE2_AIV_1", "VEC_1", "MTE3_1"}));
        };
        if (report.num_cores == 1)
            append_core(report.timeline, 0);
        else
            for (uint32_t c = 0; c < report.num_cores; ++c)
                append_core(report.multi_timeline.per_core[c], c);
        return rows;
    }

    static void AccumulateLogicalCoreStats(SimReport &report, uint32_t logical_cores)
    {
        for (uint32_t lc = 0; lc < logical_cores; ++lc) {
            report.instr_count += PtoRecorder::GetForCore(lc).size();
            report.sync_count += SyncRecorder::GetForCore(lc).size();
        }
    }

    static void DropCrossCoreSync(std::vector<MergedEntry> &merged)
    {
        merged.erase(std::remove_if(merged.begin(), merged.end(),
                                    [](const MergedEntry &e) { return e.is_sync && e.sync.cross_core; }),
                     merged.end());
    }

    static void SetCacheStats(SimReport &report, const L2CacheModel &cache)
    {
        report.cache_hits = cache.GetStats().hits;
        report.cache_misses = cache.GetStats().misses;
        report.cache_hit_rate = cache.GetStats().HitRate();
    }

    static void RunSingleCoreSimulation(SimReport &report)
    {
        report.num_cores = 1;
        AccumulateLogicalCoreStats(report, VEC_CORES_PER_AIC);

        auto merged = MergeRecordsForPhysicalCore(0);
        DropCrossCoreSync(merged);
        InlineSyncIntoMerged(merged);

        auto queues = MakePipeQueues();
        std::vector<EventChannel> channels(TotalChannels());
        FillQueuesFromMerged(queues, channels, merged);

        L2CacheModel cache;
        report.timeline = StepSimulate(queues, channels, cache);
        SetCacheStats(report, cache);
    }

    static std::vector<CorePipeline> BuildCorePipelines(uint32_t num_cores)
    {
        std::vector<CorePipeline> core_pipelines(num_cores);
        for (uint32_t c = 0; c < num_cores; ++c) {
            core_pipelines[c].core_id = c;
            core_pipelines[c].queues = MakePipeQueues();
        }
        return core_pipelines;
    }

    static void PrepareCoreMergedQueues(std::vector<CorePipeline> &core_pipelines,
                                        std::vector<std::vector<MergedEntry>> &per_core_merged,
                                        std::vector<EventChannel> &intra_channels,
                                        std::vector<EventChannel> &cross_channels)
    {
        for (uint32_t c = 0; c < core_pipelines.size(); ++c) {
            DropCrossCoreSync(per_core_merged[c]);
            InlineSyncIntoMerged(per_core_merged[c]);
            FillQueuesFromMerged(core_pipelines[c].queues, intra_channels, cross_channels, per_core_merged[c]);
        }
    }

    static void RunMultiCoreSimulation(SimReport &report, uint32_t num_cores, uint32_t cross_core_channel_count)
    {
        report.num_cores = num_cores;
        AccumulateLogicalCoreStats(report, num_cores * VEC_CORES_PER_AIC);

        auto per_core_merged = MergeRecordsPerCore(num_cores);
        auto core_pipelines = BuildCorePipelines(num_cores);
        std::vector<EventChannel> intra_channels(num_cores * EVENTS_PER_CORE);
        std::vector<EventChannel> cross_channels(cross_core_channel_count);

        PrepareCoreMergedQueues(core_pipelines, per_core_merged, intra_channels, cross_channels);

        L2CacheModel cache;
        report.multi_timeline = StepSimulateMultiCore(core_pipelines, intra_channels, cross_channels, cache);
        SetCacheStats(report, cache);
    }

    SimReport Run(const std::string &op_name)
    {
        SimReport report;
        report.op_name = op_name;
        auto &cfg = GetConfig();
        if (cfg.block_dim <= 1) {
            RunSingleCoreSimulation(report);
        } else {
            RunMultiCoreSimulation(report, cfg.block_dim, cfg.cross_core_channel_count);
        }
        return report;
    }

    // ── Pipeline table: rows=core, cols=pipe, cell=busy cycles ──
    static void PrintPipelineTable(const SimReport &report, std::ostream &os)
    {
        // Fixed column order for all pipelines across all cores
        const char *const pipe_names[] = {"Scalar", "MTE2(AIC)", "MTE1", "CUBE", "FIXP", "MTE2(AIV)", "VEC", "MTE3"};
        constexpr int kPipes = 8;

        auto rows = BuildPipelineSummary(report);

        // Header row
        os << "Pipeline     |";
        for (uint32_t c = 0; c < report.num_cores; ++c)
            os << " AIC-" << c << "   |";
        os << "\n";
        os << "-------------+";
        for (uint32_t c = 0; c < report.num_cores; ++c)
            os << "--------+";
        os << "\n";

        // Data rows
        for (int p = 0; p < kPipes; ++p) {
            os << std::setw(12) << pipe_names[p] << " |";
            for (uint32_t c = 0; c < report.num_cores; ++c) {
                const auto &aic = rows[c * 3];
                const auto &aiv0 = rows[c * 3 + 1];
                const auto &aiv1 = rows[c * 3 + 2];
                const uint64_t values[kPipes] = {
                    aic.scalar_cycles,
                    aic.mte2_aic_cycles,
                    aic.mte1_cycles,
                    aic.cube_cycles,
                    aic.fixp_cycles,
                    aiv0.mte2_aiv_cycles + aiv1.mte2_aiv_cycles,
                    aiv0.vec_cycles + aiv1.vec_cycles,
                    aiv0.mte3_cycles + aiv1.mte3_cycles,
                };
                uint64_t v = values[p];
                if (v == 0) {
                    os << "       - |";
                } else {
                    os << std::setw(7) << v << " |";
                }
            }
            os << "\n";
        }
        os << "\n";
        os << "Note: AIV pipeline rows sum AIV0 and AIV1. Use pipeline_summary.csv for per-AIV active/busy cycles.\n\n";
    }

    static void PrintText(const SimReport &report, std::ostream &os = std::cout)
    {
        os << "===== Perf-Sim Report: " << report.op_name << " =====\n";
        os << "Cores        : " << report.num_cores << "\n";
        os << "Instructions : " << report.instr_count << "\n";
        os << "Sync events  : " << report.sync_count << "\n";

        uint64_t total_cycles =
            (report.num_cores == 1) ? report.timeline.total_cycles : report.multi_timeline.total_cycles;
        os << "Total cycles : " << total_cycles << "\n";
        os << "L2 Cache hits: " << report.cache_hits << "  misses: " << report.cache_misses
           << "  hit rate: " << report.cache_hit_rate << "%\n\n";

        PrintPipelineTable(report, os);

        os << "===== End Report =====\n";
    }
