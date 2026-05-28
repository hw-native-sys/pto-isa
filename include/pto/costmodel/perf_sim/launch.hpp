/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_PERF_SIM_LAUNCH_HPP
#define PTO_PERF_SIM_LAUNCH_HPP

#include "config.hpp"
#include "recorder.hpp"
#include "reporter.hpp"
#include "tile_dep_tracker.hpp"
#include <pto/common/cpu_stub.hpp>

#define PTO_EXPAND(...) __VA_ARGS__

#ifdef __COSTMODEL

#define LAUNCH_KERNEL(func, targs, launch_cfg, ...)                                            \
    do {                                                                                       \
        ::pto::perf_sim::PtoRecorder::Clear();                                                 \
        ::pto::perf_sim::SyncRecorder::Clear();                                                \
        ::pto::perf_sim::CvSyncRecorder::Clear();                                              \
        ::pto::perf_sim::TileDepTracker::Clear();                                              \
        ::pto::perf_sim::SetLaunchConfig(PTO_EXPAND launch_cfg);                               \
        const uint32_t _blk_dim = ::pto::perf_sim::GetConfig().block_dim;                      \
        constexpr uint32_t _vec_cores = ::pto::perf_sim::VEC_CORES_PER_AIC;                    \
        for (uint32_t _core = 0; _core < _blk_dim; ++_core) {                                  \
            for (uint32_t _sub = 0; _sub < _vec_cores; ++_sub) {                               \
                ::pto::perf_sim::current_subblock_id = _sub;                                   \
                uint32_t _logical = _core * _vec_cores + _sub;                                 \
                ::pto::perf_sim::PtoRecorder::SetActiveCore(_logical);                         \
                ::pto::perf_sim::SyncRecorder::SetActiveCore(_logical);                        \
                ::pto::cpu_sim::ScopedExecutionContext _ctx(_core, _sub, _vec_cores);          \
                func targs(__VA_ARGS__);                                                       \
                ::pto::perf_sim::TileDepTracker::FinishCore();                                 \
            }                                                                                  \
        }                                                                                      \
        auto _r_ = ::pto::perf_sim::PerfSimReporter().Run(#func);                              \
        ::pto::perf_sim::PerfSimReporter::PrintText(_r_);                                      \
        ::pto::perf_sim::PerfSimReporter::WriteSwimlaneJson(                                   \
            ::pto::perf_sim::GetConfig().output_dir + "/" #func ".json", _r_);                 \
        ::pto::perf_sim::PerfSimReporter::WritePipelineSummaryCSV(                             \
            ::pto::perf_sim::GetConfig().output_dir + "/" #func "_pipeline_summary.csv", _r_); \
    } while (0)

#else

#define LAUNCH_KERNEL(func, targs, launch_cfg, ...) func targs<<<PTO_EXPAND launch_cfg>>>(__VA_ARGS__)

#endif

#endif
