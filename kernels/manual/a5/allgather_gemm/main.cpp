/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include "securec.h"
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <vector>
#include <string>
#include <iostream>

#include "ready_queue.hpp"

#ifdef DT_UNDEFINED
#define DT_UNDEFINED_SAVED DT_UNDEFINED
#undef DT_UNDEFINED
#endif
#include "test_common.h"
#ifdef DT_UNDEFINED_SAVED
#define DT_UNDEFINED DT_UNDEFINED_SAVED
#undef DT_UNDEFINED_SAVED
#endif

#ifdef AICORE
#undef AICORE
#endif
#define AICORE

#ifndef __gm__
#define __gm__
#endif

#include "common.hpp"

// ============================================================================
// Compile-time configuration
// ============================================================================
#include "gemm_config.hpp"

#ifndef CONFIG_ORIG_M
#define CONFIG_ORIG_M CONFIG_G_M
#endif
#ifndef CONFIG_ORIG_K
#define CONFIG_ORIG_K CONFIG_G_K
#endif
#ifndef CONFIG_ORIG_N
#define CONFIG_ORIG_N CONFIG_G_N
#endif

constexpr uint32_t ORIG_M = CONFIG_ORIG_M;
constexpr uint32_t ORIG_K = CONFIG_ORIG_K;
constexpr uint32_t ORIG_N = CONFIG_ORIG_N;

#ifndef CONFIG_COMPUTE_BLOCK_NUM
#define CONFIG_COMPUTE_BLOCK_NUM 20
#endif
#ifndef CONFIG_COMM_BLOCK_NUM
#define CONFIG_COMM_BLOCK_NUM 4
#endif
constexpr int COMPUTE_BLOCK_NUM = CONFIG_COMPUTE_BLOCK_NUM;
constexpr int COMM_BLOCK_NUM = CONFIG_COMM_BLOCK_NUM;
constexpr int WARMUP_ITERS = 5;
constexpr int MEASURE_ITERS = 10;
constexpr int COMPUTE_ONLY_ITERS = 5;
constexpr size_t WINDOW_GUARD_BYTES = 4096;

#include "kernel_launch.hpp"

// ============================================================================
// Per-rank resource context
// ============================================================================
struct RankResources {
    TestContext hcclTestCtx;
    aclrtStream computeStream = nullptr;
    aclrtStream commStream = nullptr;

    void *shmem_input = nullptr;
    void *chunk_flag_shmem = nullptr;
    ChunkFlagMatrix *chunk_flag_host = nullptr;
    int32_t *summary_host = nullptr;

    void *src1_dev = nullptr;
    void *output_dev = nullptr;

    size_t inputShmemBytes = 0;
    size_t chunkFlagWithSummarySize = 0;
    size_t chunkFlagMatrixSize = 0;
    size_t bSize = 0;
    size_t outputSize = 0;
    size_t aLocalSize = 0;

    int n_ranks = 0;
    int rank_id = 0;
    int numTilesPerSrc = 0;
    int optimalChunkSize = 0;
    int numChunksPerSrc = 0;

    std::vector<uint16_t> fullInputHost;
};

struct BenchmarkSamples {
    std::vector<double> compute_times_us;
    std::vector<double> sequential_times_us;
    std::vector<double> seq_comm_us;
    std::vector<double> seq_compute_us;
    std::vector<double> pipelined_times_us;
    std::vector<double> pipe_comm_us;
    std::vector<double> pipe_compute_us;
};

// ============================================================================
// Helper functions
// ============================================================================
static void LaunchCommKernel(RankResources &r)
{
    launchRingCommStreaming(reinterpret_cast<uint8_t *>(r.shmem_input), reinterpret_cast<uint8_t *>(r.chunk_flag_shmem),
                            reinterpret_cast<uint8_t *>(r.hcclTestCtx.deviceCtx), r.n_ranks, r.commStream);
}

static void LaunchComputeKernel(RankResources &r)
{
    launchAllGatherGemmComputeStreaming(
        reinterpret_cast<uint8_t *>(r.output_dev), reinterpret_cast<uint8_t *>(r.shmem_input),
        reinterpret_cast<uint8_t *>(r.src1_dev), reinterpret_cast<uint8_t *>(r.chunk_flag_shmem), r.computeStream,
        COMPUTE_BLOCK_NUM);
}

static int CreateRankStreams(RankResources &r)
{
    int status = 0;
    status |= aclrtCreateStream(&r.computeStream);
    status |= aclrtCreateStream(&r.commStream);
    return status;
}

static void InitChunkLayout(RankResources &r, int n_ranks)
{
    r.inputShmemBytes = static_cast<size_t>(G_M) * G_K * sizeof(uint16_t);
    int m_tiles = static_cast<int>(G_M / G_BASE_M);
    // NOLINTNEXTLINE - n_ranks > 0 guaranteed by caller
    int m_tiles_local = (n_ranks > 0) ? (m_tiles / n_ranks) : 0;
    int k_tiles = static_cast<int>(G_K / G_BASE_N);
    r.numTilesPerSrc = m_tiles_local * k_tiles;
    r.optimalChunkSize = ComputeOptimalChunkSize(r.numTilesPerSrc);
    r.numChunksPerSrc = (r.numTilesPerSrc + r.optimalChunkSize - 1) / r.optimalChunkSize;
    r.chunkFlagMatrixSize = ChunkFlagMatrixSize(n_ranks, r.numTilesPerSrc, r.optimalChunkSize);
    r.chunkFlagWithSummarySize = ChunkFlagMatrixWithSummarySize(n_ranks, r.numTilesPerSrc, r.optimalChunkSize);
}

static bool AllocateWindowResources(RankResources &r, int rank_id, int n_ranks)
{
    uint64_t localWinBase = r.hcclTestCtx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    (void)WindowAlloc(localWinBase, winOffset, WINDOW_GUARD_BYTES);
    r.shmem_input = WindowAlloc(localWinBase, winOffset, r.inputShmemBytes);
    aclrtMemset(r.shmem_input, r.inputShmemBytes, 0, r.inputShmemBytes);

    r.chunk_flag_shmem = WindowAlloc(localWinBase, winOffset, r.chunkFlagWithSummarySize);
    if (winOffset > r.hcclTestCtx.hostCtx.winSize) {
        std::cerr << "[ERROR] Rank " << rank_id << ": HCCL window too small! need=" << winOffset
                  << " have=" << r.hcclTestCtx.hostCtx.winSize << std::endl;
        return false;
    }
    if (rank_id == 0) {
        std::cout << "[INFO] HCCL window usage: guard=" << WINDOW_GUARD_BYTES << " input=" << r.inputShmemBytes
                  << " flags=" << r.chunkFlagWithSummarySize << " total=" << winOffset << " / "
                  << r.hcclTestCtx.hostCtx.winSize << std::endl;
    }

    aclrtMallocHost(reinterpret_cast<void **>(&r.chunk_flag_host), r.chunkFlagWithSummarySize);
    ChunkFlagMatrixInit(r.chunk_flag_host, n_ranks, r.numTilesPerSrc, r.optimalChunkSize);
    r.chunk_flag_host->my_rank = rank_id;
    r.summary_host =
        reinterpret_cast<int32_t *>(reinterpret_cast<uint8_t *>(r.chunk_flag_host) + r.chunkFlagMatrixSize);
    ChunkFlagMatrixSummaryInit(r.summary_host, n_ranks);
    aclrtMemcpy(r.chunk_flag_shmem, r.chunkFlagWithSummarySize, r.chunk_flag_host, r.chunkFlagWithSummarySize,
                ACL_MEMCPY_HOST_TO_DEVICE);
    return true;
}

static void AllocateDeviceBuffers(RankResources &r, int n_ranks)
{
    r.bSize = static_cast<size_t>(G_K) * G_N * sizeof(uint16_t);
    aclrtMalloc(&r.src1_dev, r.bSize, ACL_MEM_MALLOC_HUGE_FIRST);

    r.outputSize = static_cast<size_t>(G_M) * G_N * sizeof(float);
    aclrtMalloc(&r.output_dev, r.outputSize, ACL_MEM_MALLOC_HUGE_FIRST);

    r.aLocalSize = (n_ranks > 0) ? ((static_cast<size_t>(G_M) / n_ranks) * G_K * sizeof(uint16_t)) : 0;
    r.fullInputHost.resize(r.inputShmemBytes / sizeof(uint16_t));
}

static void SyncRankWork(RankResources &r)
{
    aclrtSynchronizeStream(r.computeStream);
    aclrtSynchronizeStream(r.commStream);
    HcclHostBarrier(r.hcclTestCtx.comm, r.hcclTestCtx.stream);
}

struct PerfStats {
    double avg = 0.0;
    double min_val = 0.0;
    double max_val = 0.0;
    double std_dev = 0.0;
};

static PerfStats CalcStats(const std::vector<double> &times)
{
    if (times.empty()) {
        return {};
    }

    double sum = 0.0;
    double mn = times[0];
    double mx = times[0];
    for (double t : times) {
        sum += t;
        mn = std::min(mn, t);
        mx = std::max(mx, t);
    }

    double avg = sum / static_cast<double>(times.size());
    double var = 0.0;
    for (double t : times) {
        double diff = t - avg;
        var += diff * diff;
    }
    return {avg, mn, mx, std::sqrt(var / static_cast<double>(times.size()))};
}

static void PrintTimingDetails(const PerfStats &comp_s, const PerfStats &seq_s, const PerfStats &pipe_s,
                               const PerfStats &seq_comm_s, const PerfStats &seq_compute_s,
                               const PerfStats &pipe_comm_s, const PerfStats &pipe_compute_s, double flops_per_rank,
                               double flops_total, double ag_bytes)
{
    auto gflops = [](double flops, double us) { return (us > 0.0) ? (flops / (us * 1e-6) / 1e9) : 0.0; };
    auto bw_gbs = [&](double us) { return (us > 0.0) ? (ag_bytes / (us * 1e-6) / (1024.0 * 1024.0 * 1024.0)) : 0.0; };

    std::cout << "\n  Compute-only:   " << std::setprecision(1) << comp_s.avg << " us"
              << "  (" << std::setprecision(0) << gflops(flops_per_rank, comp_s.avg) << " GFLOPS)" << std::endl;
    std::cout << "\n  Sequential:     " << std::setprecision(1) << seq_s.avg << " us" << std::endl;
    std::cout << "    comm:         " << std::setprecision(1) << seq_comm_s.avg << " us"
              << "  (" << std::setprecision(1) << bw_gbs(seq_comm_s.avg) << " GB/s)" << std::endl;
    std::cout << "    compute:      " << seq_compute_s.avg << " us"
              << "  (" << std::setprecision(0) << gflops(flops_per_rank, seq_compute_s.avg) << " GFLOPS)" << std::endl;
    std::cout << "\n  Pipelined:      " << std::setprecision(1) << pipe_s.avg << " us" << std::endl;
    std::cout << "    comm done:    " << pipe_comm_s.avg << " us"
              << "  (" << std::setprecision(1) << bw_gbs(pipe_comm_s.avg) << " GB/s)" << std::endl;
    std::cout << "    compute done: " << pipe_compute_s.avg << " us"
              << "  (" << std::setprecision(0) << gflops(flops_per_rank, pipe_compute_s.avg) << " GFLOPS, "
              << std::setprecision(1)
              << ((comp_s.avg > 0.0) ?
                      (gflops(flops_per_rank, pipe_compute_s.avg) / gflops(flops_per_rank, comp_s.avg) * 100.0) :
                      0.0)
              << "% of pure)" << std::endl;

    double speedup = (pipe_s.avg > 0.0) ? (seq_s.avg / pipe_s.avg) : 0.0;
    double overlap_time = (seq_comm_s.avg + seq_compute_s.avg) - pipe_s.avg;
    double overlap_eff =
        (overlap_time > 0.0) ? (overlap_time / std::min(seq_comm_s.avg, seq_compute_s.avg) * 100.0) : 0.0;

    std::cout << "\n  Speedup:        " << std::setprecision(3) << speedup << "x" << std::endl;
    std::cout << "  Time saved:     " << std::setprecision(1) << (seq_s.avg - pipe_s.avg) << " us"
              << " (" << std::setprecision(1)
              << ((seq_s.avg > 0.0) ? ((seq_s.avg - pipe_s.avg) / seq_s.avg * 100.0) : 0.0) << "%)" << std::endl;
    std::cout << "  Overlap eff:    " << std::setprecision(1) << overlap_eff << "%" << std::endl;
    std::cout << "  Throughput:     " << std::setprecision(0) << gflops(flops_total, pipe_s.avg) << " GFLOPS (total)"
              << std::endl;
    std::cout << "================================================================\n" << std::endl;
}

static void PrintPerfReport(bool is_ok, int n_ranks, const std::vector<double> &compute_times_us,
                            const std::vector<double> &sequential_times_us, const std::vector<double> &seq_comm_us,
                            const std::vector<double> &seq_compute_us, const std::vector<double> &pipelined_times_us,
                            const std::vector<double> &pipe_comm_us, const std::vector<double> &pipe_compute_us)
{
    PerfStats comp_s = CalcStats(compute_times_us);
    PerfStats seq_s = CalcStats(sequential_times_us);
    PerfStats seq_comm_s = CalcStats(seq_comm_us);
    PerfStats seq_compute_s = CalcStats(seq_compute_us);
    PerfStats pipe_s = CalcStats(pipelined_times_us);
    PerfStats pipe_comm_s = CalcStats(pipe_comm_us);
    PerfStats pipe_compute_s = CalcStats(pipe_compute_us);

    double flops_per_rank = 2.0 * static_cast<double>(ORIG_M) * ORIG_K * ORIG_N;
    double flops_total = flops_per_rank * ((n_ranks > 0) ? n_ranks : 1);
    double local_a_bytes = (n_ranks > 0) ? (static_cast<double>(G_M) / n_ranks) * G_K * sizeof(uint16_t) : 0.0;
    double ag_bytes = local_a_bytes * std::max(n_ranks - 1, 0);
    double data_gb = ag_bytes / (1024.0 * 1024.0 * 1024.0);

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "\n================================================================" << std::endl;
    std::cout << (is_ok ? "[SUCCESS]" : "[FAILED]") << " AllGather GEMM (HCCL)" << std::endl;
    std::cout << "  M=" << ORIG_M << " K=" << ORIG_K << " N=" << ORIG_N;
    if (ORIG_M != G_M || ORIG_K != G_K || ORIG_N != G_N) {
        std::cout << "  (padded " << G_M << "x" << G_K << "x" << G_N << ")";
    }
    std::cout << "  ranks=" << n_ranks << "  compute_blocks=" << COMPUTE_BLOCK_NUM << "  comm_blocks=" << COMM_BLOCK_NUM
              << std::endl;
    std::cout << "  local_a=" << (n_ranks > 0 ? (G_M / n_ranks) : 0) << "x" << G_K
              << "  comm_data=" << std::setprecision(3) << data_gb << " GB/rank" << std::endl;

    PrintTimingDetails(comp_s, seq_s, pipe_s, seq_comm_s, seq_compute_s, pipe_comm_s, pipe_compute_s, flops_per_rank,
                       flops_total, ag_bytes);
}

// ============================================================================
// Sub-functions for RunAllGatherGemmPerRank
// ============================================================================
static bool AllocateResources(RankResources &r, int rank_id, int n_ranks, const HcclRootInfo *rootInfo)
{
    if (n_ranks <= 0) {
        std::cerr << "[ERROR] n_ranks must be positive, got " << n_ranks << std::endl;
        return false;
    }
    r.rank_id = rank_id;
    r.n_ranks = n_ranks;

    if (!r.hcclTestCtx.Init(rank_id, n_ranks, n_ranks, 0, rootInfo)) {
        std::cerr << "[ERROR] Rank " << rank_id << ": HCCL TestContext Init failed\n";
        return false;
    }

    int status = CreateRankStreams(r);
    InitChunkLayout(r, n_ranks);
    if (!AllocateWindowResources(r, rank_id, n_ranks)) {
        return false;
    }
    AllocateDeviceBuffers(r, n_ranks);
    return status == 0;
}

static bool LoadInputData(RankResources &r, const std::string &dataDir)
{
    std::string b_file = dataDir + "/pe_" + std::to_string(r.rank_id) + "_b.bin";
    std::vector<uint16_t> b_host(r.bSize / sizeof(uint16_t));
    size_t b_file_size = 0;

    if (!PtoTestCommon::ReadFile(b_file, b_file_size, b_host.data(), r.bSize) || b_file_size != r.bSize) {
        std::cerr << "[ERROR] Rank " << r.rank_id << ": B file mismatch: " << b_file << std::endl;
        return false;
    }

    size_t local_elems = r.aLocalSize / sizeof(uint16_t);
    for (int src_rank = 0; src_rank < r.n_ranks; ++src_rank) {
        std::string a_file = dataDir + "/pe_" + std::to_string(src_rank) + "_a.bin";
        size_t a_file_size = 0;
        if (!PtoTestCommon::ReadFile(a_file, a_file_size,
                                     r.fullInputHost.data() + static_cast<size_t>(src_rank) * local_elems,
                                     r.aLocalSize) ||
            a_file_size != r.aLocalSize) {
            std::cerr << "[ERROR] Rank " << r.rank_id << ": A file mismatch: " << a_file << std::endl;
            return false;
        }
    }

    aclrtMemcpy(reinterpret_cast<uint8_t *>(r.shmem_input) + static_cast<size_t>(r.rank_id) * r.aLocalSize,
                r.aLocalSize, r.fullInputHost.data() + static_cast<size_t>(r.rank_id) * local_elems, r.aLocalSize,
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(r.src1_dev, r.bSize, b_host.data(), r.bSize, ACL_MEMCPY_HOST_TO_DEVICE);
    return true;
}

static void ResetChunkFlagsHost(RankResources &r)
{
    ChunkFlagMatrixReset(r.chunk_flag_host);
    ChunkFlagMatrixSummaryInit(r.summary_host, r.n_ranks);
    r.chunk_flag_host->my_rank = r.rank_id;
}

static void MarkAllChunksReadyHost(RankResources &r)
{
    int32_t *flag_base =
        reinterpret_cast<int32_t *>(reinterpret_cast<uint8_t *>(r.chunk_flag_host) + sizeof(ChunkFlagMatrix));
    for (int src_rank = 0; src_rank < r.n_ranks; ++src_rank) {
        int row_offset = src_rank * r.chunk_flag_host->stride;
        for (int chunk_idx = 0; chunk_idx < r.chunk_flag_host->num_chunks_per_src; ++chunk_idx) {
            flag_base[row_offset + chunk_idx] = r.chunk_flag_host->epoch;
        }
        r.summary_host[src_rank] = r.chunk_flag_host->num_chunks_per_src;
    }
}

static void PrepareStreamingState(RankResources &r)
{
    ResetChunkFlagsHost(r);
    aclrtMemcpy(r.chunk_flag_shmem, r.chunkFlagWithSummarySize, r.chunk_flag_host, r.chunkFlagWithSummarySize,
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemset(r.output_dev, r.outputSize, 0, r.outputSize);

    size_t local_elems = r.aLocalSize / sizeof(uint16_t);
    aclrtMemcpy(reinterpret_cast<uint8_t *>(r.shmem_input) + static_cast<size_t>(r.rank_id) * r.aLocalSize,
                r.aLocalSize, r.fullInputHost.data() + static_cast<size_t>(r.rank_id) * local_elems, r.aLocalSize,
                ACL_MEMCPY_HOST_TO_DEVICE);
}

static void PrepareComputeOnlyState(RankResources &r)
{
    ResetChunkFlagsHost(r);
    MarkAllChunksReadyHost(r);
    aclrtMemcpy(r.shmem_input, r.inputShmemBytes, r.fullInputHost.data(), r.inputShmemBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(r.chunk_flag_shmem, r.chunkFlagWithSummarySize, r.chunk_flag_host, r.chunkFlagWithSummarySize,
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemset(r.output_dev, r.outputSize, 0, r.outputSize);
}

static void RunWarmup(RankResources &r)
{
    if (r.rank_id == 0) {
        std::cout << "\n[INFO] Running warmup..." << std::endl;
    }

    for (int iter = 0; iter < WARMUP_ITERS; ++iter) {
        PrepareStreamingState(r);
        SyncRankWork(r);
        LaunchCommKernel(r);
        LaunchComputeKernel(r);
        SyncRankWork(r);
    }
}

static void RunComputeOnlyBenchmark(RankResources &r, std::vector<double> &compute_times_us)
{
    for (int iter = 0; iter < COMPUTE_ONLY_ITERS; ++iter) {
        PrepareComputeOnlyState(r);
        aclrtSynchronizeStream(r.computeStream);
        HcclHostBarrier(r.hcclTestCtx.comm, r.hcclTestCtx.stream);

        auto t0 = std::chrono::high_resolution_clock::now();
        LaunchComputeKernel(r);
        aclrtSynchronizeStream(r.computeStream);
        auto t1 = std::chrono::high_resolution_clock::now();

        compute_times_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        HcclHostBarrier(r.hcclTestCtx.comm, r.hcclTestCtx.stream);
    }
}

static void RunSequentialBenchmark(RankResources &r, std::vector<double> &sequential_times_us,
                                   std::vector<double> &seq_comm_us, std::vector<double> &seq_compute_us)
{
    for (int iter = 0; iter < MEASURE_ITERS; ++iter) {
        PrepareStreamingState(r);
        aclrtSynchronizeStream(r.computeStream);
        aclrtSynchronizeStream(r.commStream);
        HcclHostBarrier(r.hcclTestCtx.comm, r.hcclTestCtx.stream);

        auto t0 = std::chrono::high_resolution_clock::now();
        LaunchCommKernel(r);
        aclrtSynchronizeStream(r.commStream);
        auto t1 = std::chrono::high_resolution_clock::now();
        LaunchComputeKernel(r);
        aclrtSynchronizeStream(r.computeStream);
        auto t2 = std::chrono::high_resolution_clock::now();

        seq_comm_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        seq_compute_us.push_back(std::chrono::duration<double, std::micro>(t2 - t1).count());
        sequential_times_us.push_back(std::chrono::duration<double, std::micro>(t2 - t0).count());
        HcclHostBarrier(r.hcclTestCtx.comm, r.hcclTestCtx.stream);
    }
}

static void RunPipelinedBenchmark(RankResources &r, std::vector<double> &pipelined_times_us,
                                  std::vector<double> &pipe_comm_us, std::vector<double> &pipe_compute_us)
{
    aclrtEvent evComputeStart = nullptr;
    aclrtEvent evComputeEnd = nullptr;
    aclrtCreateEvent(&evComputeStart);
    aclrtCreateEvent(&evComputeEnd);

    for (int iter = 0; iter < MEASURE_ITERS; ++iter) {
        PrepareStreamingState(r);
        aclrtSynchronizeStream(r.computeStream);
        aclrtSynchronizeStream(r.commStream);
        HcclHostBarrier(r.hcclTestCtx.comm, r.hcclTestCtx.stream);

        auto t0 = std::chrono::high_resolution_clock::now();
        LaunchCommKernel(r);
        aclrtRecordEvent(evComputeStart, r.computeStream);
        LaunchComputeKernel(r);
        aclrtRecordEvent(evComputeEnd, r.computeStream);
        aclrtSynchronizeStream(r.commStream);
        auto t_comm_done = std::chrono::high_resolution_clock::now();
        aclrtSynchronizeStream(r.computeStream);
        auto t1 = std::chrono::high_resolution_clock::now();

        float compute_ms = 0.0f;
        aclrtEventElapsedTime(&compute_ms, evComputeStart, evComputeEnd);

        pipe_comm_us.push_back(std::chrono::duration<double, std::micro>(t_comm_done - t0).count());
        pipe_compute_us.push_back(static_cast<double>(compute_ms) * 1000.0);
        pipelined_times_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        HcclHostBarrier(r.hcclTestCtx.comm, r.hcclTestCtx.stream);
    }

    aclrtDestroyEvent(evComputeStart);
    aclrtDestroyEvent(evComputeEnd);
}

static BenchmarkSamples RunBenchmarks(RankResources &r)
{
    BenchmarkSamples samples;
    RunComputeOnlyBenchmark(r, samples.compute_times_us);
    RunSequentialBenchmark(r, samples.sequential_times_us, samples.seq_comm_us, samples.seq_compute_us);
    RunPipelinedBenchmark(r, samples.pipelined_times_us, samples.pipe_comm_us, samples.pipe_compute_us);
    return samples;
}

static bool VerifyOutput(RankResources &r, const std::string &dataDir)
{
    float *output_host = nullptr;
    aclrtMallocHost(reinterpret_cast<void **>(&output_host), r.outputSize);
    aclrtMemcpy(output_host, r.outputSize, r.output_dev, r.outputSize, ACL_MEMCPY_DEVICE_TO_HOST);

    std::string output_file = dataDir + "/output_rank" + std::to_string(r.rank_id) + ".bin";
    PtoTestCommon::WriteFile(output_file, output_host, r.outputSize);

    bool is_ok = false;
    std::string golden_file = dataDir + "/golden.bin";
    size_t goldenSize = static_cast<size_t>(ORIG_M) * ORIG_N * sizeof(float);
    std::vector<float> golden(goldenSize / sizeof(float));
    size_t golden_file_size = 0;

    if (PtoTestCommon::ReadFile(golden_file, golden_file_size, golden.data(), goldenSize) &&
        golden_file_size == goldenSize) {
        if (ORIG_M == G_M && ORIG_N == G_N) {
            is_ok = PtoTestCommon::ResultCmp(golden, output_host, 0.001f);
        } else {
            std::vector<float> valid_output(static_cast<size_t>(ORIG_M) * ORIG_N);
            for (uint32_t row = 0; row < ORIG_M; ++row) {
                errno_t ret = memcpy_s(valid_output.data() + row * ORIG_N, ORIG_N * sizeof(float),
                                       output_host + row * G_N, ORIG_N * sizeof(float));
                if (ret != EOK) {
                    std::cerr << "[ERROR] memcpy_s failed at row " << row << " with errno " << ret << std::endl;
                    aclrtFreeHost(output_host);
                    return false;
                }
            }
            is_ok = PtoTestCommon::ResultCmp(golden, valid_output.data(), 0.001f);
        }
    } else {
        std::cerr << "[ERROR] Rank " << r.rank_id << ": golden.bin missing or size mismatch (expected " << goldenSize
                  << " bytes); numerical verification not performed, treated as FAILED" << std::endl;
    }
    aclrtFreeHost(output_host);
    return is_ok;
}

static bool RunFunctionalVerification(RankResources &r, const std::string &dataDir)
{
    if (r.rank_id == 0) {
        std::cout << "\n[INFO] Running functional verification..." << std::endl;
    }

    PrepareStreamingState(r);
    SyncRankWork(r);
    LaunchCommKernel(r);
    LaunchComputeKernel(r);
    SyncRankWork(r);
    return VerifyOutput(r, dataDir);
}

static void PrintRank0Results(int rank_id, bool is_ok, int n_ranks, const BenchmarkSamples &samples)
{
    if (rank_id != 0) {
        return;
    }

    std::cout << (is_ok ? "[INFO] Functional run completed. Verification PASSED." :
                          "[ERROR] Functional run completed. Verification FAILED!")
              << std::endl;
    PrintPerfReport(is_ok, n_ranks, samples.compute_times_us, samples.sequential_times_us, samples.seq_comm_us,
                    samples.seq_compute_us, samples.pipelined_times_us, samples.pipe_comm_us, samples.pipe_compute_us);
}

static void Cleanup(RankResources &r)
{
    aclrtFree(r.src1_dev);
    aclrtFree(r.output_dev);
    aclrtFreeHost(r.chunk_flag_host);
    if (r.computeStream)
        aclrtDestroyStream(r.computeStream);
    if (r.commStream)
        aclrtDestroyStream(r.commStream);
}

[[noreturn]] static void CleanupAndTerminate(RankResources &r, int rank_id, bool is_ok)
{
    CommMpiBarrier();
    Cleanup(r);

    if (rank_id == 0) {
        std::cout << (is_ok ? "[SUCCESS] AllGather GEMM demo completed successfully." :
                              "[FAILED] AllGather GEMM demo FAILED.")
                  << std::endl;
    }

    // HcclCommDestroy hangs on the internal barrier in some HCCL versions
    // when used with HCCL windows (shmem_input). Force-exit after MPI
    // barrier to avoid the hang; OS reclaims all resources.
    CommMpiBarrier();
    CommMpiFinalize();
    _exit(is_ok ? 0 : 1);
}

// ============================================================================
// RunAllGatherGemmPerRank: top-level per-rank orchestration
// ============================================================================
static bool RunAllGatherGemmPerRank(int rank_id, int n_ranks, const HcclRootInfo *rootInfo, const std::string &dataDir)
{
    RankResources r;
    if (!AllocateResources(r, rank_id, n_ranks, rootInfo))
        return false;
    if (!LoadInputData(r, dataDir)) {
        Cleanup(r);
        return false;
    }

    RunWarmup(r);
    BenchmarkSamples samples = RunBenchmarks(r);
    bool is_ok = RunFunctionalVerification(r, dataDir);
    PrintRank0Results(rank_id, is_ok, n_ranks, samples);
    CleanupAndTerminate(r, rank_id, is_ok);
}

// ============================================================================
// Argument parsing and ACL initialization
// ============================================================================
struct AppArgs {
    int n_ranks = 2;
    std::string dataDir = "../out";
};

static AppArgs ParseArgs()
{
    AppArgs args;
    if (const char *env = std::getenv("N_RANKS")) {
        args.n_ranks = std::atoi(env);
    }
    if (const char *env = std::getenv("ALLGATHER_GEMM_DATA_DIR")) {
        args.dataDir = env;
    }
    return args;
}

static bool InitAcl(int rank_id, int device_id)
{
    constexpr int kAclRepeatInit = 100002;
    aclError aRet = aclInit(nullptr);
    if (aRet != ACL_SUCCESS && static_cast<int>(aRet) != kAclRepeatInit) {
        std::cerr << "[ERROR] Rank " << rank_id << ": aclInit failed: " << static_cast<int>(aRet) << std::endl;
        return false;
    }

    if (rank_id == 0)
        rtSetDevice(device_id);

    aRet = aclrtSetDevice(device_id);
    if (aRet != ACL_SUCCESS) {
        std::cerr << "[ERROR] Rank " << rank_id << ": aclrtSetDevice(" << device_id
                  << ") failed: " << static_cast<int>(aRet) << std::endl;
        return false;
    }
    return true;
}

static void PrintBanner(int n_ranks)
{
    std::cout << "\n================================================================" << std::endl;
    std::cout << "  AllGather GEMM (HCCL backend)" << std::endl;
    std::cout << "  M=" << G_M << ", K=" << G_K << ", N=" << G_N << ", pe_size=" << n_ranks << std::endl;
    if (ORIG_M != G_M || ORIG_K != G_K || ORIG_N != G_N) {
        std::cout << "  (original: M=" << ORIG_M << ", K=" << ORIG_K << ", N=" << ORIG_N << ")" << std::endl;
    }
    std::cout << "  Mode: FUNCTIONAL VERIFICATION + PERFORMANCE" << std::endl;
    std::cout << "================================================================" << std::endl;
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char **argv)
{
    if (!CommMpiInit(&argc, &argv)) {
        fprintf(stderr, "[FATAL] CommMpiInit failed. Launch with: mpirun -n <N> ./allgather_gemm\n");
        return 1;
    }

    AppArgs args = ParseArgs();
    int n_ranks = args.n_ranks;

    if (n_ranks > MAX_RING_RANKS) {
        std::cerr << "[ERROR] n_ranks exceeds MAX_RING_RANKS=" << MAX_RING_RANKS << std::endl;
        CommMpiFinalize();
        return 1;
    }

    int mpiRank = CommMpiRank();
    int mpiSize = CommMpiSize();
    if (mpiSize != n_ranks) {
        if (mpiRank == 0) {
            std::cerr << "[ERROR] MPI world size (" << mpiSize << ") != expected N_RANKS (" << n_ranks
                      << "). Launch with: mpirun -n " << n_ranks << " ./allgather_gemm" << std::endl;
        }
        CommMpiFinalize();
        return 1;
    }

    int rank_id = mpiRank;
    int device_id = rank_id;

    if (!InitAcl(rank_id, device_id)) {
        CommMpiFinalize();
        return 1;
    }

    HcclRootInfo rootInfo{};
    if (rank_id == 0) {
        HcclResult hret = HcclGetRootInfo(&rootInfo);
        if (hret != HCCL_SUCCESS) {
            std::cerr << "[ERROR] HcclGetRootInfo failed: " << hret << std::endl;
            CommMpiFinalize();
            return 1;
        }
    }

    CommMpiBcast(&rootInfo, HCCL_ROOT_INFO_BYTES, COMM_MPI_CHAR, 0);
    CommMpiBarrier();

    if (rank_id == 0)
        PrintBanner(n_ranks);

    bool ok = RunAllGatherGemmPerRank(rank_id, n_ranks, &rootInfo, args.dataDir);

    if (rank_id == 0) {
        std::cerr << "[FAILED] AllGather GEMM early init failure." << std::endl;
    }

    CommMpiFinalize();
    return ok ? 0 : 1;
}
