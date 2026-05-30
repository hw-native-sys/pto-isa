/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "args.h"
#include "comm_mpi.h"
#include "golden.h"
#include "hccl_context.h"
#include "kernel_launchers.h"
#include "layout.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"

namespace moe_combine {

namespace {

void CheckAcl(aclError ret, const std::string &where)
{
    if (ret != ACL_SUCCESS) {
        throw std::runtime_error(where + " failed: " + std::to_string(static_cast<int>(ret)));
    }
}

size_t BytesOfHalfVector(size_t elements)
{
    return elements * sizeof(uint16_t);
}

size_t BytesOfFloatVector(size_t elements)
{
    return elements * sizeof(float);
}

size_t BytesOfI32Vector(size_t elements)
{
    return elements * sizeof(int32_t);
}

double UsSince(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end)
{
    return std::chrono::duration<double, std::micro>(end - start).count();
}

} // namespace

struct PerfStats {
    double avg = 0.0;
    double min = 0.0;
    double max = 0.0;
    double stddev = 0.0;
};

struct IterationTiming {
    double prepareHostUs = 0.0;
    double combineE2eUs = 0.0;
    double totalE2eUs = 0.0;
};

PerfStats CalcStats(const std::vector<double> &samples)
{
    PerfStats stats;
    if (samples.empty()) {
        return stats;
    }

    stats.min = *std::min_element(samples.begin(), samples.end());
    stats.max = *std::max_element(samples.begin(), samples.end());
    stats.avg = std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
    double variance = 0.0;
    for (double sample : samples) {
        double delta = sample - stats.avg;
        variance += delta * delta;
    }
    stats.stddev = std::sqrt(variance / static_cast<double>(samples.size()));
    return stats;
}

std::vector<double> ExtractTimingSamples(const std::vector<IterationTiming> &timings, double IterationTiming::*field)
{
    std::vector<double> samples;
    samples.reserve(timings.size());
    for (const IterationTiming &timing : timings) {
        samples.push_back(timing.*field);
    }
    return samples;
}

void PrintOneTimingStats(const char *label, const std::vector<IterationTiming> &timings, double IterationTiming::*field)
{
    PerfStats stats = CalcStats(ExtractTimingSamples(timings, field));
    std::cout << "  " << label << ": avg=" << stats.avg << " us"
              << " max=" << stats.max << " us" << std::endl;
}

bool VerboseRuntimeLogs(const MoeCombineArgs &args)
{
    return args.runtime.debug != 0 || args.runtime.hostGoldenOnly != 0 || args.runtime.skipKernels != 0 ||
           args.runtime.combineReturnOnly != 0;
}

struct DeviceBuffers {
    void *probs = nullptr;
    void *outputC = nullptr;
    void *routeMeta = nullptr;
    void *workspace = nullptr;
    void *expertOutput = nullptr;
};

struct RuntimeState {
    MpiContext mpi;
    bool mpiActive = false;
    bool aclActive = false;
    uint32_t rank = 0;
    uint32_t size = 1;
    uint32_t device = 0;
    aclrtStream computeStream = nullptr;
    rtStream_t hcclStream = nullptr;
    HcclWindowContext hccl;
    bool hcclActive = false;
    HostInputData inputs;
    CpuGoldenData golden;
    bool dataReady = false;
    DeviceBuffers buffers;
    bool buffersAllocated = false;
    double prepareHostUs = 0.0;
    double combineE2eUs = 0.0;
    double totalE2eUs = 0.0;
};

void PrintProfileSummary(const MoeCombineArgs &args, RuntimeState *state,
                         const std::vector<IterationTiming> &localTimings)
{
    if (args.runtime.iters == 0) {
        return;
    }

    std::vector<IterationTiming> allTimings;
    size_t measuredIters = localTimings.size();
    if (state->rank == 0) {
        allTimings.resize(static_cast<size_t>(state->size) * localTimings.size());
    }
    MpiGatherBytes(&state->mpi, localTimings.data(), localTimings.size() * sizeof(IterationTiming),
                   state->rank == 0 ? allTimings.data() : nullptr, 0);

    if (state->rank != 0) {
        return;
    }

    std::vector<IterationTiming> globalTimings(measuredIters);
    for (size_t iter = 0; iter < measuredIters; ++iter) {
        IterationTiming timing;
        for (uint32_t rank = 0; rank < state->size; ++rank) {
            const IterationTiming &rankTiming = allTimings[static_cast<size_t>(rank) * measuredIters + iter];
            timing.prepareHostUs = std::max(timing.prepareHostUs, rankTiming.prepareHostUs);
            timing.combineE2eUs = std::max(timing.combineE2eUs, rankTiming.combineE2eUs);
            timing.totalE2eUs = std::max(timing.totalE2eUs, rankTiming.totalE2eUs);
        }
        globalTimings[iter] = timing;
    }

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "\n================================================================" << std::endl;
    std::cout << "[PROFILE] CombineTile" << std::endl;
    std::cout << "  M=" << args.shape.m << " K=" << args.shape.k << " ranks=" << args.shape.ep
              << " topK=" << args.shape.topK << " expertPerPe=" << args.shape.expertPerRank
              << " warmup=" << args.runtime.warmup << " measured=" << args.runtime.iters
              << " samples=" << globalTimings.size() << std::endl;
    std::cout << "  logical work: input tokens(all ranks)=" << static_cast<uint64_t>(args.shape.ep) * args.shape.m
              << " routed tokens(all ranks)=" << static_cast<uint64_t>(args.shape.ep) * args.shape.m * args.shape.topK
              << std::endl;
    PrintOneTimingStats("prepare_fixture", globalTimings, &IterationTiming::prepareHostUs);
    PrintOneTimingStats("combine_e2e", globalTimings, &IterationTiming::combineE2eUs);
    PrintOneTimingStats("total_e2e", globalTimings, &IterationTiming::totalE2eUs);
    std::cout << "  verify=" << (args.runtime.verify == 0 ? "SKIP" : "PASS") << std::endl;
    std::cout << "  note: warmup iterations are excluded; each measured sample is the max across ranks." << std::endl;
    std::cout << "================================================================\n" << std::endl;
}

void PrintStage(uint32_t rank, const char *stage, const char *state)
{
    std::cout << "rank=" << rank << " stage=" << stage << " " << state << std::endl;
}

void InitRankInfo(const MoeCombineArgs &args, int *argc, char ***argv, RuntimeState *state)
{
    if (args.runtime.rankFromMpi != 0) {
        state->mpi = InitMpiAndRank(argc, argv);
        state->mpiActive = true;
        state->rank = static_cast<uint32_t>(state->mpi.rank);
        state->size = static_cast<uint32_t>(state->mpi.size);
    } else {
        state->rank = args.runtime.rank;
        state->size = args.runtime.nranks;
    }

    if (state->size != args.shape.ep) {
        throw std::runtime_error("rank size mismatch: runtime size=" + std::to_string(state->size) +
                                 " shape EP=" + std::to_string(args.shape.ep));
    }
    if (state->rank >= state->size) {
        throw std::runtime_error("rank is outside rank size");
    }
    state->device = args.runtime.deviceBase + state->rank;
}

void RunHostGoldenOnly(const MoeCombineArgs &args, RuntimeState *state)
{
    PrintStage(state->rank, "host_golden", "begin");
    if (args.runtime.genData != 0 && state->rank == 0) {
        GenerateAllInputFiles(args);
    }
    MpiBarrier(&state->mpi);

    HostInputData inputs = GenerateOrLoadInputs(args, state->rank);
    CpuGoldenData golden = ComputeCpuGolden(args, inputs, state->rank);
    CompareResult compare = CompareOutputs(args, golden, golden.outputC, state->rank);

    std::cout << "rank=" << state->rank << " golden_owner_rows=";
    for (uint32_t rank = 0; rank < golden.ownerRows.size(); ++rank) {
        if (rank != 0) {
            std::cout << ",";
        }
        std::cout << rank << ":" << golden.ownerRows[rank];
    }
    std::cout << "\n";
    std::cout << "rank=" << state->rank << " golden_total_routes=" << golden.totalRoutes
              << " golden_invalid_routes=" << golden.invalidRoutes << " compare_elements=" << compare.elementCount
              << " compare_mismatches=" << compare.mismatchCount << "\n";
    std::cout << "rank=" << state->rank << " golden_rank_done" << std::endl;
    MpiBarrier(&state->mpi);
}

void PrepareHostData(const MoeCombineArgs &args, RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "host_data", "begin");
    }
    if (args.runtime.genData != 0 && state->rank == 0) {
        GenerateAllInputFiles(args);
    }
    MpiBarrier(&state->mpi);
    state->inputs = GenerateOrLoadInputs(args, state->rank);
    state->golden = ComputeCpuGolden(args, state->inputs, state->rank);
    state->dataReady = true;
    if (verbose) {
        std::cout << "rank=" << state->rank << " golden_total_routes=" << state->golden.totalRoutes
                  << " golden_invalid_routes=" << state->golden.invalidRoutes << std::endl;
        PrintStage(state->rank, "host_data", "done");
    }
}

void BindDeviceContinuous(const MoeCombineArgs &args, RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "bind_device", "begin");
    }
    if (!InitAclAndBindDevice(state->rank, state->device)) {
        throw std::runtime_error("rank " + std::to_string(state->rank) + " failed to bind device " +
                                 std::to_string(state->device));
    }
    state->aclActive = true;
    if (verbose) {
        PrintStage(state->rank, "bind_device", "done");
    }
}

void CreateStreams(const MoeCombineArgs &args, RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "create_streams", "begin");
    }
    if (aclrtCreateStream(&state->computeStream) != ACL_SUCCESS) {
        throw std::runtime_error("rank " + std::to_string(state->rank) + " aclrtCreateStream failed");
    }
    if (rtStreamCreate(&state->hcclStream, kRtStreamPriorityDefault) != 0) {
        throw std::runtime_error("rank " + std::to_string(state->rank) + " rtStreamCreate failed");
    }
    if (verbose) {
        PrintStage(state->rank, "create_streams", "done");
    }
}

void InitHccl(RuntimeState *state, const MoeCombineArgs &args, const PeerWindowLayout &peerWindowLayout)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "hccl_root_info", "begin");
    }
    HcclRootInfo rootInfo{};
    if (state->rank == 0 && !InitHcclRootInfo(&rootInfo)) {
        throw std::runtime_error("rank 0 failed to get HCCL root info");
    }
    MpiBroadcast(&state->mpi, &rootInfo, HCCL_ROOT_INFO_BYTES, 0);
    MpiBarrier(&state->mpi);
    if (verbose) {
        PrintStage(state->rank, "hccl_root_info", "done");
    }

    if (verbose) {
        PrintStage(state->rank, "hccl_window", "begin");
    }
    state->hccl =
        InitHcclWindowContext(args.shape, peerWindowLayout, state->rank, state->size, &rootInfo, state->hcclStream);
    state->hcclActive = true;
    if (verbose) {
        std::cout << "rank=" << state->rank << " size=" << state->size << " device=" << state->device
                  << " window_base=" << state->hccl.hostDeviceContext.windowsIn[state->rank]
                  << " peer_window_offset=" << state->hccl.peerWindowOffset
                  << " peer_window=" << reinterpret_cast<uint64_t>(state->hccl.peerWindow)
                  << " win_size=" << state->hccl.hostDeviceContext.winSize << std::endl;
        PrintStage(state->rank, "hccl_window", "done");
    }
}

void AllocateLocalBuffers(const MoeCombineArgs &args, const WorkspaceLayout &workspaceLayout,
                          const CombineRouteMetaLayout &routeMetaLayout, RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "allocate_buffers", "begin");
    }
    const MoeCombineShape &shape = args.shape;
    size_t probsBytes = BytesOfFloatVector(static_cast<size_t>(shape.m) * shape.topK);
    size_t outputBytes = BytesOfHalfVector(static_cast<size_t>(shape.m) * shape.k);
    size_t expertOutputBytes = BytesOfHalfVector(static_cast<size_t>(shape.maxOutputSize) * shape.k);
    CheckAcl(aclrtMalloc(&state->buffers.probs, probsBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "rank " + std::to_string(state->rank) + " aclrtMalloc probs");
    CheckAcl(aclrtMalloc(&state->buffers.outputC, outputBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "rank " + std::to_string(state->rank) + " aclrtMalloc outputC");
    CheckAcl(aclrtMalloc(&state->buffers.routeMeta, routeMetaLayout.totalBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "rank " + std::to_string(state->rank) + " aclrtMalloc routeMeta");
    CheckAcl(aclrtMalloc(&state->buffers.workspace, workspaceLayout.totalBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "rank " + std::to_string(state->rank) + " aclrtMalloc workspace");
    CheckAcl(aclrtMalloc(&state->buffers.expertOutput, expertOutputBytes, ACL_MEM_MALLOC_HUGE_FIRST),
             "rank " + std::to_string(state->rank) + " aclrtMalloc expertOutput");
    state->buffersAllocated = true;
    if (verbose) {
        PrintStage(state->rank, "allocate_buffers", "done");
    }
}

void CopyInputsToDevice(const MoeCombineArgs &args, RuntimeState *state)
{
    if (!state->dataReady) {
        throw std::runtime_error("host data is not ready before device copy");
    }
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "copy_inputs", "begin");
    }
    const MoeCombineShape &shape = args.shape;
    size_t probsBytes = BytesOfFloatVector(state->inputs.probs.size());
    CheckAcl(aclrtMemcpy(state->buffers.probs, probsBytes, state->inputs.probs.data(), probsBytes,
                         ACL_MEMCPY_HOST_TO_DEVICE),
             "rank " + std::to_string(state->rank) + " copy probs");
    CheckAcl(aclrtMemset(state->buffers.outputC, BytesOfHalfVector(static_cast<size_t>(shape.m) * shape.k), 0,
                         BytesOfHalfVector(static_cast<size_t>(shape.m) * shape.k)),
             "rank " + std::to_string(state->rank) + " clear outputC");
    if (verbose) {
        PrintStage(state->rank, "copy_inputs", "done");
    }
}

void ClearDeviceState(const MoeCombineArgs &args, const WorkspaceLayout &workspaceLayout,
                      const CombineRouteMetaLayout &routeMetaLayout, const PeerWindowLayout &peerWindowLayout,
                      RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "clear_device_state", "begin");
    }
    CheckAcl(aclrtMemset(state->buffers.workspace, workspaceLayout.totalBytes, 0, workspaceLayout.totalBytes),
             "rank " + std::to_string(state->rank) + " clear workspace");
    CheckAcl(aclrtMemset(state->buffers.routeMeta, routeMetaLayout.totalBytes, 0, routeMetaLayout.totalBytes),
             "rank " + std::to_string(state->rank) + " clear routeMeta");
    void *peerWindowClearBase = state->hccl.localWindowBase;
    size_t peerWindowDataBytes = static_cast<size_t>(state->hccl.peerWindowOffset + peerWindowLayout.totalBytes);
    CheckAcl(aclrtMemset(peerWindowClearBase, peerWindowDataBytes, 0, peerWindowDataBytes),
             "rank " + std::to_string(state->rank) + " clear peerWindow data");
    CheckAcl(aclrtSynchronizeStream(state->computeStream), "rank " + std::to_string(state->rank) + " sync clear");
    if (verbose) {
        PrintStage(state->rank, "clear_device_state", "done");
    }
}

std::vector<float> CopyDeviceHalfToFloat(void *devicePtr, size_t elementCount, uint32_t rank, const std::string &name)
{
    std::vector<uint16_t> halfData(elementCount);
    CheckAcl(aclrtMemcpy(halfData.data(), BytesOfHalfVector(halfData.size()), devicePtr,
                         BytesOfHalfVector(halfData.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(rank) + " copy " + name);
    return HalfBitsToFloatVector(halfData);
}

uint64_t CompareFloatBuffer(const MoeCombineArgs &args, const std::string &name, const std::vector<float> &actual,
                            const std::vector<float> &expected, uint32_t rank)
{
    uint64_t mismatches = 0;
    size_t elementCount = actual.size() < expected.size() ? actual.size() : expected.size();
    size_t firstMismatch = elementCount;
    float firstActual = 0.0f;
    float firstExpected = 0.0f;
    if (actual.size() != expected.size()) {
        mismatches =
            actual.size() > expected.size() ? actual.size() - expected.size() : expected.size() - actual.size();
        firstMismatch = elementCount;
    }
    for (size_t i = 0; i < elementCount; ++i) {
        float actualValue = actual[i];
        float expectedValue = expected[i];
        float diff = std::fabs(actualValue - expectedValue);
        float tol = static_cast<float>(args.atol + args.rtol * std::fabs(expectedValue));
        if (diff > tol) {
            if (firstMismatch == elementCount) {
                firstMismatch = i;
                firstActual = actualValue;
                firstExpected = expectedValue;
            }
            ++mismatches;
        }
    }
    std::cout << "rank=" << rank << " buffer=" << name << " elements=" << actual.size() << " mismatches=" << mismatches;
    if (mismatches != 0) {
        std::cout << " first_index=" << firstMismatch << " actual=" << firstActual << " expected=" << firstExpected;
    }
    std::cout << std::endl;
    return mismatches;
}

void ValidateRouteMetaCumsum(const MoeCombineShape &shape, const CpuGoldenData &golden, uint32_t rank)
{
    size_t expertNumPadded = static_cast<size_t>(ExpertNumPadded(shape));
    size_t expectedElems = static_cast<size_t>(shape.ep) * expertNumPadded;
    if (golden.peerTokenPerExpert.size() != expectedElems) {
        throw std::runtime_error(
            "rank " + std::to_string(rank) + " routeMeta peerTokenPerExpert size mismatch: actual=" +
            std::to_string(golden.peerTokenPerExpert.size()) + " expected=" + std::to_string(expectedElems));
    }
    if (golden.cumsumPerExpert.size() != expectedElems) {
        throw std::runtime_error("rank " + std::to_string(rank) + " routeMeta cumsumPerExpert size mismatch: actual=" +
                                 std::to_string(golden.cumsumPerExpert.size()) +
                                 " expected=" + std::to_string(expectedElems));
    }
    for (uint32_t src = 0; src < shape.ep; ++src) {
        int32_t running = 0;
        size_t base = static_cast<size_t>(src) * expertNumPadded;
        for (size_t expert = 0; expert < expertNumPadded; ++expert) {
            running += golden.peerTokenPerExpert[base + expert];
            int32_t actual = golden.cumsumPerExpert[base + expert];
            if (actual != running) {
                throw std::runtime_error("rank " + std::to_string(rank) +
                                         " routeMeta cumsumPerExpert must be inclusive prefix: src=" +
                                         std::to_string(src) + " expert=" + std::to_string(expert) +
                                         " actual=" + std::to_string(actual) + " expected=" + std::to_string(running));
            }
        }
    }
}

void CopyRouteMetaToDevice(const MoeCombineShape &shape, const CombineRouteMetaLayout &layout, RuntimeState *state)
{
    ValidateRouteMetaCumsum(shape, state->golden, state->rank);
    auto *routeMetaBase = reinterpret_cast<uint8_t *>(state->buffers.routeMeta);
    CheckAcl(aclrtMemcpy(routeMetaBase + layout.cumsumPerExpert, BytesOfI32Vector(state->golden.cumsumPerExpert.size()),
                         state->golden.cumsumPerExpert.data(), BytesOfI32Vector(state->golden.cumsumPerExpert.size()),
                         ACL_MEMCPY_HOST_TO_DEVICE),
             "rank " + std::to_string(state->rank) + " copy fixture cumsumPerExpert");
    CheckAcl(aclrtMemcpy(routeMetaBase + layout.dispatchOffset, BytesOfI32Vector(state->golden.dispatchOffset.size()),
                         state->golden.dispatchOffset.data(), BytesOfI32Vector(state->golden.dispatchOffset.size()),
                         ACL_MEMCPY_HOST_TO_DEVICE),
             "rank " + std::to_string(state->rank) + " copy fixture dispatchOffset");
    CheckAcl(
        aclrtMemcpy(routeMetaBase + layout.prevSumBeforeRank, BytesOfI32Vector(state->golden.prevSumBeforeRank.size()),
                    state->golden.prevSumBeforeRank.data(), BytesOfI32Vector(state->golden.prevSumBeforeRank.size()),
                    ACL_MEMCPY_HOST_TO_DEVICE),
        "rank " + std::to_string(state->rank) + " copy fixture prevSumBeforeRank");
    CheckAcl(
        aclrtMemcpy(routeMetaBase + layout.peerTokenPerExpert,
                    BytesOfI32Vector(state->golden.peerTokenPerExpert.size()), state->golden.peerTokenPerExpert.data(),
                    BytesOfI32Vector(state->golden.peerTokenPerExpert.size()), ACL_MEMCPY_HOST_TO_DEVICE),
        "rank " + std::to_string(state->rank) + " copy fixture peerTokenPerExpert");
    CheckAcl(aclrtMemcpy(routeMetaBase + layout.expandedRowIdx, BytesOfI32Vector(state->golden.expandedRowIdx.size()),
                         state->golden.expandedRowIdx.data(), BytesOfI32Vector(state->golden.expandedRowIdx.size()),
                         ACL_MEMCPY_HOST_TO_DEVICE),
             "rank " + std::to_string(state->rank) + " copy fixture expandedRowIdx");
}

std::vector<uint16_t> CopyExpertFixtureToDevice(const MoeCombineShape &shape, RuntimeState *state)
{
    size_t expertElements = static_cast<size_t>(shape.maxOutputSize) * shape.k;
    std::vector<uint16_t> dispatchedHalf = FloatVectorToHalfBits(state->golden.dispatchedA);
    CheckAcl(aclrtMemcpy(state->buffers.expertOutput, BytesOfHalfVector(expertElements), dispatchedHalf.data(),
                         BytesOfHalfVector(expertElements), ACL_MEMCPY_HOST_TO_DEVICE),
             "rank " + std::to_string(state->rank) + " copy fixture expertOutput");
    return dispatchedHalf;
}

void VerifyCombineFixtureCopy(const MoeCombineArgs &args, const std::vector<uint16_t> &dispatchedHalf,
                              RuntimeState *state)
{
    if (args.runtime.debug < 2) {
        return;
    }
    size_t expertElements = static_cast<size_t>(args.shape.maxOutputSize) * args.shape.k;
    std::vector<float> expertOutput =
        CopyDeviceHalfToFloat(state->buffers.expertOutput, expertElements, state->rank, "expertOutput");
    uint64_t goldenMismatches =
        CompareFloatBuffer(args, "expertOutput", expertOutput, state->golden.dispatchedA, state->rank);
    if (goldenMismatches != 0) {
        throw std::runtime_error("rank " + std::to_string(state->rank) + " combine fixture mismatch");
    }
    WriteBinaryFile(RankBinaryFile(args, state->rank, "actual_dispatchedA_head"), dispatchedHalf);
}

void PrepareCombineFixture(const MoeCombineArgs &args, const CombineRouteMetaLayout &routeMetaLayout,
                           RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "prepare_combine_fixture", "begin");
    }
    auto prepareStart = std::chrono::steady_clock::now();
    CopyRouteMetaToDevice(args.shape, routeMetaLayout, state);
    std::vector<uint16_t> dispatchedHalf = CopyExpertFixtureToDevice(args.shape, state);
    auto prepareEnd = std::chrono::steady_clock::now();
    state->prepareHostUs = UsSince(prepareStart, prepareEnd);
    VerifyCombineFixtureCopy(args, dispatchedHalf, state);
    MpiBarrier(&state->mpi);
    if (verbose) {
        PrintStage(state->rank, "prepare_combine_fixture", "done");
    }
}

struct CombineReturnDump {
    std::vector<float> ptrD;
    std::vector<int32_t> combineDoneSignal;
};

void CopyCombineReturnToHost(const MoeCombineArgs &args, const PeerWindowLayout &peerWindowLayout, RuntimeState *state,
                             CombineReturnDump *dump)
{
    const MoeCombineShape &shape = args.shape;
    size_t expandedRows = static_cast<size_t>(shape.m) * shape.topK;
    dump->ptrD.assign(expandedRows * shape.k, 0.0f);
    dump->combineDoneSignal.assign(shape.ep, 0);
    auto *peerBase = reinterpret_cast<uint8_t *>(state->hccl.peerWindow);
    std::vector<uint16_t> ptrDHalf(dump->ptrD.size());
    CheckAcl(aclrtMemcpy(ptrDHalf.data(), BytesOfHalfVector(ptrDHalf.size()), peerBase + peerWindowLayout.ptrD,
                         BytesOfHalfVector(ptrDHalf.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy ptrD");
    CheckAcl(aclrtMemcpy(dump->combineDoneSignal.data(), BytesOfI32Vector(dump->combineDoneSignal.size()),
                         peerBase + peerWindowLayout.combineDoneSignal,
                         BytesOfI32Vector(dump->combineDoneSignal.size()), ACL_MEMCPY_DEVICE_TO_HOST),
             "rank " + std::to_string(state->rank) + " copy combineDoneSignal");
    dump->ptrD = HalfBitsToFloatVector(ptrDHalf);
}

void ClearCombineReturnState(const MoeCombineArgs &args, const PeerWindowLayout &peerWindowLayout, RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "clear_combine_state", "begin");
    }
    const MoeCombineShape &shape = args.shape;
    size_t expandedRows = static_cast<size_t>(shape.m) * shape.topK;
    auto *peerBase = reinterpret_cast<uint8_t *>(state->hccl.peerWindow);
    CheckAcl(aclrtMemset(peerBase + peerWindowLayout.ptrD, BytesOfHalfVector(expandedRows * shape.k), 0,
                         BytesOfHalfVector(expandedRows * shape.k)),
             "rank " + std::to_string(state->rank) + " clear ptrD");
    CheckAcl(aclrtMemset(state->buffers.outputC, BytesOfHalfVector(static_cast<size_t>(shape.m) * shape.k), 0,
                         BytesOfHalfVector(static_cast<size_t>(shape.m) * shape.k)),
             "rank " + std::to_string(state->rank) + " clear combine outputC");
    CheckAcl(aclrtSynchronizeStream(state->computeStream),
             "rank " + std::to_string(state->rank) + " sync clear combine state");
    if (verbose) {
        PrintStage(state->rank, "clear_combine_state", "done");
    }
}

void PrintCombineReturnSegments(const MoeCombineArgs &args, RuntimeState *state)
{
    if (args.runtime.debug < 2) {
        return;
    }
    const MoeCombineShape &shape = args.shape;
    size_t expertNumPadded = ExpertNumPadded(shape);
    for (uint32_t dst = 0; dst < shape.ep; ++dst) {
        for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
            uint32_t globalExpert = state->rank * shape.expertPerRank + localExpert;
            int32_t rows = state->golden.peerTokenPerExpert[static_cast<size_t>(dst) * expertNumPadded + globalExpert];
            if (rows <= 0) {
                continue;
            }
            int32_t srcStart =
                state->golden.dispatchOffset[localExpert] +
                state->golden.prevSumBeforeRank[static_cast<size_t>(dst) * shape.expertPerRank + localExpert];
            int32_t dstStart =
                globalExpert == 0 ?
                    0 :
                    state->golden.cumsumPerExpert[static_cast<size_t>(dst) * expertNumPadded + globalExpert - 1];
            std::cout << "rank=" << state->rank << " combine_return dst=" << dst << " local_expert=" << localExpert
                      << " rows=" << rows << " src_start=" << srcStart << " dst_start=" << dstStart << std::endl;
        }
    }
}

void LaunchCombineAndMeasure(const MoeCombineArgs &args, RuntimeState *state)
{
    uint32_t launchBlocks = args.shape.aivBlocks == 0 ? 1 : args.shape.aivBlocks;
    MpiBarrier(&state->mpi);
    auto combineStart = std::chrono::steady_clock::now();
    LaunchMoeCombineKernel(
        args.shape, state->rank, reinterpret_cast<uint8_t *>(state->buffers.expertOutput),
        reinterpret_cast<uint8_t *>(state->buffers.probs), reinterpret_cast<uint8_t *>(state->buffers.outputC),
        reinterpret_cast<uint8_t *>(state->buffers.routeMeta), reinterpret_cast<uint8_t *>(state->hccl.peerWindow),
        reinterpret_cast<uint8_t *>(state->hccl.deviceContext), reinterpret_cast<uint8_t *>(state->buffers.workspace),
        state->computeStream, launchBlocks);
    CheckAcl(aclrtSynchronizeStream(state->computeStream),
             "rank " + std::to_string(state->rank) + " combine stream sync");
    auto combineEnd = std::chrono::steady_clock::now();
    state->combineE2eUs = UsSince(combineStart, combineEnd);
    MpiBarrier(&state->mpi);
}

void DumpAndCheckCombineReturn(const MoeCombineArgs &args, const PeerWindowLayout &peerWindowLayout,
                               RuntimeState *state)
{
    if (args.runtime.combineReturnOnly != 0 || args.runtime.debug >= 2) {
        CombineReturnDump dump;
        CopyCombineReturnToHost(args, peerWindowLayout, state, &dump);
        if (args.runtime.debug != 0) {
            std::cout << "rank=" << state->rank << " combine_done_signal=";
            for (size_t i = 0; i < dump.combineDoneSignal.size(); ++i) {
                if (i != 0) {
                    std::cout << ",";
                }
                std::cout << dump.combineDoneSignal[i];
            }
            std::cout << std::endl;
            WriteBinaryFile(RankBinaryFile(args, state->rank, "actual_ptrD_head"), FloatVectorToHalfBits(dump.ptrD));
        }
        uint64_t ptrDMismatches = CompareFloatBuffer(args, "ptrD", dump.ptrD, state->golden.ptrD, state->rank);
        if (ptrDMismatches != 0) {
            throw std::runtime_error("rank " + std::to_string(state->rank) + " combine return mismatch");
        }
    }
}

void PrintCombineCompletionLogs(const MoeCombineArgs &args, RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose && args.runtime.combineReturnOnly != 0) {
        std::cout << "rank=" << state->rank << " combine_return_done" << std::endl;
        std::cout << "rank=" << state->rank << " combine_wait_done peers=" << args.shape.ep << std::endl;
    } else if (verbose) {
        std::cout << "rank=" << state->rank << " combine_return_done" << std::endl;
        std::cout << "rank=" << state->rank << " combine_wait_done peers=" << args.shape.ep << std::endl;
        std::cout << "rank=" << state->rank << " restore_done" << std::endl;
    }
}

void RunCombine(const MoeCombineArgs &args, const PeerWindowLayout &peerWindowLayout, RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "combine", "begin");
    }
    ClearCombineReturnState(args, peerWindowLayout, state);
    PrintCombineReturnSegments(args, state);
    LaunchCombineAndMeasure(args, state);
    DumpAndCheckCombineReturn(args, peerWindowLayout, state);
    PrintCombineCompletionLogs(args, state);
    if (verbose) {
        PrintStage(state->rank, "combine", "done");
    }
}

void VerifyAndDump(const MoeCombineArgs &args, RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    if (verbose) {
        PrintStage(state->rank, "verify", "begin");
    }
    size_t elements = static_cast<size_t>(args.shape.m) * args.shape.k;
    std::vector<float> actualOutputC = CopyDeviceHalfToFloat(state->buffers.outputC, elements, state->rank, "outputC");
    if (args.runtime.verify == 0) {
        if (verbose) {
            std::cout << "rank=" << state->rank << " verify=SKIP mismatch_count=0" << std::endl;
            PrintStage(state->rank, "verify", "done");
        }
        return;
    }
    if (args.runtime.debug != 0) {
        WriteBinaryFile(RankBinaryFile(args, state->rank, "outputC"), FloatVectorToHalfBits(actualOutputC));
    }
    CompareResult compare = CompareOutputs(args, state->golden, actualOutputC, state->rank);
    if (verbose || compare.mismatchCount != 0) {
        std::cout << "rank=" << state->rank << " buffer=outputC elements=" << compare.elementCount
                  << " mismatch_count=" << compare.mismatchCount;
    }
    if (compare.mismatchCount != 0) {
        uint64_t row = compare.firstMismatchIndex / args.shape.k;
        uint64_t col = compare.firstMismatchIndex % args.shape.k;
        std::cout << " first_index=" << compare.firstMismatchIndex << " row=" << row << " col=" << col
                  << " actual=" << compare.actual << " expected=" << compare.expected;
    }
    if (verbose || compare.mismatchCount != 0) {
        std::cout << std::endl;
    }
    if (compare.mismatchCount != 0) {
        std::cout << "rank=" << state->rank << " verify=FAIL mismatch_count=" << compare.mismatchCount << std::endl;
        throw std::runtime_error("rank " + std::to_string(state->rank) + " outputC mismatch");
    }
    if (verbose) {
        std::cout << "rank=" << state->rank << " verify=PASS mismatch_count=0" << std::endl;
        PrintStage(state->rank, "verify", "done");
    }
}

void PrintPerformanceConfig(const MoeCombineArgs &args, const WorkspaceLayout &workspaceLayout,
                            const CombineRouteMetaLayout &routeMetaLayout, const PeerWindowLayout &peerWindowLayout,
                            uint32_t rank)
{
    if (!VerboseRuntimeLogs(args)) {
        return;
    }
    uint32_t aivBlocks = args.shape.aivBlocks == 0 ? 1 : args.shape.aivBlocks;
    uint32_t peerShards = args.shape.ep < aivBlocks ? args.shape.ep : aivBlocks;
    std::cout << "rank=" << rank << " aiv_blocks=" << aivBlocks << " tile_cols=" << kMoeCombineTileCols
              << " peer_window_bytes=" << peerWindowLayout.totalBytes
              << " route_meta_bytes=" << routeMetaLayout.totalBytes << " workspace_bytes=" << workspaceLayout.totalBytes
              << " combine_peer_shards=" << peerShards << std::endl;
}

void FreeDeviceBuffers(RuntimeState *state)
{
    if (!state->buffersAllocated) {
        return;
    }
    if (state->buffers.probs != nullptr) {
        aclrtFree(state->buffers.probs);
    }
    if (state->buffers.outputC != nullptr) {
        aclrtFree(state->buffers.outputC);
    }
    if (state->buffers.routeMeta != nullptr) {
        aclrtFree(state->buffers.routeMeta);
    }
    if (state->buffers.workspace != nullptr) {
        aclrtFree(state->buffers.workspace);
    }
    if (state->buffers.expertOutput != nullptr) {
        aclrtFree(state->buffers.expertOutput);
    }
    state->buffers = DeviceBuffers{};
    state->buffersAllocated = false;
}

void DestroyRuntimeStreams(RuntimeState *state)
{
    if (state->hcclStream != nullptr) {
        rtStreamDestroy(state->hcclStream);
        state->hcclStream = nullptr;
    }
    if (state->computeStream != nullptr) {
        aclrtDestroyStream(state->computeStream);
        state->computeStream = nullptr;
    }
}

void Cleanup(RuntimeState *state)
{
    if (state == nullptr) {
        return;
    }
    if (state->mpiActive) {
        try {
            MpiBarrier(&state->mpi);
        } catch (const std::exception &ex) {
            std::cerr << "[WARN] rank=" << state->rank << " cleanup MPI barrier failed: " << ex.what() << "\n";
        }
    }
    if (state->hcclActive) {
        DestroyHcclWindowContext(&state->hccl);
        state->hcclActive = false;
    }
    FreeDeviceBuffers(state);
    DestroyRuntimeStreams(state);
    if (state->aclActive) {
        aclrtResetDevice(static_cast<int32_t>(state->device));
        aclFinalize();
        state->aclActive = false;
    }
    if (state->mpiActive) {
        FinalizeMpi(&state->mpi);
        state->mpiActive = false;
    }
}

void PrintLayoutSummary(const WorkspaceLayout &workspaceLayout, const CombineRouteMetaLayout &routeMetaLayout,
                        const PeerWindowLayout &peerWindowLayout, uint64_t hcclBuffSizeMb)
{
    std::cout << "workspace_bytes=" << workspaceLayout.totalBytes << "\n";
    std::cout << "route_meta_bytes=" << routeMetaLayout.totalBytes << "\n";
    std::cout << "peer_window_bytes=" << peerWindowLayout.totalBytes << "\n";
    std::cout << "HCCL_BUFFSIZE=" << hcclBuffSizeMb << std::endl;
}

void InitDeviceRuntime(const MoeCombineArgs &args, const WorkspaceLayout &workspaceLayout,
                       const CombineRouteMetaLayout &routeMetaLayout, const PeerWindowLayout &peerWindowLayout,
                       RuntimeState *state)
{
    PrepareHostData(args, state);
    BindDeviceContinuous(args, state);
    CreateStreams(args, state);
    InitHccl(state, args, peerWindowLayout);
    AllocateLocalBuffers(args, workspaceLayout, routeMetaLayout, state);
    CopyInputsToDevice(args, state);
}

bool HandleHostGoldenExit(const MoeCombineArgs &args, RuntimeState *state)
{
    if (args.runtime.hostGoldenOnly != 0) {
        RunHostGoldenOnly(args, state);
        Cleanup(state);
        return true;
    }
    return false;
}

bool HandleSkipKernelExit(const MoeCombineArgs &args, RuntimeState *state)
{
    if (args.runtime.skipKernels != 0) {
        MpiBarrier(&state->mpi);
        std::cout << "rank=" << state->rank << " skip_kernels_done" << std::endl;
        Cleanup(state);
        return true;
    }
    return false;
}

void RecordMeasuredTiming(bool isWarmup, RuntimeState *state, std::vector<IterationTiming> *measureTimings)
{
    state->totalE2eUs = state->combineE2eUs;
    if (isWarmup) {
        return;
    }
    IterationTiming timing;
    timing.prepareHostUs = state->prepareHostUs;
    timing.combineE2eUs = state->combineE2eUs;
    timing.totalE2eUs = state->totalE2eUs;
    measureTimings->push_back(timing);
}

bool RunOneIteration(const MoeCombineArgs &args, const WorkspaceLayout &workspaceLayout,
                     const CombineRouteMetaLayout &routeMetaLayout, const PeerWindowLayout &peerWindowLayout,
                     bool isWarmup, RuntimeState *state, std::vector<IterationTiming> *measureTimings)
{
    ClearDeviceState(args, workspaceLayout, routeMetaLayout, peerWindowLayout, state);
    PrepareCombineFixture(args, routeMetaLayout, state);
    RunCombine(args, peerWindowLayout, state);
    if (args.runtime.combineReturnOnly != 0) {
        MpiBarrier(&state->mpi);
        std::cout << "rank=" << state->rank << " run_done" << std::endl;
        return true;
    }
    VerifyAndDump(args, state);
    RecordMeasuredTiming(isWarmup, state, measureTimings);
    return false;
}

int RunIterations(const MoeCombineArgs &args, const WorkspaceLayout &workspaceLayout,
                  const CombineRouteMetaLayout &routeMetaLayout, const PeerWindowLayout &peerWindowLayout,
                  RuntimeState *state)
{
    bool verbose = VerboseRuntimeLogs(args);
    PrintPerformanceConfig(args, workspaceLayout, routeMetaLayout, peerWindowLayout, state->rank);
    uint32_t totalIterations = args.runtime.warmup + args.runtime.iters;
    std::vector<IterationTiming> measureTimings;
    measureTimings.reserve(args.runtime.iters);
    for (uint32_t iter = 0; iter < totalIterations; ++iter) {
        bool isWarmup = iter < args.runtime.warmup;
        if (verbose && !isWarmup) {
            std::cout << "rank=" << state->rank << " iteration=" << (iter - args.runtime.warmup)
                      << " phase=measure begin" << std::endl;
        }
        if (RunOneIteration(args, workspaceLayout, routeMetaLayout, peerWindowLayout, isWarmup, state,
                            &measureTimings)) {
            return 0;
        }
        if (verbose && !isWarmup) {
            std::cout << "rank=" << state->rank << " iteration=" << (iter - args.runtime.warmup)
                      << " phase=measure done" << std::endl;
        }
    }
    PrintProfileSummary(args, state, measureTimings);
    MpiBarrier(&state->mpi);
    return 0;
}

int RunMoeCombine(int argc, char **argv, RuntimeState *state)
{
    MoeCombineArgs args = ParseArgs(argc, argv);
    ValidateArgs(args);
    WorkspaceLayout workspaceLayout = ComputeWorkspaceLayout(args.shape);
    CombineRouteMetaLayout routeMetaLayout = ComputeCombineRouteMetaLayout(args.shape);
    PeerWindowLayout peerWindowLayout = ComputePeerWindowLayout(args.shape);
    uint64_t hcclBuffSizeMb = args.runtime.hcclBuffSizeMb == 0 ? EstimateHcclBuffSizeMb(args.shape, peerWindowLayout) :
                                                                 args.runtime.hcclBuffSizeMb;
    PrintRunSummary(args);
    PrintLayoutSummary(workspaceLayout, routeMetaLayout, peerWindowLayout, hcclBuffSizeMb);
    InitRankInfo(args, &argc, &argv, state);
    if (VerboseRuntimeLogs(args)) {
        std::cout << "rank=" << state->rank << " size=" << state->size << " device=" << state->device << " start"
                  << std::endl;
    }
    if (HandleHostGoldenExit(args, state)) {
        return 0;
    }
    InitDeviceRuntime(args, workspaceLayout, routeMetaLayout, peerWindowLayout, state);
    if (HandleSkipKernelExit(args, state)) {
        return 0;
    }
    int ret = RunIterations(args, workspaceLayout, routeMetaLayout, peerWindowLayout, state);
    if (VerboseRuntimeLogs(args) && args.runtime.combineReturnOnly == 0) {
        std::cout << "rank=" << state->rank << " run_done" << std::endl;
    }
    Cleanup(state);
    return ret;
}

} // namespace moe_combine

int main(int argc, char **argv)
{
    moe_combine::RuntimeState state;
    try {
        return moe_combine::RunMoeCombine(argc, argv, &state);
    } catch (const std::exception &ex) {
        std::cerr << "[ERROR] rank=" << state.rank << " " << ex.what() << std::endl;
        moe_combine::Cleanup(&state);
        return 1;
    }
}
