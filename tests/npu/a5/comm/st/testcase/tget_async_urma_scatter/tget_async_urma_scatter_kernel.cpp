/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>
#include <pto/pto-inst.hpp>

#include "../common.hpp"
#include "tget_async_urma_scatter_kernel.h"

namespace {
constexpr uint32_t kMaxDescriptors = 512U;
constexpr uint32_t kRemoteOffset = 256U;
constexpr uint32_t kLocalOffset = kRemoteOffset + 16U * kMaxDescriptors;
constexpr uint32_t kRecordsOffset = 32768U;
constexpr uint32_t kSourceOffset = 65536U;
constexpr uint32_t kDataBytes = 8U * 1024U * 1024U;
constexpr uint32_t kDestinationOffset = kSourceOffset + kDataBytes;
constexpr uint32_t kWorkspaceBytes = kDestinationOffset + kDataBytes;
struct TransferRecord {
    uint32_t peer, offset, segments, length;
};
inline uint32_t ExpectedWord(uint32_t peer, uint32_t offset) { return 0x10000000U + peer * 0x01000000U + offset / 4U; }

#ifdef PTO_URMA_SUPPORTED
using namespace pto::comm;
using namespace pto::comm::urma;
namespace ud = pto::comm::urma::detail;

template <bool Get>
struct Descriptors {
    using Remote = std::conditional_t<Get, ScatterSource, GatherDestination>;
    using Local = std::conditional_t<Get, ScatterSegment, GatherSegment>;
    __gm__ Remote* remote;
    __gm__ Local* local;
    AICORE AsyncEvent Submit(uint32_t ranges, uint32_t segments, const AsyncSession& session)
    {
        DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(remote), ranges * sizeof(Remote));
        DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(local), segments * sizeof(Local));
        dsb(DSB_DDR);
        if constexpr (Get) {
            return TGET_ASYNC<DmaEngine::URMA>(local, segments, remote, ranges, session);
        } else {
            return TPUT_ASYNC<DmaEngine::URMA>(remote, ranges, local, segments, session);
        }
    }
    AICORE void Set(uint32_t index, TransferRecord record, __gm__ uint8_t* buffer, __gm__ uint8_t* workspace)
    {
        remote[index].addr = reinterpret_cast<decltype(remote[0].addr)>(
            UrmaPeerMrBaseAddr(workspace, record.peer) + (Get ? kSourceOffset : kDestinationOffset) + record.offset);
        remote[index].segCount = record.segments;
        remote[index].peer = record.peer;
        for (uint32_t segment = 0U; segment < record.segments; ++segment) {
            local[index * record.segments + segment].addr =
                buffer + (Get ? kDestinationOffset : kSourceOffset) + record.offset + segment * (record.length + 64U);
            local[index * record.segments + segment].length = record.length;
        }
    }
};

AICORE bool CheckReads(__gm__ uint8_t* buffer, __gm__ TransferRecord* records, uint32_t count)
{
    for (uint32_t i = 0U; i < count; ++i) {
        const TransferRecord r{records[i].peer, records[i].offset, records[i].segments, records[i].length};
        for (uint32_t segment = 0U; segment < r.segments; ++segment) {
            for (uint32_t byte = 0U; byte < r.length; byte += 4U) {
                const auto* address = reinterpret_cast<__gm__ uint32_t*>(
                    buffer + kDestinationOffset + r.offset + segment * (r.length + 64U) + byte);
                const uint32_t expected =
                    0x10000000U + r.peer * 0x01000000U + (r.offset + segment * r.length + byte) / 4U;
                if (ld_dev(const_cast<__gm__ uint32_t*>(address), 0) != expected) {
                    return false;
                }
            }
        }
    }
    return true;
}

template <bool Get>
AICORE bool RunDescriptors(
    __gm__ uint8_t* buffer, __gm__ uint8_t* workspace, UrmaDescriptorCase testCase, uint32_t ranks, bool sharedPool)
{
    using C = UrmaDescriptorCase;
    using D = Descriptors<Get>;
    D descriptors{
        reinterpret_cast<__gm__ typename D::Remote*>(buffer + kRemoteOffset),
        reinterpret_cast<__gm__ typename D::Local*>(buffer + kLocalOffset)};
    __gm__ auto* records = reinterpret_cast<__gm__ TransferRecord*>(buffer + kRecordsOffset);
    AsyncSession session;
    pto::Tile<pto::TileType::Vec, uint8_t, 1, sizeof(UrmaGatherSimtScratch)> scratch;
    TASSIGN(scratch, 0x0);
    if (!BuildAsyncSession<DmaEngine::URMA>(workspace, scratch, session)) {
        return false;
    }
    __gm__ auto* wq = ud::GetWqContextAt(session, 1U, session.qpIdxBase);
    const uint32_t depth = wq->depth;
    const uint32_t initialHead = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0);
    if (testCase == C::Invalid) {
        AsyncEvent missing;
        if constexpr (Get) {
            missing = TGET_ASYNC<DmaEngine::URMA>(nullptr, 1U, nullptr, 1U, session);
        } else {
            missing = TPUT_ASYNC<DmaEngine::URMA>(nullptr, 1U, nullptr, 1U, session);
        }
        if (!missing.postFailed || missing.Wait(session) || missing.Test(session)) {
            return false;
        }
        for (uint32_t bad = 0U; bad < 8U; ++bad) {
            descriptors.Set(0U, {1U, 0U, 1U, 64U}, buffer, workspace);
            if (bad == 0U) {
                descriptors.local[0].length = 0U;
            }
            if (bad == 1U) {
                descriptors.local[0].addr = nullptr;
            }
            if (bad == 2U) {
                descriptors.remote[0].addr = nullptr;
            }
            if (bad == 3U) {
                descriptors.remote[0].peer = ranks;
            }
            if (bad == 4U) {
                descriptors.local[0].length = 256U * 1024U * 1024U + 1U;
            }
            if (bad == 5U) {
                descriptors.local[0].addr = buffer + kWorkspaceBytes - 32U;
            }
            if (bad == 6U) {
                descriptors.remote[0].addr = reinterpret_cast<decltype(descriptors.remote[0].addr)>(
                    UrmaPeerMrBaseAddr(workspace, 1U) + kWorkspaceBytes - 32U);
            }
            if (bad == 7U) {
                descriptors.remote[0].segCount = 2U;
            }
            const auto event = descriptors.Submit(1U, 1U, session);
            if (!event.postFailed || event.Wait(session) || event.Test(session) ||
                ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0) != initialHead) {
                return false;
            }
        }
        return true;
    }
    const bool ring = testCase == C::Ring;
    const bool large = testCase == C::LargeFinal || testCase == C::LargeEmptyFinal;
    const bool prefix = testCase == C::Prefix || testCase == C::EmptyFinal || testCase == C::Deferred || large || ring;
    const bool empty = testCase == C::EmptyFinal || testCase == C::Deferred || testCase == C::LargeEmptyFinal || ring;
    const uint32_t rounds = ring ? 2U * depth + 5U : (large ? 2U : (prefix ? 4U : 1U));
    uint32_t total = 0U, offset = 0U, queueMask = 0U;
    for (uint32_t round = 0U; round < rounds; ++round) {
        const bool last = round + 1U == rounds;
        const uint32_t segments =
            testCase == C::FullSge ?
                12U :
                (testCase == C::Split13 ? 13U : (testCase == C::Split48 ? 48U : (testCase == C::Multiple ? 2U : 1U)));
        const uint32_t length = large && round == 0U ? 4U * 1024U * 1024U : 64U;
        uint32_t count = (testCase == C::Multiple || testCase == C::Split13 || testCase == C::Split48) ? 7U : 1U;
        if (ring && round == 0U) {
            count = depth - 1U;
        }
        if (last && empty) {
            count = 0U;
        }
        if (count > kMaxDescriptors || count * segments > kMaxDescriptors) {
            return false;
        }
        session.cfg.urma.cqePolicy = prefix && !(ring && round == 0U) ?
                                         (last ? UrmaCqePolicy::Final : UrmaCqePolicy::NoCqe) :
                                         UrmaCqePolicy::OneCqe;
        session.submitMode = testCase == C::Deferred || (ring && round != 0U) ?
                                 (last ? AsyncSubmitMode::DEFER_AND_SUBMIT : AsyncSubmitMode::DEFER) :
                                 AsyncSubmitMode::IMMEDIATE;
        session.batchSize = testCase == C::Deferred ? 2U : UINT32_MAX;
        for (uint32_t i = 0U; i < count; ++i) {
            const TransferRecord record{1U + total % (ranks - 1U), offset, segments, length};
            if (offset + segments * (length + 64U) > kDataBytes || total >= 1024U) {
                return false;
            }
            records[total].peer = record.peer;
            records[total].offset = record.offset;
            records[total].segments = record.segments;
            records[total++].length = record.length;
            descriptors.Set(i, record, buffer, workspace);
            offset += segments * (length + 64U);
            const uint32_t sqes = Get ? segments : (segments + 11U) / 12U;
            for (uint32_t j = 0U; j < sqes; ++j) {
                const uint32_t queue = (session.rt.urma.nextWqeSeq + i * sqes + j) % session.cfg.urma.qpCount;
                queueMask |= 1U << ((sharedPool ? 0U : record.peer - 1U) * session.cfg.urma.qpCount + queue);
            }
        }
        if (ring && round == 0U) {
            descriptors.Set(count, {1U, offset, 1U, 64U}, buffer, workspace);
            const auto rejected = descriptors.Submit(count + 1U, count + 1U, session);
            if (!rejected.postFailed || rejected.Wait(session) ||
                ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0) != initialHead) {
                return false;
            }
        }
        auto event = descriptors.Submit(count, count * segments, session);
        if (event.postFailed) {
            return false;
        }
        if (session.cfg.urma.cqePolicy == UrmaCqePolicy::NoCqe) {
            if (!event.valid() || event.Test(session) || event.Wait(session)) {
                return false;
            }
            if (testCase == C::Deferred && round < 2U &&
                ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0) != initialHead + 2U * round) {
                return false;
            }
        } else {
            uint32_t expected = count * (Get ? segments : (segments + 11U) / 12U);
            if (session.cfg.urma.cqePolicy == UrmaCqePolicy::Final) {
                expected = 0U;
                for (uint32_t mask = queueMask; mask != 0U; mask >>= 1U) {
                    expected += mask & 1U;
                }
            }
            if (event.CompletionRecordCount(session) != expected) {
                return false;
            }
            (void)event.Test(session);
            if (!event.Wait(session)) {
                return false;
            }
            // Consume all READ results here, before a stream/MPI barrier can mask a missing fence.
            if constexpr (Get) {
                if (!CheckReads(buffer, records, total)) {
                    return false;
                }
            }
        }
    }
    if (ring) {
        const uint32_t head = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0);
        const uint32_t recoveries = (rounds - 3U) / (depth - 1U);
        if (head != initialHead + total + recoveries + 1U ||
            ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->tailAddr), 0) != head ||
            wq->submittedWqeCount != depth - 1U + recoveries + 1U) {
            return false;
        }
    }
    if (ring) {
        // Mirror Gather PUT's CQ owner-wrap coverage, independently of SQ BB recovery.
        auto* cq = ud::GetCqContextAt(session, 1U, session.qpIdxBase);
        session.cfg.urma.cqePolicy = UrmaCqePolicy::OneCqe;
        session.submitMode = AsyncSubmitMode::IMMEDIATE;
        for (uint32_t i = 0U; i <= cq->depth; ++i) {
            if (!descriptors.Submit(1U, 1U, session).Wait(session)) {
                return false;
            }
        }
        if (ld_dev(reinterpret_cast<__gm__ uint32_t*>(cq->tailAddr), 0) <= cq->depth) {
            return false;
        }
    }
    reinterpret_cast<__gm__ uint32_t*>(buffer)[1] = total;
    return true;
}
#endif

template <bool Get>
__global__ AICORE void UrmaDescriptorKernel(
    __gm__ uint8_t* buffer, __gm__ uint8_t* workspace, uint32_t rank, uint32_t ranks, UrmaDescriptorCase testCase,
    bool sharedPool)
{
    pipe_barrier(PIPE_ALL);
#ifdef PTO_URMA_SUPPORTED
    if (rank == 0U) {
        reinterpret_cast<__gm__ int32_t*>(buffer)[0] =
            RunDescriptors<Get>(buffer, workspace, testCase, ranks, sharedPool) ? 1 : -1;
    }
#endif
    pipe_barrier(PIPE_ALL);
}

bool RunRank(
    int rank, int ranks, int devices, int firstDevice, int firstRank, int root, UrmaDescriptorCase testCase, bool get,
    uint32_t queues, bool sharedPool)
{
    UrmaTestContext context;
    if (!context.Setup(
            rank, ranks, devices, firstDevice, root, kWorkspaceBytes,
            sharedPool ? UrmaLayout::SHARED_POOL : UrmaLayout::PER_PEER, 1U, queues, 64U)) {
        return false;
    }
    std::vector<uint32_t> data(kWorkspaceBytes / 4U, 0xA5A5A5A5U);
    data[0] = data[1] = 0U;
    for (uint32_t offset = 0U; offset < kDataBytes; offset += 4U) {
        data[(kSourceOffset + offset) / 4U] = ExpectedWord(rank - firstRank, offset);
    }
    bool success =
        aclrtMemcpy(context.devBuf, kWorkspaceBytes, data.data(), kWorkspaceBytes, ACL_MEMCPY_HOST_TO_DEVICE) ==
        ACL_SUCCESS;
    CommMpiBarrier();
    auto* buffer = reinterpret_cast<uint8_t*>(context.devBuf);
    auto* workspace = reinterpret_cast<uint8_t*>(context.urmaMgr.GetWorkspaceAddr());
    if (get) {
        UrmaDescriptorKernel<true>
            <<<1, nullptr, context.stream>>>(buffer, workspace, rank - firstRank, ranks, testCase, sharedPool);
    } else {
        UrmaDescriptorKernel<false>
            <<<1, nullptr, context.stream>>>(buffer, workspace, rank - firstRank, ranks, testCase, sharedPool);
    }
    const int sync = aclrtSynchronizeStream(context.stream);
    CommMpiBarrier();
    success = aclrtMemcpy(data.data(), kWorkspaceBytes, context.devBuf, kWorkspaceBytes, ACL_MEMCPY_DEVICE_TO_HOST) ==
                  ACL_SUCCESS &&
              success && sync == ACL_SUCCESS;
    uint32_t metadata[2] = {data[0], data[1]};
    CommMpiBcast(metadata, sizeof(metadata), COMM_MPI_CHAR, root);
    std::vector<TransferRecord> records(1024U);
    if (rank == root) {
        std::memcpy(records.data(), data.data() + kRecordsOffset / 4U, records.size() * sizeof(TransferRecord));
    }
    CommMpiBcast(records.data(), records.size() * sizeof(TransferRecord), COMM_MPI_CHAR, root);
    success = success && metadata[0] == 1U && metadata[1] <= records.size();
    for (uint32_t i = 0U; success && i < metadata[1]; ++i) {
        const auto r = records[i];
        if ((get && rank != root) || (!get && rank - firstRank != static_cast<int>(r.peer))) {
            continue;
        }
        for (uint32_t segment = 0U; segment < r.segments; ++segment) {
            for (uint32_t byte = 0U; byte < r.length; byte += 4U) {
                const uint32_t local = r.offset + segment * (r.length + 64U) + byte;
                const uint32_t remote = r.offset + segment * r.length + byte;
                success = success && data[(kDestinationOffset + (get ? local : remote)) / 4U] ==
                                         ExpectedWord(get ? r.peer : 0U, get ? remote : local);
            }
            if (get) {
                success =
                    success &&
                    data[(kDestinationOffset + r.offset + segment * (r.length + 64U) + r.length) / 4U] == 0xA5A5A5A5U;
            }
        }
    }
    if (!success) {
        std::cerr << "Descriptor case=" << static_cast<uint32_t>(testCase) << " get=" << get << " rank=" << rank
                  << " status=" << metadata[0] << " sync=" << sync << std::endl;
    }
    context.Cleanup();
    return success;
}
} // namespace

bool RunUrmaDescriptorCase(UrmaDescriptorCase testCase, bool get, uint32_t queues, bool sharedPool, int ranks)
{
    return RunUrmaTestMpiLaunch(
        ranks, ranks, 0, 0, [=](int rank, int n, int devices, int firstDevice, int firstRank, int root) {
            return RunRank(rank, n, devices, firstDevice, firstRank, root, testCase, get, queues, sharedPool);
        });
}
