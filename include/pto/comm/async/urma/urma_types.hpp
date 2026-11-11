/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_URMA_TYPES_HPP
#define PTO_COMM_ASYNC_URMA_TYPES_HPP

#include <cstddef>
#include <cstdint>

namespace pto {
namespace comm {
namespace urma {

// ============================================================================
// Constants
// ============================================================================
constexpr uint64_t kUrmaCyclesPerUs = 1000ULL;
constexpr uint64_t kUrmaPollCqTimeoutUs = 10ULL * 1000ULL * 1000ULL;
constexpr uint64_t kUrmaPollCqTimeoutCycles = kUrmaPollCqTimeoutUs * kUrmaCyclesPerUs;

constexpr uint32_t kUrmaPollCycleStride = 256U;
constexpr uint64_t kCacheLineSize = 64;
constexpr size_t kUrmaEidBytes = 16;

// Sizes the per-AIV jetty arrays. Kept here so a host translation unit can
// include the URMA workspace types without pulling the AICORE tile headers
// that comm_types.hpp includes.
constexpr uint32_t kUrmaMaxJettiesPerCore = 8U;

constexpr uint32_t kUrmaCompletionRecordMinCqDepth = 128U;

constexpr uint64_t kUrmaMaxWqeTransferBytes = 256ULL * 1024ULL * 1024ULL;

constexpr uint32_t kUrmaAutoAivCount = UINT32_MAX;

constexpr uint32_t kUrmaAutoJettiesPerCore = 0U;

constexpr uint32_t kUrmaMaxRowsPerPeer = 256;

constexpr uint32_t kUrmaSqeSizeBytes = 48;
constexpr uint32_t kUrmaSgeSizeBytes = 16;
constexpr uint32_t kUrmaSqeRmtEidLOffset = 16;
constexpr uint32_t kUrmaSqeRmtEidHOffset = 24;
constexpr uint32_t kUrmaSqeRmtAddrLOffset = 40;
constexpr uint32_t kUrmaSqeRmtAddrHOffset = 44;

constexpr uint32_t kUrmaNotifySetSlotCount = 8U;
constexpr uint32_t kUrmaNotifyResourceCachelineBytes = 64U;
constexpr uint32_t kUrmaMaxSgesPerSqe = 12U;

// Bounds for the plan-free SIMT gather fast path.
constexpr uint32_t kUrmaGatherSimtWarpSize = 32U;
constexpr uint32_t kUrmaGatherSimtWarps = 32U;
constexpr uint32_t kUrmaGatherSimtMaxQueues = 16U;
// Largest payload-SQE count routed to the scalar path. A 4-rank A5 sweep over
// 1/2/6/12 SGEs and 1/4 queues puts the robust single-threshold crossover at
// two SQEs: low-SGE N=1/2 favors scalar, while N>=4 favors SIMT overall.
constexpr uint32_t kUrmaGatherScalarSqeThreshold = 2U;

constexpr uint32_t kUrmaGatherSimtMaxSlotBbs = (kUrmaSqeSizeBytes + kUrmaMaxSgesPerSqe * kUrmaSgeSizeBytes + 63U) / 64U;
static_assert(
    kUrmaGatherSimtMaxSlotBbs * 8U <= kUrmaGatherSimtWarpSize,
    "A fixed SQE slot must fit one 64-bit word per lane of a single warp");

constexpr uint32_t kDefaultInflightGatherEventsPerAiv = 64U;
constexpr uint16_t kUrmaGatherInlineEvent = UINT16_MAX - 2U;
constexpr uint16_t kUrmaGatherNoCompletionEvent = UINT16_MAX - 1U;
constexpr uint16_t kUrmaGatherSlotEvent = UINT16_MAX;

struct UrmaNotifyResourceRegion {
    int32_t setValues[kUrmaNotifySetSlotCount];
    uint32_t nextSetSlot;
    uint32_t setRingStarted;
    int32_t faaResult;
    uint8_t reserved
        [kUrmaNotifyResourceCachelineBytes - sizeof(setValues) - sizeof(nextSetSlot) - sizeof(setRingStarted) -
         sizeof(faaResult)];
};

static_assert(
    sizeof(UrmaNotifyResourceRegion) == kUrmaNotifyResourceCachelineBytes,
    "URMA notify resource region layout mismatch");
static_assert(alignof(UrmaNotifyResourceRegion) == alignof(uint32_t), "URMA notify resource region alignment mismatch");

// ============================================================================
// UrmaOpcode — URMA operation codes (binary-compatible with hcomm UB ABI)
// ============================================================================
enum class UrmaOpcode : uint32_t {
    SEND = 0,
    SEND_WITH_IMM,
    SEND_WITH_INV,
    WRITE,
    WRITE_WITH_IMM,
    WRITE_WITH_NOTIFY,
    READ,
    CAS,
    ATOMIC_SWAP,
    ATOMIC_STORE,
    ATOMIC_LOAD,
    FAA = 0xb,
    NOP = 0x11
};

// ============================================================================
// UrmaLayout — how the SQ/CQ context table is indexed
//
// PER_PEER:    one private run of jettiesPerCore queues per AIV on every peer.
//              Queue aiv * jettiesPerCore + slot reaches only that peer. The
//              runs are created at init, so a block never shares a WQ. A ctx
//              row is (peer, queue).
// SHARED_POOL: a pool of jetties, each reaching every peer. Peer and queue are
//              chosen independently, so a ctx row is a jetty. The pool is
//              carved into one exclusive run per AIV. The default run contains
//              one queue; an explicit jettiesPerCore > 1 gives each AIV more
//              queue slots. Host-side weighted placement distributes all queue
//              instances over the available rails. Device code rotates only
//              within the current AIV's run.
// ============================================================================
enum class UrmaLayout : uint32_t { PER_PEER = 0, SHARED_POOL = 1 };

// ============================================================================
// UrmaInfo — device-side workspace root
// ============================================================================
struct UrmaInfo {
    uint32_t rowsPerPeer; // SQ/CQ ctx rows occupied per peer; >= 1
    uint32_t rankCount;
    uint64_t sqPtr;
    uint64_t rqPtr;
    uint64_t scqPtr;
    uint64_t rcqPtr;
    uint64_t memPtr;
    uint64_t notifyPoolPtr;
    UrmaLayout layout;
    uint32_t jettyCount;
    uint32_t jettiesPerCore;
    uint64_t gatherRuntimeBase;
    uint32_t gatherRuntimeStride;
    uint32_t gatherCompletionSlotStride;
    uint32_t gatherQueueStageCapacity;
    uint64_t localMrBase;
    uint64_t localMrLength;
};

enum class UrmaGatherQueueState : uint8_t {
    Clean = 0,
    OpenNoCqePrefix = 1,
    FinalQueued = 2,
};

enum class UrmaGatherSlotState : uint8_t {
    Allocated = 0,
    Published = 1,
    Retired = 2,
};

struct alignas(16) UrmaGatherAivRuntimeHeader {
    uint32_t queueStageCapacity;
    uint32_t reserved[3];
};
static_assert(sizeof(UrmaGatherAivRuntimeHeader) == 16U, "URMA gather runtime header layout mismatch");

struct alignas(16) UrmaGatherCompletionPoolState {
    uint32_t allocateSequence;
    uint32_t reclaimSequence;
    uint32_t nextGeneration;
    uint32_t reserved;
};
static_assert(sizeof(UrmaGatherCompletionPoolState) == 16U, "URMA gather completion pool layout mismatch");

struct alignas(16) UrmaGatherQueueStage {
    uint32_t publishedBbSequence;
    uint32_t nextBbSequence;
    uint32_t publishedCqeSequence;
    uint32_t nextCqeSequence;
    uint32_t stagedOperationCount;
    uint32_t noCqePrefixBbCount;
    uint32_t lastPeer;
    UrmaGatherQueueState state;
    uint8_t initialized;
    uint8_t reserved[2];
};
static_assert(sizeof(UrmaGatherQueueStage) == 32U, "URMA gather queue stage layout mismatch");

struct alignas(16) UrmaGatherQueueSubmission {
    uint32_t peer;
    uint16_t jettyIndex;
    uint16_t stageIndex;
    uint32_t nextBbSequence;
    uint32_t remainingSqeCount;
    uint32_t payloadBbCount;
    uint32_t requestedCqeCount;
    uint32_t beginBbSequence;
    uint32_t endBbSequence;
    uint32_t beginCqeSequence;
    uint32_t endCqeSequence;
    uint8_t finalizeOnly;
    uint8_t reserved[3];
};
static_assert(sizeof(UrmaGatherQueueSubmission) == 48U, "URMA gather queue submission layout mismatch");

struct alignas(16) UrmaGatherSubmissionPlan {
    uint32_t queueCount;
    uint32_t payloadSqeCount;
    uint64_t completionSlotAddress;
    uint32_t completionSlotGeneration;
    uint32_t nextWqeSeq;
    uint8_t slotBbCount;
};
static_assert(sizeof(UrmaGatherSubmissionPlan) == 32U, "URMA gather submission plan layout mismatch");

// ============================================================================
// UB scratch for the plan-free SIMT gather fast path
// ============================================================================
// Everything the two SIMT passes share lives here instead of in GM: the
// per-queue ring coordinates the fill needs, and the atomics that hand out SQE
// slots. Keeping the counters in UB is what makes a per-SQE atomic affordable.
struct UrmaGatherSimtQueueEntry {
    uint64_t sqBaseAddress;
    uint32_t workQueueDepth;
    uint32_t baseBbSequence;
    uint32_t sqeCount;
    uint32_t fillCounter;
};
static_assert(sizeof(UrmaGatherSimtQueueEntry) == 24U, "URMA gather SIMT queue entry layout mismatch");

struct UrmaGatherSimtActiveStage {
    uint32_t stageKey;
    uint32_t sqeCount;
    uint32_t queueIndex;
    uint32_t peer;
};
static_assert(sizeof(UrmaGatherSimtActiveStage) == 16U, "URMA gather SIMT active-stage layout mismatch");

struct UrmaGatherSimtScratch {
    UrmaGatherSimtActiveStage activeStages[kUrmaGatherSimtMaxQueues];
    uint32_t warpSegmentSum[kUrmaGatherSimtWarps];
    uint32_t warpSegmentBase[kUrmaGatherSimtWarps];
    uint32_t warpSqeSum[kUrmaGatherSimtWarps];
    uint32_t warpSqeBase[kUrmaGatherSimtWarps];
    uint32_t maxSegmentCount;
    uint32_t rejected;
    uint32_t reserved[6];
    UrmaGatherSimtQueueEntry queues[kUrmaGatherSimtMaxQueues];
};
static_assert(sizeof(UrmaGatherSimtScratch) == 1184U, "URMA gather SIMT scratch layout mismatch");

struct UrmaCompletionTarget {
    uint32_t peer;
    uint16_t jettyIndex;
    uint16_t reserved;
    uint32_t targetBbSequence;
    uint32_t beginCqeSequence;
    uint32_t endCqeSequence;
};
static_assert(sizeof(UrmaCompletionTarget) == 20U, "URMA gather completion target layout mismatch");

struct UrmaGatherCompletionSlotHeader {
    uint32_t generation;
    uint32_t targetCount;
    uint32_t slotBytes;
    uint8_t cqePolicy;
    UrmaGatherSlotState state;
    uint16_t reserved;
};
static_assert(sizeof(UrmaGatherCompletionSlotHeader) == 16U, "URMA gather completion slot layout mismatch");

// ============================================================================
// UrmaMemInfo — per-peer memory authentication info
// ============================================================================
struct UrmaMemInfo {
    uint64_t addr;
    uint64_t eidAddr;
    uint32_t tpn;
    uint32_t tid;
    uint32_t rmtTokenValue;
    uint32_t len;
    uint32_t localTokenId;
    uint32_t notifyTokenId;
    uint8_t tokenValueValid;
    uint8_t rmtJettyType;
    uint8_t targetHint;
    uint8_t reserved[5];
};

// ============================================================================
// UrmaDbMode — doorbell mode for URMA queues
// ============================================================================
enum class UrmaDbMode : int32_t { INVALID_DB = -1, HW_DB = 0, SW_DB };

// ============================================================================
// UrmaWQCtx — send/receive work queue context
// ============================================================================
struct UrmaWQCtx {
    uint64_t bufAddr;
    uint64_t headAddr;
    uint64_t tailAddr;
    uint64_t dbAddr;
    uint32_t wqn;
    uint32_t wqeShiftSize;
    uint32_t depth;
    UrmaDbMode dbMode;
    uint32_t sl;
    uint32_t reserved[2];
    // Keep the mutable CQ-accounting counter at byte 60 in each 64-byte row.
    uint32_t submittedWqeCount;
};

static_assert(sizeof(UrmaWQCtx) == 64U, "URMA WQ context layout mismatch");

// ============================================================================
// UrmaCqCtx — completion queue context
// ============================================================================
struct UrmaCqCtx {
    uint64_t bufAddr;
    uint64_t headAddr;
    uint64_t tailAddr;
    uint64_t dbAddr;
    uint32_t cqn;
    uint32_t cqeShiftSize;
    uint32_t depth;
    UrmaDbMode dbMode;
};

// Independent producer requirements for the WQ BB ring and CQE ring.
// The producer sequences already include any software-staged entries.
struct UrmaQueueCapacityRequest {
    uint32_t bbProducerSequence;
    uint32_t cqeProducerSequence;
    uint32_t requiredBbCount;
    uint32_t requiredCqeCount;
};

// ============================================================================
// UrmaSqeCtx — 48-byte SQE (ABI-compatible with hcomm UB JFS WQE layout)
// ============================================================================
struct UrmaSqeCtx {
    /* byte 0 - 4 */
    uint32_t sqeBbIdx : 16;
    uint32_t flag : 8;
    uint32_t rsv0 : 3;
    uint32_t nf : 1;
    uint32_t tokenEn : 1;
    uint32_t rmtJettyType : 2;
    uint32_t owner : 1;
    /* byte 4 - 8 */
    uint32_t targetHint : 8;
    uint32_t opcode : 8;
    uint32_t rsv1 : 6;
    uint32_t inlineMsgLen : 10;
    /* byte 8 - 12 */
    uint32_t tpId : 24;
    uint32_t sgeNum : 8;
    /* byte 12 - 16 */
    uint32_t rmtJettyOrSegId : 20;
    uint32_t rsv2 : 12;
    /* byte 16 - 32 */
    uint64_t rmtEidL;
    uint64_t rmtEidH;
    /* byte 32 - 36 */
    uint32_t rmtTokenValue;
    /* byte 36 - 40 */
    uint32_t udfType : 8;
    uint32_t reduceDataType : 4;
    uint32_t reduceOpcode : 4;
    uint32_t rsv3 : 16;
    /* byte 40 - 48 */
    uint32_t rmtAddrLOrTokenId;
    uint32_t rmtAddrHOrTokenValue;
};

// ============================================================================
// UrmaSgeCtx — scatter-gather entry
// ============================================================================
struct UrmaSgeCtx {
    uint32_t len;
    uint32_t tokenId;
    uint64_t va;
};

// ============================================================================
// UrmaJfcCqeCtx — 64-byte CQE (ABI-compatible with hcomm UB JFC CQE layout)
// ============================================================================
struct UrmaJfcCqeCtx {
    /* DW0 */
    uint32_t sR : 1;
    uint32_t isJetty : 1;
    uint32_t owner : 1;
    uint32_t inlineEn : 1;
    uint32_t opcode : 3;
    uint32_t fd : 1;
    uint32_t rsv : 8;
    uint32_t substatus : 8;
    uint32_t status : 8;
    /* DW1 */
    uint32_t entryIdx : 16;
    uint32_t localNumL : 16;
    /* DW2 */
    uint32_t localNumH : 4;
    uint32_t rmtIdx : 20;
    uint32_t rsv1 : 8;
    /* DW3 */
    uint32_t tpn : 24;
    uint32_t rsv2 : 8;
    /* DW4 */
    uint32_t byteCnt;
    /* DW5 ~ DW6 */
    uint32_t userDataL;
    uint32_t userDataH;
    /* DW7 ~ DW10 */
    uint32_t rmtEid[4];
    /* DW11 ~ DW12 */
    uint32_t dataL;
    uint32_t dataH;
    /* DW13 ~ DW15 */
    uint32_t inlineData[3];
};

} // namespace urma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_URMA_TYPES_HPP
