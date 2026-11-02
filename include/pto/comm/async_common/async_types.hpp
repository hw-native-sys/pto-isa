/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_ASYNC_TYPES_HPP
#define PTO_COMM_ASYNC_ASYNC_TYPES_HPP

#include <cstdint>
#include <climits>
#include "pto/comm/async_common/sdma_constants.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {
namespace sdma {

// ============================================================================
// Public SDMA constants (used by kernel code for buffer sizing, etc.)
// ============================================================================
constexpr uint32_t kSdmaFlagLength = 128U;
constexpr uint32_t kUbAlignSize = 256U;
constexpr uint32_t kSdmaEventRecordBytes = 16U;
constexpr uint32_t kSdmaEventSlotCount = kSdmaFlagLength / kSdmaEventRecordBytes;
constexpr uint32_t SDMA_FLAG_LENGTH = kSdmaFlagLength;
constexpr uint32_t UB_ALIGN_SIZE = kUbAlignSize;
constexpr uint32_t SDMA_EVENT_RECORD_BYTES = kSdmaEventRecordBytes;
constexpr uint32_t SDMA_EVENT_SLOT_COUNT = kSdmaEventSlotCount;

// ============================================================================
// SdmaBaseConfig: static transfer parameters supplied by caller context.
// ============================================================================
struct SdmaBaseConfig {
    uint64_t block_bytes;       // Block size per SQE
    uint64_t comm_block_offset; // Transfer offset for this operation
    uint32_t queue_num;         // Number of queues per core
};
using sdma_base_config_t = SdmaBaseConfig;

// ============================================================================
// Context types for SDMA async operations
// ============================================================================
struct TmpBuffer {
    __ubuf__ uint8_t* addr;
    uint32_t size;
};

struct SdmaExecContext {
    __gm__ uint8_t* contextGm;
    TmpBuffer tmpBuf;
    uint32_t syncId;
    uint32_t channelGroupIdx;
    SdmaBaseConfig baseConfig;
};

struct SdmaEventContext {
    TmpBuffer tmpBuf;
    uint32_t syncId;
};

namespace detail {

struct SdmaRuntimeContext {
    uint64_t nextPostId;
    // Single completion watermark: every queue in the confirmedQueueCount prefix is known to have
    // retired all posts up to confirmedPostId. A narrower observation leaves it untouched, which
    // only costs an extra GM poll.
    uint64_t confirmedPostId;
    uint8_t confirmedQueueCount;
    // Widest queue prefix a flag payload slot reuse may have to wait on. Slots are reused in postId
    // order, so this bounds the queues an old slot's flag could still be in flight on. A session
    // resumed on an existing workspace seeds it from the prefix its predecessor committed, because
    // the slots it inherits were fenced by that session rather than by this one.
    uint8_t lastPostQueueCount;
    // Explicit stand-in for the padding the compiler would otherwise insert before
    // usedQueueCount. Keeping it named means the layout is stated rather than implied.
    uint8_t resv[2];
    // Cumulative queue prefix used by this session. Every Post fences these queues. A resumed session
    // starts from its predecessor's final prefix rather than zero, which keeps the prefix monotonic
    // across every session that shares one workspace. It is persisted alongside nextPostId.
    uint32_t usedQueueCount;
    // Data SQEs staged by TPUT_ASYNC(DEFER) but not published yet.
    uint32_t batchStagedDataSqeCount;
    // Successful non-empty TPUT_ASYNC calls in the current physical batch.
    uint32_t batchStagedOperationCount;
    __gm__ uint8_t* postDoneBase;
};

} // namespace detail

// ============================================================================
// SdmaSession: bundles ExecContext + EventContext for convenient async usage.
// ============================================================================
struct SdmaSession {
    SdmaExecContext execCtx{};
    SdmaEventContext eventCtx{};
    mutable detail::SdmaRuntimeContext runtimeCtx{};
    uint8_t valid{0};
    uint8_t resv[7]{};
};

constexpr uint32_t kAutoChannelGroupIdx = UINT32_MAX;
constexpr uint64_t kDefaultSdmaBlockBytes = 1024 * 1024;

// Session configuration only the SDMA backend reads. Fields shared with another
// engine stay in AsyncSession itself.
//
// No default member initializers here, nor in the URMA and RDMA counterparts below: they are union
// alternatives, and assigning to an alternative only begins its lifetime when its type is trivially
// default constructible. A default member initializer would make the implicit default constructor
// non-trivial and turn every engine switch into undefined behaviour. The defaults these used to
// carry live in the builder signatures, which are their only writers anyway.
struct SdmaSessionConfig {
    uint32_t channelGroupIdx;
    uint32_t queueNum;
    uint64_t blockBytes;
    uint64_t commBlockOffset;
};

} // namespace sdma

namespace urma {

// Session configuration only the URMA backend reads.
struct UrmaSessionConfig {
    uint32_t qpCount;
};

namespace detail {

struct UrmaRuntimeContext {
    uint32_t batchStagedWqeCount;
    uint32_t batchStagedOperationCount;
    // WQE sequence within the logical batch. It is retained across automatic
    // physical submissions so Jetty selection keeps rotating.
    uint32_t batchLogicalWqeCount;
    uint32_t batchPeer;
    uint32_t railCycleCursor;
};

} // namespace detail
} // namespace urma

namespace rdma {

// Session configuration only the RDMA backend reads.
struct RdmaSessionConfig {
    RdmaBackend backend;
    uint32_t myPe;
};

} // namespace rdma

// ============================================================================
// Engine-exclusive session payloads.
//
// AsyncSession::engine is the discriminant for both unions below: a builder
// must set engine and the matching alternative together via
// detail::ActivateEngineState, and only that alternative may be read
// afterwards. Fields used by more than one engine are deliberately left
// outside, so the common paths need no union qualification.
//
// Neither union declares a constructor, which makes both aggregates: the cfg{}
// and rt{} initializers in AsyncSession then start the first alternative, and
// DmaEngine defaults to SDMA to match.
//
// Every alternative of both unions must stay trivially default constructible,
// which is why none of them carries a default member initializer. Adding one
// still compiles: it just stops the assignment in ActivateEngineState from
// beginning that alternative's lifetime, leaving every later read undefined.
// ============================================================================
union EngineSessionConfig {
    sdma::SdmaSessionConfig sdma;
    urma::UrmaSessionConfig urma;
    rdma::RdmaSessionConfig rdma;
};

namespace detail {

// RDMA keeps no cross-post runtime state, so it has no alternative here.
union EngineRuntimeContext {
    sdma::detail::SdmaRuntimeContext sdma;
    urma::detail::UrmaRuntimeContext urma;
};

} // namespace detail

// ============================================================================
// AsyncSession: engine-agnostic session for async DMA operations.
// Users build via comm::BuildAsyncSession<engine>() and pass to
// TPUT_ASYNC / TGET_ASYNC / event.Wait() without knowing engine internals.
// ============================================================================
struct AsyncSession {
    DmaEngine engine{DmaEngine::SDMA};
    uint8_t valid{0};
    AsyncSubmitMode submitMode{AsyncSubmitMode::IMMEDIATE};
    // Explicit stand-in for the padding the compiler would otherwise insert before
    // batchSize. Keeping it named means the layout is stated rather than implied.
    uint8_t resv{0};
    // Zero disables batching. UINT32_MAX effectively disables threshold-based
    // auto-submit while retaining explicit mode-controlled submission.
    uint32_t batchSize{UINT32_MAX};

    __gm__ uint8_t* contextGm{nullptr};

    // SDMA and RDMA.
    __ubuf__ uint8_t* tmpBufAddr{nullptr};
    uint32_t tmpBufSize{0};
    uint32_t syncId{0};

    // URMA and RDMA.
    uint32_t destRankId{0};
    uint32_t qpIdxBase{0};

    // Engine-exclusive configuration. Written once at build time.
    EngineSessionConfig cfg{};

    // Engine-exclusive persistent runtime state. The backend threads this across
    // successive posts/waits, so it is mutated even through const references
    // (mirrors sdma::SdmaSession::runtimeCtx). Kept separate from cfg so that
    // making it mutable does not also expose the configuration to writes.
    mutable detail::EngineRuntimeContext rt{};
};

namespace detail {

// The only place that switches a session between engines. Setting AsyncSession::engine on its own
// does not move either union's active alternative, and assigning a whole AsyncSession leaves both on
// SDMA, so each builder has to activate its own alternative explicitly. Routing that through one
// helper is what keeps the discriminant and the two unions from drifting apart.
template <DmaEngine engine>
PTO_INTERNAL void ActivateEngineState(AsyncSession& session)
{
    session.engine = engine;
    if constexpr (engine == DmaEngine::SDMA) {
        session.cfg.sdma = sdma::SdmaSessionConfig{};
        session.rt.sdma = sdma::detail::SdmaRuntimeContext{};
    } else if constexpr (engine == DmaEngine::URMA) {
        session.cfg.urma = urma::UrmaSessionConfig{};
        session.rt.urma = urma::detail::UrmaRuntimeContext{};
    } else {
        // RDMA keeps no cross-post runtime state, so session.rt has no alternative to activate and
        // is simply never read for it.
        session.cfg.rdma = rdma::RdmaSessionConfig{};
    }
}

} // namespace detail

} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_ASYNC_TYPES_HPP
