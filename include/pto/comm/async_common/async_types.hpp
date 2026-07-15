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

// ============================================================================
// SdmaSession: bundles ExecContext + EventContext for convenient async usage.
// ============================================================================
struct SdmaSession {
    SdmaExecContext execCtx{};
    SdmaEventContext eventCtx{};
    bool valid{false};
};

constexpr uint32_t kAutoChannelGroupIdx = UINT32_MAX;
constexpr uint64_t kDefaultSdmaBlockBytes = 1024 * 1024;

} // namespace sdma

// ============================================================================
// URMA context types for async operations (HCCP V2 Jetty, NPU_ARCH 3510 only)
// ============================================================================
namespace urma {

struct UrmaExecContext {
    __gm__ uint8_t* contextGm{nullptr};
    uint32_t destRankId{0};
    uint32_t qpIdx{0};
};

struct UrmaEventContext {
    __gm__ uint8_t* contextGm{nullptr};
};

struct UrmaSession {
    UrmaExecContext execCtx{};
    UrmaEventContext eventCtx{};
    bool valid{false};
};

} // namespace urma

// ============================================================================
// AsyncSession: engine-agnostic session for async DMA operations.
// Users build via comm::BuildAsyncSession<engine>() and pass to
// TPUT_ASYNC / TGET_ASYNC / event.Wait() without knowing engine internals.
// ============================================================================
struct AsyncSession {
    DmaEngine engine{DmaEngine::SDMA};
    sdma::SdmaSession sdmaSession{};
    urma::UrmaSession urmaSession{};
    bool valid{false};
};

} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_ASYNC_TYPES_HPP
