/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_SDMA_SDMA_TYPES_HPP
#define PTO_COMM_ASYNC_SDMA_SDMA_TYPES_HPP

#include <cstdint>
#include "pto/common/arch_macro.hpp"
#include "pto/comm/async/async_types.hpp"

namespace pto {
namespace comm {
namespace sdma {

// NPU-internal SDMA SQE constants
constexpr uint64_t kRtStarsSqeTypeSdma = 11ULL;
#ifdef PTO_NPU_ARCH_A5
constexpr uint64_t kCreditTimeDefault = 254ULL;
#else
constexpr uint64_t kCreditTimeDefault = 240ULL;
#endif
constexpr uint32_t kSqDepth = 2048U;
constexpr uint32_t kSdmaMaxChannel = 48U;

constexpr uint64_t RT_STARS_SQE_TYPE_SDMA = kRtStarsSqeTypeSdma;
constexpr uint64_t K_CREDIT_TIME_DEFAULT = kCreditTimeDefault;
constexpr uint32_t SQ_DEPTH = kSqDepth;
constexpr uint32_t SDMA_MAX_CHAN = kSdmaMaxChannel;

// ============================================================================
// SDMA Configuration Structure (full, with dynamic fields)
// ============================================================================
struct SdmaConfig {
    uint64_t block_bytes;       // Block size per SQE (typically 1MB)
    uint64_t per_core_bytes;    // Total bytes to transfer per core
    uint64_t comm_block_offset; // Offset for current core's data
    uint32_t queue_num;         // Number of queues per core
    uint32_t iter_num;          // Number of iterations (SQEs) needed
};
using sdma_config_t = SdmaConfig;

// ============================================================================
// Workspace Layout Structure
// ============================================================================
struct WorkspaceLayout {
    __gm__ uint8_t *send_workspace; // Local send flag workspace
    __gm__ uint8_t *recv_workspace; // Local receive flag workspace
};
using workspace_layout_t = WorkspaceLayout;

// ============================================================================
// Batch Write Flag Info Structure
// ============================================================================
struct BatchWriteFlagInfo {
    uint32_t flag;
    uint32_t totalQueueNum;
    uint8_t reserved[56]; // Padding to 64 bytes
};
using batch_write_flag_info_t = BatchWriteFlagInfo;

// ============================================================================
// Batch Write Channel Info Structure
// ============================================================================
struct BatchWriteChannelInfo {
    uint32_t sq_head;        // Send Queue head
    uint32_t sq_tail;        // Send Queue tail
    uint64_t sq_base;        // SQ buffer base address
    uint64_t sq_reg_base;    // SQ register base address
    uint32_t sq_depth;       // SQ depth
    uint32_t sq_id;          // SQ ID
    uint32_t cq_id;          // CQ ID
    uint32_t logic_cq_id;    // Logic CQ ID
    uint64_t cqe_addr;       // CQE address
    uint32_t report_cqe_num; // CQE report number
    uint32_t stream_id;      // Stream ID
    uint32_t dev_id;         // Device ID
    uint8_t reserved[4];     // Padding to 64 bytes
};
using batch_write_channel_info_t = BatchWriteChannelInfo;

// ============================================================================
// Batch Write Item (SQE) Structure
// ============================================================================
#ifdef PTO_NPU_ARCH_A5

struct BatchWriteItem {
    // Header (bytes 0-7)
    uint8_t type : 6;
    uint8_t lock : 1;
    uint8_t unlock : 1;
    uint8_t ie : 1;
    uint8_t preP : 1;
    uint8_t postP : 1;
    uint8_t wrCqe : 1;
    uint8_t ptrMode : 1;
    uint8_t rttMode : 1;
    uint8_t headUpdate : 1;
    uint8_t reserved0 : 1;
    uint16_t numBlocks;
    uint16_t rtStreamId;
    uint16_t taskId;

    // Words 2-3 (bytes 8-15)
    uint32_t res1;
    uint16_t res2;
    uint8_t kernelCredit;
    uint8_t res3;

    // Word 4 (bytes 16-19): A5 bit-field layout
    uint32_t opcode : 8;
    uint32_t sssv : 1;
    uint32_t dssv : 1;
    uint32_t sns : 1;
    uint32_t dns : 1;
    uint32_t sro : 1;
    uint32_t dro : 1;
    uint32_t stride : 2;
    uint32_t ie2 : 1;
    uint32_t compEn : 1;
    uint32_t res4 : 14;

    // Word 5 (bytes 20-23)
    uint16_t sqeId;
    uint8_t mapamPartId;
    uint8_t mpamns : 1;
    uint8_t pmg : 2;
    uint8_t qos : 4;
    uint8_t d2dOffsetFlag : 1;

    // Word 6 (bytes 24-27)
    uint16_t srcStreamId;
    uint16_t srcSubStreamId;

    // Word 7 (bytes 28-31): dstStreamId replaces A2/A3 length
    uint16_t dstStreamId;
    uint16_t dstSubStreamId;

    // Words 8-11 (bytes 32-47): src/dst addresses (same position as A2/A3)
    uint32_t srcAddrLow;
    uint32_t srcAddrHigh;
    uint32_t dstAddrLow;
    uint32_t dstAddrHigh;

    // Word 12 (bytes 48-51): transfer length (replaces A2/A3 linkType+reserved)
    uint32_t lengthMove;

    // Words 13-15 (bytes 52-63)
    uint32_t srcOffsetLow;
    uint32_t dstOffsetLow;
    uint16_t srcOffsetHigh;
    uint16_t dstOffsetHigh;
};

#else // A2/A3

struct BatchWriteItem {
    uint8_t type : 6;
    uint16_t res1 : 10;
    uint16_t blockDim;
    uint16_t rtStreamId;
    uint16_t taskId;

    uint32_t res3;

    uint16_t res4;
    uint8_t kernel_credit;
    uint8_t ptr_mode : 1;
    uint8_t res5 : 7;

    uint32_t opcode : 8;
    uint32_t ie2 : 1;
    uint32_t sssv : 1;
    uint32_t dssv : 1;
    uint32_t sns : 1;
    uint32_t dns : 1;
    uint32_t qos : 4;
    uint32_t sro : 1;
    uint32_t dro : 1;
    uint32_t partid : 8;
    uint32_t mpam : 1;
    uint32_t res6 : 4;

    uint16_t src_streamid;
    uint16_t src_sub_streamid;

    uint16_t dst_streamid;
    uint16_t dst_sub_streamid;

    uint32_t length;

    uint32_t srcAddrLow;
    uint32_t srcAddrHigh;
    uint32_t dstAddrLow;
    uint32_t dstAddrHigh;

    uint8_t linkType;
    uint8_t reserved[3];
    uint32_t reslast[3];
};

#endif // PTO_NPU_ARCH_A5
using batch_write_item_t = BatchWriteItem;

// ============================================================================
// SDMA Async Event Record (single slot)
// ============================================================================
struct SdmaEventRecord {
    uint32_t flag;         // Set by flag SQE (non-zero indicates completion)
    uint32_t sq_tail;      // Tail value to commit on completion
    uint64_t channel_info; // __gm__ batch_write_channel_info_t* (stored as uint64_t)
};
using sdma_event_record_t = SdmaEventRecord;

} // namespace sdma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_SDMA_SDMA_TYPES_HPP
