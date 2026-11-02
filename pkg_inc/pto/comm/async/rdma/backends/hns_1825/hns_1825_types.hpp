/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// HNS_1825 PTO queue descriptors, NIC-visible WQE/CQE layouts, and constants.

#ifndef PTO_COMM_ASYNC_RDMA_BACKENDS_HNS_1825_TYPES_HPP
#define PTO_COMM_ASYNC_RDMA_BACKENDS_HNS_1825_TYPES_HPP

#include <cstdint>

#include "pto/comm/async/rdma/rdma_workspace_types.hpp"

namespace pto {
namespace comm {
namespace rdma {
namespace hns_1825 {

// One zero-initialized, 64-byte record per peer/QP. All device accesses
// to these fields use scalar atomics; session-local UB is never shared.
struct QueueState {
    uint64_t reserveHead;
    uint64_t readyHead;
    uint64_t postedHead;
    uint64_t completedHead;
    uint64_t postLock;
    uint64_t cqLock;
    uint64_t error;
    uint64_t reserved;
};

struct SqContext {
    uint64_t bufAddr;
    uint64_t headAddr;
    uint64_t tailAddr;
    uint64_t dbAddr;
    uint64_t amoAddr;
    uint64_t dbSwAddr;
    uint64_t stateAddr;
    uint32_t wqn;
    uint32_t wqeSize;
    uint32_t depth;
    uint32_t sl;
    uint32_t amoLkey;
    RdmaDbMode dbMode;
    uint8_t mtuShift{4};
    uint8_t dbCos{0x7};
    uint8_t reserved[6];
};

struct CqContext {
    uint64_t bufAddr;
    uint64_t headAddr;
    uint64_t tailAddr;
    uint64_t dbAddr;
    uint64_t dbSwAddr;
    uint32_t cqn;
    uint32_t cqeSize;
    uint32_t depth;
    RdmaDbMode dbMode;
};

struct WqeCtrlSeg {
    uint8_t owner_sl;
    uint8_t df_tsl;
    uint16_t wf_bdsl;
    uint32_t cl_pi;
    uint64_t db;
};

union WqeTaskCommonSeg {
    struct {
        uint32_t xrc_srqn : 18;
        uint32_t ext : 1;
        uint32_t dif : 1;
        uint32_t rsvd : 3;
        uint32_t so : 1;
        uint32_t opcode : 5;
        uint32_t signal : 1;
        uint32_t fence : 1;
        uint32_t se : 1;
    } bs;
    uint32_t value;
};

struct WqeRdmaTaskSeg {
    WqeTaskCommonSeg com_tsk;
    uint32_t data_len;
    uint32_t imm_data;
    union {
        struct {
            uint32_t last_ext_len : 8;
            uint32_t cmd_len : 8;
            uint32_t pi : 16;
        } bs;
        uint32_t value;
    } dw3;
    uint64_t va;
    uint32_t rkey;
    uint32_t ulp;
};

struct WqeDataSeg {
    uint64_t buf_addr;
    uint32_t r_len;
    uint32_t le_key;
};

struct Cqe {
    uint32_t owner_id_qpn;
    uint32_t op_sr_wqebb;
    uint32_t byte_cnt;
    uint32_t imm_data;
    uint32_t rsvd_dw5;
    uint32_t wqe_num;
    uint32_t vlan_queue_index;
    uint8_t syndrome;
    uint8_t rsvd;
    uint16_t wqe_counter;
};

union SqDoorbell {
    struct {
        uint64_t qpn : 20;
        uint64_t cntx_size : 2;
        uint64_t rsvd0 : 1;
        uint64_t c : 1;
        uint64_t cos : 3;
        uint64_t type : 5;
        uint64_t pi : 8;
        uint64_t rsvd1 : 8;
        uint64_t xrc_vld : 1;
        uint64_t rsvd2 : 1;
        uint64_t mtu_shift : 3;
        uint64_t sgid_index : 7;
        uint64_t sub_type : 4;
    } bs;
    uint64_t value;
};

constexpr uint32_t kWriteReadWqeSize = 64;
constexpr uint32_t kWqebbSize = 64;
constexpr uint32_t kDefaultCqeSize = 64;
constexpr uint32_t kPollCqThreshold = 10;
constexpr uint64_t kMaxTransferBytes = 0x7fffffffULL;
constexpr uint32_t kMaxSyncId = 7;
constexpr uint32_t kPollCqTimeoutError = 0x10000;
constexpr uint32_t kSessionBuildError = 0x20000;
constexpr uint32_t kInvalidArgumentError = 0x20001;
constexpr uint32_t kInvalidContextError = 0x20002;
constexpr uint32_t kCqeError = 0x20003;
// Handles retain their peer/u32-index encoding. Stop before the index can wrap
// into the zero/no-op handle or alias a live event from an earlier generation.
constexpr uint32_t kQueueIndexExhaustedError = 0x20004;
constexpr uint64_t kMaxQueueIndex = 0xffffffffULL;
constexpr uint32_t kCqProgressBatch = 8;
constexpr uint32_t kPostSendBatch = 32;
// One registered 4 KiB operand ring per peer. Reuse follows the signal WQE's CQ completion.
constexpr uint32_t kNotifyOperandSlots = 1024;

static_assert(sizeof(QueueState) == 64, "unexpected HNS1825 queue-state layout");
static_assert(sizeof(SqContext) == 88 && alignof(SqContext) == 8, "unexpected HNS1825 SQ context layout");
static_assert(sizeof(CqContext) == 56 && alignof(CqContext) == 8, "unexpected HNS1825 CQ context layout");

#if defined(PTO_NPU_ARCH_A2A3) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
constexpr uint64_t kCycleToTimeBase = 50;
#else
constexpr uint64_t kCycleToTimeBase = 1000;
#endif
constexpr uint64_t kPollCqTimeoutUs = 60ULL * 1000 * 1000;
constexpr uint64_t kPollCqTimeoutCycles = kPollCqTimeoutUs * kCycleToTimeBase;

// Backend-local aliases preserve the vocabulary used by the verified HNS WQE
// implementation while keeping NIC layouts inside the HNS1825 backend.
using RoceSqCtx = SqContext;
using RoceCqCtx = CqContext;
using Hns1825WqeCtrlSeg = WqeCtrlSeg;
using Hns1825WqeRdmaTaskSeg = WqeRdmaTaskSeg;
using Hns1825WqeDataSeg = WqeDataSeg;
using Hns1825Cqe = Cqe;
using Hns1825SqDb = SqDoorbell;
constexpr uint32_t kHns1825WriteReadWqeSize = kWriteReadWqeSize;
constexpr uint32_t kHns1825WqebbSize = kWqebbSize;
constexpr uint32_t kHns1825DefaultCqeSize = kDefaultCqeSize;
constexpr uint32_t kHns1825PollCqThreshold = kPollCqThreshold;
constexpr uint64_t kHns1825MaxTransferBytes = kMaxTransferBytes;
constexpr uint32_t kHns1825MaxSyncId = kMaxSyncId;
constexpr uint32_t kHns1825PollCqTimeoutError = kPollCqTimeoutError;
constexpr uint32_t kHns1825SessionBuildError = kSessionBuildError;
constexpr uint32_t kHns1825InvalidArgumentError = kInvalidArgumentError;
constexpr uint32_t kHns1825InvalidContextError = kInvalidContextError;
constexpr uint32_t kHns1825CqeError = kCqeError;
constexpr uint64_t kHns1825CycleToTimeBase = kCycleToTimeBase;
constexpr uint64_t kHns1825PollCqTimeoutCycles = kPollCqTimeoutCycles;

static_assert(sizeof(WqeCtrlSeg) == 16, "unexpected HNS1825 WQE control-segment layout");
static_assert(sizeof(WqeRdmaTaskSeg) == 32, "unexpected HNS1825 WQE task-segment layout");
static_assert(sizeof(WqeDataSeg) == 16, "unexpected HNS1825 WQE data-segment layout");
static_assert(sizeof(Cqe) == 32, "unexpected HNS1825 CQE layout");
static_assert(sizeof(SqDoorbell) == 8, "unexpected HNS1825 SQ doorbell layout");

} // namespace hns_1825
} // namespace rdma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_RDMA_BACKENDS_HNS_1825_TYPES_HPP
