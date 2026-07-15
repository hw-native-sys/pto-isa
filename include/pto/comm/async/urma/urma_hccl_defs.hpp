/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_URMA_HCCL_DEFS_HPP
#define PTO_COMM_ASYNC_URMA_HCCL_DEFS_HPP

// Host-only definitions for HCCL-based URMA workspace setup.
// Contains:
//   1. Supplemental HCCL API declarations (HcclCommMemReg, etc.) not exposed
//      in the standard CANN public headers.
//   2. Minimal ChannelEntity / SqContext / CqContext / RegedBufferEntity
//      layout (ABI-compatible with hcomm) — hcomm_res_entity_defs.h is not
//      in the standard CANN include path so we define them here.

#include <cstdint>

#include "hccl/hccl_res.h"
#include "hccl/hccl_rank_graph.h"

// ============================================================================
// Supplemental HCCL API declarations
// ============================================================================
#ifdef __cplusplus
extern "C" {
#endif

#ifndef HCCL_MEM_HANDLE_DEFINED
#define HCCL_MEM_HANDLE_DEFINED
typedef void* HcclMemHandle;
#endif

extern HcclResult HcclCommMemReg(HcclComm comm, const char* memTag, const CommMem* mem, HcclMemHandle* memHandle);

extern HcclResult HcclChannelGetRemoteMems(
    HcclComm comm, ChannelHandle channel, uint32_t* memNum, CommMem** remoteMems, char*** memTags);

#ifdef __cplusplus
}
#endif

// ============================================================================
// ChannelEntity / SqContext / CqContext / RegedBufferEntity
// (ABI-compatible with hcomm/hcomm_res_entity_defs.h)
// ============================================================================
#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    REGED_BUFFER_INVALID = -1,
    REGED_BUFFER_IPC = 0,
    REGED_BUFFER_RMA = 1,
} RegedBufferType;

typedef struct {
    int32_t type;
    union {
        struct {
            uint32_t tokenId;
            uint32_t tokenValue;
        } ub;
        uint8_t raws[24];
    } memInfo;
} ProtectionInfo;

typedef struct {
    RegedBufferType type;
    union {
        struct {
            uint64_t addr;
            uint64_t size;
            ProtectionInfo protectionInfo;
        } rma;
        uint8_t raws[56];
    } bufferInfo;
} RegedBufferEntity;

typedef struct {
    int32_t type;
    union {
        struct {
            uint64_t sqVa;
            uint64_t headAddr;
            uint64_t tailAddr;
            uint64_t dbVa;
            uint32_t jfsID;
            uint32_t wqeSize;
            uint32_t sqDepth;
            uint32_t tpID;
            uint8_t remoteEID[16];
        } ubJfs;
        uint8_t raws[120];
    } contextInfo;
} SqContext;

typedef struct {
    int32_t type;
    union {
        struct {
            uint64_t scqVa;
            uint64_t headAddr;
            uint64_t tailAddr;
            uint64_t dbVa;
            uint32_t jfcID;
            uint32_t cqeSize;
            uint32_t cqDepth;
        } ubJfc;
        uint8_t raws[120];
    } contextInfo;
} CqContext;

typedef struct {
    CommAbiHeader abiHeader;
    CommEngine engine;
    CommProtocol protocol;
    uint32_t localNotifyNum;
    uint32_t remoteNotifyNum;
    uint32_t localBufferNum;
    uint32_t remoteBufferNum;
    uint32_t sqNum;
    uint32_t cqNum;
    void* localNotifyAddr;
    void* remoteNotifyAddr;
    RegedBufferEntity* localBufferAddr;
    RegedBufferEntity* remoteBufferAddr;
    SqContext* sqContextAddr;
    CqContext* cqContextAddr;
    uint8_t reserve[160];
} ChannelEntity;

#ifdef __cplusplus
}
#endif

#endif // PTO_COMM_ASYNC_URMA_HCCL_DEFS_HPP
