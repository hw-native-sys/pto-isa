/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_URMA_WORKSPACE_TABLES_HPP
#define PTO_COMM_ASYNC_URMA_WORKSPACE_TABLES_HPP

#if defined(__CCE_KT_TEST__)
#error "urma_workspace_tables.hpp is a host-only header and cannot be included in device code."
#endif

#include <cstdint>

#include "hccl/hccl.h"
#include "hccl/hccl_res.h"

#include "pto/comm/async/urma/urma_types.hpp"
#include "pto/comm/async/urma/urma_hccl_defs.hpp"

namespace pto {
namespace comm {
namespace urma {
namespace detail {

inline uint32_t Log2U32(uint32_t n) { return (n <= 1) ? 0 : __builtin_ctz(n); }

inline void FillWqCtx(UrmaWQCtx& wq, const SqContext& sq)
{
    wq.wqn = sq.contextInfo.ubJfs.jfsID;
    wq.bufAddr = sq.contextInfo.ubJfs.sqVa;
    wq.wqeShiftSize = Log2U32(sq.contextInfo.ubJfs.wqeSize);
    wq.depth = sq.contextInfo.ubJfs.sqDepth;
    wq.headAddr = sq.contextInfo.ubJfs.headAddr;
    wq.tailAddr = sq.contextInfo.ubJfs.tailAddr;
    wq.dbMode = UrmaDbMode::SW_DB;
    wq.dbAddr = sq.contextInfo.ubJfs.dbVa;
    wq.sl = 0;
}

inline void FillCqCtx(UrmaCqCtx& cqCtx, const CqContext& cq)
{
    cqCtx.cqn = cq.contextInfo.ubJfc.jfcID;
    cqCtx.bufAddr = cq.contextInfo.ubJfc.scqVa;
    cqCtx.cqeShiftSize = Log2U32(cq.contextInfo.ubJfc.cqeSize);
    cqCtx.depth = cq.contextInfo.ubJfc.cqDepth;
    cqCtx.headAddr = cq.contextInfo.ubJfc.headAddr;
    cqCtx.tailAddr = cq.contextInfo.ubJfc.tailAddr;
    cqCtx.dbMode = UrmaDbMode::SW_DB;
    cqCtx.dbAddr = cq.contextInfo.ubJfc.dbVa;
}

inline void FillMemInfo(
    UrmaMemInfo& mem, const SqContext& sq, const RegedBufferEntity& symRemoteBuf, uint64_t symRmaAddr,
    uint32_t symRmaSize)
{
    mem.tokenValueValid = true;
    mem.rmtJettyType = 1;
    mem.targetHint = 0;
    mem.tpn = sq.contextInfo.ubJfs.tpID;
    mem.tid = symRemoteBuf.bufferInfo.rma.protectionInfo.memInfo.ub.tokenId;
    mem.rmtTokenValue = symRemoteBuf.bufferInfo.rma.protectionInfo.memInfo.ub.tokenValue;
    mem.len = symRmaSize;
    mem.addr = symRmaAddr;
}

} // namespace detail
} // namespace urma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_URMA_WORKSPACE_TABLES_HPP
