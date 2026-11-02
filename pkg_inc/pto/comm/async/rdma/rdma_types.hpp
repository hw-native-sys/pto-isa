/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// RDMA device context and request types. The Host-safe workspace layout lives
// in rdma_workspace_types.hpp; HNS1825 NIC-visible layouts live under
// async/rdma/backends/hns_1825/.

#ifndef PTO_COMM_ASYNC_RDMA_TYPES_HPP
#define PTO_COMM_ASYNC_RDMA_TYPES_HPP

#include "pto/comm/async_common/async_types.hpp"
#include "pto/comm/async/rdma/rdma_workspace_types.hpp"

namespace pto {
namespace comm {
namespace rdma {

constexpr uint32_t kRdmaScratchBytes = 256U;

struct RdmaTmpBuffer {
    __ubuf__ uint8_t* addr{nullptr};
    uint32_t size{0};
};

namespace detail {

template <typename ScratchTile>
PTO_INTERNAL bool MakeTmpBufferFromTile(ScratchTile& scratchTile, RdmaTmpBuffer& tmpBuf)
{
    static_assert(is_tile_data_v<ScratchTile>, "scratchTile must be a pto::Tile type");
    static_assert(ScratchTile::Loc == TileType::Vec, "scratchTile must be in Vec(UB) memory");
    tmpBuf.addr = reinterpret_cast<__ubuf__ uint8_t*>(scratchTile.data());
    tmpBuf.size = static_cast<uint32_t>(ScratchTile::Numel * sizeof(typename ScratchTile::DType));
    return tmpBuf.addr != nullptr && tmpBuf.size >= sizeof(uint64_t);
}

} // namespace detail

struct RdmaExecContext {
    __gm__ uint8_t* contextGm{nullptr};
    RdmaBackend backend{RdmaBackend::NONE};
    uint32_t destRankId{0};
    uint32_t qpIdx{0};
    uint32_t myPe{0};
    RdmaTmpBuffer tmpBuf{};
    uint32_t syncId{0};
};

struct RdmaEventContext {
    __gm__ uint8_t* contextGm{nullptr};
    RdmaTmpBuffer tmpBuf{};
    RdmaBackend backend{RdmaBackend::NONE};
    uint32_t syncId{0};
};

struct RdmaSession {
    RdmaExecContext execCtx{};
    RdmaEventContext eventCtx{};
    uint8_t valid{0};
    uint8_t resv[7]{};
};

enum class RdmaOpcode : uint32_t {
    OP_RDMA_READ = 0,
    OP_RDMA_WRITE,
    OP_RDMA_WRITE_WITH_IMM,
};

struct RdmaSge {
    __gm__ uint8_t* addr;
    uint64_t length;
    uint32_t lkey;
};

struct RdmaSendWr {
    __gm__ uint8_t* remote_addr;
    __gm__ uint8_t* local_addr;
    uint64_t message_len;
    uint32_t send_flags;
    uint32_t imm_data;
    uint32_t rkey;
    uint32_t lkey;
};

} // namespace rdma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_RDMA_TYPES_HPP
