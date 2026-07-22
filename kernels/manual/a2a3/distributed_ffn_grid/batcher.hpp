/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software; you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to the License for details.
You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// GM-simulated Batcher for the WSE-FFN tile-graph demo.
//
// In the SVG tile graph an external **Batcher** module owns the FULL input and the
// FULL DRAM-resident weights, splits them column-parallel, broadcasts x to every
// core, and collects the H-sharded output.  A2/A3 has no such hardware, so this
// header simulates the Batcher entirely in GM:
//
//   LoadResidentWeights  -> full w_gate/w_up/w_down live in GM (DRAM-resident),
//                           mirroring the SVG "DRAM 常驻 全量权重" store.
//   DistributeWeights    -> slice those full weights column-parallel and write a
//                           contiguous per-column shard into a per-col GM region.
//                           Each core then TLOADs its own shard (DRAM->L1 stream),
//                           exactly like a core streaming its Batcher-delivered
//                           weight tile.  This host-side slice loop IS the
//                           GM-simulated Batcher distribution.
//   BroadcastInput       -> full x into GM; every column in a row reads the same
//                           x (broadcast, "复制 broadcast -> N 核").
//   CollectOutput        -> cores write their y shards (AllGather) / the EAST
//                           reduce writes the per-row sum (ReduceSum) straight
//                           into the Batcher y region of GM; collection is then
//                           just a D2H for verification.
//
// The compute kernel never sees "full" tensors -- it only reads its column's
// contiguous shard and writes its output shard, addressing them by (row, col).
// Reducing the per-cell duplication (x was copied per col, weights per row) to
// per-row x / per-col weights also matches the SVG memory model.

#ifndef DISTRIBUTED_FFN_GRID_BATCHER_HPP
#define DISTRIBUTED_FFN_GRID_BATCHER_HPP

#include <cstring>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "ffn_config.hpp"
// test_common.h (PtoTestCommon::ReadFile) and the aclrt* runtime API are expected
// to be available from the including TU (both main_*.cpp include them first).

// Host-side Batcher: owns the GM arena for full tensors + per-col shards + output.
struct FfnBatcher {
    // GM arena (Batcher storage, simulated by GM).
    void *x_full_dev = nullptr;          // [gridRows, T, H]            half
    void *w_gate_full_dev = nullptr;     // [H, F]                      half  (resident)
    void *w_up_full_dev = nullptr;       // [H, F]                      half  (resident)
    void *w_down_full_dev = nullptr;     // [F, H]                      half  (resident)
    void *w_gate_shards_dev = nullptr;   // [gridCols, H, Fi]           half  (distributed)
    void *w_up_shards_dev = nullptr;     // [gridCols, H, Fi]           half  (distributed)
    void *w_down_shards_dev = nullptr;   // [gridCols, ...] per mode    half  (distributed)
    void *y_full_dev = nullptr;          // [gridRows, T, H]            float (collected)

    // Cached host copies of the full weights (loaded once, sliced by Distribute).
    std::vector<uint8_t> host_w_gate_full;
    std::vector<uint8_t> host_w_up_full;
    std::vector<uint8_t> host_w_down_full;

    size_t rows = 0;
    size_t cols = 0;
    std::string split_mode = "reduce"; // "reduce" ([Fi,H] w_down shard) or "allgather" ([F,Hc])

    static void *MallocOrDie(size_t bytes, const char *what)
    {
        void *p = nullptr;
        if (aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS || p == nullptr) {
            std::cerr << "[ERROR] aclrtMalloc(" << what << ", " << bytes << ") failed" << std::endl;
            return nullptr;
        }
        aclrtMemset(p, bytes, 0, bytes);
        return p;
    }

    bool Allocate(size_t gridRows, size_t gridCols, const std::string &splitMode)
    {
        rows = gridRows;
        cols = gridCols;
        split_mode = splitMode;

        x_full_dev        = MallocOrDie(static_cast<size_t>(FFN_BATCHER_X_BYTES), "batcher.x_full");
        w_gate_full_dev   = MallocOrDie(static_cast<size_t>(FFN_BATCHER_W_GATE_FULL_BYTES), "batcher.w_gate_full");
        w_up_full_dev     = MallocOrDie(static_cast<size_t>(FFN_BATCHER_W_UP_FULL_BYTES), "batcher.w_up_full");
        w_down_full_dev   = MallocOrDie(static_cast<size_t>(FFN_BATCHER_W_DOWN_FULL_BYTES), "batcher.w_down_full");
        w_gate_shards_dev = MallocOrDie(static_cast<size_t>(FFN_BATCHER_W_GATE_SHARD_REGION_BYTES), "batcher.w_gate_shards");
        w_up_shards_dev   = MallocOrDie(static_cast<size_t>(FFN_BATCHER_W_UP_SHARD_REGION_BYTES), "batcher.w_up_shards");
        w_down_shards_dev = MallocOrDie(static_cast<size_t>(FFN_BATCHER_W_DOWN_SHARD_REGION_BYTES), "batcher.w_down_shards");
        y_full_dev        = MallocOrDie(static_cast<size_t>(FFN_BATCHER_Y_BYTES), "batcher.y_full");

        if (!x_full_dev || !w_gate_full_dev || !w_up_full_dev || !w_down_full_dev || !w_gate_shards_dev ||
            !w_up_shards_dev || !w_down_shards_dev || !y_full_dev) {
            return false;
        }
        std::cout << "[INFO] Batcher GM arena allocated (split_mode=" << split_mode << "): "
                  << "x_full=" << FFN_BATCHER_X_BYTES << "B, w_*_full=" << FFN_BATCHER_W_GATE_FULL_BYTES
                  << "B each, y_full=" << FFN_BATCHER_Y_BYTES << "B" << std::endl;
        return true;
    }

    // Read w_*_full.bin into host caches AND H2D into the resident full-weight GM.
    bool LoadResidentWeights(const std::string &dataDir)
    {
        struct Spec {
            const char *name;
            std::vector<uint8_t> *host;
            void *dev;
            size_t bytes;
        };
        Spec specs[] = {
            {"w_gate_full.bin", &host_w_gate_full, w_gate_full_dev, static_cast<size_t>(FFN_BATCHER_W_GATE_FULL_BYTES)},
            {"w_up_full.bin",   &host_w_up_full,   w_up_full_dev,   static_cast<size_t>(FFN_BATCHER_W_UP_FULL_BYTES)},
            {"w_down_full.bin", &host_w_down_full, w_down_full_dev, static_cast<size_t>(FFN_BATCHER_W_DOWN_FULL_BYTES)},
        };
        for (const auto &s : specs) {
            s.host->resize(s.bytes);
            std::string path = dataDir + "/" + s.name;
            size_t fileSize = 0;
            if (!PtoTestCommon::ReadFile(path, fileSize, s.host->data(), s.bytes) || fileSize != s.bytes) {
                std::cerr << "[ERROR] Batcher resident weight load mismatch: " << path << " (got " << fileSize
                          << " bytes, expected " << s.bytes << ")" << std::endl;
                return false;
            }
            if (aclrtMemcpy(s.dev, s.bytes, s.host->data(), s.bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
                std::cerr << "[ERROR] Batcher H2D resident weight failed: " << s.name << std::endl;
                return false;
            }
        }
        return true;
    }

    // Gather columns [colStart, colStart+shardCols) of a row-major [rows, cols] half
    // tensor into a contiguous [rows, shardCols] half dst (per-row memcpy).
    static void GatherColShard(const uint8_t *srcHalf, int rows, int cols, int colStart, int shardCols,
                               uint8_t *dstHalf)
    {
        const size_t bytes = static_cast<size_t>(shardCols) * FFN_HALF_ELEM_BYTES;
        for (int i = 0; i < rows; ++i) {
            const uint8_t *srcRow = srcHalf + (static_cast<size_t>(i) * cols + colStart) * FFN_HALF_ELEM_BYTES;
            uint8_t *dstRow = dstHalf + static_cast<size_t>(i) * shardCols * FFN_HALF_ELEM_BYTES;
            std::memcpy(dstRow, srcRow, bytes);
        }
    }

    // Slice the cached full weights column-parallel and H2D each per-col contiguous
    // shard into its GM region.  This loop is the GM-simulated Batcher distribution.
    bool DistributeWeights()
    {
        const int H = FFN_MODEL_TILE;
        const int Fi = FFN_FFN_TILE;
        const int F = FFN_FFN_TOTAL_TILE;
        const int Hc = FFN_MODEL_SHARD_TILE;
        const size_t gateShardBytes = static_cast<size_t>(FFN_W_GATE_BYTES); // [H, Fi]
        const size_t downShardBytes = static_cast<size_t>(FFN_W_DOWN_BYTES); // [F,Hc] or [Fi,H]

        std::vector<uint8_t> hostShard(std::max(gateShardBytes, downShardBytes));

        for (size_t col = 0; col < cols; ++col) {
            const int colStart = static_cast<int>(col);

            // w_gate / w_up: column-parallel along the intermediate dim F -> [H, Fi].
            GatherColShard(host_w_gate_full.data(), H, F, colStart * Fi, Fi, hostShard.data());
            if (aclrtMemcpy(static_cast<uint8_t *>(w_gate_shards_dev) + col * gateShardBytes, gateShardBytes,
                            hostShard.data(), gateShardBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
                std::cerr << "[ERROR] Batcher H2D w_gate shard col=" << col << " failed" << std::endl;
                return false;
            }
            GatherColShard(host_w_up_full.data(), H, F, colStart * Fi, Fi, hostShard.data());
            if (aclrtMemcpy(static_cast<uint8_t *>(w_up_shards_dev) + col * gateShardBytes, gateShardBytes,
                            hostShard.data(), gateShardBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
                std::cerr << "[ERROR] Batcher H2D w_up shard col=" << col << " failed" << std::endl;
                return false;
            }

            // w_down: split direction depends on the variant.
            //   allgather: columns [col*Hc:(col+1)*Hc] of [F, H] -> [F, Hc]
            //   reduce:    rows    [col*Fi:(col+1)*Fi] of [F, H] -> [Fi, H] (contiguous block)
            if (split_mode == "allgather") {
                GatherColShard(host_w_down_full.data(), F, H, colStart * Hc, Hc, hostShard.data());
                if (aclrtMemcpy(static_cast<uint8_t *>(w_down_shards_dev) + col * downShardBytes, downShardBytes,
                                hostShard.data(), downShardBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
                    std::cerr << "[ERROR] Batcher H2D w_down(allgather) shard col=" << col << " failed" << std::endl;
                    return false;
                }
            } else {
                const size_t byteOff = static_cast<size_t>(colStart) * Fi * H * FFN_HALF_ELEM_BYTES;
                if (aclrtMemcpy(static_cast<uint8_t *>(w_down_shards_dev) + col * downShardBytes, downShardBytes,
                                host_w_down_full.data() + byteOff, downShardBytes,
                                ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
                    std::cerr << "[ERROR] Batcher H2D w_down(reduce) shard col=" << col << " failed" << std::endl;
                    return false;
                }
            }
        }
        std::cout << "[INFO] Batcher distributed " << cols << " per-col weight shards into GM" << std::endl;
        return true;
    }

    // Read x_full.bin -> GM x_full (each row's x is read by every col = broadcast).
    bool BroadcastInput(const std::string &dataDir)
    {
        std::vector<uint8_t> hostX(static_cast<size_t>(FFN_BATCHER_X_BYTES));
        std::string path = dataDir + "/x_full.bin";
        size_t fileSize = 0;
        if (!PtoTestCommon::ReadFile(path, fileSize, hostX.data(), hostX.size()) || fileSize != hostX.size()) {
            std::cerr << "[ERROR] Batcher x_full load mismatch: " << path << " (got " << fileSize << " bytes, expected "
                      << hostX.size() << ")" << std::endl;
            return false;
        }
        if (aclrtMemcpy(x_full_dev, hostX.size(), hostX.data(), hostX.size(), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            std::cerr << "[ERROR] Batcher H2D x_full failed" << std::endl;
            return false;
        }
        std::cout << "[INFO] Batcher broadcast input x_full into GM (" << rows << " rows)" << std::endl;
        return true;
    }

    // Output collection: cores already wrote y shards / the reduce wrote per-row
    // sums straight into y_full_dev, so collection needs no extra GM work -- the
    // host just reads y_full_dev back in VerifyOutput().
    uint8_t *YFull() const { return reinterpret_cast<uint8_t *>(y_full_dev); }
    uint8_t *XFull() const { return reinterpret_cast<uint8_t *>(x_full_dev); }
    uint8_t *WGateShards() const { return reinterpret_cast<uint8_t *>(w_gate_shards_dev); }
    uint8_t *WUpShards() const { return reinterpret_cast<uint8_t *>(w_up_shards_dev); }
    uint8_t *WDownShards() const { return reinterpret_cast<uint8_t *>(w_down_shards_dev); }

    void Release()
    {
#define FFN_BATCHER_FREE(p) \
    do { \
        if (p) { \
            aclrtFree(p); \
            p = nullptr; \
        } \
    } while (0)
        FFN_BATCHER_FREE(x_full_dev);
        FFN_BATCHER_FREE(w_gate_full_dev);
        FFN_BATCHER_FREE(w_up_full_dev);
        FFN_BATCHER_FREE(w_down_full_dev);
        FFN_BATCHER_FREE(w_gate_shards_dev);
        FFN_BATCHER_FREE(w_up_shards_dev);
        FFN_BATCHER_FREE(w_down_shards_dev);
        FFN_BATCHER_FREE(y_full_dev);
#undef FFN_BATCHER_FREE
        std::vector<uint8_t>().swap(host_w_gate_full);
        std::vector<uint8_t>().swap(host_w_up_full);
        std::vector<uint8_t>().swap(host_w_down_full);
    }

    ~FfnBatcher() { Release(); }
};

#endif // DISTRIBUTED_FFN_GRID_BATCHER_HPP
