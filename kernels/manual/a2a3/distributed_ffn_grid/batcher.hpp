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

    // ----- pure 1D N-cut (方案①) 32-cell arena (only when split_mode == "pure_ncut") -----
    // Flat layout: one broadcast x [T,H]; w_gate/w_up split along I into `cells`
    // contiguous [H,I_shard] shards; w_down split along H into `cells` [I,H_shard]
    // shards; collected y [T,H] (each cell writes its [T,H_shard] H-slice).  Kept as
    // separate members so the legacy 2D-decomposition path is byte-for-byte unchanged.
    void *ncut_x_full_dev = nullptr;        // [T, H]            half (broadcast)
    void *ncut_w_gate_full_dev = nullptr;   // [H, I]            half (resident)
    void *ncut_w_up_full_dev = nullptr;     // [H, I]            half (resident)
    void *ncut_w_down_full_dev = nullptr;   // [I, H]            half (resident)
    void *ncut_w_gate_shards_dev = nullptr; // [cells, H, I_shard] half (distributed)
    void *ncut_w_up_shards_dev = nullptr;   // [cells, H, I_shard] half (distributed)
    void *ncut_w_down_shards_dev = nullptr; // [cells, I, H_shard] half (distributed) -- AllGather H-column shard
    void *ncut_y_full_dev = nullptr;        // [T, H]            float (collected)
    size_t ncut_cells = 0;
    // ReduceSum (Option B, pure N-cut) uses the same broadcast x and gate/up I-column
    // shards as AllGather, but cuts W_down along I (its K-axis) into ROW shards
    // [I_shard, H]; only this member is ReduceSum-specific.
    void *ncut_reduce_w_down_shards_dev = nullptr; // [cells, I_shard, H] half (W_down row-shard)

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

    // ===================== pure 1D N-cut (方案①) 32-cell path =====================
    // Flat distribution: x broadcast to all cells; w_gate/w_up split along the
    // intermediate I into `cells` contiguous [H, I_shard] shards (shard index =
    // cell = row*cols + col); w_down split along the output H into `cells`
    // [I, H_shard] shards.  The kernel addresses cell c's shard at
    // shards + c * FFN_NCUT_W_*_SHARD_BYTES and its y H-slice at yFull + c*H_shard*4.
    bool AllocateNcut(size_t gridRows, size_t gridCols)
    {
        rows = gridRows;
        cols = gridCols;
        ncut_cells = gridRows * gridCols;
        split_mode = "pure_ncut";

        ncut_x_full_dev        = MallocOrDie(static_cast<size_t>(FFN_NCUT_X_BYTES), "batcher.ncut_x_full");
        ncut_w_gate_full_dev   = MallocOrDie(static_cast<size_t>(FFN_NCUT_H * FFN_NCUT_I * FFN_HALF_ELEM_BYTES), "batcher.ncut_w_gate_full");
        ncut_w_up_full_dev     = MallocOrDie(static_cast<size_t>(FFN_NCUT_H * FFN_NCUT_I * FFN_HALF_ELEM_BYTES), "batcher.ncut_w_up_full");
        ncut_w_down_full_dev   = MallocOrDie(static_cast<size_t>(FFN_NCUT_I * FFN_NCUT_H * FFN_HALF_ELEM_BYTES), "batcher.ncut_w_down_full");
        ncut_w_gate_shards_dev = MallocOrDie(ncut_cells * static_cast<size_t>(FFN_NCUT_W_GATE_SHARD_BYTES), "batcher.ncut_w_gate_shards");
        ncut_w_up_shards_dev   = MallocOrDie(ncut_cells * static_cast<size_t>(FFN_NCUT_W_UP_SHARD_BYTES), "batcher.ncut_w_up_shards");
        ncut_w_down_shards_dev = MallocOrDie(ncut_cells * static_cast<size_t>(FFN_NCUT_W_DOWN_SHARD_BYTES), "batcher.ncut_w_down_shards");
        ncut_y_full_dev        = MallocOrDie(static_cast<size_t>(FFN_NCUT_T * FFN_NCUT_H * 4), "batcher.ncut_y_full");

        if (!ncut_x_full_dev || !ncut_w_gate_full_dev || !ncut_w_up_full_dev || !ncut_w_down_full_dev ||
            !ncut_w_gate_shards_dev || !ncut_w_up_shards_dev || !ncut_w_down_shards_dev || !ncut_y_full_dev) {
            return false;
        }
        std::cout << "[INFO] Batcher ncut arena allocated: cells=" << ncut_cells << " T=" << FFN_NCUT_T
                  << " H=" << FFN_NCUT_H << " I=" << FFN_NCUT_I << " (I_shard=" << FFN_NCUT_I_SHARD
                  << " H_shard=" << FFN_NCUT_H_SHARD << ")" << std::endl;
        return true;
    }

    // Load the full DRAM-resident weights + broadcast x, then slice them flat-32-way.
    // This host slice loop IS the GM-simulated Batcher distribution for the pure
    // N-cut topology.  Returns false on any file/memcpy mismatch.
    bool LoadAndDistributeNcut(const std::string &dataDir)
    {
        const int H = FFN_NCUT_H;
        const int I = FFN_NCUT_I;
        const int IShard = FFN_NCUT_I_SHARD;
        const int HShard = FFN_NCUT_H_SHARD;
        const size_t elem = FFN_HALF_ELEM_BYTES;

        // Full resident weights -> GM (DRAM-resident, per the SVG memory model).
        std::vector<uint8_t> hostWG(H * I * elem);
        std::vector<uint8_t> hostWU(H * I * elem);
        std::vector<uint8_t> hostWD(I * H * elem);
        auto loadFull = [&](const char *name, void *dev, std::vector<uint8_t> &host, size_t bytes) -> bool {
            std::string path = dataDir + "/" + name;
            size_t fileSize = 0;
            if (!PtoTestCommon::ReadFile(path, fileSize, host.data(), bytes) || fileSize != bytes) {
                std::cerr << "[ERROR] ncut load mismatch: " << path << " (got " << fileSize << " expected " << bytes << ")" << std::endl;
                return false;
            }
            return aclrtMemcpy(dev, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
        };
        if (!loadFull("w_gate_full.bin", ncut_w_gate_full_dev, hostWG, H * I * elem) ||
            !loadFull("w_up_full.bin", ncut_w_up_full_dev, hostWU, H * I * elem) ||
            !loadFull("w_down_full.bin", ncut_w_down_full_dev, hostWD, I * H * elem)) {
            std::cerr << "[ERROR] ncut resident weight load failed" << std::endl;
            return false;
        }

        // Slice w_gate/w_up along I (column-parallel on the intermediate dim):
        // cell c owns columns [c*IShard, (c+1)*IShard) of [H, I].
        std::vector<uint8_t> shardW(std::max(FFN_NCUT_W_GATE_SHARD_BYTES, FFN_NCUT_W_DOWN_SHARD_BYTES));
        for (size_t c = 0; c < ncut_cells; ++c) {
            const int colStart = static_cast<int>(c) * IShard;
            GatherColShard(hostWG.data(), H, I, colStart, IShard, shardW.data());
            if (aclrtMemcpy(static_cast<uint8_t *>(ncut_w_gate_shards_dev) + c * FFN_NCUT_W_GATE_SHARD_BYTES,
                            FFN_NCUT_W_GATE_SHARD_BYTES, shardW.data(), FFN_NCUT_W_GATE_SHARD_BYTES,
                            ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
                return false;
            }
            GatherColShard(hostWU.data(), H, I, colStart, IShard, shardW.data());
            if (aclrtMemcpy(static_cast<uint8_t *>(ncut_w_up_shards_dev) + c * FFN_NCUT_W_UP_SHARD_BYTES,
                            FFN_NCUT_W_UP_SHARD_BYTES, shardW.data(), FFN_NCUT_W_UP_SHARD_BYTES,
                            ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
                return false;
            }
            // w_down along H (output dim): cell c owns columns [c*HShard,(c+1)*HShard) of [I, H].
            GatherColShard(hostWD.data(), I, H, static_cast<int>(c) * HShard, HShard, shardW.data());
            if (aclrtMemcpy(static_cast<uint8_t *>(ncut_w_down_shards_dev) + c * FFN_NCUT_W_DOWN_SHARD_BYTES,
                            FFN_NCUT_W_DOWN_SHARD_BYTES, shardW.data(), FFN_NCUT_W_DOWN_SHARD_BYTES,
                            ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
                return false;
            }
        }

        // Broadcast input: single [T, H] x read by all cells.
        std::vector<uint8_t> hostX(static_cast<size_t>(FFN_NCUT_X_BYTES));
        std::string xpath = dataDir + "/x_full.bin";
        size_t xSize = 0;
        if (!PtoTestCommon::ReadFile(xpath, xSize, hostX.data(), hostX.size()) || xSize != hostX.size()) {
            std::cerr << "[ERROR] ncut x_full load mismatch: " << xpath << std::endl;
            return false;
        }
        if (aclrtMemcpy(ncut_x_full_dev, hostX.size(), hostX.data(), hostX.size(), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            return false;
        }
        std::cout << "[INFO] Batcher ncut distributed " << ncut_cells << " per-cell weight shards + broadcast x"
                  << std::endl;
        return true;
    }

    uint8_t *NcutXFull() const { return reinterpret_cast<uint8_t *>(ncut_x_full_dev); }
    uint8_t *NcutWGateShards() const { return reinterpret_cast<uint8_t *>(ncut_w_gate_shards_dev); }
    uint8_t *NcutWUpShards() const { return reinterpret_cast<uint8_t *>(ncut_w_up_shards_dev); }
    uint8_t *NcutWDownShards() const { return reinterpret_cast<uint8_t *>(ncut_w_down_shards_dev); }
    uint8_t *NcutYFull() const { return reinterpret_cast<uint8_t *>(ncut_y_full_dev); }

    // ===== ReduceSum (Option B, pure N-cut) path =====
    // Allocate the same broadcast x + gate/up I-column shards + full resident
    // weights + collected y as AllGather, plus the W_down I-row shards.  (The
    // AllGather W_down H-column shard ncut_w_down_shards_dev is left null.)
    bool AllocateNcutReduce(size_t gridRows, size_t gridCols)
    {
        rows = gridRows;
        cols = gridCols;
        ncut_cells = gridRows * gridCols;
        split_mode = "ncut_reduce";

        const size_t elem = FFN_HALF_ELEM_BYTES;
        ncut_x_full_dev = MallocOrDie(static_cast<size_t>(FFN_NCUT_X_BYTES), "batcher.ncut_x_full");
        ncut_w_gate_full_dev = MallocOrDie(static_cast<size_t>(FFN_NCUT_H * FFN_NCUT_I) * elem, "batcher.ncut_w_gate_full");
        ncut_w_up_full_dev = MallocOrDie(static_cast<size_t>(FFN_NCUT_H * FFN_NCUT_I) * elem, "batcher.ncut_w_up_full");
        ncut_w_down_full_dev = MallocOrDie(static_cast<size_t>(FFN_NCUT_I * FFN_NCUT_H) * elem, "batcher.ncut_w_down_full");
        ncut_w_gate_shards_dev =
            MallocOrDie(ncut_cells * static_cast<size_t>(FFN_NCUT_W_GATE_SHARD_BYTES), "batcher.ncut_w_gate_shards");
        ncut_w_up_shards_dev =
            MallocOrDie(ncut_cells * static_cast<size_t>(FFN_NCUT_W_UP_SHARD_BYTES), "batcher.ncut_w_up_shards");
        ncut_reduce_w_down_shards_dev =
            MallocOrDie(ncut_cells * static_cast<size_t>(FFN_RS_W_DOWN_SHARD_BYTES), "batcher.ncut_reduce_w_down_shards");
        ncut_y_full_dev = MallocOrDie(static_cast<size_t>(FFN_NCUT_T * FFN_NCUT_H * 4), "batcher.ncut_y_full");
        if (!ncut_x_full_dev || !ncut_w_gate_full_dev || !ncut_w_up_full_dev || !ncut_w_down_full_dev ||
            !ncut_w_gate_shards_dev || !ncut_w_up_shards_dev || !ncut_reduce_w_down_shards_dev || !ncut_y_full_dev) {
            return false;
        }
        std::cout << "[INFO] Batcher ncut-reduce arena allocated: cells=" << ncut_cells << " T=" << FFN_NCUT_T
                  << " H=" << FFN_NCUT_H << " I=" << FFN_NCUT_I << " (I_shard=" << FFN_NCUT_I_SHARD << ")" << std::endl;
        return true;
    }

    // Load full weights + broadcast x, then slice gate/up along I (column-parallel)
    // and W_down along I (row-parallel, ROW shard [I_shard, H]) flat-32-way.
    bool LoadAndDistributeNcutReduce(const std::string &dataDir)
    {
        const int H = FFN_NCUT_H;
        const int I = FFN_NCUT_I;
        const int IShard = FFN_NCUT_I_SHARD;
        const size_t elem = FFN_HALF_ELEM_BYTES;

        std::vector<uint8_t> hostWG(H * I * elem);
        std::vector<uint8_t> hostWU(H * I * elem);
        std::vector<uint8_t> hostWD(I * H * elem);
        auto loadFull = [&](const char *name, void *dev, std::vector<uint8_t> &host, size_t bytes) -> bool {
            std::string path = dataDir + "/" + name;
            size_t fileSize = 0;
            if (!PtoTestCommon::ReadFile(path, fileSize, host.data(), bytes) || fileSize != bytes) {
                std::cerr << "[ERROR] ncut-reduce load mismatch: " << path << std::endl;
                return false;
            }
            return aclrtMemcpy(dev, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
        };
        if (!loadFull("w_gate_full.bin", ncut_w_gate_full_dev, hostWG, H * I * elem) ||
            !loadFull("w_up_full.bin", ncut_w_up_full_dev, hostWU, H * I * elem) ||
            !loadFull("w_down_full.bin", ncut_w_down_full_dev, hostWD, I * H * elem)) {
            return false;
        }

        std::vector<uint8_t> shardW(std::max(static_cast<int>(FFN_NCUT_W_GATE_SHARD_BYTES), FFN_RS_W_DOWN_SHARD_BYTES));
        for (size_t c = 0; c < ncut_cells; ++c) {
            const int istart = static_cast<int>(c) * IShard;
            // gate/up: column-parallel along I -> [H, I_shard].
            GatherColShard(hostWG.data(), H, I, istart, IShard, shardW.data());
            if (aclrtMemcpy(static_cast<uint8_t *>(ncut_w_gate_shards_dev) + c * FFN_NCUT_W_GATE_SHARD_BYTES,
                            FFN_NCUT_W_GATE_SHARD_BYTES, shardW.data(), FFN_NCUT_W_GATE_SHARD_BYTES,
                            ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
                return false;
            }
            GatherColShard(hostWU.data(), H, I, istart, IShard, shardW.data());
            if (aclrtMemcpy(static_cast<uint8_t *>(ncut_w_up_shards_dev) + c * FFN_NCUT_W_UP_SHARD_BYTES,
                            FFN_NCUT_W_UP_SHARD_BYTES, shardW.data(), FFN_NCUT_W_UP_SHARD_BYTES,
                            ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
                return false;
            }
            // w_down: row-parallel along I -> rows [istart, istart+I_shard) of [I, H] = [I_shard, H]
            // (contiguous row block in row-major [I, H]).
            const size_t rowOff = static_cast<size_t>(istart) * H * elem;
            if (aclrtMemcpy(static_cast<uint8_t *>(ncut_reduce_w_down_shards_dev) + c * FFN_RS_W_DOWN_SHARD_BYTES,
                            FFN_RS_W_DOWN_SHARD_BYTES, hostWD.data() + rowOff, FFN_RS_W_DOWN_SHARD_BYTES,
                            ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
                return false;
            }
        }

        std::vector<uint8_t> hostX(static_cast<size_t>(FFN_NCUT_X_BYTES));
        std::string xpath = dataDir + "/x_full.bin";
        size_t xSize = 0;
        if (!PtoTestCommon::ReadFile(xpath, xSize, hostX.data(), hostX.size()) || xSize != hostX.size()) {
            std::cerr << "[ERROR] ncut-reduce x_full load mismatch: " << xpath << std::endl;
            return false;
        }
        if (aclrtMemcpy(ncut_x_full_dev, hostX.size(), hostX.data(), hostX.size(), ACL_MEMCPY_HOST_TO_DEVICE) !=
            ACL_SUCCESS) {
            return false;
        }
        std::cout << "[INFO] Batcher ncut-reduce distributed " << ncut_cells
                  << " per-cell gate/up col-shards + W_down row-shards + broadcast x" << std::endl;
        return true;
    }

    uint8_t *NcutReduceWDownShards() const { return reinterpret_cast<uint8_t *>(ncut_reduce_w_down_shards_dev); }

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
        FFN_BATCHER_FREE(ncut_x_full_dev);
        FFN_BATCHER_FREE(ncut_w_gate_full_dev);
        FFN_BATCHER_FREE(ncut_w_up_full_dev);
        FFN_BATCHER_FREE(ncut_w_down_full_dev);
        FFN_BATCHER_FREE(ncut_w_gate_shards_dev);
        FFN_BATCHER_FREE(ncut_w_up_shards_dev);
        FFN_BATCHER_FREE(ncut_w_down_shards_dev);
        FFN_BATCHER_FREE(ncut_reduce_w_down_shards_dev);
        FFN_BATCHER_FREE(ncut_y_full_dev);
#undef FFN_BATCHER_FREE
        std::vector<uint8_t>().swap(host_w_gate_full);
        std::vector<uint8_t>().swap(host_w_up_full);
        std::vector<uint8_t>().swap(host_w_down_full);
    }

    ~FfnBatcher() { Release(); }
};

#endif // DISTRIBUTED_FFN_GRID_BATCHER_HPP
