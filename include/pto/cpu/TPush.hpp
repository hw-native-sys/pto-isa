/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPUSH_HPP
#define TPUSH_HPP

#include <pto/common/fifo.hpp>
#include <pto/cpu/TStore.hpp>
#include <pto/cpu/TLoad.hpp>

namespace pto {

// Operation types for TSync - identifies the producer/consumer operation
enum class TSyncOpType : uint8_t
{
    TSTORE_C2GM,  // Store (Cube core operation via PIPE_FIX) - GM path
    TSTORE_V2GM,  // Store (Vector core operation via PIPE_MTE3) - GM path
    TMOV_C2UB,    // TMOV from L0C to UB (Cube core operation via PIPE_FIX) - UB path
    TINSERT_V2L1, // TINSERT from UB to L1 (Vector core operation via PIPE_MTE3) - UB path
                  // TINSERT uses copy_ubuf_to_cbuf which goes through MTE3 pipe
                  // Cube consumer waits on PIPE_MTE1 (L1 side receives via MTE1)
    TLOAD         // Load operation (consumer operation)
};

template <TSyncOpType ProducerOp, TSyncOpType ConsumerOp>
struct TSyncTraits {
    // GM path: Cube produces via TSTORE_C2GM (PIPE_FIX) - consumer waits on PIPE_MTE2
    static constexpr bool is_cube_to_vec_gm = (ProducerOp == TSyncOpType::TSTORE_C2GM);
    // UB path: Cube produces via TMOV_C2UB (PIPE_FIX) - consumer waits on PIPE_V
    static constexpr bool is_cube_to_vec_ub = (ProducerOp == TSyncOpType::TMOV_C2UB);
    // Unified Cube-to-Vec detection
    static constexpr bool is_cube_to_vec = is_cube_to_vec_gm || is_cube_to_vec_ub;

    // GM path: Vector produces via TSTORE_V2GM (PIPE_MTE3)
    static constexpr bool is_vec_to_cube_gm = (ProducerOp == TSyncOpType::TSTORE_V2GM);
    // UB path: Vector produces via TINSERT_V2L1 (PIPE_MTE3) - Cube waits on PIPE_MTE1
    static constexpr bool is_vec_to_cube_ub = (ProducerOp == TSyncOpType::TINSERT_V2L1);
    // Unified Vec-to-Cube detection
    static constexpr bool is_vec_to_cube = is_vec_to_cube_gm || is_vec_to_cube_ub;

    static_assert(ConsumerOp == TSyncOpType::TLOAD, "Consumer operation must be TLOAD");
    static_assert(is_cube_to_vec || is_vec_to_cube,
                  "Producer must be TSTORE_C2GM, TMOV_C2UB (Cube) or TSTORE_V2GM, TINSERT_V2L1 (Vector)");
};

template <uint16_t FlagID, typename DataFIFO, TSyncOpType ProducerOp, TSyncOpType ConsumerOp>
struct TFIFOSync {
    using Traits = TSyncTraits<ProducerOp, ConsumerOp>;
    // static constexpr bool is_c2v = Traits::is_cube_to_vec;
    // static constexpr bool is_c2v_gm = Traits::is_cube_to_vec_gm;
    // static constexpr bool is_c2v_ub = Traits::is_cube_to_vec_ub;
    // static constexpr bool is_v2c = Traits::is_vec_to_cube;
    // static constexpr bool is_v2c_gm = Traits::is_vec_to_cube_gm;
    // static constexpr bool is_v2c_ub = Traits::is_vec_to_cube_ub;
    // static constexpr int VEC_CORE_ID_OFFSET = 16;

    static constexpr int fifoSize = DataFIFO::fifoDepth;
    static constexpr int syncPeriod = DataFIFO::fifoPeriod;

    // -------------------------------------------------------------------------
    // Producer Interface
    // -------------------------------------------------------------------------
    struct Producer {
        volatile int tile_id;

        PTO_INTERNAL Producer()
        {
            tile_id = 0;
        }

        PTO_INTERNAL void set_tile_id(int t_id, int sub_t_id)
        {
            tile_id = t_id;
        }

        PTO_INTERNAL int get_tile_id()
        {
            return tile_id;
        }

        PTO_INTERNAL void allocate()
        {
            // For multithreaded operations code locking data block of fifo buffer should be added here
        }

        PTO_INTERNAL void record()
        {
            tile_id = (tile_id + 1) % fifoSize;
            // Add notification routine from producer to notifier for multithreaded scenario
        }
    };

    // -------------------------------------------------------------------------
    // Consumer Interface
    // -------------------------------------------------------------------------
    struct Consumer {
        volatile int tile_id;
        volatile int sub_tile_id;

        PTO_INTERNAL Consumer()
        {
            tile_id = 0;
        }

        PTO_INTERNAL void set_tile_id(int tid, int sub_tid)
        {
            tile_id = tid;
        }

        PTO_INTERNAL int get_tile_id()
        {
            return tile_id;
        }

        PTO_INTERNAL void wait()
        {
            // Add wait routine if fifo buffer is empty.
        }

        PTO_INTERNAL void free()
        {
            tile_id = (tile_id + 1) % fifoSize;
            // Add notification to producer that will notify that particular block in FIFO is free now
        }
    };
};

template <typename PipeProd, typename TileData, typename DataFiFo>
PTO_INTERNAL void TPUSH_IMPL(PipeProd &prod, TileData &tile, DataFiFo &fifo)
{
    // 1. Cross-Core: Wait for space
    prod.allocate();

    // 2. Address Calculation
    __gm__ typename DataFiFo::DType *addr = fifo.getBasePtr() + TileData::Numel * (prod.get_tile_id() % fifo.fifoDepth);
    constexpr unsigned int cols = TileData::Cols;
    constexpr unsigned int rows = TileData::Rows;
    GlobalTensor<typename TileData::DType, Shape<1, 1, 1, rows, cols>,
                 Stride<rows * cols, rows * cols, rows * cols, cols, 1>>
        gt(addr);
    TSTORE(gt, tile);

    // 3. Cross-Core: Commit & Signal
    prod.record();
}

} // namespace pto

#endif
