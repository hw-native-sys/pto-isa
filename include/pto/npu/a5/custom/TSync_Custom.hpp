/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSYNC_CUSTOM_HPP
#define TSYNC_CUSTOM_HPP

#include <pto/common/type.hpp>
#include <pto/common/utils.hpp>

#define VEC_CORE_ID_OFFSET 16

namespace pto {

// Operation types for TSync - identifies the producer/consumer operation
enum class SyncOpType : uint8_t {
    TSTORE_C2GM,  // Store (Cube core operation via PIPE_FIX) - GM path
    TSTORE_V2GM,  // Store (Vector core operation via PIPE_MTE3) - GM path
    TMOV_C2UB,    // TMOV from L0C to UB (Cube core operation via PIPE_FIX) - UB path
    TINSERT_V2L1, // TINSERT from UB to L1 (Vector core operation via PIPE_MTE3) - UB path
                  // TINSERT uses copy_ubuf_to_cbuf which goes through MTE3 pipe
                  // Cube consumer waits on PIPE_MTE1 (L1 side receives via MTE1)
    TLOAD         // Load operation (consumer operation)
};

// -----------------------------------------------------------------------------
// Compile-time direction inference based on producer/consumer ops
// GM path:
//   TSTORE_C2GM (producer) + TLOAD (consumer) = Cube to Vector via PIPE_FIX
//   TSTORE_V2GM (producer) + TLOAD (consumer) = Vector to Cube via PIPE_MTE3
// UB path:
//   TMOV_C2UB (producer) + TLOAD (consumer) = Cube to Vector via PIPE_FIX
//   TINSERT_V2L1 (producer) + TLOAD (consumer) = Vector to Cube via PIPE_MTE3
//   TINSERT (UB->L1) uses MTE3 on Vec side, Cube waits on MTE1
// -----------------------------------------------------------------------------
template <SyncOpType ProducerOp, SyncOpType ConsumerOp>
struct SyncTraits {
    // GM path: Cube produces via TSTORE_C2GM (PIPE_FIX) - consumer waits on PIPE_MTE2
    static constexpr bool is_cube_to_vec_gm = (ProducerOp == SyncOpType::TSTORE_C2GM);
    // UB path: Cube produces via TMOV_C2UB (PIPE_FIX) - consumer waits on PIPE_V
    static constexpr bool is_cube_to_vec_ub = (ProducerOp == SyncOpType::TMOV_C2UB);
    // Unified Cube-to-Vec detection
    static constexpr bool is_cube_to_vec = is_cube_to_vec_gm || is_cube_to_vec_ub;
    // GM path: Vector produces via TSTORE_V2GM (PIPE_MTE3)
    static constexpr bool is_vec_to_cube_gm = (ProducerOp == SyncOpType::TSTORE_V2GM);
    // UB path: Vector produces via TINSERT_V2L1 (PIPE_MTE3) - Cube waits on PIPE_MTE1
    static constexpr bool is_vec_to_cube_ub = (ProducerOp == SyncOpType::TINSERT_V2L1);
    // Unified Vec-to-Cube detection
    static constexpr bool is_vec_to_cube = is_vec_to_cube_gm || is_vec_to_cube_ub;

    static_assert(ConsumerOp == SyncOpType::TLOAD, "Consumer operation must be TLOAD");
    static_assert(
        is_cube_to_vec || is_vec_to_cube,
        "Producer must be TSTORE_C2GM, TMOV_C2UB (Cube) or TSTORE_V2GM, TINSERT_V2L1 (Vector)");
};

namespace detail {
template <int N>
struct FlagIDTag {
    static constexpr int value = N;
};

// Base counter starts at 0 (user IDs start from 0 to 12)
constexpr int kUserFlagIDStart = 0;
constexpr int kMaxFlagID = 12;
constexpr int kNumUserFlags = kMaxFlagID - kUserFlagIDStart + 1; // 12 flags
} // namespace detail

// -----------------------------------------------------------------------------
// TSync_Custom - Lightweight synchronization primitive for intra-core dependencies
//
// Supports both GM path (TSTORE->TLOAD) and UB path (TMOV->TLOAD, TINSERT->TLOAD)
//
// PIPE MAPPINGS:
//   1) GM -> L1: MTE2
//   2) L1 -> L0A/L0B: MTE1
//   3) L0C -> GM: MTE3 (via PIPE_FIX for TSTORE)
//   4) L0C -> UB: PIPE_FIX (TMOV)
//   5) UB -> GM: MTE3 (TSTORE Vec)
//   6) GM -> UB: MTE2 (TLOAD Vec)
//   7) UB -> L1: MTE3 (TINSERT/copy_ubuf_to_cbuf)
//   Vec operations: PIPE_V
//
// CROSS-CORE SYNC PATTERNS (asymmetric due to 1 Cube, 2 Vec subblocks):
//   Cube -> Vec forward sync:
//     - GM path (TSTORE->TLOAD): Cube sets PIPE_FIX, Vec waits PIPE_MTE2
//     - UB path (TMOV->VecOps): Cube sets PIPE_FIX, Vec waits PIPE_V
//     - Cube sets flag_id AND flag_id+16; Vec waits flag_id only
//
//   Vec -> Cube forward sync:
//     - GM path (TSTORE->TLOAD): Vec sets PIPE_MTE3, Cube waits PIPE_MTE2
//     - UB path (TINSERT->L1): Vec sets PIPE_MTE3, Cube waits PIPE_MTE1
//     - Vec sets flag_id only; Cube waits flag_id AND flag_id+16
//
//   Backward sync (consumer frees buffer for producer):
//     - Cube->Vec: Vec sets PIPE_MTE2/PIPE_V, Cube waits PIPE_FIX on both
//     - Vec->Cube: Cube sets PIPE_MTE1 on both, Vec waits PIPE_MTE3
//
// -----------------------------------------------------------------------------
template <SyncOpType ProducerOp, SyncOpType ConsumerOp>
struct TSync_Custom {
    using Traits = SyncTraits<ProducerOp, ConsumerOp>;
    static constexpr bool is_c2v = Traits::is_cube_to_vec;
    static constexpr bool is_c2v_gm = Traits::is_cube_to_vec_gm;
    static constexpr bool is_c2v_ub = Traits::is_cube_to_vec_ub;
    static constexpr bool is_v2c = Traits::is_vec_to_cube;
    static constexpr bool is_v2c_gm = Traits::is_vec_to_cube_gm;
    static constexpr bool is_v2c_ub = Traits::is_vec_to_cube_ub;

    uint16_t flag_id; // FFTS flag ID for cross-core synchronization

    // -----------------------------------------------------------------------------
    // Forward dependency: record (producer) and wait (consumer)
    // -----------------------------------------------------------------------------

    // -----------------------------------------------------------------------------
    // record - Producer signals that data is ready
    // Cube producers: set BOTH flag_id AND flag_id + 16 (one for each Vec subblock)
    // Vec producers: set flag_id only (hardware maps to flag_id+16 for subblock 1)
    // -----------------------------------------------------------------------------
    AICORE inline void record() const
    {
        if constexpr (is_c2v) {
            // Cube -> Vec: Cube sets BOTH flags on PIPE_FIX
            set_intra_block(PIPE_FIX, flag_id);
            set_intra_block(PIPE_FIX, flag_id + VEC_CORE_ID_OFFSET);
        } else { // is_v2c (both gm and ub)
            // Vec -> Cube: Vec sets flag_id only on PIPE_MTE3
            // Each Vec subblock executes this; hardware maps subblock 1's flag to flag_id+16
            set_intra_block(PIPE_MTE3, flag_id);
        }
    }

    // -----------------------------------------------------------------------------
    // wait - Consumer waits for data to be ready
    // Vec consumers: wait on flag_id only (each subblock waits independently)
    // Cube consumers: wait on BOTH flag_id AND flag_id + 16
    // -----------------------------------------------------------------------------
    AICORE inline void wait() const
    {
        if constexpr (is_c2v_gm) {
            // Cube -> Vec (GM path): Vec waits on PIPE_MTE2 (data loaded from GM)
            wait_intra_block(PIPE_MTE2, flag_id);
        } else if constexpr (is_c2v_ub) {
            // Cube -> Vec (UB path): Vec waits on PIPE_V before vector ops on UB data
            // Cube sets PIPE_FIX, Vec waits PIPE_V (Vec does vector ops, not TLOAD)
            wait_intra_block(PIPE_V, flag_id);
        } else if constexpr (is_v2c_gm) {
            // Vec -> Cube (GM path): Cube waits on PIPE_MTE2, BOTH flags
            wait_intra_block(PIPE_MTE2, flag_id);
            wait_intra_block(PIPE_MTE2, flag_id + VEC_CORE_ID_OFFSET);
        } else { // is_v2c_ub
            // Vec -> Cube (UB path - TINSERT): Cube waits on PIPE_MTE1, BOTH flags
            wait_intra_block(PIPE_MTE1, flag_id);
            wait_intra_block(PIPE_MTE1, flag_id + VEC_CORE_ID_OFFSET);
        }
    }
    // -----------------------------------------------------------------------------
    // Backward dependency: allocate (producer) and free (consumer)
    // -----------------------------------------------------------------------------

    // -----------------------------------------------------------------------------
    // allocate - Producer waits for buffer space to be available
    // Cube producers: wait on BOTH flag_id+1 AND flag_id+1+16 (Vec consumer signals)
    // Vec producers: wait on flag_id+1 only (Cube consumer signals both)
    // -----------------------------------------------------------------------------
    AICORE inline void allocate() const
    {
        if constexpr (is_c2v) {
            // Cube producer waits for Vec consumer to free buffer
            // Vec signals on flag_id+1 only, but Cube must wait on BOTH
            // (because Vec0 signals flag_id+1, Vec1 signals flag_id+1+16 from Cube's view)
            wait_intra_block(PIPE_FIX, flag_id + 1);
            wait_intra_block(PIPE_FIX, flag_id + 1 + VEC_CORE_ID_OFFSET);
        } else { // is_v2c (both gm and ub)
            // Vec producer waits for Cube consumer to free buffer
            // Cube signals on BOTH, Vec waits on flag_id+1 only
            wait_intra_block(PIPE_MTE3, flag_id + 1);
        }
    }

    // -----------------------------------------------------------------------------
    // free - Consumer signals that buffer space is available
    // Vec consumers: set flag_id+1 only (hardware maps to flag_id+1+16 for subblock 1)
    // Cube consumers: set BOTH flag_id+1 AND flag_id+1+16
    // -----------------------------------------------------------------------------
    AICORE inline void free() const
    {
        if constexpr (is_c2v_gm) {
            // Vec consumer frees buffer for Cube - signals on PIPE_MTE2, flag_id+1 only
            set_intra_block(PIPE_MTE2, flag_id + 1);
        } else if constexpr (is_c2v_ub) {
            // Vec consumer frees buffer for Cube - signals on PIPE_V, flag_id+1 only
            // Vec signals after vector ops complete (PIPE_V)
            set_intra_block(PIPE_V, flag_id + 1);
        } else { // is_v2c (both gm and ub)
            // Cube consumer frees buffer for Vec - signals BOTH flags on PIPE_MTE1
            set_intra_block(PIPE_MTE1, flag_id + 1);
            set_intra_block(PIPE_MTE1, flag_id + 1 + VEC_CORE_ID_OFFSET);
        }
    }
};
} // namespace pto

#endif // TSYNC_HPP
