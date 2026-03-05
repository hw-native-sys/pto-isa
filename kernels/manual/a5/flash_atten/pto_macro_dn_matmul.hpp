/*
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MACRO_MATMUL_HPP
#define PTO_MACRO_MATMUL_HPP

#include <pto/pto-inst.hpp>
#include <pto/common/memory.hpp>

#define CUBE_K_256 256
#define CUBE_K_128 128
#define CUBE_K_64 64
#define CUBE_K_SMALLEST 32

namespace pto {

/**
 * Layout type for matrix multiplication operations.
 * First letter represents the layout of matrix A, second letter represents matrix B.
 * N = Normal (Row-major), T = Transposed (Column-major)
 */
enum class layout_t
{
    NN, // Matrix A: Normal, Matrix B: Normal
    NT, // Matrix A: Normal, Matrix B: Transposed
    TN, // Matrix A: Transposed, Matrix B: Normal
    TT, // Matrix A: Transposed, Matrix B: Transposed
    NONE
};

enum class AccMode
{
    Init,           // auto phase, first slice initializes, rest accumulate
    Acc,            // auto phase, all slices accumulate into existing C
    InitPartialSum, // explicitly partial, first slice initializes
    InitFinalSum,   // explicitly final, first slice initializes
    AccPartialSum,  // explicitly partial, all slices accumulate
    AccFinalSum,    // explicitly final, all slices accumulate
};

#define L0A_BUF0 ((__ca__ half *)(__ca__ char *)0x0)
#define L0A_BUF1 ((__ca__ half *)(__ca__ char *)0x8000)
#define L0B_BUF0 ((__ca__ half *)(__ca__ char *)0x0)
#define L0B_BUF1 ((__ca__ half *)(__ca__ char *)0x8000)
#define L0C_BUF0 ((__ca__ half *)(__ca__ char *)0x0)
#define L0C_BUF1 ((__ca__ half *)(__ca__ char *)0x20000)

#define LAST_LOOP(x, n) ((x) == ((n)-1))
#define UNIT_FLAG_ENABLE(i, n) (LAST_LOOP(i, n) ? 3 : 2)

AICORE inline uint64_t getPingPong(uint32_t flip)
{
    static uint64_t pingpong = 0;
    if (flip) {
        pingpong = 1 - pingpong;
    }
    return pingpong;
}

// Memory constraints (L0 ping-pong is 32 KiB per buffer in this implementation).
// Tuning knob: if you change L0 layout or buffer addresses, re-check these constraints.
constexpr uint32_t MEM_BUFFER_SIZE_BYTES = 64 * 1024 / 2; // 64KB per buffer with pingpong (32KB)
constexpr uint32_t HALF_SIZE_BYTES = 2;                   // sizeof(half) = 2 bytes

/**
 * Calculate the largest Cube_K value that fits in the 64KB memory buffer.
 * Checks if both Cube_M * Cube_K (left matrix) and Cube_K * Cube_N (right matrix)
 * can fit within the 64KB buffer.
 *
 * @param Cube_M - The tile dimension M
 * @param Cube_N - The tile dimension N
 * @return - Largest Cube_K value (32, 64, 128, or 256) that fits in memory
 */
// Choose the largest Cube_K that fits both L0A (Cube_M x Cube_K) and L0B (Cube_K x Cube_N)
// so TMATMUL stays compute-dense while respecting L0 ping-pong capacity.
AICORE inline constexpr uint32_t calculateFittingCubeK(uint32_t Cube_M, uint32_t Cube_N)
{
    uint32_t bestCubeK = CUBE_K_SMALLEST; // Default to smallest value

    // Test candidates from largest to smallest to find the largest that fits
    if (Cube_M * CUBE_K_256 * HALF_SIZE_BYTES <= MEM_BUFFER_SIZE_BYTES &&
        CUBE_K_256 * Cube_N * HALF_SIZE_BYTES <= MEM_BUFFER_SIZE_BYTES) {
        bestCubeK = CUBE_K_256;
    } else if (Cube_M * CUBE_K_128 * HALF_SIZE_BYTES <= MEM_BUFFER_SIZE_BYTES &&
               CUBE_K_128 * Cube_N * HALF_SIZE_BYTES <= MEM_BUFFER_SIZE_BYTES) {
        bestCubeK = CUBE_K_128;
    } else if (Cube_M * CUBE_K_64 * HALF_SIZE_BYTES <= MEM_BUFFER_SIZE_BYTES &&
               CUBE_K_64 * Cube_N * HALF_SIZE_BYTES <= MEM_BUFFER_SIZE_BYTES) {
        bestCubeK = CUBE_K_64;
    }

    return bestCubeK;
}

// Deduce layout_t from SLayouts
template <typename TileDataA, typename TileDataB>
AICORE inline constexpr layout_t deduce_layout()
{
    if constexpr (TileDataA::SFractal == SLayout::RowMajor && TileDataB::SFractal == SLayout::RowMajor)
        return layout_t::NN;
    if constexpr (TileDataA::SFractal == SLayout::RowMajor && TileDataB::SFractal == SLayout::ColMajor)
        return layout_t::NT;
    if constexpr (TileDataA::SFractal == SLayout::ColMajor && TileDataB::SFractal == SLayout::RowMajor)
        return layout_t::TN;
    if constexpr (TileDataA::SFractal == SLayout::ColMajor && TileDataB::SFractal == SLayout::ColMajor)
        return layout_t::TT;
    return layout_t::NONE;
}

struct MatmulCallConfig {
    bool useAcc;    // true -> TMATMUL_UF_ACC, false -> TMATMUL_UF
    AccPhase phase; // UF mapping
};

AICORE inline MatmulCallConfig resolve_acc_mode(AccMode mode, bool isFirstSlice, bool isLastSlice)
{
    switch (mode) {
        case AccMode::Init:
            return MatmulCallConfig{!isFirstSlice, AccPhase::Unknown};
        case AccMode::Acc:
            return MatmulCallConfig{true, AccPhase::Unknown};
        case AccMode::InitPartialSum:
            return MatmulCallConfig{!isFirstSlice, AccPhase::Partial};
        case AccMode::InitFinalSum:
            return MatmulCallConfig{!isFirstSlice, AccPhase::Final};
        case AccMode::AccPartialSum:
            return MatmulCallConfig{true, AccPhase::Partial};
        case AccMode::AccFinalSum:
            return MatmulCallConfig{true, AccPhase::Final};
    }
    return MatmulCallConfig{!isFirstSlice, AccPhase::Partial};
}

template <unsigned Cube_M, unsigned Tile_K, unsigned Cube_N, layout_t LAYOUT = layout_t::NONE, typename TileDataA,
          typename TileDataB, typename TileDataC>
AICORE inline void pto_macro_matmul(TileDataA &aMatTile, TileDataB &bMatTile, TileDataC &cAccTile,
                                    AccMode accMode = AccMode::Init)
{
    constexpr layout_t layout = deduce_layout<TileDataA, TileDataB>();

    static_assert(layout != layout_t::NONE, "Deduced layout is NONE, check tile SLayouts");
    // Assert that template LAYOUT matches deduced layout if LAYOUT is not NONE
    if constexpr (LAYOUT != layout_t::NONE) {
        static_assert(LAYOUT == layout,
                      "Layout mismatch: template LAYOUT does not match deduced layout from tile SLayouts. "
                      "Check SLayout of TileDataA and TileDataB.");
    }

    // // Only NT layout supports inner K loop (Tile_K/Cube_K > 1) due to TEXTRACT memory layout constraints
    // // For NN, TN, and TT layouts, outer K loop is not supported, so Tile_K must equal Cube_K
    // static_assert(layout == layout_t::NT || Tile_K == Cube_K,
    //                 "Only NT layout supports inner K loop (Tile_K/Cube_K > 1). "
    //                 "For NN, TN, and TT layouts, Tile_K must equal Cube_K (K loop at L1 level not supported).");

    // Ping-pong is used to overlap TEXTRACT (L1->L0) with TMATMUL on alternating buffers.
    uint64_t pingpong = getPingPong(0);
    const uint64_t Cube_K =
        calculateFittingCubeK(Cube_M, Cube_N) > Tile_K ? Tile_K : calculateFittingCubeK(Cube_M, Cube_N);
    const uint64_t kSegments = (uint64_t)(Tile_K / Cube_K);
    for (uint64_t k = 0; k < kSegments; k++) {
        using LeftTile = TileLeft<half, Cube_M, Cube_K, Cube_M, Cube_K>;
        LeftTile al0Tiles[2] = {LeftTile(), LeftTile()};
        using RightTile = TileRight<half, Cube_K, Cube_N, Cube_K, Cube_N>;
        RightTile bl0Tiles[2] = {RightTile(), RightTile()};

        TASSIGN(al0Tiles[0], (uint64_t)L0A_BUF0);
        TASSIGN(al0Tiles[1], (uint64_t)L0A_BUF1);
        TASSIGN(bl0Tiles[0], (uint64_t)L0B_BUF0);
        TASSIGN(bl0Tiles[1], (uint64_t)L0B_BUF1);

        // Wait until previous TMATMUL finishes using this L0 buffer before overwriting it via TEXTRACT.
        wait_flag(PIPE_M, PIPE_MTE1, pingpong);

        if (layout == layout_t::NT) {
            TASSIGN(aMatTile, (uint64_t)aMatTile.data() + k * Cube_K * Cube_M * sizeof(typename TileDataA::DType));
            TASSIGN(bMatTile, (uint64_t)bMatTile.data() + k * Cube_K * Cube_N * sizeof(typename TileDataB::DType));
        }

        // TEXTRACT slices the current Cube_K panel into L0A/L0B.
        TEXTRACT(al0Tiles[pingpong], aMatTile, 0, 0);
        TEXTRACT(bl0Tiles[pingpong], bMatTile, 0, 0);

        set_flag(PIPE_MTE1, PIPE_M, pingpong);
        wait_flag(PIPE_MTE1, PIPE_M, pingpong);

        const bool isLast = (k + 1 == kSegments);
        MatmulCallConfig cfg = resolve_acc_mode(accMode, k == 0, isLast);
        if (cfg.useAcc) {
            if (cfg.phase == AccPhase::Final) {
                TMATMUL_ACC<AccPhase::Final>(cAccTile, al0Tiles[pingpong], bl0Tiles[pingpong]);
            } else if (cfg.phase == AccPhase::Partial) {
                TMATMUL_ACC<AccPhase::Partial>(cAccTile, al0Tiles[pingpong], bl0Tiles[pingpong]);
            } else {
                TMATMUL_ACC(cAccTile, al0Tiles[pingpong], bl0Tiles[pingpong]);
            }
        } else {
            if (cfg.phase == AccPhase::Final) {
                TMATMUL<AccPhase::Final>(cAccTile, al0Tiles[pingpong], bl0Tiles[pingpong]);
            } else if (cfg.phase == AccPhase::Partial) {
                TMATMUL<AccPhase::Partial>(cAccTile, al0Tiles[pingpong], bl0Tiles[pingpong]);
            } else {
                TMATMUL(cAccTile, al0Tiles[pingpong], bl0Tiles[pingpong]);
            }
        }
        set_flag(PIPE_M, PIPE_MTE1, pingpong);
        pingpong = getPingPong(1);
    }
}
} // namespace pto

#endif // PTO_MACRO_MATMUL_H
