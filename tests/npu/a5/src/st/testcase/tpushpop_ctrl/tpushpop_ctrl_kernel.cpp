/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>

#include "ctrl_test_config.h"

using namespace pto;
#ifdef __CCE_AICORE__
using CtrlTile = Tile<TileType::Ctrl, uint32_t, 1, 1>;
using PredicateTile = Tile<TileType::Vec, uint32_t, 1, 8>;

AICORE inline uint32_t laneValue(uint32_t lane, uint32_t message)
{
    // The four pairs are (0,0), (0,7), (2,0), (0x80000000,0x40).
    if (lane == 0) {
        return message == 2 ? 2 : (message == 3 ? 0x80000000U : 0);
    }
    return message == 1 ? 7 : (message == 3 ? 0x40 : 0);
}

#endif

template <int Key>
__global__ AICORE void runTPushPopCtrl(__gm__ uint32_t* out)
{
    using Case = CtrlTest::Case<Key>;
    constexpr auto MERGE =
        Case::MERGE_NONE ? CtrlMergeMode::NONE : (Case::MERGE_AND ? CtrlMergeMode::AND : CtrlMergeMode::OR);
    constexpr auto RATIO = Case::SINGLE_LANE == 0 ?
                               VecCubeRatio::V1C1_VEC0 :
                               (Case::SINGLE_LANE == 1 ? VecCubeRatio::V1C1_VEC1 : VecCubeRatio::V2C1_VECS);
    using DataConfig = CtrlPipeConfig<MERGE, RATIO, Case::FORWARD, Case::BACKWARD>;
    using EchoConfig = CtrlPipeConfig<CtrlMergeMode::OR, VecCubeRatio::V2C1_VECS, Case::FORWARD, Case::BACKWARD>;
    using GuardConfig = CtrlPipeConfig<>;
    using ProducerTile = std::conditional_t<Case::VEC_PRODUCER, PredicateTile, CtrlTile>;
    // The data pipes use flags 0/2; fully dependent guards use flags 4/6.
    using V2C = TPipe<0, Direction::DIR_V2C_CTRL, 4, 2, 2, false, false, DataConfig>;
    using C2V = TPipe<2, Direction::DIR_C2V_CTRL, 4, 2, 2, false, false, EchoConfig>;
    using GuardV2C = TPipe<4, Direction::DIR_V2C_CTRL, 4, 2, 2, false, false, GuardConfig>;
    using GuardC2V = TPipe<6, Direction::DIR_C2V_CTRL, 4, 2, 2, false, false, GuardConfig>;

    // Reuse one physical AIC and its two AIVs for three logical jobs. Scoping
    // each job forces all pending credits to drain before the flag IDs recur.
    for (uint32_t job = 0; job < CtrlTest::JOBS; ++job) {
        V2C v2c(nullptr, 0, 0x100);
        C2V c2v(nullptr, 0x140, 0);
        GuardV2C guardV2C(nullptr, 0, 0x180);
        GuardC2V guardC2V(nullptr, 0x1c0, 0);
        CtrlTile token;
        token.data() = 1;
        // Two messages are in flight before either response is consumed. This
        // fills the initial depth without allocation waits and then wraps.
        for (uint32_t batch = 0; batch < 2; ++batch) {
#ifdef __DAV_VEC__
            const uint32_t lane = get_subblockid();
            for (uint32_t slot = 0; slot < 2; ++slot) {
                const uint32_t message = batch * 2 + slot;
                ProducerTile predicate;
                if constexpr (Case::VEC_PRODUCER) {
                    TASSIGN(predicate, 0);
                    TEXPANDS(predicate, laneValue(lane, message));
                    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
                } else {
                    predicate.data() = (uint64_t{0xfedcba98} << 32) | laneValue(lane, message);
                }
                // Delay AIV1 independently so an early AIV0 flag cannot
                // hide a missing wait for the second producer lane.
                if (lane == 1) {
                    uint64_t tick = 0;
                    for (uint32_t delay = 0; delay < 16; ++delay) {
                        asm volatile("MOV %0, SYS_CNT\n" : "+l"(tick));
                    }
                }
                // Both AIVs call TPUSH; the pipe must ignore an unselected producer.
                TPUSH<V2C, ProducerTile, TileSplitAxis::TILE_NO_SPLIT>(v2c, predicate);
                TPUSH<GuardV2C, CtrlTile, TileSplitAxis::TILE_NO_SPLIT>(guardV2C, token);
            }
            for (uint32_t slot = 0; slot < 2; ++slot) {
                CtrlTile ready, response;
                TPOP<GuardC2V, CtrlTile, TileSplitAxis::TILE_NO_SPLIT>(guardC2V, ready);
                TFREE<GuardC2V, TileSplitAxis::TILE_NO_SPLIT>(guardC2V);
                TPOP<C2V, CtrlTile, TileSplitAxis::TILE_NO_SPLIT>(c2v, response);
                TFREE<C2V, TileSplitAxis::TILE_NO_SPLIT>(c2v);
                const uint32_t index = job * 4 + batch * 2 + slot;
                out[(lane + 1) * CtrlTest::OUTPUT_STRIDE + index] = static_cast<uint32_t>(response.data());
            }
#endif
#ifdef __DAV_CUBE__
            uint32_t merged[2];
            for (uint32_t slot = 0; slot < 2; ++slot) {
                CtrlTile ready, predicate;
                TPOP<GuardV2C, CtrlTile, TileSplitAxis::TILE_NO_SPLIT>(guardV2C, ready);
                TFREE<GuardV2C, TileSplitAxis::TILE_NO_SPLIT>(guardV2C);
                TPOP<V2C, CtrlTile, TileSplitAxis::TILE_NO_SPLIT>(v2c, predicate);
                TFREE<V2C, TileSplitAxis::TILE_NO_SPLIT>(v2c);
                merged[slot] = static_cast<uint32_t>(predicate.data());
                out[job * 4 + batch * 2 + slot] = merged[slot];
            }
            for (uint32_t slot = 0; slot < 2; ++slot) {
                CtrlTile response;
                // Full-width, distinct echoes detect stale slots and require
                // identical raw-word broadcast to the two vector consumers.
                response.data() = 0xa5000000U + job * 0x100 + (batch * 2 + slot) * 0x10 + merged[slot];
                TPUSH<C2V, CtrlTile, TileSplitAxis::TILE_NO_SPLIT>(c2v, response);
                TPUSH<GuardC2V, CtrlTile, TileSplitAxis::TILE_NO_SPLIT>(guardC2V, token);
            }
#endif
        }
    }
}

template <int Key>
void LaunchTPushPopCtrl(uint32_t* out, void* stream)
{
    runTPushPopCtrl<Key><<<1, nullptr, stream>>>(out);
}

#define INSTANTIATE_CTRL_CASE(key) template void LaunchTPushPopCtrl<key>(uint32_t*, void*);
INSTANTIATE_CTRL_CASE(1)
INSTANTIATE_CTRL_CASE(2)
INSTANTIATE_CTRL_CASE(3)
INSTANTIATE_CTRL_CASE(4)
INSTANTIATE_CTRL_CASE(5)
INSTANTIATE_CTRL_CASE(6)
INSTANTIATE_CTRL_CASE(7)
INSTANTIATE_CTRL_CASE(8)
INSTANTIATE_CTRL_CASE(9)
INSTANTIATE_CTRL_CASE(10)
INSTANTIATE_CTRL_CASE(11)
INSTANTIATE_CTRL_CASE(12)
#undef INSTANTIATE_CTRL_CASE
