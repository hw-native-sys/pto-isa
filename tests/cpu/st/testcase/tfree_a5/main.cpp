/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <array>
#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>

// Exercise the production A5 dispatchers with host-side receive/notification
// fixtures. Keep the CPU dispatchers available under their original names.
#undef TPOP_HPP
#define TPOP_IMPL A5_TPOP_IMPL
#include <pto/npu/a5/TPop.hpp>
#undef TPOP_IMPL
#define TFREE_IMPL A5_TFREE_IMPL
#include <pto/npu/a5/TFree.hpp>
#undef TFREE_IMPL

using namespace pto;

namespace {
using ReceiveTile = Tile<TileType::Vec, float, 16, 16>;
using SlotTensor = GlobalTensor<float, Shape<1, 1, 1, 16, 16>, Stride<256, 256, 256, 16, 1>>;

template <uint32_t Slots = 2, bool AutoFree = false>
struct CreditPipe {
    static constexpr uint32_t SyncPeriod = Slots <= 2 ? Slots : Slots / 2;
    static constexpr bool is_c2v_gm = true;
    static constexpr bool is_v2c_gm = false;
    static constexpr bool is_both_gm = false;

    struct RingFiFo {
        static constexpr uint32_t SLOT_NUM = Slots;
        static constexpr uint32_t SLOT_SIZE = 16 * 16 * sizeof(float);
    };

    struct Consumer {
        uint32_t tileIndex = 0;
        uint32_t freeIndex = 0;
        uint32_t notifications = 0;
        bool freeEnabled = true;

        bool getWaitStatus() const { return false; }
        bool getFreeStatus() const { return freeEnabled; }
        template <TileSplitAxis Split>
        void wait() const
        {}
        template <TileSplitAxis Split>
        void free()
        {
            notifications++;
        }
        template <typename TileCons, TileSplitAxis Split, typename Fifo>
        bool pop(Fifo&, TileCons&, int32_t)
        {
            return AutoFree;
        }
    } cons;

    struct {
        void* GM_SLOT_BUFFER = nullptr;
    } fifo;

    static bool shouldNotifyFree(uint32_t index) { return (index + 1) % SyncPeriod == 0; }
};

template <typename Pipe>
void pop(Pipe& pipe)
{
    ReceiveTile tile;
    A5_TPOP_IMPL<Pipe, ReceiveTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, tile);
}

template <typename Pipe>
void free(Pipe& pipe)
{
    A5_TFREE_IMPL<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
}

TEST(TFREE_A5Test, DeferredPairReturnsOneCredit)
{
    CreditPipe<> pipe;
    pop(pipe);
    pop(pipe);
    free(pipe);
    EXPECT_EQ(pipe.cons.notifications, 0u);
    free(pipe);
    EXPECT_EQ(pipe.cons.notifications, 1u);
    EXPECT_EQ(pipe.cons.tileIndex, 2u);
    EXPECT_EQ(pipe.cons.freeIndex, 2u);

    // Complete the four-pop pattern from the two-matmul native reproducer.
    pop(pipe);
    free(pipe);
    pop(pipe);
    free(pipe);
    EXPECT_EQ(pipe.cons.notifications, 2u);
}

template <uint32_t Slots>
void checkRepeatedBatches()
{
    CreditPipe<Slots> pipe;
    for (uint32_t batch = 0; batch < 32; ++batch) {
        for (uint32_t slot = 0; slot < Slots; ++slot) {
            pop(pipe);
        }
        for (uint32_t slot = 0; slot < Slots; ++slot) {
            free(pipe);
        }
        EXPECT_EQ(pipe.cons.notifications, (batch + 1) * Slots / pipe.SyncPeriod);
        EXPECT_EQ(pipe.cons.tileIndex, (batch + 1) * Slots);
        EXPECT_EQ(pipe.cons.freeIndex, pipe.cons.tileIndex);
    }
}

TEST(TFREE_A5Test, RepeatedBatchesRespectSparsePeriod)
{
    checkRepeatedBatches<1>();
    checkRepeatedBatches<2>();
    checkRepeatedBatches<8>();
}

TEST(TFREE_A5Test, AutoFreeAndExplicitSubblockPopDoNotDoubleNotify)
{
    CreditPipe<2, true> pipe;
    ReceiveTile tile;
    A5_TPOP_IMPL<decltype(pipe), ReceiveTile, TileSplitAxis::TILE_UP_DOWN>(pipe, tile, 0);
    A5_TPOP_IMPL<decltype(pipe), ReceiveTile, TileSplitAxis::TILE_UP_DOWN>(pipe, tile, 1);
    EXPECT_EQ(pipe.cons.notifications, 1u);
    EXPECT_EQ(pipe.cons.freeIndex, 2u);
    free(pipe);
    free(pipe);
    EXPECT_EQ(pipe.cons.notifications, 1u);
}

TEST(TFREE_A5Test, AutoFreeWithoutExplicitSubblockRetiresEveryPop)
{
    CreditPipe<2, true> pipe;
    pop(pipe);
    EXPECT_EQ(pipe.cons.freeIndex, 1u);
    pop(pipe);
    EXPECT_EQ(pipe.cons.notifications, 1u);
    free(pipe);
    EXPECT_EQ(pipe.cons.notifications, 1u);
}

TEST(TFREE_A5Test, FreeWithNoOutstandingPopDoesNotNotify)
{
    CreditPipe<1> pipe;
    free(pipe);
    EXPECT_EQ(pipe.cons.notifications, 0u);
    pop(pipe);
    free(pipe);
    free(pipe);
    EXPECT_EQ(pipe.cons.notifications, 1u);
    EXPECT_EQ(pipe.cons.freeIndex, 1u);
}

TEST(TFREE_A5Test, DisabledNotificationStillRetiresEntries)
{
    CreditPipe<> pipe;
    pipe.cons.freeEnabled = false;
    pop(pipe);
    pop(pipe);
    free(pipe);
    free(pipe);
    EXPECT_EQ(pipe.cons.notifications, 0u);
    EXPECT_EQ(pipe.cons.freeIndex, 2u);

    pipe.cons.freeEnabled = true;
    pop(pipe);
    pop(pipe);
    free(pipe);
    free(pipe);
    EXPECT_EQ(pipe.cons.notifications, 1u);
    EXPECT_EQ(pipe.cons.freeIndex, 4u);
}

TEST(TFREE_A5Test, GlobalViewsRetainAddressingUntilFreed)
{
    CreditPipe<> pipe;
    std::array<float, 2 * 16 * 16> slots{};
    pipe.fifo.GM_SLOT_BUFFER = slots.data();
    SlotTensor first;
    SlotTensor second;
    A5_TPOP_IMPL<decltype(pipe), SlotTensor, TileSplitAxis::TILE_NO_SPLIT>(pipe, first);
    A5_TPOP_IMPL<decltype(pipe), SlotTensor, TileSplitAxis::TILE_NO_SPLIT>(pipe, second);
    EXPECT_EQ(first.data(), slots.data());
    EXPECT_EQ(second.data(), slots.data() + 16 * 16);
    EXPECT_EQ(pipe.cons.freeIndex, 0u);

    A5_TFREE_IMPL<decltype(pipe), SlotTensor, TileSplitAxis::TILE_NO_SPLIT>(pipe, first);
    EXPECT_EQ(pipe.cons.notifications, 0u);
    A5_TFREE_IMPL<decltype(pipe), SlotTensor, TileSplitAxis::TILE_NO_SPLIT>(pipe, second);
    EXPECT_EQ(pipe.cons.notifications, 1u);
    EXPECT_EQ(pipe.cons.tileIndex, 2u);
}
} // namespace
