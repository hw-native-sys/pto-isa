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
#include <cstdint>
#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>

using namespace pto;

// Producer runs until the next push needs a free credit, then the consumer pops one tile.
// That is the schedule that wraps a live slot when constructor credits are posted ahead of
// the startup window. GitHub issue 390: depth 2, four C2V pushes, TILE_UP_DOWN.
template <uint32_t SlotNum>
struct CreditSim {
    bool overwrite = false;
    bool deadlock = false;
    uint32_t waits = 0;
    uint32_t notifies = 0;
    uint32_t creditsLeft = 0;
    uint32_t maxInFlight = 0;
};

template <uint32_t SlotNum>
CreditSim<SlotNum> SimulateFreeCredits(uint32_t pushCount, uint32_t initialCredits)
{
    using Credit = TPipeFreeCredit<SlotNum>;
    CreditSim<SlotNum> result;
    result.creditsLeft = initialCredits;
    std::array<uint8_t, SlotNum> live{};
    uint32_t produced = 0;
    uint32_t consumed = 0;
    const uint32_t stepLimit = pushCount * 4 + 2;
    for (uint32_t step = 0; step < stepLimit && (produced < pushCount || consumed < pushCount); ++step) {
        if (produced < pushCount && (!Credit::shouldWaitFree(produced) || result.creditsLeft > 0)) {
            if (Credit::shouldWaitFree(produced)) {
                --result.creditsLeft;
                ++result.waits;
            }
            const uint32_t slot = produced % SlotNum;
            if (live[slot] != 0) {
                result.overwrite = true;
            }
            live[slot] = 1;
            ++produced;
            const uint32_t inFlight = produced - consumed;
            if (inFlight > result.maxInFlight) {
                result.maxInFlight = inFlight;
            }
            continue;
        }
        if (consumed < produced) {
            live[consumed % SlotNum] = 0;
            if (Credit::shouldNotifyFree(consumed)) {
                ++result.creditsLeft;
                ++result.notifies;
            }
            ++consumed;
            continue;
        }
        result.deadlock = true;
        break;
    }
    if (produced != pushCount || consumed != pushCount) {
        result.deadlock = true;
    }
    return result;
}

template <uint32_t SlotNum>
void ExpectStartupWindowKeepsSlots(uint32_t pushCount)
{
    using Credit = TPipeFreeCredit<SlotNum>;
    const CreditSim<SlotNum> sim = SimulateFreeCredits<SlotNum>(pushCount, Credit::kInitialFreeCredits);
    ASSERT_FALSE(sim.overwrite) << "SlotNum=" << SlotNum << " pushes=" << pushCount;
    ASSERT_FALSE(sim.deadlock) << "SlotNum=" << SlotNum << " pushes=" << pushCount;
    ASSERT_LE(sim.maxInFlight, SlotNum) << "SlotNum=" << SlotNum << " pushes=" << pushCount;
    ASSERT_EQ(sim.waits, Credit::countProducerWaits(pushCount));
    ASSERT_EQ(sim.notifies, Credit::countConsumerNotifies(pushCount));
    ASSERT_EQ(sim.creditsLeft, Credit::countPendingFreeCredits(pushCount));
}

TEST(TPipeFreeCreditTest, depth2_four_pushes_do_not_wrap_a_live_slot)
{
    using Credit = TPipeFreeCredit<2>;
    static_assert(Credit::kInitialFreeCredits == 0);
    static_assert(Credit::SyncPeriod == 2);
    static_assert(Credit::countProducerWaits(4) == 1);
    static_assert(Credit::countConsumerNotifies(4) == 2);
    static_assert(Credit::countPendingFreeCredits(4) == 1);

    const CreditSim<2> fixed = SimulateFreeCredits<2>(4, Credit::kInitialFreeCredits);
    EXPECT_FALSE(fixed.overwrite);
    EXPECT_FALSE(fixed.deadlock);
    EXPECT_EQ(fixed.maxInFlight, 2u);
    EXPECT_EQ(fixed.creditsLeft, 1u);

    // Two 256-column blocks never wrap; both credit models leave that control exact.
    const CreditSim<2> twoPushes = SimulateFreeCredits<2>(2, Credit::kInitialFreeCredits);
    EXPECT_FALSE(twoPushes.overwrite);
    EXPECT_EQ(twoPushes.maxInFlight, 2u);
}

TEST(TPipeFreeCreditTest, sync_period_constructor_credits_overwrite_the_second_slot)
{
    using Credit = TPipeFreeCredit<2>;
    const CreditSim<2> buggy = SimulateFreeCredits<2>(4, Credit::SyncPeriod);
    EXPECT_TRUE(buggy.overwrite);
    EXPECT_GT(buggy.maxInFlight, 2u);

    const CreditSim<2> control = SimulateFreeCredits<2>(2, Credit::SyncPeriod);
    EXPECT_FALSE(control.overwrite);
    EXPECT_EQ(control.maxInFlight, 2u);
}

TEST(TPipeFreeCreditTest, startup_window_matches_drain_for_depths_1_through_5)
{
    constexpr uint32_t kMaxPushes = 1024;
    for (uint32_t pushes = 0; pushes <= kMaxPushes; ++pushes) {
        ExpectStartupWindowKeepsSlots<1>(pushes);
        ExpectStartupWindowKeepsSlots<2>(pushes);
        ExpectStartupWindowKeepsSlots<3>(pushes);
        ExpectStartupWindowKeepsSlots<4>(pushes);
        ExpectStartupWindowKeepsSlots<5>(pushes);
    }
}
