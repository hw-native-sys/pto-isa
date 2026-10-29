/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Test TWAIT operations on CPU
// TWAIT: Blocking wait for signal condition
// Uses std::thread for multi-threaded synchronization (unlike NPU which uses multi-process)

#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include "twait_kernel.h"

// ============================================================================
// TWAIT Tests
// ============================================================================

// Basic TWAIT test: wait for signal == expected value
TEST(TWait, Basic) { ASSERT_TRUE(RunTWaitBasic()); }

// TWAIT with all comparison operators (EQ, NE, GT, GE, LT, LE)
TEST(TWait, CompareOps) { ASSERT_TRUE(RunTWaitCompare()); }

// TWAIT with atomic add: multiple threads contribute, wait for threshold
TEST(TWait, Atomic_2Threads) { ASSERT_TRUE(RunTWaitAtomic(2)); }

TEST(TWait, Atomic_4Threads) { ASSERT_TRUE(RunTWaitAtomic(4)); }

TEST(TWait, Atomic_8Threads) { ASSERT_TRUE(RunTWaitAtomic(8)); }

// TWAIT 2D signal matrix
TEST(TWait, Matrix2D_4x8) { ASSERT_TRUE((RunTWaitMatrix<4, 8>())); }

TEST(TWait, Matrix2D_7x13) { ASSERT_TRUE((RunTWaitMatrix<7, 13>())); }

// TWAIT multi-phase update
TEST(TWait, MultiPhase) { ASSERT_TRUE(RunTWaitMultiPhase()); }

// TWAIT sub-region of signal matrix
TEST(TWait, SubRegion_4x8_of_16cols) { ASSERT_TRUE((RunTWaitSubRegion<16, 4, 8>())); }

namespace {
class ScopedSpinBudget {
public:
    explicit ScopedSpinBudget(const char* value)
    {
        const char* previous = std::getenv("PTO_CPU_SIM_TWAIT_MAX_SPINS");
        if (previous != nullptr) {
            previous_ = previous;
        }
        if (value == nullptr) {
            unsetenv("PTO_CPU_SIM_TWAIT_MAX_SPINS");
        } else {
            setenv("PTO_CPU_SIM_TWAIT_MAX_SPINS", value, 1);
        }
    }

    ~ScopedSpinBudget()
    {
        if (previous_) {
            setenv("PTO_CPU_SIM_TWAIT_MAX_SPINS", previous_->c_str(), 1);
        } else {
            unsetenv("PTO_CPU_SIM_TWAIT_MAX_SPINS");
        }
    }

private:
    std::optional<std::string> previous_;
};
} // namespace

TEST(TWait, ExpiredBudgetCannotReportSuccessfulWait)
{
    ScopedSpinBudget budget("1");
    alignas(64) std::atomic<int32_t> value{0};
    pto::comm::Signal signal(reinterpret_cast<int32_t*>(&value));
    try {
        pto::comm::TWAIT(signal, 1, pto::comm::WaitCmp::EQ);
        FAIL() << "TWAIT returned while the signal predicate was false";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("PTO_CPU_SIM_TWAIT_TIMEOUT"), std::string::npos);
    }
    EXPECT_EQ(value.load(), 0);
}

TEST(TWait, SatisfiedPredicateSucceedsWithFiniteBudget)
{
    ScopedSpinBudget budget("1");
    alignas(64) std::atomic<int32_t> value{1};
    pto::comm::Signal signal(reinterpret_cast<int32_t*>(&value));
    EXPECT_NO_THROW(pto::comm::TWAIT(signal, 1, pto::comm::WaitCmp::EQ));
}

TEST(TWait, EveryStridedSignalMustSatisfyPredicate)
{
    ScopedSpinBudget budget("1");
    alignas(64) std::atomic<int32_t> values[8];
    for (auto& value : values) {
        value.store(1);
    }
    values[5].store(0);
    pto::comm::Signal2D<2, 2> signals(reinterpret_cast<int32_t*>(values), 4);
    EXPECT_THROW(pto::comm::TWAIT(signals, 1, pto::comm::WaitCmp::EQ), std::runtime_error);
    values[5].store(1);
    EXPECT_NO_THROW(pto::comm::TWAIT(signals, 1, pto::comm::WaitCmp::EQ));
}

TEST(TWait, DefaultWaitAcquiresPublishedPayload)
{
    ScopedSpinBudget budget(nullptr);
    alignas(64) std::atomic<int32_t> value{0};
    int payload = 0;
    pto::comm::Signal signal(reinterpret_cast<int32_t*>(&value));
    std::thread producer([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        payload = 42;
        value.store(1, std::memory_order_release);
    });
    pto::comm::TWAIT(signal, 1, pto::comm::WaitCmp::EQ);
    EXPECT_EQ(payload, 42);
    producer.join();
}
