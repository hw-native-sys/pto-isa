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
