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

#include "acl/acl.h"
#include "ctrl_test_config.h"

template <int Key>
void LaunchTPushPopCtrl(uint32_t* out, void* stream);

class TPushPopCtrlTest : public testing::Test {
protected:
    void SetUp() override
    {
        ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
        ASSERT_EQ(aclrtSetDevice(0), ACL_SUCCESS);
        ASSERT_EQ(aclrtCreateStream(&stream_), ACL_SUCCESS);
        ASSERT_EQ(
            aclrtMalloc(reinterpret_cast<void**>(&deviceOut_), sizeof(output_), ACL_MEM_MALLOC_HUGE_FIRST),
            ACL_SUCCESS);
        ASSERT_EQ(aclrtMemset(deviceOut_, sizeof(output_), 0xcd, sizeof(output_)), ACL_SUCCESS);
    }

    void TearDown() override
    {
        if (deviceOut_ != nullptr) {
            aclrtFree(deviceOut_);
        }
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
        }
        aclrtResetDevice(0);
        aclFinalize();
    }

    template <int Key>
    void check()
    {
        using Case = CtrlTest::Case<Key>;
        LaunchTPushPopCtrl<Key>(deviceOut_, stream_);
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
        ASSERT_EQ(
            aclrtMemcpy(output_.data(), sizeof(output_), deviceOut_, sizeof(output_), ACL_MEMCPY_DEVICE_TO_HOST),
            ACL_SUCCESS);
        constexpr std::array<uint32_t, 4> LANE0 = {0, 0, 2, 0x80000000U};
        constexpr std::array<uint32_t, 4> LANE1 = {0, 7, 0, 0x40};
        for (uint32_t job = 0; job < CtrlTest::JOBS; ++job) {
            for (uint32_t message = 0; message < CtrlTest::MESSAGES_PER_JOB; ++message) {
                SCOPED_TRACE(testing::Message() << "job=" << job << " message=" << message);
                uint32_t expected;
                if constexpr (Case::SINGLE_LANE == 0) {
                    expected = Case::MERGE_NONE ? LANE0[message] : (LANE0[message] != 0);
                } else if constexpr (Case::SINGLE_LANE == 1) {
                    expected = Case::MERGE_NONE ? LANE1[message] : (LANE1[message] != 0);
                } else if constexpr (Case::MERGE_AND) {
                    expected = (LANE0[message] != 0) && (LANE1[message] != 0);
                } else {
                    expected = (LANE0[message] != 0) || (LANE1[message] != 0);
                }
                const uint32_t index = job * 4 + message;
                EXPECT_EQ(output_[index], expected) << "AIC merge";
                const uint32_t echo = 0xa5000000U + job * 0x100 + message * 0x10 + expected;
                EXPECT_EQ(output_[CtrlTest::OUTPUT_STRIDE + index], echo) << "AIV0 broadcast";
                EXPECT_EQ(output_[2 * CtrlTest::OUTPUT_STRIDE + index], echo) << "AIV1 broadcast";
            }
        }
    }

private:
    aclrtStream stream_ = nullptr;
    uint32_t* deviceOut_ = nullptr;
    std::array<uint32_t, CtrlTest::OUTPUT_WORDS> output_{};
};

TEST_F(TPushPopCtrlTest, tpipe_or) { check<1>(); }

TEST_F(TPushPopCtrlTest, tpipe_and) { check<2>(); }

TEST_F(TPushPopCtrlTest, tpipe_vec0_none) { check<3>(); }

TEST_F(TPushPopCtrlTest, tpipe_vec1_none) { check<4>(); }

TEST_F(TPushPopCtrlTest, tpipe_forward_off) { check<5>(); }

TEST_F(TPushPopCtrlTest, tpipe_backward_off) { check<6>(); }

TEST_F(TPushPopCtrlTest, tpipe_both_off) { check<7>(); }

TEST_F(TPushPopCtrlTest, tpipe_vec_predicate) { check<8>(); }

TEST_F(TPushPopCtrlTest, tpipe_vec0_or) { check<9>(); }

TEST_F(TPushPopCtrlTest, tpipe_vec1_or) { check<10>(); }

TEST_F(TPushPopCtrlTest, tpipe_vec0_and) { check<11>(); }

TEST_F(TPushPopCtrlTest, tpipe_vec1_and) { check<12>(); }
