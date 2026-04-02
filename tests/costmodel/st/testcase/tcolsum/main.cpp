/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

class TCOLSUMTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, bool IsBinary, float profiling,
          float accuracy>
void LaunchTCOLSUM(T *out, T *src, void *stream);

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, bool IsBinary, float profiling,
          float accuracy>
void test_tcolsum()
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    LaunchTCOLSUM<T, kGRows_, kGCols_, kTRows_, kTCols_, IsBinary, profiling, accuracy>(nullptr, nullptr, stream);
    aclrtSynchronizeStream(stream);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

TEST_F(TCOLSUMTest, case_float_64x64_64x64_64x64)
{
    test_tcolsum<float, 64, 64, 64, 64, true, 140.0f, 1.0f>();
}
TEST_F(TCOLSUMTest, case_float_1x3072_1x3072_1x3072)
{
    test_tcolsum<float, 1, 3072, 1, 3072, true, 14.0f, 1.0f>();
}
TEST_F(TCOLSUMTest, case_half_16x256_16x256_16x256)
{
    test_tcolsum<aclFloat16, 16, 256, 16, 256, false, 266.0f, 1.0f>();
}
