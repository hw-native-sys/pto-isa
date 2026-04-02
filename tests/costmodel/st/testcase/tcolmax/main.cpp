/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

class TCOLMAXTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, float profiling, float accuracy>
void LaunchTColMax(T *out, T *src, void *stream);

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, float profiling, float accuracy>
void test_tcolmax()
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    LaunchTColMax<T, kGRows_, kGCols_, kTRows_, kTCols_, profiling, accuracy>(nullptr, nullptr, stream);
    aclrtSynchronizeStream(stream);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

TEST_F(TCOLMAXTest, case_float_64x64)
{
    test_tcolmax<float, 64, 64, 64, 64, 1130.0f, 1.0f>();
}
TEST_F(TCOLMAXTest, case_half_64x64)
{
    test_tcolmax<aclFloat16, 64, 64, 64, 64, 1256.0f, 1.0f>();
}
TEST_F(TCOLMAXTest, case_int16_64x64)
{
    test_tcolmax<int16_t, 64, 64, 64, 64, 1256.0f, 1.0f>();
}
TEST_F(TCOLMAXTest, case_half_16x256)
{
    test_tcolmax<aclFloat16, 16, 256, 16, 256, 266.0f, 1.0f>();
}
TEST_F(TCOLMAXTest, case_float_1x3072_1x3072_1x3072)
{
    test_tcolmax<float, 1, 3072, 1, 3072, 14.0f, 1.0f>();
}
