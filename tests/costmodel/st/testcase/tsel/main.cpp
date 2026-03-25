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
#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>

using namespace std;
using namespace PtoTestCommon;

class TSELTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

template <typename T, int kTRows_, int kTCols_, float profiling, float accuracy>
void LaunchTSel(void *stream);

template <typename T, int kTRows_, int kTCols_, float profiling, float accuracy>
void test_tsel()
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    LaunchTSel<T, kTRows_, kTCols_, profiling, accuracy>(stream);
    aclrtSynchronizeStream(stream);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

// 34 + (4-1)*20 = 94
TEST_F(TSELTest, case_half_4x128)
{
    test_tsel<aclFloat16, 4, 128, 94.0f, 0.0f>();
}

// 34 + (1-1)*20 = 34
TEST_F(TSELTest, case_half_1x128)
{
    test_tsel<aclFloat16, 1, 128, 34.0f, 0.0f>();
}

// 34 + (4-1)*20 = 94  (same formula, FP32 vsel has same per_repeat)
TEST_F(TSELTest, case_float_4x64)
{
    test_tsel<float, 4, 64, 94.0f, 0.0f>();
}
