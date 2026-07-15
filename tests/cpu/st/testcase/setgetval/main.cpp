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
#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>

using namespace std;
using namespace PtoTestCommon;

class SETGETVALTest : public testing::Test {
public:
protected:
    void SetUp() override {}

    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}
template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void LaunchSetGetVal(T* src0, void* stream);

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void test_setgetval()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t srcByteSize = kGRows_ * kGCols_ * sizeof(T);
    T* srcHost;
    T* srcDevice;

    aclrtMallocHost((void**)(&srcHost), srcByteSize);

    aclrtMalloc((void**)&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    LaunchSetGetVal<T, kGRows_, kGCols_, kTRows_, kTCols_>(srcDevice, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(srcHost, srcByteSize, srcDevice, srcByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(srcDevice);
    T value_4 = srcHost[4];
    T value_5 = srcHost[5];
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    bool res = false;
    if ((value_4 - 12.34 < 0.00001) && value_5 - 12.34 < 0.00001) {
        res = true;
    }
    EXPECT_TRUE(res);
}

TEST_F(SETGETVALTest, case1) { test_setgetval<float, 32, 32, 32, 32>(); }
