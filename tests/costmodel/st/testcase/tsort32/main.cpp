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

template <typename T0, typename T1, int kGRows, int kGCols, int kTRows, int kTCols, int validRow, int validCol,
          float profiling, float accuracy>
void launchTSort32(void *stream);

class TSORT32Test : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

template <typename T0, typename T1, int kGRows, int kGCols, int kTRows, int kTCols, int validRow, int validCol,
          float profiling, float accuracy>
void TSort32Test()
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    launchTSort32<T0, T1, kGRows, kGCols, kTRows, kTCols, validRow, validCol, profiling, accuracy>(stream);
    aclrtSynchronizeStream(stream);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

// test0: int16_t, 16x16, R=0: 14 + 15*18 = 284
TEST_F(TSORT32Test, test0)
{
    TSort32Test<int16_t, uint32_t, 16, 16, 16, 16, 16, 16, 284.0f, 1.0f>();
}

// test1: float, 8x32: costmodel=172
TEST_F(TSORT32Test, test1)
{
    TSort32Test<float, uint32_t, 8, 32, 8, 32, 8, 32, 172.0f, 1.0f>();
}

// test2: int32_t, 7x32: costmodel=150
TEST_F(TSORT32Test, test2)
{
    TSort32Test<int32_t, uint32_t, 7, 32, 7, 32, 7, 32, 150.0f, 1.0f>();
}

// test3: aclFloat16->half, 32x16, R=0: 14 + 31*18 = 572
TEST_F(TSORT32Test, test3)
{
    TSort32Test<aclFloat16, uint32_t, 32, 16, 32, 16, 32, 16, 572.0f, 1.0f>();
}
