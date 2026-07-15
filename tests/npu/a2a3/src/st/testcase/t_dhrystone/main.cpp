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
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

class TDHRYSTONETest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

template <int iterations>
void LaunchTDhrystone(void* stream);

template <int iterations>
void test_tdhrystone()
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    LaunchTDhrystone<iterations>(stream);
    aclrtSynchronizeStream(stream);

    EXPECT_TRUE(true);
}

TEST_F(TDHRYSTONETest, case_1000i) { test_tdhrystone<1000>(); }
TEST_F(TDHRYSTONETest, case_2000i) { test_tdhrystone<2000>(); }
TEST_F(TDHRYSTONETest, case_3000i) { test_tdhrystone<3000>(); }
TEST_F(TDHRYSTONETest, case_4000i) { test_tdhrystone<4000>(); }
TEST_F(TDHRYSTONETest, case_5000i) { test_tdhrystone<5000>(); }
TEST_F(TDHRYSTONETest, case_6000i) { test_tdhrystone<6000>(); }
TEST_F(TDHRYSTONETest, case_7000i) { test_tdhrystone<7000>(); }
TEST_F(TDHRYSTONETest, case_8000i) { test_tdhrystone<8000>(); }