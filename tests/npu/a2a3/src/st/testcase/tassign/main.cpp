/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

#include "test_common.h"
#include <acl/acl.h>
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <uint32_t caseId>
void launchTASSIGNTestCase(void* out, void* src, int offset, aclrtStream stream);

class TASSIGNTest : public testing::Test {
public:
    aclrtStream stream;
    void* dstHost;
    void* dstDevice;
    void* srcDevice;

protected:
    void SetUp() override
    {
        aclInit(nullptr);
        aclrtSetDevice(0);
        aclrtCreateStream(&stream);
    }

    void TearDown() override
    {
        aclrtDestroyStream(stream);
        aclrtResetDevice(0);
        aclFinalize();
    }

    template <uint32_t caseId, typename T, int offset>
    bool TAssignTestFramework()
    {
        size_t ptrByteSize = sizeof(T*);
        aclrtMallocHost(&dstHost, ptrByteSize);
        aclrtMalloc(&srcDevice, ptrByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&dstDevice, ptrByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

        launchTASSIGNTestCase<caseId>(dstDevice, srcDevice, offset, stream);
        aclrtSynchronizeStream(stream);
        aclrtMemcpy(dstHost, ptrByteSize, dstDevice, ptrByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
        bool ret = reinterpret_cast<uintptr_t>((T*)srcDevice + offset) == *(reinterpret_cast<uintptr_t*>(dstHost));
        aclrtFree(dstDevice);
        aclrtFree(srcDevice);
        aclrtFreeHost(dstHost);
        return ret;
    }
};

TEST_F(TASSIGNTest, case1)
{
    bool ret = TAssignTestFramework<1, float, 4>();
    EXPECT_TRUE(ret);
}
