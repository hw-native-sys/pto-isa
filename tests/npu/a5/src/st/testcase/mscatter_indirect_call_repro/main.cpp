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
#include <functional>
#include "acl/acl.h"
#include "mscatter_indirect_call_repro_common.h"

using namespace std;
using namespace PtoTestCommon;

class MSCATTERIndirectCallReproTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

static std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    return std::string("../") + testInfo->test_suite_name() + "." + testInfo->name();
}

template <typename T, typename TIdx, typename Launcher>
void run_mscatter_repro_test(size_t srcCount, size_t idxCount, size_t outCount, Launcher launcher)
{
    size_t srcByteSize = srcCount * sizeof(T);
    size_t idxByteSize = idxCount * sizeof(TIdx);
    size_t outByteSize = outCount * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *srcHost, *outHost;
    TIdx *idxHost;
    T *srcDevice, *outDevice;
    TIdx *idxDevice;

    aclrtMallocHost((void **)(&srcHost), srcByteSize);
    aclrtMallocHost((void **)(&idxHost), idxByteSize);
    aclrtMallocHost((void **)(&outHost), outByteSize);

    aclrtMalloc((void **)&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&idxDevice, idxByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&outDevice, outByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/src.bin", srcByteSize, srcHost, srcByteSize);
    ReadFile(GetGoldenDir() + "/indices.bin", idxByteSize, idxHost, idxByteSize);

    aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(idxDevice, idxByteSize, idxHost, idxByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    aclrtMemset(outDevice, outByteSize, 0, outByteSize);

    launcher(outDevice, srcDevice, idxDevice, stream);

    aclrtSynchronizeStream(stream);   // EXPECTED to hang on current toolchain; see kernel.cpp header.
    aclrtMemcpy(outHost, outByteSize, outDevice, outByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", outHost, outByteSize);

    aclrtFree(srcDevice);
    aclrtFree(idxDevice);
    aclrtFree(outDevice);

    aclrtFreeHost(srcHost);
    aclrtFreeHost(idxHost);
    aclrtFreeHost(outHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(outCount);
    std::vector<T> devFinal(outCount);
    ReadFile(GetGoldenDir() + "/golden.bin", outByteSize, golden.data(), outByteSize);
    ReadFile(GetGoldenDir() + "/output.bin", outByteSize, devFinal.data(), outByteSize);

    bool ret = ResultCmp<T>(golden, devFinal, 0.0f);
    EXPECT_TRUE(ret);
}

#define DECLARE_LAUNCH(NAME, THOST, TIDX) \
    void Launch_##NAME(THOST *out, THOST *src, TIDX *indices, void *stream);

DECLARE_LAUNCH(indirect_call_elem2d_float_8x32_random_256size, float, int32_t)

// Mirrors MSCATTERTest.case_elem2d_float_8x32_random_256size from
// tests/.../mscatter/main.cpp, but routed through a kernel that interposes one
// volatile fn-pointer indirect call (BLR) before the MSCATTER body.
//
// Reference direct-call test: PASSes in ~5 seconds.
// This indirect-call variant: chip hangs (no SIMT scheduler staged for BLR
// target). When the toolchain learns to emit SIMT-aware indirect call
// lowering, this test will start passing without code change.
TEST_F(MSCATTERIndirectCallReproTest, case_indirect_call_elem2d_float_8x32_random_256size)
{
    constexpr size_t kSrcCount = 8 * 32;     // 256
    constexpr size_t kIdxCount = 8 * 32;     // 256
    constexpr size_t kOutCount = 256;
    run_mscatter_repro_test<float, int32_t>(
        kSrcCount, kIdxCount, kOutCount,
        Launch_indirect_call_elem2d_float_8x32_random_256size);
}
