/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "acl/acl.h"
#include "test_common.h"
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>
#include <climits>
#include <vector>

using namespace PtoTestCommon;

void LaunchTPushPopC2VMaxTrack(float* outVec, float* outGMax, float* outRMax, float* srcA, float* srcB, void* stream);
void LaunchTPushPopC2VMaxTrackMOnly(
    float* outVec, float* outGMax, float* outRMax, float* srcA, float* srcB, void* stream);

class TPushPopC2VMaxTrackTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

using LaunchFn = void (*)(float*, float*, float*, float*, float*, void*);

// run_st.py: gen_data writes build/<Suite.Case>/ ; binary runs from build/bin/.
// Prefer path next to the executable so a non-bin cwd still finds goldens.
static bool IsDir(const std::string& path)
{
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

static std::string Dirname(const std::string& path)
{
    const auto pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return ".";
    }
    if (pos == 0) {
        return "/";
    }
    return path.substr(0, pos);
}

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseDir = std::string(testInfo->test_suite_name()) + "." + testInfo->name();

    std::vector<std::string> candidates;
    candidates.push_back("../" + caseDir); // cwd == build/bin (run_st.py)
    candidates.push_back(caseDir);         // cwd == build
    candidates.push_back("./" + caseDir);

    char exeBuf[PATH_MAX];
    const ssize_t n = readlink("/proc/self/exe", exeBuf, sizeof(exeBuf) - 1);
    if (n > 0) {
        exeBuf[n] = '\0';
        candidates.insert(candidates.begin(), Dirname(std::string(exeBuf)) + "/../" + caseDir);
    }

    for (const auto& cand : candidates) {
        if (IsDir(cand)) {
            return cand;
        }
    }

    char cwdBuf[PATH_MAX];
    const char* cwd = getcwd(cwdBuf, sizeof(cwdBuf)) ? cwdBuf : "<unknown>";
    ADD_FAILURE() << "Golden dir not found for " << caseDir << " (cwd=" << cwd
                  << "). Run via run_st.py so gen_data.py creates build/" << caseDir
                  << "/ (build_st.py does not generate goldens).";
    return "../" + caseDir;
}

void TPushPopC2VMaxTrackTestFunc(
    uint32_t aRows, uint32_t aCols, uint32_t bRows, uint32_t bCols, uint32_t cRows, uint32_t cCols, LaunchFn launch)
{
    constexpr uint32_t groupNum = 8;
    const uint32_t gMaxCols = cCols / groupNum;

    const size_t aCount = static_cast<size_t>(aRows) * aCols;
    const size_t bCount = static_cast<size_t>(bRows) * bCols;
    const size_t vecCount = static_cast<size_t>(cRows) * cCols;
    const size_t gMaxCount = static_cast<size_t>(cRows) * gMaxCols;
    const size_t rMaxCount = static_cast<size_t>(cRows);

    size_t aBytes = aCount * sizeof(float);
    size_t bBytes = bCount * sizeof(float);
    size_t vecBytes = vecCount * sizeof(float);
    size_t gMaxBytes = gMaxCount * sizeof(float);
    size_t rMaxBytes = rMaxCount * sizeof(float);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    float* srcAHost = nullptr;
    float* srcBHost = nullptr;
    float* outVecHost = nullptr;
    float* outGMaxHost = nullptr;
    float* outRMaxHost = nullptr;
    float* srcADevice = nullptr;
    float* srcBDevice = nullptr;
    float* outVecDevice = nullptr;
    float* outGMaxDevice = nullptr;
    float* outRMaxDevice = nullptr;

    aclrtMallocHost((void**)(&srcAHost), aBytes);
    aclrtMallocHost((void**)(&srcBHost), bBytes);
    aclrtMallocHost((void**)(&outVecHost), vecBytes);
    aclrtMallocHost((void**)(&outGMaxHost), gMaxBytes);
    aclrtMallocHost((void**)(&outRMaxHost), rMaxBytes);

    aclrtMalloc((void**)&srcADevice, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcBDevice, bBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&outVecDevice, vecBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&outGMaxDevice, gMaxBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&outRMaxDevice, rMaxBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aBytes, srcAHost, aBytes);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bBytes, srcBHost, bBytes);

    aclrtMemcpy(srcADevice, aBytes, srcAHost, aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcBDevice, bBytes, srcBHost, bBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    launch(outVecDevice, outGMaxDevice, outRMaxDevice, srcADevice, srcBDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outVecHost, vecBytes, outVecDevice, vecBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(outGMaxHost, gMaxBytes, outGMaxDevice, gMaxBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(outRMaxHost, rMaxBytes, outRMaxDevice, rMaxBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    const std::string goldenDir = GetGoldenDir();
    WriteFile(goldenDir + "/output_vec.bin", outVecHost, vecBytes);
    WriteFile(goldenDir + "/output_gmax.bin", outGMaxHost, gMaxBytes);
    WriteFile(goldenDir + "/output_rmax.bin", outRMaxHost, rMaxBytes);

    aclrtFree(srcADevice);
    aclrtFree(srcBDevice);
    aclrtFree(outVecDevice);
    aclrtFree(outGMaxDevice);
    aclrtFree(outRMaxDevice);
    aclrtFreeHost(srcAHost);
    aclrtFreeHost(srcBHost);
    aclrtFreeHost(outVecHost);
    aclrtFreeHost(outGMaxHost);
    aclrtFreeHost(outRMaxHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> goldenVec(vecCount);
    std::vector<float> goldenGMax(gMaxCount);
    std::vector<float> goldenRMax(rMaxCount);
    std::vector<float> outVec(vecCount);
    std::vector<float> outGMax(gMaxCount);
    std::vector<float> outRMax(rMaxCount);

    ReadFile(goldenDir + "/golden_vec.bin", vecBytes, goldenVec.data(), vecBytes);
    ReadFile(goldenDir + "/golden_gmax.bin", gMaxBytes, goldenGMax.data(), gMaxBytes);
    ReadFile(goldenDir + "/golden_rmax.bin", rMaxBytes, goldenRMax.data(), rMaxBytes);
    ReadFile(goldenDir + "/output_vec.bin", vecBytes, outVec.data(), vecBytes);
    ReadFile(goldenDir + "/output_gmax.bin", gMaxBytes, outGMax.data(), gMaxBytes);
    ReadFile(goldenDir + "/output_rmax.bin", rMaxBytes, outRMax.data(), rMaxBytes);

    EXPECT_TRUE(ResultCmp(goldenVec, outVec, 0.001f));
    EXPECT_TRUE(ResultCmp(goldenGMax, outGMax, 0.001f));
    EXPECT_TRUE(ResultCmp(goldenRMax, outRMax, 0.001f));
}

TEST_F(TPushPopC2VMaxTrackTest, case_maxtrack_512_128_128_512)
{
    // A[512,128] @ B[128,512] -> C[512,512], mLoops=nLoops=4
    TPushPopC2VMaxTrackTestFunc(512, 128, 128, 512, 512, 512, LaunchTPushPopC2VMaxTrack);
}

TEST_F(TPushPopC2VMaxTrackTest, case_maxtrack_512_128_128_128)
{
    // A[512,128] @ B[128,128] -> C[512,128], mLoops=4, nLoops=1
    TPushPopC2VMaxTrackTestFunc(512, 128, 128, 128, 512, 128, LaunchTPushPopC2VMaxTrackMOnly);
}
