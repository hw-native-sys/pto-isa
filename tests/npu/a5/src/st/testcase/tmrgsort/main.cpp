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
#include "acl/acl.h"
#include <gtest/gtest.h>
#include <vector>
#include <string>

using namespace std;
using namespace PtoTestCommon;

using DataType = float;

template <
    typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kTCols_src1, int kTCols_src2, int kTCols_src3,
    int TOPK, int LISTNUM, bool EXHAUSTED>
void LanchTMrgsortMulti(DataType* out, DataType* src0, DataType* src1, DataType* src2, DataType* src3, void* stream);

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, uint32_t blockLen>
void LanchTMrgsortSingle(DataType* out, DataType* src, void* stream);

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int topk>
void LanchTMrgsortTopK(DataType* out, DataType* src, void* stream);

class TMRGSORTTest : public testing::Test {
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

template <
    typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kTCols_src1, int kTCols_src2, int kTCols_src3,
    int TOPK, int LISTNUM, bool EXHAUSTED>
void MultiSort(
    size_t outputFileSize, size_t inputFileSize, std::vector<DataType*> hostList, std::vector<DataType*> deviceList,
    void* stream)
{
    DataType *dstHost = nullptr, *tmpHost = nullptr, *src0Host = nullptr, *src1Host = nullptr, *src2Host = nullptr,
             *src3Host = nullptr;
    DataType *dstDevice = nullptr, *tmpDevice = nullptr, *src0Device = nullptr, *src1Device = nullptr,
             *src2Device = nullptr, *src3Device = nullptr;

    aclrtMallocHost((void**)(&dstHost), outputFileSize);
    aclrtMalloc((void**)(&dstDevice), outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMallocHost((void**)(&src0Host), inputFileSize);
    aclrtMallocHost((void**)(&src1Host), inputFileSize);

    aclrtMalloc((void**)(&src0Device), inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)(&src1Device), inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    hostList.push_back(dstHost);
    hostList.push_back(tmpHost);
    hostList.push_back(src0Host);
    hostList.push_back(src1Host);
    deviceList.push_back(dstDevice);
    deviceList.push_back(tmpDevice);
    deviceList.push_back(src0Device);
    deviceList.push_back(src1Device);

    ReadFile(GetGoldenDir() + "/input0.bin", inputFileSize, src0Host, inputFileSize);
    ReadFile(GetGoldenDir() + "/input1.bin", inputFileSize, src1Host, inputFileSize);

    aclrtMemcpy(src0Device, inputFileSize, src0Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, inputFileSize, src1Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    constexpr const int LIST_NUM_3 = 3;
    constexpr const int LIST_NUM_4 = 4;
    if constexpr (LISTNUM >= LIST_NUM_3) {
        aclrtMallocHost((void**)(&src2Host), inputFileSize);
        aclrtMalloc((void**)(&src2Device), inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        hostList.push_back(src2Host);
        deviceList.push_back(src1Device);
        ReadFile(GetGoldenDir() + "/input2.bin", inputFileSize, src2Host, inputFileSize);
        aclrtMemcpy(src2Device, inputFileSize, src2Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    if constexpr (LISTNUM == LIST_NUM_4) {
        aclrtMallocHost((void**)(&src3Host), inputFileSize);
        aclrtMalloc((void**)&src3Device, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        hostList.push_back(src3Host);
        deviceList.push_back(src3Device);
        ReadFile(GetGoldenDir() + "/input3.bin", inputFileSize, src3Host, inputFileSize);
        aclrtMemcpy(src3Device, inputFileSize, src3Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    LanchTMrgsortMulti<
        T, kGRows_, kGCols_, kTRows_, kTCols_, kTCols_src1, kTCols_src2, kTCols_src3, TOPK, LISTNUM, EXHAUSTED>(
        dstDevice, src0Device, src1Device, src2Device, src3Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outputFileSize, dstDevice, outputFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, outputFileSize);
}

void HandleOutputData(std::vector<DataType>& golden, std::vector<DataType>& devFinal)
{
    size_t goldenSize = golden.size();
    size_t i = goldenSize - 1;
    while (i > 0) {
        if (golden[i] == 0.0f) {
            devFinal[i] = 0.0f;
            i -= 1;
        } else {
            return;
        }
    }
}

template <
    typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kTCols_src1, int kTCols_src2, int kTCols_src3,
    int TOPK, int LISTNUM, bool EXHAUSTED>
void TMrgsortMulti()
{
    size_t inputFileSize = kGRows_ * kGCols_ * sizeof(DataType);
    size_t outputFileSize = LISTNUM * kGRows_ * kGCols_ * sizeof(DataType);
    std::vector<DataType*> hostList;
    std::vector<DataType*> deviceList;

    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    MultiSort<T, kGRows_, kGCols_, kTRows_, kTCols_, kTCols_src1, kTCols_src2, kTCols_src3, TOPK, LISTNUM, EXHAUSTED>(
        outputFileSize, inputFileSize, hostList, deviceList, stream);

    for (auto ptr : deviceList) {
        aclrtFree(ptr);
    }

    for (auto ptr : hostList) {
        aclrtFreeHost(ptr);
    }

    aclrtDestroyStream(stream);

    aclrtResetDevice(0);
    aclFinalize();

    std::vector<DataType> golden(outputFileSize / sizeof(DataType));
    std::vector<DataType> devFinal(outputFileSize / sizeof(DataType));
    ReadFile(GetGoldenDir() + "/golden.bin", outputFileSize, golden.data(), outputFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", outputFileSize, devFinal.data(), outputFileSize);

    if constexpr (EXHAUSTED) {
        HandleOutputData(golden, devFinal);
    }
    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, uint32_t blockLen>
void TMrgsortSingle()
{
    size_t inputFileSize = kGRows_ * kGCols_ * sizeof(DataType);
    size_t outputFileSize = kGRows_ * kGCols_ * sizeof(DataType);

    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    DataType *dstHost, *src0Host;
    DataType *dstDevice, *src0Device;

    aclrtMallocHost((void**)(&dstHost), outputFileSize);
    aclrtMallocHost((void**)(&src0Host), inputFileSize);

    aclrtMalloc((void**)&dstDevice, outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input0.bin", inputFileSize, src0Host, inputFileSize);

    aclrtMemcpy(src0Device, inputFileSize, src0Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LanchTMrgsortSingle<T, kGRows_, kGCols_, kTRows_, kTCols_, blockLen>(dstDevice, src0Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outputFileSize, dstDevice, outputFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, outputFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<DataType> golden(outputFileSize / sizeof(DataType));
    std::vector<DataType> devFinal(outputFileSize / sizeof(DataType));
    ReadFile(GetGoldenDir() + "/golden.bin", outputFileSize, golden.data(), outputFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", outputFileSize, devFinal.data(), outputFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int topk>
void TMrgsortTopk()
{
    size_t inputFileSize = kGRows_ * kGCols_ * sizeof(DataType);
    size_t outputFileSize = kGRows_ * kGCols_ * sizeof(DataType);

    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    DataType *dstHost, *src0Host;
    DataType *dstDevice, *src0Device;

    aclrtMallocHost((void**)(&dstHost), outputFileSize);
    aclrtMallocHost((void**)(&src0Host), inputFileSize);

    aclrtMalloc((void**)&dstDevice, outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input0.bin", inputFileSize, src0Host, inputFileSize);

    aclrtMemcpy(src0Device, inputFileSize, src0Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LanchTMrgsortTopK<T, kGRows_, kGCols_, kTRows_, kTCols_, topk>(dstDevice, src0Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outputFileSize, dstDevice, outputFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, outputFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<DataType> golden(outputFileSize / sizeof(DataType));
    std::vector<DataType> devFinal(outputFileSize / sizeof(DataType));
    ReadFile(GetGoldenDir() + "/golden.bin", outputFileSize, golden.data(), outputFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", outputFileSize, devFinal.data(), outputFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TMRGSORTTest, case_multi1) { TMrgsortMulti<float, 1, 128, 1, 128, 128, 128, 128, 512, 4, false>(); }

TEST_F(TMRGSORTTest, case_multi2) { TMrgsortMulti<uint16_t, 1, 128, 1, 128, 128, 128, 128, 512, 4, false>(); }

TEST_F(TMRGSORTTest, case_multi3) { TMrgsortMulti<float, 1, 128, 1, 128, 128, 128, 64, 448, 4, false>(); }

TEST_F(TMRGSORTTest, case_multi4) { TMrgsortMulti<float, 1, 128, 1, 128, 128, 64, 0, 128, 3, false>(); }

TEST_F(TMRGSORTTest, case_exhausted1) { TMrgsortMulti<float, 1, 64, 1, 64, 64, 0, 0, 128, 2, true>(); }

TEST_F(TMRGSORTTest, case_exhausted2) { TMrgsortMulti<uint16_t, 1, 256, 1, 256, 256, 256, 0, 768, 3, true>(); }

TEST_F(TMRGSORTTest, case_single1) { TMrgsortSingle<float, 1, 256, 1, 256, 64>(); }

TEST_F(TMRGSORTTest, case_single2) { TMrgsortSingle<float, 1, 320, 1, 256, 64>(); }

TEST_F(TMRGSORTTest, case_single3) { TMrgsortSingle<float, 1, 512, 1, 512, 64>(); }

TEST_F(TMRGSORTTest, case_single4) { TMrgsortSingle<float, 1, 640, 1, 512, 64>(); }

TEST_F(TMRGSORTTest, case_single5) { TMrgsortSingle<uint16_t, 1, 256, 1, 256, 64>(); }

TEST_F(TMRGSORTTest, case_single6) { TMrgsortSingle<uint16_t, 1, 320, 1, 256, 64>(); }

TEST_F(TMRGSORTTest, case_single7) { TMrgsortSingle<uint16_t, 1, 512, 1, 512, 64>(); }

TEST_F(TMRGSORTTest, case_single8) { TMrgsortSingle<uint16_t, 1, 1024, 1, 1024, 256>(); }

TEST_F(TMRGSORTTest, case_topk1) { TMrgsortTopk<float, 1, 2048, 1, 2048, 1024>(); }

TEST_F(TMRGSORTTest, case_topk2) { TMrgsortTopk<float, 1, 2048, 1, 2048, 2048>(); }

TEST_F(TMRGSORTTest, case_topk3) { TMrgsortTopk<float, 1, 1280, 1, 1280, 512>(); }

TEST_F(TMRGSORTTest, case_topk4) { TMrgsortTopk<uint16_t, 1, 2048, 1, 2048, 1024>(); }

TEST_F(TMRGSORTTest, case_topk5) { TMrgsortTopk<uint16_t, 1, 2048, 1, 2048, 2048>(); }

TEST_F(TMRGSORTTest, case_topk6) { TMrgsortTopk<uint16_t, 1, 1280, 1, 1280, 512>(); }