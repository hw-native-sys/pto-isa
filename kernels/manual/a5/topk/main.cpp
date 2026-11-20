/*
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

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

// Identical definition of pto::TopKAlgo (host TU cannot include the device-only TopK.hpp).
namespace pto {
enum class TopKAlgo { ITER_SELECT, RADIX_SELECT };
}
using pto::TopKAlgo;

template <typename T, int N, int K>
void launchTopK(T* outVal, int32_t* outIdx, T* scores, int32_t* srcIdx, bool largest, TopKAlgo algo, void* stream);

#ifndef TOPK_GOLDEN_BASE
#define TOPK_GOLDEN_BASE "."
#endif

namespace {

constexpr uint32_t AlignUp(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }

struct ValIdx {
    float v;
    int32_t i;
};

bool CheckOutput(const vector<ValIdx>& golden, const vector<ValIdx>& device, bool largest)
{
    (void)largest;
    // The interface must return the k results already sorted, so compare element-wise directly.
    if (device.size() != golden.size()) {
        ERROR_LOG("output size mismatch: dev %zu vs golden %zu", device.size(), golden.size());
        return false;
    }
    for (size_t j = 0; j < golden.size(); ++j) {
        if (device[j].v != golden[j].v || device[j].i != golden[j].i) {
            ERROR_LOG(
                "top[%zu]: device(v=%a, idx=%d) vs golden(v=%a, idx=%d)", j, device[j].v, device[j].i, golden[j].v,
                golden[j].i);
            return false;
        }
    }
    return true;
}

std::vector<uint8_t> ReadBin(const std::string& path, size_t expectBytes)
{
    std::vector<uint8_t> data(expectBytes);
    size_t fileSize = expectBytes;
    if (!ReadFile(path, fileSize, data.data(), expectBytes)) {
        data.clear();
    }
    return data;
}

// Golden-dir driven case: input.bin + golden_{value,index}.bin (largest=true, fp32).
template <int N, int K>
void test_topk_file(const std::string& goldenDirName, TopKAlgo algo)
{
    constexpr uint32_t KP = AlignUp(K, 8);
    const std::string dir = std::string(TOPK_GOLDEN_BASE) + "/" + goldenDirName;
    const size_t inBytes = static_cast<size_t>(N) * sizeof(float);
    const size_t outVBytes = static_cast<size_t>(KP) * sizeof(float);
    const size_t outIBytes = static_cast<size_t>(KP) * sizeof(int32_t);

    vector<uint8_t> inHost = ReadBin(dir + "/input.bin", inBytes);
    ASSERT_EQ(inHost.size(), inBytes) << "missing input.bin in " << dir;
    vector<uint8_t> goldenV = ReadBin(dir + "/golden_value.bin", outVBytes);
    ASSERT_EQ(goldenV.size(), outVBytes) << "missing golden_value.bin in " << dir;
    vector<int32_t> goldenI(outIBytes / sizeof(int32_t));
    size_t goldenIBytes = outIBytes;
    ASSERT_TRUE(ReadFile(dir + "/golden_index.bin", goldenIBytes, goldenI.data(), outIBytes));

    vector<ValIdx> golden;
    for (int32_t j = 0; j < K; ++j) {
        float v;
        std::memcpy(&v, &goldenV[j * sizeof(float)], sizeof(v));
        golden.push_back({v, goldenI[j]});
    }

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    // srcIdx tile: element positions 0..N-1
    vector<int32_t> srcIdxHost(N);
    for (int32_t j = 0; j < N; ++j) {
        srcIdxHost[j] = j;
    }

    uint8_t* inDev;
    uint8_t* outVDev;
    int32_t* outIDev;
    int32_t* srcIdxDev;
    aclrtMalloc((void**)&inDev, inBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&outVDev, outVBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&outIDev, outIBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcIdxDev, inBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(outVDev, outVBytes, 0, outVBytes);
    aclrtMemset(outIDev, outIBytes, 0, outIBytes);
    aclrtMemcpy(inDev, inBytes, inHost.data(), inBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcIdxDev, inBytes, srcIdxHost.data(), inBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    launchTopK<float, N, K>(
        reinterpret_cast<float*>(outVDev), outIDev, reinterpret_cast<float*>(inDev), srcIdxDev, true, algo, stream);

    aclrtSynchronizeStream(stream);

    vector<uint8_t> outVHost(outVBytes);
    vector<int32_t> outIHost(outIBytes / sizeof(int32_t));
    aclrtMemcpy(outVHost.data(), outVBytes, outVDev, outVBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(outIHost.data(), outIBytes, outIDev, outIBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(dir + "/output_value.bin", outVHost.data(), outVBytes);
    WriteFile(dir + "/output_index.bin", outIHost.data(), outIBytes);

    aclrtFree(inDev);
    aclrtFree(outVDev);
    aclrtFree(outIDev);
    aclrtFree(srcIdxDev);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    vector<ValIdx> device;
    for (int32_t j = 0; j < K; ++j) {
        float v;
        std::memcpy(&v, &outVHost[j * sizeof(float)], sizeof(v));
        device.push_back({v, outIHost[j]});
    }
    EXPECT_TRUE(CheckOutput(golden, device, true));
}

} // namespace

class TopKTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// iter_sort
TEST_F(TopKTest, iter_1x128_k3) { test_topk_file<128, 3>("TopKTest.radix_1x128_k3", TopKAlgo::ITER_SELECT); }

TEST_F(TopKTest, iter_1x256_k8) { test_topk_file<256, 8>("TopKTest.radix_1x256_k8", TopKAlgo::ITER_SELECT); }

TEST_F(TopKTest, iter_1x256_k12) { test_topk_file<256, 12>("TopKTest.radix_1x256_k12", TopKAlgo::ITER_SELECT); }

TEST_F(TopKTest, iter_1x288_k8) { test_topk_file<288, 8>("TopKTest.radix_1x288_k8", TopKAlgo::ITER_SELECT); }

TEST_F(TopKTest, iter_1x384_k6) { test_topk_file<384, 6>("TopKTest.radix_1x384_k6", TopKAlgo::ITER_SELECT); }

TEST_F(TopKTest, iter_1x512_k12) { test_topk_file<512, 12>("TopKTest.radix_1x512_k12", TopKAlgo::ITER_SELECT); }

TEST_F(TopKTest, iter_1x896_k16) { test_topk_file<896, 16>("TopKTest.radix_1x896_k16", TopKAlgo::ITER_SELECT); }

// radix_sort
TEST_F(TopKTest, radix_1x4096_k512) { test_topk_file<4096, 512>("TopKTest.radix_1x4096_k512", TopKAlgo::RADIX_SELECT); }

TEST_F(TopKTest, radix_1x16384_k256)
{
    test_topk_file<16384, 256>("TopKTest.radix_1x16384_k256", TopKAlgo::RADIX_SELECT);
}

TEST_F(TopKTest, radix_1x16384_k2048)
{
    test_topk_file<16384, 2048>("TopKTest.radix_1x16384_k2048", TopKAlgo::RADIX_SELECT);
}
