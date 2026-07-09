/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the \"License\").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN \"AS IS\" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <acl/acl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

void LaunchPagedAttentionHighPerf(
    uint8_t *sync, uint8_t *qGm, uint8_t *kGm, uint8_t *vGm, uint8_t *blockTablesGm, uint8_t *maskGm,
    uint8_t *deqScale1Gm, uint8_t *offset1Gm, uint8_t *deqScale2Gm, uint8_t *offset2Gm, uint8_t *razorOffset,
    uint8_t *scaleGm, uint8_t *logNGm, uint8_t *eyeGm, uint8_t *oGm, uint8_t *sGm, uint8_t *pGm, uint8_t *oTmpGm,
    uint8_t *goGm, uint8_t *oCoreTmpGm, uint8_t *lGm, uint8_t *gmK16, uint8_t *gmV16, uint8_t *tilingParaGm,
    uint32_t blockDim, void *stream);

namespace {
constexpr int kBatch = 1;
constexpr int kNumHeads = 32;
constexpr int kNumKvHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kSeqLen = 128;
constexpr int kBlockSize = 128;
constexpr int kBlockDim = 24;
constexpr size_t kQBytes = kBatch * kNumHeads * kHeadDim * sizeof(uint16_t);
constexpr size_t kKvBytes = (kSeqLen / kBlockSize) * kBlockSize * kNumKvHeads * kHeadDim * sizeof(uint16_t);
constexpr size_t kBlockTableBytes = kBatch * (kSeqLen / kBlockSize) * sizeof(int32_t);
constexpr size_t kOutBytes = kQBytes;
constexpr size_t kSBytes = 6291456;
constexpr size_t kPBytes = 3145728;
constexpr size_t kOTmpBytes = 12582912;
constexpr size_t kGoBytes = 6291456;
constexpr size_t kOCoreTmpBytes = 7471104;
constexpr size_t kLBytes = 58368;
constexpr size_t kK16Bytes = 100663296;
constexpr size_t kV16Bytes = 100663296;
constexpr float kAtol = 2.0e-2f;
constexpr float kRtol = 5.0e-3f;
const int32_t kTiling[] = {
    1, 32, 128, 1, 128, 1, 1035273459, 8, 1, 2, 0, 0, 1, 0, 0, 0, 0, 44, 17, 4, 2, 1, 128,
    128, 1, 128, 0, 0, 1, 128, -1, 1, 32, 0, 0, 0, 0, 0, 128, 128, 128, 128, 0, 0, 1, 128,
    16, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};


uint16_t FloatToHalf(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = bits & 0x7fffff;
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant = (mant | 0x800000) >> (1 - exp);
        return static_cast<uint16_t>(sign | ((mant + 0x1000) >> 13));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | ((mant + 0x1000) >> 13));
}

float HalfToFloat(uint16_t h)
{
    uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x03ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ff;
            bits = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000 | (mant << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

float ValuePattern(int token, int kvHead, int dim)
{
    int raw = (token * 17 + kvHead * 31 + dim * 7) % 257;
    return (static_cast<float>(raw) / 128.0f - 1.0f) * 0.25f;
}

std::vector<uint16_t> MakeValueInput()
{
    std::vector<uint16_t> value(kKvBytes / sizeof(uint16_t));
    for (int token = 0; token < kSeqLen; ++token) {
        for (int kvHead = 0; kvHead < kNumKvHeads; ++kvHead) {
            for (int dim = 0; dim < kHeadDim; ++dim) {
                int idx = (token * kNumKvHeads + kvHead) * kHeadDim + dim;
                value[idx] = FloatToHalf(ValuePattern(token, kvHead, dim));
            }
        }
    }
    return value;
}

std::vector<uint16_t> MakeUniformAttentionGolden(const std::vector<uint16_t> &value)
{
    std::vector<uint16_t> golden(kOutBytes / sizeof(uint16_t));
    constexpr int headsPerKv = kNumHeads / kNumKvHeads;
    for (int head = 0; head < kNumHeads; ++head) {
        int kvHead = head / headsPerKv;
        for (int dim = 0; dim < kHeadDim; ++dim) {
            float sum = 0.0f;
            for (int token = 0; token < kSeqLen; ++token) {
                int idx = (token * kNumKvHeads + kvHead) * kHeadDim + dim;
                sum += HalfToFloat(value[idx]);
            }
            golden[head * kHeadDim + dim] = FloatToHalf(sum / static_cast<float>(kSeqLen));
        }
    }
    return golden;
}

bool CompareOutput(const std::vector<uint16_t> &actual, const std::vector<uint16_t> &golden)
{
    int mismatches = 0;
    float maxDiff = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        float a = HalfToFloat(actual[i]);
        float e = HalfToFloat(golden[i]);
        float diff = std::fabs(a - e);
        maxDiff = std::max(maxDiff, diff);
        if (diff > kAtol + kRtol * std::fabs(e)) {
            if (mismatches < 8) {
                std::cerr << "mismatch idx=" << i << " actual=" << a << " expected=" << e << " diff=" << diff
                          << std::endl;
            }
            ++mismatches;
        }
    }
    std::cout << "max_diff=" << maxDiff << " mismatches=" << mismatches << std::endl;
    return mismatches == 0;
}

void Check(aclError ret, const char *what)
{
    if (ret != ACL_SUCCESS) {
        std::cerr << what << " failed, ret=" << ret << std::endl;
        std::abort();
    }
}

void *MallocDevice(size_t bytes)
{
    void *ptr = nullptr;
    Check(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    Check(aclrtMemset(ptr, bytes, 0, bytes), "aclrtMemset");
    return ptr;
}
} // namespace

int main(int argc, char **argv)
{
    int deviceId = 0;
    if (argc > 1) {
        deviceId = std::atoi(argv[1]);
    }
    Check(aclInit(nullptr), "aclInit");
    Check(aclrtSetDevice(deviceId), "aclrtSetDevice");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");

    std::vector<uint16_t> valueHost = MakeValueInput();
    std::vector<uint16_t> goldenHost = MakeUniformAttentionGolden(valueHost);

    void *q = MallocDevice(kQBytes);
    void *k = MallocDevice(kKvBytes);
    void *v = MallocDevice(kKvBytes);
    void *blockTable = MallocDevice(kBlockTableBytes);
    void *out = MallocDevice(kOutBytes);
    void *s = MallocDevice(kSBytes);
    void *p = MallocDevice(kPBytes);
    void *oTmp = MallocDevice(kOTmpBytes);
    void *go = MallocDevice(kGoBytes);
    void *oCoreTmp = MallocDevice(kOCoreTmpBytes);
    void *l = MallocDevice(kLBytes);
    void *k16 = MallocDevice(kK16Bytes);
    void *v16 = MallocDevice(kV16Bytes);
    void *tiling = MallocDevice(sizeof(kTiling));
    void *nullTensor = MallocDevice(1);
    int32_t blockId = 0;
    Check(aclrtMemcpy(v, kKvBytes, valueHost.data(), kKvBytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy value");
    Check(aclrtMemcpy(blockTable, sizeof(blockId), &blockId, sizeof(blockId), ACL_MEMCPY_HOST_TO_DEVICE), "copy block table");
    Check(aclrtMemcpy(tiling, sizeof(kTiling), kTiling, sizeof(kTiling), ACL_MEMCPY_HOST_TO_DEVICE), "copy tiling");

    LaunchPagedAttentionHighPerf(
        nullptr, reinterpret_cast<uint8_t *>(q), reinterpret_cast<uint8_t *>(k), reinterpret_cast<uint8_t *>(v),
        reinterpret_cast<uint8_t *>(blockTable), reinterpret_cast<uint8_t *>(nullTensor),
        reinterpret_cast<uint8_t *>(nullTensor), reinterpret_cast<uint8_t *>(nullTensor),
        reinterpret_cast<uint8_t *>(nullTensor), reinterpret_cast<uint8_t *>(nullTensor),
        reinterpret_cast<uint8_t *>(nullTensor), reinterpret_cast<uint8_t *>(nullTensor),
        reinterpret_cast<uint8_t *>(nullTensor), reinterpret_cast<uint8_t *>(nullTensor), reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<uint8_t *>(s), reinterpret_cast<uint8_t *>(p), reinterpret_cast<uint8_t *>(oTmp),
        reinterpret_cast<uint8_t *>(go), reinterpret_cast<uint8_t *>(oCoreTmp), reinterpret_cast<uint8_t *>(l),
        reinterpret_cast<uint8_t *>(k16), reinterpret_cast<uint8_t *>(v16), reinterpret_cast<uint8_t *>(tiling), kBlockDim,
        stream);
    aclError runRet = aclrtSynchronizeStream(stream);
    std::cout << "paged_attention_highperf run ret=" << runRet << std::endl;

    std::vector<uint16_t> actualHost(goldenHost.size());
    bool compareOk = false;
    if (runRet == ACL_SUCCESS) {
        Check(aclrtMemcpy(actualHost.data(), kOutBytes, out, kOutBytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy output");
        compareOk = CompareOutput(actualHost, goldenHost);
    }

    aclrtFree(q); aclrtFree(k); aclrtFree(v); aclrtFree(blockTable); aclrtFree(out); aclrtFree(s); aclrtFree(p);
    aclrtFree(oTmp); aclrtFree(go); aclrtFree(oCoreTmp); aclrtFree(l); aclrtFree(k16); aclrtFree(v16);
    aclrtFree(tiling); aclrtFree(nullTensor);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    std::cout << ((runRet == ACL_SUCCESS && compareOk) ? "test success" : "test failed") << std::endl;
    return (runRet == ACL_SUCCESS && compareOk) ? 0 : 1;
}
