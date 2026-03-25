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

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kTCols_src1, int kTCols_src2,
          int kTCols_src3, int TOPK, int LISTNUM, bool EXHAUSTED, float profiling, float accuracy>
void LanchTMrgsortMulti(void *stream);

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, uint32_t blockLen, float profiling,
          float accuracy>
void LanchTMrgsortSingle(void *stream);

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int topk, float profiling, float accuracy>
void LanchTMrgsortTopK(void *stream);

class TMRGSORTTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kTCols_src1, int kTCols_src2,
          int kTCols_src3, int TOPK, int LISTNUM, bool EXHAUSTED, float profiling, float accuracy>
void TMrgsortMulti()
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    LanchTMrgsortMulti<T, kGRows_, kGCols_, kTRows_, kTCols_, kTCols_src1, kTCols_src2, kTCols_src3, TOPK, LISTNUM,
                       EXHAUSTED, profiling, accuracy>(stream);
    aclrtSynchronizeStream(stream);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, uint32_t blockLen, float profiling,
          float accuracy>
void TMrgsortSingle()
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    LanchTMrgsortSingle<T, kGRows_, kGCols_, kTRows_, kTCols_, blockLen, profiling, accuracy>(stream);
    aclrtSynchronizeStream(stream);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int topk, float profiling, float accuracy>
void TMrgsortTopk()
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    LanchTMrgsortTopK<T, kGRows_, kGCols_, kTRows_, kTCols_, topk, profiling, accuracy>(stream);
    aclrtSynchronizeStream(stream);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

// multi case: costmodel=20
TEST_F(TMRGSORTTest, case_multi1)
{
    TMrgsortMulti<float, 1, 128, 1, 128, 128, 128, 128, 512, 4, false, 20.0f, 1.0f>();
}

TEST_F(TMRGSORTTest, case_multi2)
{
    TMrgsortMulti<uint16_t, 1, 128, 1, 128, 128, 128, 128, 512, 4, false, 20.0f, 1.0f>();
}

TEST_F(TMRGSORTTest, case_exhausted1)
{
    TMrgsortMulti<float, 1, 64, 1, 64, 64, 0, 0, 128, 2, true, 20.0f, 1.0f>();
}

TEST_F(TMRGSORTTest, case_exhausted2)
{
    TMrgsortMulti<uint16_t, 1, 256, 1, 256, 256, 256, 0, 768, 3, true, 20.0f, 1.0f>();
}

// single case: costmodel output
TEST_F(TMRGSORTTest, case_single1)
{
    TMrgsortSingle<float, 1, 256, 1, 256, 64, 20.0f, 1.0f>();
}

TEST_F(TMRGSORTTest, case_single3)
{
    TMrgsortSingle<float, 1, 512, 1, 512, 64, 26.0f, 1.0f>();
}

TEST_F(TMRGSORTTest, case_single5)
{
    TMrgsortSingle<uint16_t, 1, 256, 1, 256, 64, 20.0f, 1.0f>();
}

TEST_F(TMRGSORTTest, case_single7)
{
    TMrgsortSingle<uint16_t, 1, 512, 1, 512, 64, 26.0f, 1.0f>();
}

TEST_F(TMRGSORTTest, case_single8)
{
    TMrgsortSingle<uint16_t, 1, 1024, 1, 1024, 256, 20.0f, 1.0f>();
}

// topk case: final TMRGSORT on dstTile costmodel=20
TEST_F(TMRGSORTTest, case_topk2)
{
    TMrgsortTopk<float, 1, 2048, 1, 2048, 2048, 20.0f, 1.0f>();
}

TEST_F(TMRGSORTTest, case_topk5)
{
    TMrgsortTopk<uint16_t, 1, 2048, 1, 2048, 2048, 20.0f, 1.0f>();
}
