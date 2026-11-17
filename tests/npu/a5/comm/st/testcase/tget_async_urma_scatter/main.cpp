/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h>
#include "../comm_mpi.h"
#include "tget_async_urma_scatter_kernel.h"

struct DescriptorCase {
    const char* name;
    UrmaDescriptorCase scenario;
    bool get = true;
    uint32_t queues = 1U;
    bool sharedPool = false;
};
class UrmaGatherScatter : public testing::TestWithParam<DescriptorCase> {};
TEST_P(UrmaGatherScatter, Transfer)
{
    SKIP_IF_RANKS_LT(2);
    const auto p = GetParam();
    ASSERT_TRUE(RunUrmaDescriptorCase(p.scenario, p.get, p.queues, p.sharedPool));
}
using C = UrmaDescriptorCase;
INSTANTIATE_TEST_SUITE_P(
    Descriptors, UrmaGatherScatter,
    testing::Values(
        // GET counterparts of existing Gather PUT cases.
        DescriptorCase{"Get_SinglePeer_InlineEvent", C::Single},
        DescriptorCase{"Get_MultiQueue", C::Multiple, true, 4U}, DescriptorCase{"Get_TwelveSegments", C::FullSge},
        DescriptorCase{"Get_SimtSplit13Sge", C::Split13, true, 4U},
        DescriptorCase{"Get_SimtSplit48Sge", C::Split48, true, 4U},
        DescriptorCase{"Get_NoCqePrefix_FinalClose", C::Prefix, true, 4U},
        DescriptorCase{"Get_SubmitModes", C::Deferred}, DescriptorCase{"Get_RingAndCapacityBoundary", C::Ring},
        DescriptorCase{"Get_SharedPool_SIMT", C::Multiple, true, 4U, true},
        DescriptorCase{"Get_SharedPool_SimtSplit48Sge", C::Split48, true, 4U, true},
        DescriptorCase{"Get_SharedPool_NoCqeFinal", C::Prefix, true, 4U, true},
        DescriptorCase{"Get_SharedPool_RingAndCapacityBoundary", C::Ring, true, 1U, true},
        // Missing common contracts use the same harness for both directions.
        DescriptorCase{"Get_EmptyFinal", C::EmptyFinal, true, 4U},
        DescriptorCase{"Put_EmptyFinal", C::EmptyFinal, false, 4U},
        DescriptorCase{"Get_LargePrefix_SmallFinal", C::LargeFinal},
        DescriptorCase{"Put_LargePrefix_SmallFinal", C::LargeFinal, false},
        DescriptorCase{"Get_LargePrefix_EmptyFinal", C::LargeEmptyFinal, true, 4U},
        DescriptorCase{"Put_LargePrefix_EmptyFinal", C::LargeEmptyFinal, false, 4U},
        DescriptorCase{"Get_InvalidDescriptors", C::Invalid},
        DescriptorCase{"Put_InvalidDescriptors", C::Invalid, false}),
    [](const testing::TestParamInfo<DescriptorCase>& info) { return info.param.name; });

TEST(TGetAsyncUrmaScatter, MultiPeer_NoCqeFinal_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE(RunUrmaDescriptorCase(C::Prefix, true, 4U, false, 4));
}
TEST(TGetAsyncUrmaScatter, MultiPeer_EmptyFinal_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE(RunUrmaDescriptorCase(C::EmptyFinal, true, 4U, false, 4));
}
TEST(TGetAsyncUrmaScatter, SharedPool_MultiPeer_SIMT_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE(RunUrmaDescriptorCase(C::Multiple, true, 2U, true, 4));
}
// Three peers rotate over two queues, including a queue closed by a Final NOP.
TEST(TGetAsyncUrmaScatter, SharedPool_MultiPeer_NoCqeFinal_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE(RunUrmaDescriptorCase(C::Prefix, true, 2U, true, 4));
}
TEST(TGetAsyncUrmaScatter, SharedPool_MultiPeer_EmptyFinal_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE(RunUrmaDescriptorCase(C::EmptyFinal, true, 2U, true, 4));
}
int main(int argc, char** argv)
{
    CommMpiInit(&argc, &argv);
    testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    CommMpiFinalize();
    return result;
}
