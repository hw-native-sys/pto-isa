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
#include <cstdlib>

#include "tput_async_urma_gather_kernel.h"
#include "../comm_mpi.h"

// ============================================================================
// TPUT_ASYNC source-side gather via URMA.
//
// The _4Ranks suffix is what run_st.py matches when it selects the 4-rank
// round; without it these never leave the 2-rank round, where they skip.
// ============================================================================
TEST(TPutAsyncUrmaGather, Gather_MultiPeer_TwoSegments_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE(RunPutAsyncUrmaGather(4, 4, 0, 0));
}

TEST(TPutAsyncUrmaGather, Gather_MultiPeer_MixedSegments_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE(RunPutAsyncUrmaGatherMixed(4, 4, 0, 0));
}

// Two ranks means one destination, which keeps the submission on a single
// queue: the only shape that produces an inline event instead of a slot.
TEST(TPutAsyncUrmaGather, Gather_SinglePeer_InlineEvent)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunPutAsyncUrmaGatherSinglePeer(2, 2, 0, 0));
}

TEST(TPutAsyncUrmaGather, Gather_FullSge_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE(RunPutAsyncUrmaGatherFullSge(4, 4, 0, 0));
}

TEST(TPutAsyncUrmaGather, Gather_SimtSplit13Sge_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE(RunPutAsyncUrmaGatherSimtSplit(4, 4, 0, 0, 13U, false));
}

TEST(TPutAsyncUrmaGather, Gather_SimtSplit48Sge_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE(RunPutAsyncUrmaGatherSimtSplit(4, 4, 0, 0, 48U, false));
}

TEST(TPutAsyncUrmaGather, Gather_MultiQueue_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE(RunPutAsyncUrmaGatherMultiQueue(4, 4, 0, 0));
}

TEST(TPutAsyncUrmaGather, Gather_LargeBatch_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE(RunPutAsyncUrmaGatherLargeBatch(4, 4, 0, 0));
}

TEST(TPutAsyncUrmaGather, Gather_SubmitModes)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunPutAsyncUrmaGatherSubmitModes(2, 2, 0, 0, false));
}

TEST(TPutAsyncUrmaGather, Gather_RingAndCapacityBoundary)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunPutAsyncUrmaGatherRingBoundary(2, 2, 0, 0, false));
}

TEST(TPutAsyncUrmaGather, Gather_CompletionSlotGenerationReuse)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunPutAsyncUrmaGatherSlotReuse(2, 2, 0, 0, false));
}

TEST(TPutAsyncUrmaGather, Gather_MultiAivConcurrent)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunPutAsyncUrmaGatherMultiAiv(2, 2, 0, 0, false));
}

TEST(TPutAsyncUrmaGather, Gather_ByteBoundary256MiB)
{
    if (std::getenv("PTO_URMA_GATHER_RUN_LARGE_BOUNDARY") == nullptr) {
        GTEST_SKIP() << "Set PTO_URMA_GATHER_RUN_LARGE_BOUNDARY=1 to run the 512 MiB MR test.";
    }
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunPutAsyncUrmaGatherByteBoundary(2, 2, 0, 0));
}

// A single remote peer is enough to exercise SHARED_POOL stage indexing and
// per-SQE target metadata on machines where cross-layer pool links are absent.
TEST(TPutAsyncUrmaGather, Gather_SharedPool_SinglePeer_ScalarInline)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunPutAsyncUrmaGatherSharedPool(2, 2, 0, 0, 1U, 1U, 1U));
}

// K=2 is the inclusive scalar boundary; both SQEs share one inline-event queue.
TEST(TPutAsyncUrmaGather, Gather_SharedPool_TwoSqe_ScalarBoundary)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunPutAsyncUrmaGatherSharedPool(2, 2, 0, 0, 2U, 1U, 1U));
}

TEST(TPutAsyncUrmaGather, Gather_SharedPool_SinglePeer_SIMT)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunPutAsyncUrmaGatherSharedPool(2, 2, 0, 0, 16U, 1U, 1U));
}

// One destination still routes to SIMT because 48 SGEs produce four SQEs.
TEST(TPutAsyncUrmaGather, Gather_SharedPool_SinglePeer_SimtSplit48Sge)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunPutAsyncUrmaGatherSharedPool(2, 2, 0, 0, 1U, 48U, 1U));
}

// Multi-peer SHARED_POOL requires a layer-1 topology, just like the Pool_*
// coverage in tput_async_urma. Selecting SHARED_POOL in Setup is the complete
// test contract; no extra environment switch is required.
TEST(TPutAsyncUrmaGather, Gather_SharedPool_MixedWidths_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE(RunPutAsyncUrmaGatherSharedPoolMixed(4, 4, 0, 0));
}

TEST(TPutAsyncUrmaGather, Gather_SharedPool_SixSge_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE(RunPutAsyncUrmaGatherSharedPool(4, 4, 0, 0, 2U, 6U, 1U));
}

TEST(TPutAsyncUrmaGather, Gather_SharedPool_FullSge_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE(RunPutAsyncUrmaGatherSharedPool(4, 4, 0, 0, 1U, 12U, 1U));
}

TEST(TPutAsyncUrmaGather, Gather_SharedPool_SimtSplit13Sge_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE(RunPutAsyncUrmaGatherSimtSplit(4, 4, 0, 0, 13U, true));
}

TEST(TPutAsyncUrmaGather, Gather_SharedPool_SimtSplit48Sge_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE(RunPutAsyncUrmaGatherSimtSplit(4, 4, 0, 0, 48U, true));
}

TEST(TPutAsyncUrmaGather, Gather_SharedPool_LargeBatchMultiQueue_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE(RunPutAsyncUrmaGatherSharedPool(4, 4, 0, 0, 16U, 1U, 4U));
}

// Three 3-SQE rounds rotate over four queues. The Final round carries payload
// on three queues and must close the fourth with a CQE-producing NOP, covering
// shared-stage lastPeer recovery as well as NoCqe/Final state transitions.
TEST(TPutAsyncUrmaGather, Gather_SharedPool_NoCqeFinalMultiQueue_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE(RunPutAsyncUrmaGatherSharedPoolNoCqeFinal(4, 4, 0, 0));
}

TEST(TPutAsyncUrmaGather, Gather_SharedPool_SubmitModes_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE(RunPutAsyncUrmaGatherSubmitModes(4, 4, 0, 0, true));
}

TEST(TPutAsyncUrmaGather, Gather_SharedPool_RingAndCapacityBoundary)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunPutAsyncUrmaGatherRingBoundary(2, 2, 0, 0, true));
}

TEST(TPutAsyncUrmaGather, Gather_SharedPool_CompletionSlotGenerationReuse)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunPutAsyncUrmaGatherSlotReuse(2, 2, 0, 0, true));
}

TEST(TPutAsyncUrmaGather, Gather_SharedPool_MultiAivConcurrent)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunPutAsyncUrmaGatherMultiAiv(2, 2, 0, 0, true));
}

TEST(TPutAsyncUrmaGather, Gather_NoCqePrefix_FinalClose_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE(RunPutAsyncUrmaGatherNoCqeFinal(4, 4, 0, 0));
}

// Same three rounds, but every round takes its own completion: isolates
// multi-round queue reuse from the NoCqe policy itself.
TEST(TPutAsyncUrmaGather, Gather_MultiRound_OneCqe_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE(RunPutAsyncUrmaGatherMultiRound(4, 4, 0, 0));
}

TEST(TPutAsyncUrmaGather, Gather_MultiRound_SingleQueue)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunPutAsyncUrmaGatherMultiRoundSingleQueue(2, 2, 0, 0));
}

TEST(TPutAsyncUrmaGather, Gather_PerfK_4Ranks)
{
    if (std::getenv("PTO_URMA_GATHER_RUN_PERF") == nullptr) {
        GTEST_SKIP() << "Set PTO_URMA_GATHER_RUN_PERF=1 to run the crossover benchmark.";
    }
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE(RunPutAsyncUrmaGatherPerf(4, 4, 0, 0));
}

TEST(TPutAsyncUrmaGather, Gather_SplitPerf_4Ranks)
{
    if (std::getenv("PTO_URMA_GATHER_RUN_SPLIT_PERF") == nullptr) {
        GTEST_SKIP() << "Set PTO_URMA_GATHER_RUN_SPLIT_PERF=1 to run the split benchmark.";
    }
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE(RunPutAsyncUrmaGatherSplitPerf(4, 4, 0, 0));
}

int main(int argc, char** argv)
{
    CommMpiInit(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int ret = RUN_ALL_TESTS();
    CommMpiFinalize();
    return ret;
}
