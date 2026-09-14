/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include <utility>

#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>

using namespace std::chrono_literals;
using namespace pto::cpu_sim;

namespace {
using Storage = std::unique_ptr<void, decltype(&std::free)>;
std::map<std::string, Storage> registry;
std::mutex registryMutex;
int stringHookCalls = 0;
bool returnNull = false;
ffts::EventStorage injectedStorage{};

void* sharedStorage(std::string key, size_t size)
{
    std::lock_guard lock(registryMutex);
    ++stringHookCalls;
    if (returnNull)
        return nullptr;
    auto it = registry.find(key);
    if (it == registry.end()) {
        it = registry.emplace(key, Storage(std::calloc(1, size), &std::free)).first;
    }
    return it->second.get();
}
void* injected(uint64_t, size_t) { return &injectedStorage; }
void* nullStorage(uint64_t, size_t) { return nullptr; }

void finish(std::future<void>& result)
{
    if (result.wait_for(5s) != std::future_status::ready)
        std::abort();
    result.get();
}

class FFTSHooksTest : public testing::Test {
protected:
    void SetUp() override
    {
        registry.clear();
        injectedStorage = {};
        stringHookCalls = 0;
        returnNull = false;
        register_hooks(nullptr, nullptr);
        reset_execution_context();
    }
    void TearDown() override
    {
        register_hooks(nullptr, nullptr);
        reset_execution_context();
        registry.clear();
    }
    static void signal() { ffts::signal<true>(pto::getFFTSMsg(2, 0)); }
    static void wait() { ffts::wait<false>(0); }
    static void checkIsolation(uint64_t cookie, uint32_t block)
    {
        signal();
        set_task_cookie(cookie);
        set_execution_context(block, 0, 2);
        std::promise<void> started;
        auto startedFuture = started.get_future();
        auto pending = std::async(std::launch::async, [cookie, block, &started] {
            set_task_cookie(cookie);
            set_execution_context(block, 0, 2);
            started.set_value();
            wait();
        });
        finish(startedFuture);
        EXPECT_EQ(pending.wait_for(20ms), std::future_status::timeout);
        signal();
        finish(pending);
    }
};

TEST_F(FFTSHooksTest, StringProviderSharesStateBetweenWorkers)
{
    int value = 0;
    auto pending = std::async(std::launch::async, [&] {
        wait();
        EXPECT_EQ(value, 123);
    });
    value = 123;
    signal();
    finish(pending);
    EXPECT_EQ(registry.size(), 1u);
}

TEST_F(FFTSHooksTest, StringProviderIsolatesTaskCookies) { checkIsolation(1, 0); }
TEST_F(FFTSHooksTest, StringProviderIsolatesBlocks) { checkIsolation(0, 1); }

TEST_F(FFTSHooksTest, InjectedProviderTakesPrecedence)
{
    register_hooks(nullptr, reinterpret_cast<void*>(injected));
    signal();
    wait();
    EXPECT_EQ(stringHookCalls, 0);
}

TEST_F(FFTSHooksTest, NullInjectedProviderDoesNotFallBack)
{
    register_hooks(nullptr, reinterpret_cast<void*>(nullStorage));
    EXPECT_THROW(signal(), std::runtime_error);
    EXPECT_EQ(stringHookCalls, 0);
}

TEST_F(FFTSHooksTest, NullStringProviderIsRejected)
{
    returnNull = true;
    EXPECT_THROW(signal(), std::runtime_error);
}

TEST_F(FFTSHooksTest, AmbiguousKernelRoleIsRejected)
{
    EXPECT_THROW(__builtin_cce_ffts_cross_core_sync(PIPE_FIX, pto::getFFTSMsg(2, 0)), std::runtime_error);
    EXPECT_THROW(__builtin_cce_wait_flag_dev(0), std::runtime_error);
}

TEST_F(FFTSHooksTest, OverflowDoesNotPartiallyBroadcast)
{
    auto& state = ffts::getEventState(0);
    for (int fullLane = 0; fullLane < 2; ++fullLane) {
        state.cubeToVector[fullLane].store(UINT32_MAX);
        state.cubeToVector[1 - fullLane].store(7);
        EXPECT_THROW(signal(), std::overflow_error);
        EXPECT_EQ(state.cubeToVector[fullLane].load(), UINT32_MAX);
        EXPECT_EQ(state.cubeToVector[1 - fullLane].load(), 7u);
    }
}

TEST_F(FFTSHooksTest, VectorOverflowDoesNotWrap)
{
    auto& state = ffts::getEventState(0);
    state.vectorToCube[0].store(UINT32_MAX);
    EXPECT_THROW(ffts::signal<false>(pto::getFFTSMsg(2, 0)), std::overflow_error);
    EXPECT_EQ(state.vectorToCube[0].load(), UINT32_MAX);
    EXPECT_EQ(state.vectorToCube[1].load(), 0u);
}
} // namespace

// This executable deliberately has no numeric-key provider, so dlsym selects the fallback.
extern "C" __attribute__((visibility("default"))) void* pto_cpu_sim_get_shared_storage(std::string key, size_t size)
{
    return sharedStorage(std::move(key), size);
}
