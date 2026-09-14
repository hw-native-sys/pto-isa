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
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>

#include <utility>

#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>

using namespace std::chrono_literals;

namespace {
thread_local uint32_t laneId = 0;
thread_local uint32_t deviceId = 0;
thread_local uint32_t groupId = 0;
using Storage = std::unique_ptr<void, decltype(&std::free)>;
std::map<std::tuple<uint32_t, uint32_t, uint64_t>, Storage> registry;
std::mutex registryMutex;

void* sharedStorage(uint64_t key, size_t size)
{
    std::lock_guard lock(registryMutex);
    auto id = std::make_tuple(deviceId, groupId, key);
    auto it = registry.find(id);
    if (it == registry.end()) {
        it = registry.emplace(id, Storage(std::calloc(1, size), &std::free)).first;
    }
    return it->second.get();
}
uint32_t subblock() { return laneId; }
void* missingStorage(uint64_t, size_t) { return nullptr; }

struct Kernel {
    void* handle;
    void (*signal)(uint16_t);
    void (*wait)(int);
    void (*base)(uint64_t);
    void (*hooks)(void*, void*);
    int (*pipeline)(int*, int*, int);

    explicit Kernel(const char* path) : handle(dlopen(path, RTLD_NOW | RTLD_LOCAL))
    {
        if (!handle) {
            std::fprintf(stderr, "%s\n", dlerror());
            std::abort();
        }
        signal = symbol<decltype(signal)>("signalEvent");
        wait = symbol<decltype(wait)>("waitEvent");
        base = symbol<decltype(base)>("setBase");
        hooks = symbol<decltype(hooks)>("registerTestHooks");
        pipeline = symbol<decltype(pipeline)>("pipeline");
    }
    template <class Fn>
    Fn symbol(const char* name)
    {
        auto fn = reinterpret_cast<Fn>(dlsym(handle, name));
        if (!fn) {
            std::fprintf(stderr, "Missing kernel symbol %s: %s\n", name, dlerror());
            std::abort();
        }
        return fn;
    }
    ~Kernel() { dlclose(handle); }
    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;
};

// A missing notification must fail the test instead of hanging future destruction.
template <class T>
T finish(std::future<T>& future)
{
    if (future.wait_for(5s) != std::future_status::ready) {
        std::fprintf(stderr, "FFTS worker did not finish within 5 seconds\n");
        std::abort();
    }
    return future.get();
}

class FFTSTest : public testing::TestWithParam<bool> {
protected:
    Kernel cube{FFTS_AIC_PATH};
    Kernel vec0{FFTS_AIV_PATH};
    Kernel vec1{FFTS_AIV_SECOND_PATH};

    void SetUp() override
    {
        registry.clear();
        laneId = deviceId = groupId = 0;
        for (auto* kernel : {&cube, &vec0, &vec1}) {
            kernel->hooks(
                GetParam() ? reinterpret_cast<void*>(subblock) : nullptr,
                GetParam() ? reinterpret_cast<void*>(sharedStorage) : nullptr);
        }
    }
    void TearDown() override { registry.clear(); }

    static uint16_t message(int event) { return pto::getFFTSMsg(FFTS_MODE_VAL, event); }
    std::future<void> waitAsync(Kernel& kernel, int event, uint32_t lane = 0)
    {
        const auto device = deviceId;
        const auto group = groupId;
        std::promise<void> started;
        auto startedFuture = started.get_future();
        auto pending = std::async(
            std::launch::async, [&kernel, event, lane, device, group, started = std::move(started)]() mutable {
                laneId = lane;
                deviceId = device;
                groupId = group;
                started.set_value();
                kernel.wait(event);
            });
        finish(startedFuture);
        return pending;
    }
};

TEST_P(FFTSTest, BroadcastRetainsRepeatedNotifications)
{
    for (int i = 0; i < 3; ++i) {
        cube.signal(message(15));
    }
    // Raw PTOAS initializes the hardware base on each core; it must not reset credits.
    vec0.base(0x1234);
    vec1.base(0);
    auto a = std::async(std::launch::async, [&] {
        laneId = 0;
        for (int i = 0; i < 3; ++i)
            vec0.wait(15);
    });
    auto b = std::async(std::launch::async, [&] {
        laneId = 1;
        for (int i = 0; i < 3; ++i)
            vec1.wait(15);
    });
    finish(a);
    finish(b);
    auto next = waitAsync(vec0, 15);
    EXPECT_EQ(next.wait_for(20ms), std::future_status::timeout);
    cube.signal(message(15));
    finish(next);
}

TEST_P(FFTSTest, JoinConsumesOneCreditFromEachLane)
{
    laneId = 0;
    vec0.signal(message(0));
    vec0.signal(message(0));
    auto first = waitAsync(cube, 0);
    EXPECT_EQ(first.wait_for(20ms), std::future_status::timeout);
    laneId = 1;
    vec1.signal(message(0));
    finish(first);
    auto second = waitAsync(cube, 0);
    EXPECT_EQ(second.wait_for(20ms), std::future_status::timeout);
    vec1.signal(message(0));
    finish(second);
}

TEST_P(FFTSTest, EventDeviceAndGroupIsolation)
{
    cube.signal(message(1));
    deviceId = 1;
    cube.signal(message(0));
    deviceId = 0;
    groupId = 1;
    cube.signal(message(0));
    groupId = 0;
    auto pending = waitAsync(vec0, 0);
    EXPECT_EQ(pending.wait_for(20ms), std::future_status::timeout);
    cube.signal(message(0));
    finish(pending);
}

TEST_P(FFTSTest, RuntimeResetDiscardsCredits)
{
    cube.signal(message(0));
    // A run boundary: no active workers retain pointers into this registry.
    registry.clear();
    auto pending = waitAsync(vec0, 0);
    EXPECT_EQ(pending.wait_for(20ms), std::future_status::timeout);
    cube.signal(message(0));
    finish(pending);
}

TEST_P(FFTSTest, CrossLibraryPipelinePreservesDataVisibility)
{
    int data[3]{};
    int results[6]{};
    auto producer = std::async(std::launch::async, [&] { return cube.pipeline(data, results, 3000); });
    auto consumer0 = std::async(std::launch::async, [&] {
        laneId = 0;
        return vec0.pipeline(data, results, 3000);
    });
    auto consumer1 = std::async(std::launch::async, [&] {
        laneId = 1;
        return vec1.pipeline(data, results, 3000);
    });
    EXPECT_EQ(finish(producer), 0);
    EXPECT_EQ(finish(consumer0), 0);
    EXPECT_EQ(finish(consumer1), 0);
}

TEST_P(FFTSTest, RejectsUnsupportedProtocolAndMissingContext)
{
    EXPECT_THROW(pto::getFFTSMsg(1, 0), std::invalid_argument);
    EXPECT_THROW(pto::getFFTSMsg(2, 16), std::invalid_argument);
    EXPECT_THROW(pto::getFFTSMsg(2, 0, 2), std::invalid_argument);
    for (uint16_t msg : {0x11, 0x22, 0x1021}) {
        EXPECT_THROW(cube.signal(msg), std::invalid_argument);
    }
    EXPECT_THROW(cube.wait(-1), std::invalid_argument);
    EXPECT_THROW(cube.wait(16), std::invalid_argument);
    laneId = 2;
    EXPECT_THROW(vec0.signal(message(0)), std::invalid_argument);
    EXPECT_THROW(vec0.wait(0), std::invalid_argument);
    cube.hooks(reinterpret_cast<void*>(subblock), reinterpret_cast<void*>(missingStorage));
    EXPECT_THROW(cube.signal(message(0)), std::runtime_error);
}

INSTANTIATE_TEST_SUITE_P(InjectedOrResolvedHooks, FFTSTest, testing::Bool());
} // namespace

// Simulate the existing runtime ABI, discoverable by independently loaded kernels.
extern "C" __attribute__((visibility("default"))) uint32_t pto_sim_get_subblock_id() { return subblock(); }
extern "C" __attribute__((visibility("default"))) void* pto_sim_get_pipe_shared_state(uint64_t key, size_t size)
{
    return sharedStorage(key, size);
}
