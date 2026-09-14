/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_FFTS_HPP
#define PTO_CPU_FFTS_HPP

#include <atomic>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <thread>
#include <type_traits>

#include <pto/common/cpu_stub.hpp>

#define FFTS_BASE_COUNT_WIDTH 0xf
#define FFTS_MODE_VAL 0x2
#define FFTS_MODE_WIDTH 0x3
#define FFTS_MODE_OFFSET 4
#define FFTS_EVENT_ID_WIDTH 0xf
#define FFTS_EVENT_ID_OFFSET 8

namespace pto {
inline uint16_t getFFTSMsg(uint16_t mode, uint16_t eventId, uint16_t baseCount = 1)
{
    if (mode != FFTS_MODE_VAL || baseCount != 1 || eventId > FFTS_EVENT_ID_WIDTH) {
        throw std::invalid_argument("CPU FFTS requires mode 2, base count 1 and event ID in [0, 15]");
    }
    return baseCount | (mode << FFTS_MODE_OFFSET) | (eventId << FFTS_EVENT_ID_OFFSET);
}

namespace cpu_sim::ffts {
struct EventState {
    std::atomic<uint32_t> cubeToVector[2]{};
    std::atomic<uint32_t> vectorToCube[2]{};
};
static_assert(std::is_trivially_destructible_v<EventState>);

struct EventStorage {
    alignas(std::atomic_ref<uint32_t>::required_alignment) uint32_t initialized;
    alignas(EventState) unsigned char payload[sizeof(EventState)];
};

inline EventState& getEventState(int eventId)
{
    if (eventId < 0 || eventId > FFTS_EVENT_ID_WIDTH) {
        throw std::invalid_argument("CPU FFTS requires an event ID in [0, 15]");
    }
    // This key is outside the TPipe flag/direction namespace. The runtime owns
    // device/group isolation and clears storage between runs, with all workers stopped.
    constexpr uint64_t KEY_PREFIX = 0xff46465453000000ULL;
    auto pipeHook = injected_pipe_shared_state_hook;
    if (pipeHook == nullptr) {
        pipeHook = ResolvePipeSharedStateHook();
    }
    void* raw = nullptr;
    if (pipeHook != nullptr) {
        raw = pipeHook(KEY_PREFIX | static_cast<uint64_t>(eventId), sizeof(EventStorage));
    } else if (auto hook = ResolveSharedStorageHook(); hook != nullptr) {
        const auto key = "pto-ffts-" + std::to_string(get_task_cookie()) + "-" + std::to_string(get_block_idx()) + "-" +
                         std::to_string(eventId);
        raw = hook(key, sizeof(EventStorage));
    }
    if (raw == nullptr) {
        throw std::runtime_error("CPU FFTS requires runtime-owned shared storage; register hooks before execution");
    }
    auto& storage = *static_cast<EventStorage*>(raw);
    std::atomic_ref<uint32_t> initialized(storage.initialized);
    uint32_t expected = 0;
    if (initialized.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
        new (storage.payload) EventState{};
        initialized.store(2, std::memory_order_release);
    } else {
        while (initialized.load(std::memory_order_acquire) != 2) {
            std::this_thread::yield();
        }
    }
    return *std::launder(reinterpret_cast<EventState*>(storage.payload));
}

inline uint32_t vectorLane()
{
    const auto lane = get_subblockid();
    if (lane >= 2) {
        throw std::invalid_argument("CPU FFTS mode 2 requires AIV subblock 0 or 1");
    }
    return lane;
}

inline void publish(std::atomic<uint32_t>& credits)
{
    auto value = credits.load(std::memory_order_relaxed);
    do {
        if (value == std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("CPU FFTS credit counter overflow");
        }
    } while (!credits.compare_exchange_weak(value, value + 1, std::memory_order_release, std::memory_order_relaxed));
}

inline void consume(std::atomic<uint32_t>& credits)
{
    auto value = credits.load(std::memory_order_acquire);
    for (;;) {
        if (value == 0) {
            std::this_thread::yield();
            value = credits.load(std::memory_order_acquire);
        } else if (credits.compare_exchange_weak(value, value - 1, std::memory_order_acquire)) {
            return;
        }
    }
}

// Role is a template argument so AIC/AIV code cannot interpose the opposite
// implementation when kernels are linked together or loaded from separate DSOs.
template <bool IsCube>
inline void signal(uint16_t message)
{
    constexpr uint16_t MESSAGE_MASK = 0x0f3f;
    if ((message & ~MESSAGE_MASK) != 0 || ((message >> FFTS_MODE_OFFSET) & FFTS_MODE_WIDTH) != FFTS_MODE_VAL ||
        (message & FFTS_BASE_COUNT_WIDTH) != 1) {
        throw std::invalid_argument("CPU FFTS supports mode 2 with base count 1 only");
    }
    auto& state = getEventState((message >> FFTS_EVENT_ID_OFFSET) & FFTS_EVENT_ID_WIDTH);
    if constexpr (IsCube) {
        // Each group has one AIC publisher. Check both lanes before changing
        // either counter; the AIV consumers can only free capacity meanwhile.
        for (auto& credits : state.cubeToVector) {
            if (credits.load(std::memory_order_relaxed) == std::numeric_limits<uint32_t>::max()) {
                throw std::overflow_error("CPU FFTS credit counter overflow");
            }
        }
        publish(state.cubeToVector[0]);
        publish(state.cubeToVector[1]);
    } else {
        publish(state.vectorToCube[vectorLane()]);
    }
}

template <bool IsCube>
inline void wait(int eventId)
{
    auto& state = getEventState(eventId);
    if constexpr (IsCube) {
        consume(state.vectorToCube[0]);
        consume(state.vectorToCube[1]);
    } else {
        consume(state.cubeToVector[vectorLane()]);
    }
}
} // namespace cpu_sim::ffts
} // namespace pto

// CPU execution uses runtime-owned state, not the hardware FFTS MMIO address.
// Setting this address must not clear credits published by another core.
static inline void set_ffts_base_addr(uint64_t) {}

static inline void __builtin_cce_ffts_cross_core_sync(int, uint16_t message)
{
#if defined(__DAV_CUBE__) && !defined(__DAV_VEC__)
    pto::cpu_sim::ffts::signal<true>(message);
#elif defined(__DAV_VEC__) && !defined(__DAV_CUBE__)
    pto::cpu_sim::ffts::signal<false>(message);
#else
    (void)message;
    throw std::runtime_error("CPU FFTS requires exactly one kernel role: __DAV_CUBE__ or __DAV_VEC__");
#endif
}

static inline void __builtin_cce_wait_flag_dev(int eventId)
{
#if defined(__DAV_CUBE__) && !defined(__DAV_VEC__)
    pto::cpu_sim::ffts::wait<true>(eventId);
#elif defined(__DAV_VEC__) && !defined(__DAV_CUBE__)
    pto::cpu_sim::ffts::wait<false>(eventId);
#else
    (void)eventId;
    throw std::runtime_error("CPU FFTS requires exactly one kernel role: __DAV_CUBE__ or __DAV_VEC__");
#endif
}

static inline void ffts_cross_core_sync(int pipe, uint16_t message)
{
    __builtin_cce_ffts_cross_core_sync(pipe, message);
}

static inline void wait_flag_dev(int eventId) { __builtin_cce_wait_flag_dev(eventId); }

#endif // PTO_CPU_FFTS_HPP
