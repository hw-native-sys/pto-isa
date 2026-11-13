/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_LINX_RUNTIME_HPP
#define PTO_CPU_LINX_RUNTIME_HPP
#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <map>
#include <mutex>
#include <limits>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>
namespace pto::linx_cpu {
inline thread_local unsigned peIndex = 0;
inline unsigned threadIndex() { return peIndex; }
struct MemoryRegion {
    uintptr_t address;
    std::span<std::byte> storage;
};
struct CoreConfig {
    unsigned peCount = 4;
    std::span<const MemoryRegion> memory;
};
struct CoreContext {
    std::mutex mutex;
    std::condition_variable changed;
    unsigned arrived = 0, generation = 0, finished = 0;
    std::exception_ptr error;
    unsigned activeMask = 15;
    std::span<const MemoryRegion> memory;
    unsigned barrierArrived = 0, barrierGeneration = 0;
    struct PeerCopy {
        void* destination = nullptr;
        std::vector<std::byte> source;
        uint64_t peer = 0;
        unsigned mask = 0;
    };
    std::array<PeerCopy, 4> copies;
    // Shared producers rendezvous only with their mask's selected PEs. Keep
    // completed generations for nonselected consumers that arrive later.
    struct SharedExchange {
        std::array<std::vector<std::byte>, 4> sources;
        std::vector<std::byte> result;
        unsigned arrived = 0;
        bool ready = false;
    };
    std::array<std::array<std::size_t, 4>, 16> sharedSequence{};
    std::map<std::pair<unsigned, std::size_t>, SharedExchange> sharedExchanges;
    // Absolute Shared slots are Core-wide, including for PEs that did not
    // execute the producer instruction.
    // Explicit slots retain ordered generations until every PE has acquired
    // them; a fast producer must not overwrite a slow consumer's input.
    std::map<std::pair<unsigned, std::size_t>, std::vector<std::byte>> sharedSlots;
    std::array<std::size_t, 64> slotPublished{};
    std::array<std::array<std::size_t, 4>, 64> slotConsumed{};
    unsigned slotWaiters = 0;
};
inline thread_local CoreContext* currentCore = nullptr;
inline std::byte* globalAddress(const void* pointer, std::size_t offset, std::size_t length)
{
    const auto base = reinterpret_cast<uintptr_t>(pointer);
    if (offset > std::numeric_limits<uintptr_t>::max() - base)
        throw std::out_of_range("Linx GM address overflow");
    const auto address = base + offset;
    if (length > std::numeric_limits<uintptr_t>::max() - address)
        throw std::out_of_range("Linx GM access wraps address space");
    if (currentCore)
        for (const auto& region : currentCore->memory)
            if ((base >= region.address && base - region.address < region.storage.size()) ||
                (address >= region.address && address - region.address < region.storage.size()) ||
                (length == 0 && address == region.address + region.storage.size())) {
                const auto index = address - region.address;
                if (index > region.storage.size() || length > region.storage.size() - index)
                    throw std::out_of_range("Linx GM access exceeds mapped region");
                return region.storage.data() + index;
            }
    if (address < 4096)
        throw std::out_of_range("Linx GM low address requires an explicit MemoryRegion");
    return reinterpret_cast<std::byte*>(address);
}
inline void syncCore()
{
    if (!currentCore)
        return;
    auto& core = *currentCore;
    std::unique_lock lock(core.mutex);
    if (core.error)
        std::rethrow_exception(core.error);
    if (core.finished)
        throw std::invalid_argument("Linx barrier participant already exited");
    const auto generation = core.barrierGeneration;
    core.barrierArrived |= 1u << threadIndex();
    if (core.barrierArrived == core.activeMask) {
        core.barrierArrived = 0;
        ++core.barrierGeneration;
        core.changed.notify_all();
    } else {
        core.changed.wait(lock, [&] { return core.barrierGeneration != generation || core.error || core.finished; });
        if (core.error)
            std::rethrow_exception(core.error);
        if (core.barrierGeneration == generation)
            throw std::invalid_argument("Linx barrier participant exited before arrival");
    }
}
// Host launcher for kernels containing Core4 collectives. The kernel itself
// retains its native instruction calls and observes get_thread_idx() == 0..3.
template <typename Function>
void runCore(Function fn, CoreConfig config = {})
{
    if (config.peCount < 1 || config.peCount > 4)
        throw std::invalid_argument("Linx Core requires one to four PEs");
    for (std::size_t i = 0; i < config.memory.size(); ++i) {
        const auto& region = config.memory[i];
        if (region.storage.empty() || region.storage.size() > std::numeric_limits<uintptr_t>::max() - region.address)
            throw std::invalid_argument("invalid Linx GM mapping");
        for (std::size_t j = 0; j < i; ++j) {
            const auto& other = config.memory[j];
            if (region.address < other.address + other.storage.size() &&
                other.address < region.address + region.storage.size())
                throw std::invalid_argument("overlapping Linx GM mappings");
        }
    }
    CoreContext core;
    core.activeMask = (1u << config.peCount) - 1;
    core.memory = config.memory;
    std::array<std::thread, 4> workers;
    unsigned launched = 0;
    try {
        for (unsigned pe = 0; pe < config.peCount; ++pe) {
            workers[pe] = std::thread([&, pe] {
                peIndex = pe;
                currentCore = &core;
                try {
                    fn();
                } catch (...) {
                    std::lock_guard lock(core.mutex);
                    if (!core.error)
                        core.error = std::current_exception();
                    core.changed.notify_all();
                }
                {
                    std::lock_guard lock(core.mutex);
                    core.finished |= 1u << pe;
                    if ((core.arrived || core.barrierArrived) && !core.error)
                        core.error = std::make_exception_ptr(std::invalid_argument("divergent Linx Core4 collective"));
                    core.changed.notify_all();
                }
                currentCore = nullptr;
            });
            ++launched;
        }
    } catch (...) {
        std::lock_guard lock(core.mutex);
        core.error = std::current_exception();
        core.changed.notify_all();
    }
    for (unsigned pe = 0; pe < launched; ++pe)
        workers[pe].join();
    if (core.error)
        std::rethrow_exception(core.error);
}
} // namespace pto::linx_cpu
#endif
