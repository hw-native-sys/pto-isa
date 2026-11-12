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
#include <stdexcept>
#include <thread>
#include <vector>
namespace pto::linx_cpu {
inline thread_local unsigned peIndex = 0;
inline unsigned threadIndex() { return peIndex; }
struct CoreContext {
    std::mutex mutex;
    std::condition_variable changed;
    unsigned arrived = 0, generation = 0, finished = 0;
    std::exception_ptr error;
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
    std::map<unsigned, std::vector<std::byte>> sharedSlots;
    unsigned slotWaiters = 0;
};
inline thread_local CoreContext* currentCore = nullptr;
// Host launcher for kernels containing Core4 collectives. The kernel itself
// retains its native instruction calls and observes get_thread_idx() == 0..3.
template <typename Function>
void runCore(Function fn)
{
    CoreContext core;
    std::array<std::thread, 4> workers;
    for (unsigned pe = 0; pe < 4; ++pe)
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
                if (core.arrived && !core.error)
                    core.error = std::make_exception_ptr(std::invalid_argument("divergent Linx Core4 collective"));
                core.changed.notify_all();
            }
            currentCore = nullptr;
        });
    for (auto& thread : workers)
        thread.join();
    if (core.error)
        std::rethrow_exception(core.error);
}
} // namespace pto::linx_cpu
#endif
