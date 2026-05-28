/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_PERF_SIM_CACHE_MODEL_HPP
#define PTO_PERF_SIM_CACHE_MODEL_HPP

#include "config.hpp"
#include <cstdint>
#include <list>
#include <unordered_map>

namespace pto::perf_sim {

struct CacheLine {
    uint64_t addr;
    std::list<uint64_t>::iterator lru_iter;
};

struct CacheAccessResult {
    bool hit;
    uint64_t adjusted_duration;
};

class L2CacheModel {
public:
    explicit L2CacheModel(const PerfSimConfig &config = GetConfig())
        : line_size_(config.l2_cache_line),
          capacity_(config.has_l2 ? config.l2_size / config.l2_cache_line : 0),
          l2_read_bw_(config.l2_read_bw),
          gm_read_bw_(config.gm_read_bw),
          hit_latency_(config.l2_hit_latency),
          miss_extra_latency_(config.l2_miss_extra_latency)
    {}

    CacheAccessResult Access(uint64_t addr, uint64_t size, uint64_t base_duration)
    {
        if (capacity_ == 0) {
            return {false, base_duration};
        }

        uint64_t line_addr = AlignToLine(addr);
        bool hit = Lookup(line_addr);

        if (hit) {
            stats_.hits++;
            // Hit: use L2 bandwidth (faster)
            uint64_t l2_duration = (size + l2_read_bw_ - 1) / l2_read_bw_ + hit_latency_;
            return {true, std::min(l2_duration, base_duration)};
        } else {
            // Miss: GM bandwidth + extra latency, allocate cache line
            Allocate(line_addr);
            stats_.misses++;
            return {false, base_duration + miss_extra_latency_};
        }
    }

    struct Stats {
        uint64_t hits = 0;
        uint64_t misses = 0;
        double HitRate() const
        {
            auto total = hits + misses;
            return total > 0 ? 100.0 * hits / total : 0.0;
        }
    };

    const Stats &GetStats() const
    {
        return stats_;
    }
    void Reset()
    {
        cache_.clear();
        lru_.clear();
        stats_ = {};
    }

private:
    uint64_t line_size_;
    uint64_t capacity_;
    uint32_t l2_read_bw_;
    uint32_t gm_read_bw_;
    uint32_t hit_latency_;
    uint32_t miss_extra_latency_;

    std::unordered_map<uint64_t, CacheLine> cache_;
    std::list<uint64_t> lru_; // front = least recently used
    Stats stats_;

    uint64_t AlignToLine(uint64_t addr) const
    {
        return (addr / line_size_) * line_size_;
    }

    bool Lookup(uint64_t line_addr)
    {
        auto it = cache_.find(line_addr);
        if (it == cache_.end())
            return false;
        // Update LRU: move to back
        lru_.erase(it->second.lru_iter);
        lru_.push_back(line_addr);
        it->second.lru_iter = std::prev(lru_.end());
        return true;
    }

    void Allocate(uint64_t line_addr)
    {
        // Evict LRU if at capacity
        while (cache_.size() >= capacity_ && !lru_.empty()) {
            auto victim = lru_.front();
            lru_.pop_front();
            cache_.erase(victim);
        }
        auto [it, _] = cache_.emplace(line_addr, CacheLine{});
        lru_.push_back(line_addr);
        it->second.lru_iter = std::prev(lru_.end());
    }
};

} // namespace pto::perf_sim

#endif
