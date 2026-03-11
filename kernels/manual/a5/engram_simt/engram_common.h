/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ENGRAM_COMMON_H
#define ENGRAM_COMMON_H

#include <cstdint>

constexpr uint32_t kGoldenRatio = 0x9e3779b9u;

constexpr uint32_t kHashPrimes[16] = {0x85ebca6b, 0xc2b2ae35, 0x27d4eb2f, 0x165667b1, 0x8d2a4c8a, 0x1b873593,
                                      0xcc9e2d51, 0xe6546b64, 0x9e3779b9, 0x7f4a7c15, 0x2c1b3c6d, 0x5851f42d,
                                      0x4ca5cf08, 0x32fd6b73, 0xa3f8c72e, 0x1f3b7e89};

inline uint32_t compute_ngram_key_host(const uint32_t *ids, int pos, int ngram_size)
{
    if (pos < ngram_size - 1)
        return 0;
    uint32_t key = 0;
    for (int i = 0; i < ngram_size; ++i)
        key = key * 31 + ids[pos - ngram_size + 1 + i];
    return key;
}

inline uint32_t multi_head_hash_host(uint32_t key, uint32_t table_size, int head)
{
    uint32_t h = key;
    h ^= kHashPrimes[head % 16];
    h = (h ^ (h >> 16)) * kGoldenRatio;
    h ^= kHashPrimes[(head + 1) % 16];
    h = (h ^ (h >> 13)) * kHashPrimes[(head + 2) % 16];
    h ^= (h >> 16);
    return h % table_size;
}

#endif
