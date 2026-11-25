/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef FLASH_MLA_BF16_API_H_
#define FLASH_MLA_BF16_API_H_

#include <cstdint>

// Sparse indices address physical tokens in cache; -1 entries are masked.
// Query and dequantized K/V use one 512-feature vector, including RoPE.
struct FlashMLABf16Config {
    int32_t batchSize = 1;
    int32_t queryLength = 1;
    int32_t numHeads = 64;
    int32_t headDim = 512;
    int32_t pageSize = 256;
    int32_t numPages = 1;
    float softmaxScale = 0.0441941738f;
    int32_t topK = 0;
    int32_t extraTopK = 0;
    int32_t extraNumPages = 0;
    int32_t extraPageSize = 256;
};

// q: BF16 [B,Q,H,512]. cache/extraCache: [pages,pageSize,288], consisting
// of packed E2M1[256] followed by E4M3[32], one scale per 16 features.
// Each 16-feature group uses its finite signed or zero E4M3 scale.
// K/V dequantization and P are BF16. QK/PV accumulate in FP32, and online
// maxima, denominator, alpha rescaling and running output remain FP32.
// indices: int32 [B,Q,topK]; optional topkLength: int32 [B] in [0,topK].
// extraIndices/extraTopkLength follow the same contract for extraTopK.
// The caller validates device indices, prefix lengths, and finite Q/cache data.
// sink: optional float32 [H]. output: BF16 [B,Q,H,512], lse: FP32 [B,H,Q].
// Empty selections produce zero output and positive-infinity LSE.
// launchCount is bounded by available AICs; jobs are distributed persistently.
extern "C" void LaunchFlashMLABf16(
    void* output, float* lse, const uint16_t* q, const uint8_t* cache, const float* sink, const int32_t* indices,
    const int32_t* topkLength, const uint8_t* extraCache, const int32_t* extraIndices, const int32_t* extraTopkLength,
    const FlashMLABf16Config& config, uint32_t launchCount, void* stream);

#endif // FLASH_MLA_BF16_API_H_
