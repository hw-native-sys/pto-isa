/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
#ifndef PAGED_HATTENTION_H
#define PAGED_HATTENTION_H

#include <cstdint>

namespace AtbOps {
constexpr int32_t BLOCK_SIZE = 16;
constexpr int32_t BLOCK_SIZE_32 = 32;
constexpr int32_t TILING_PARA_SIZE = 17;
constexpr int32_t TILING_HEAD_SIZE = 44;
constexpr int32_t TILING_HEAD_SIZE_NZ = 128;
constexpr int32_t TILING_HEAD_SIZE_910A = 192;
constexpr int32_t TILING_PARA_SIZE_NZ = 8;
constexpr int32_t M_LIMIT = 128;
constexpr int32_t FLOAT_LIMIT = 64;
constexpr int32_t MAX_EMBEDDING = 576;
constexpr int32_t ND_BATCH_LIMIT = INT32_MAX;
constexpr int32_t BLOCK_LIMIT = 128 * 128;
constexpr int32_t BLOCK_LIMIT_NO_PINGPONG = 128 * 256;
constexpr int32_t BLOCK_LIMIT_NO_PINGPONG_UINT8 = 128 * 256 * 2;
constexpr int32_t NZ_BLOCK_SIZE = 16;
constexpr int32_t TILING_KEY_ID = 16;
constexpr int32_t MLA_BLOCK_SIZE_LIMIT = 128;
constexpr int32_t MLA_THRESHOLD = 256;
constexpr int32_t PREFILL_BATCH = 27;
constexpr int32_t PARALLEL_MAX_HEAD = 256;
constexpr int32_t PARALLEL_MAX_BLK_SIZE = 128;
constexpr int32_t PARALLEL_MAX_BATCH = 2000;
constexpr int32_t WORKSPACE_BLOCK_SIZE_DB = 65536;  // 128 * 256 * 2

enum class TilingKeyType {
    TILING_HALF_DATA = 0,
    TILING_BF16_DATA = 1,
    TILING_INT8_DATA = 2,
    TILING_INT8_CUBE_QUANT = 4,
    TILING_INT8_VEC_QUANT = 8,
    TILING_INT8_VEC_QUANTBF16 = 9,
    TILING_QUANT_FP16OUT = 12,
    TILING_QUANT_BF16OUT = 14
};

enum class CalcType { CALC_TYPE_DEFAULT = 0, CALC_TYPE_MIX = 1, CALC_TYPE_PREFILL = 2 };

enum class DataShapeType { BSND = 0, BNSD = 1 };

enum class CompressType { COMPRESS_TYPE_UNDEFINED = 0, COMPRESS_TYPE_KVHEAD = 1 };

enum class PagedAttnVariant { DEFAULT = 0, MULTI_LATENT = 1 };

using PagedAttentionInfo = struct PagedAttentionTilingParams {
    int32_t numTokens = 0;
    int32_t numHeads = 0;
    int32_t embeddingSize = 0;
    int32_t embeddingSizeV = 0;
    int32_t numBlocks = 0;
    int32_t blockSize = 0;
    int32_t maxNumBlocksPerQuery = 0;
    float tor = 0;
    int32_t kvHeads = 0;
    int32_t maxPromptLen = 0;
    int32_t batchStride = 0;
    int32_t headStride = 0;
    TilingKeyType type = TilingKeyType::TILING_HALF_DATA;
    int32_t batch = 0;
    int32_t isMaskSquare = 0;
    int32_t *batchRunStatus{nullptr};
    int32_t *kvSeqLen{nullptr};
    int32_t modCoef{-1};
    int32_t divCoef{1};
    int32_t *qSeqLen{nullptr};
    int32_t qHeadOriginal = 0;
    int32_t compressHead = 0;
    int32_t tBlockAlign = 16;  // L1 tile alignment: 16 for fp16, 32 for int8
    int32_t dataShapeType = 0;
};

using AddrOffsets = struct AddressOffsetInfo {
    uint64_t addrQSeqOffset = 0;
    uint64_t addrOSeqOffset = 0;
    uint64_t addrOFdSeqOffset = 0;
    uint64_t addrLSeqOffset = 0;
};

}  // namespace AtbOps

#endif
// PAGED_HATTENTION_H
