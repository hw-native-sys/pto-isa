#!/usr/bin/env python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

"""Describe page-aligned KV pools and translate local sparse indices."""

import numpy as np


def pool_layout(mode, batch, logical_length, page_size):
    """Validate index capacity before callers allocate any fixture tensors."""
    if mode not in ("per-batch", "shared"):
        raise ValueError("--kv-pool must be per-batch or shared")
    if min(batch, logical_length, page_size) < 1:
        raise ValueError("KV pool batch, logical length, and page size must be positive")
    count = batch if mode == "per-batch" else 1
    pages_per_pool = (logical_length + page_size - 1) // page_size
    tokens_per_pool = pages_per_pool * page_size
    if count * tokens_per_pool > np.iinfo(np.int32).max:
        raise ValueError("KV pool allocation exceeds the nonnegative int32 sparse index capacity")
    return {
        "mode": mode,
        "count": count,
        "logical_length_per_pool": logical_length,
        "pages_per_pool": pages_per_pool,
        "page_size": page_size,
        "allocated_tokens_per_pool": tokens_per_pool,
        "num_pages": count * pages_per_pool,
    }


def offset_pool_indices(indices, layout):
    """Translate valid local indices in place, retaining every -1 sentinel."""
    if layout["mode"] == "per-batch":
        offsets = np.arange(indices.shape[0], dtype=np.int64) * layout["allocated_tokens_per_pool"]
        np.add(indices, offsets[:, None, None], out=indices, where=indices >= 0, casting="unsafe")
    return indices
