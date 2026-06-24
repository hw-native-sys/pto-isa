#!/usr/bin/python3
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

import os
import numpy as np
np.random.seed(19)


def gen_golden_data(param):
    src_type = param.src_type
    index_type = param.index_type
    g_shape0 = param.g_shape0
    g_shape1 = param.g_shape1
    g_shape2 = param.g_shape2
    g_shape3 = param.g_shape3
    g_shape4 = param.g_shape4
    g_whole_shape0 = param.g_whole_shape0
    g_whole_shape1 = param.g_whole_shape1
    g_whole_shape2 = param.g_whole_shape2
    g_whole_shape3 = param.g_whole_shape3
    g_whole_shape4 = param.g_whole_shape4
    topk = param.topk

    valid_row = g_shape0 * g_shape1 * g_shape2 * g_shape3
    valid_col = g_shape4
    rows = g_whole_shape0 * g_whole_shape1 * g_whole_shape2 * g_whole_shape3
    cols = g_whole_shape4
    
    new_data = np.zeros((rows, cols)).astype(src_type)
    for i in range(valid_row):
        data = np.random.uniform(i, i + valid_col, size=valid_col).astype(src_type)
        new_data[i, :valid_col] = data

    x1_gm = np.zeros((rows, cols * 2)) 
    for i in range(valid_row):
        counter = 0
        for j in range(valid_col):
            original_value = new_data[i, j]
            x1_gm[i, j * 2] = original_value
            x1_gm[i, j * 2 + 1] = counter
            counter += 1

    topk_values = np.zeros((rows, topk)).astype(src_type)
    topk_indices = np.zeros((rows, topk)).astype(index_type)
    idx = np.arange(valid_col).astype(np.uint32)
    for i in range(valid_row):
        row = new_data[i, :valid_col]
        sorted_indices = np.lexsort((idx, -row))
        indices_sorted = sorted_indices[:topk]
        values = row[indices_sorted]

        topk_values[i] = values
        topk_indices[i] = indices_sorted

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    new_data.tofile("./input/x1_gm.bin")
    idx.tofile("./input/x1_idx.bin")
    topk_indices.tofile("./output/golden_i.bin")
    topk_values.tofile("./output/golden_d.bin")


class TopkParams:
    def __init__(self, src_type, index_type, g_shape0, g_shape1, g_shape2, g_shape3, g_shape4,
                 g_whole_shape0, g_whole_shape1, g_whole_shape2, g_whole_shape3, g_whole_shape4, topk):
        self.src_type = src_type
        self.index_type = index_type
        self.g_shape0 = g_shape0
        self.g_shape1 = g_shape1
        self.g_shape2 = g_shape2
        self.g_shape3 = g_shape3
        self.g_shape4 = g_shape4
        self.g_whole_shape0 = g_whole_shape0
        self.g_whole_shape1 = g_whole_shape1
        self.g_whole_shape2 = g_whole_shape2
        self.g_whole_shape3 = g_whole_shape3
        self.g_whole_shape4 = g_whole_shape4
        self.topk = topk

if __name__ == "__main__":

    case_params_list = [
        TopkParams(np.float32, np.int32, 1, 1, 1, 4800, 1024, 1, 1, 1, 4800, 1280, 1000)
    ]
    gen_golden_data(case_params_list[0])
