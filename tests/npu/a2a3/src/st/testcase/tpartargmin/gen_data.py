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


def gen_tile(dtype, size):
    if dtype in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32):
        dtype_info = np.iinfo(dtype)
        return np.random.randint(dtype_info.min, dtype_info.max, size=size).astype(dtype)
    else:
        dtype_info = np.finfo(dtype)
        return np.random.uniform(low=dtype_info.min, high=dtype_info.max, size=size).astype(dtype)


def gen_golden_data(param):
    idx_dtype, val_dtype = param.idx_dtype, param.val_dtype
    valid_height0, valid_width0 = param.valid_size0
    valid_height1, valid_width1 = param.valid_size1
    valid_height_big, valid_width_big = max(valid_height0, valid_height1), max(valid_width0, valid_width1)
    valid_height_small, valid_width_small = min(valid_height0, valid_height1), min(valid_width0, valid_width1)

    src_val0 = gen_tile(val_dtype, param.src_val0_size)
    src_idx0 = gen_tile(idx_dtype, param.src_idx0_size)
    src_val1 = gen_tile(val_dtype, param.src_val1_size)
    src_idx1 = gen_tile(idx_dtype, param.src_idx1_size)
    dst_val = np.full(param.dst_val_size, 0).astype(val_dtype)
    dst_idx = np.full(param.dst_idx_size, 0).astype(idx_dtype)

    src_val_big, src_idx_big = src_val0, src_idx0
    if valid_height0 < valid_height1 or valid_width0 < valid_width1:
        src_val_big, src_idx_big = src_val1, src_idx1

    dst_val[:valid_height_big, :valid_width_big] = src_val_big[:valid_height_big, :valid_width_big]
    dst_idx[:valid_height_big, :valid_width_big] = src_idx_big[:valid_height_big, :valid_width_big]

    src0 = src_val0[0:valid_height_small, 0:valid_width_small]
    src1 = src_val1[0:valid_height_small, 0:valid_width_small]
    mask = src0 < src1
    dst_val[:valid_height_small, :valid_width_small] = np.where(mask, src0, src1)
    dst_val = dst_val.astype(val_dtype)
    src_idx_part0 = src_idx0[:valid_height_small, :valid_width_small]
    src_idx_part1 = src_idx1[:valid_height_small, :valid_width_small]
    dst_idx[:valid_height_small, :valid_width_small] = np.where(mask, src_idx_part0, src_idx_part1)
    dst_idx = dst_idx.astype(idx_dtype)

    dst_val.tofile('dst_val.bin')
    dst_idx.tofile('dst_idx.bin')
    src_val0.tofile('src_val0.bin')
    src_idx0.tofile('src_idx0.bin')
    src_val1.tofile('src_val1.bin')
    src_idx1.tofile('src_idx1.bin')


class TestParams:
    TEST_NAME = 'TPARTARGMINTest'
    DTYPE_STR_TABLE = {
        np.float32: 'float',
        np.float16: 'half',
        np.int32: 'int32',
        np.uint32: 'uint32',
        np.int16: 'int16',
        np.uint16: 'uint16',
        np.int8: 'int8',
        np.uint8: 'uint8',
    }

    def __init__(self, val_dtype, idx_dtype, comment, dst_val_size, dst_idx_size, src_val0_size, src_idx0_size,
        src_val1_size, src_idx1_size, valid_size0, valid_size1):
        self.val_dtype, self.idx_dtype = val_dtype, idx_dtype
        self.dst_val_size = dst_val_size
        self.dst_idx_size = dst_idx_size
        self.src_val0_size = src_val0_size
        self.src_idx0_size = src_idx0_size
        self.src_val1_size = src_val1_size
        self.src_idx1_size = src_idx1_size
        self.valid_size0 = valid_size0
        self.valid_size1 = valid_size1
        val_dtype_str, idx_dtype_str = self.DTYPE_STR_TABLE[val_dtype], self.DTYPE_STR_TABLE[idx_dtype]
        self.name = f'{self.TEST_NAME}.case_{val_dtype_str}_{idx_dtype_str}_{comment}'


if __name__ == "__main__":
    case_list = [
        TestParams(np.float32, np.uint32, 'same_size',
            (4, 8), (4, 8), (4, 8), (4, 8), (4, 8), (4, 8), (4, 8), (4, 8)),
        TestParams(np.float32, np.uint32, 'row_diff_0',
            (4, 8), (4, 8), (4, 8), (4, 8), (4, 8), (4, 8), (3, 8), (4, 8)),
        TestParams(np.float32, np.uint32, 'row_diff_1',
            (4, 8), (4, 8), (4, 8), (4, 8), (4, 8), (4, 8), (4, 8), (3, 8)),
        TestParams(np.float32, np.uint32, 'col_diff_0',
            (4, 8), (4, 8), (4, 8), (4, 8), (4, 8), (4, 8), (4, 7), (4, 8)),
        TestParams(np.float32, np.uint32, 'col_diff_1',
            (4, 8), (4, 8), (4, 8), (4, 8), (4, 8), (4, 8), (4, 8), (4, 7)),
        TestParams(np.float32, np.uint32, 'small_0',
            (4, 8), (4, 8), (4, 8), (4, 8), (4, 8), (4, 8), (3, 7), (4, 8)),
        TestParams(np.float32, np.uint32, 'small_1',
            (4, 8), (4, 8), (4, 8), (4, 8), (4, 8), (4, 8), (4, 8), (3, 7)),
        TestParams(np.float32, np.uint32, 'same_size_unaligned',
            (4, 8), (4, 8), (4, 8), (4, 8), (4, 8), (4, 8), (4, 7), (4, 7)),
        TestParams(np.float32, np.uint32, 'row_diff_unaligned_0',
            (4, 8), (4, 8), (4, 8), (4, 8), (4, 8), (4, 8), (3, 7), (4, 7)),
        TestParams(np.float32, np.uint32, 'row_diff_unaligned_1',
            (4, 8), (4, 8), (4, 8), (4, 8), (4, 8), (4, 8), (4, 7), (3, 7)),
        TestParams(np.float32, np.uint32, 'col_diff_unaligned_0',
            (4, 8), (4, 8), (4, 8), (4, 8), (4, 8), (4, 8), (4, 5), (4, 7)),
        TestParams(np.float32, np.uint32, 'col_diff_unaligned_1',
            (4, 8), (4, 8), (4, 8), (4, 8), (4, 8), (4, 8), (4, 7), (4, 5)),
        TestParams(np.float32, np.uint32, 'same_size_32k',
            (128, 64), (128, 64), (128, 64), (128, 64), (128, 64), (128, 64), (128, 64), (128, 64)),
        TestParams(np.float32, np.uint32, 'row_diff_32k_0',
            (128, 64), (128, 64), (128, 64), (128, 64), (128, 64), (128, 64), (111, 64), (128, 64)),
        TestParams(np.float32, np.uint32, 'row_diff_32k_1',
            (128, 64), (128, 64), (128, 64), (128, 64), (128, 64), (128, 64), (128, 64), (111, 64)),
        TestParams(np.float32, np.uint32, 'col_diff_32k_0',
            (128, 64), (128, 64), (128, 64), (128, 64), (128, 64), (128, 64), (128, 57), (128, 64)),
        TestParams(np.float32, np.uint32, 'col_diff_32k_1',
            (128, 64), (128, 64), (128, 64), (128, 64), (128, 64), (128, 64), (128, 64), (128, 57)),
        TestParams(np.float32, np.uint32, 'small_32k_0',
            (128, 64), (128, 64), (128, 64), (128, 64), (128, 64), (128, 64), (111, 57), (128, 64)),
        TestParams(np.float32, np.uint32, 'small_32k_1',
            (128, 64), (128, 64), (128, 64), (128, 64), (128, 64), (128, 64), (128, 64), (111, 57)),
        TestParams(np.float16, np.uint16, 'same_size_32k',
            (128, 128), (128, 128), (128, 128), (128, 128), (128, 128), (128, 128), (128, 128), (128, 128)),
        TestParams(np.float16, np.uint16, 'row_diff_32k_0',
            (128, 128), (128, 128), (128, 128), (128, 128), (128, 128), (128, 128), (111, 128), (128, 128)),
        TestParams(np.float16, np.uint16, 'row_diff_32k_1',
            (128, 128), (128, 128), (128, 128), (128, 128), (128, 128), (128, 128), (128, 128), (111, 128)),
        TestParams(np.float16, np.uint16, 'col_diff_32k_0',
            (128, 128), (128, 128), (128, 128), (128, 128), (128, 128), (128, 128), (128, 111), (128, 128)),
        TestParams(np.float16, np.uint16, 'col_diff_32k_1',
            (128, 128), (128, 128), (128, 128), (128, 128), (128, 128), (128, 128), (128, 128), (128, 111)),
        TestParams(np.float32, np.uint32, 'tile_diff',
            (4, 8), (4, 16), (4, 24), (4, 32), (4, 40), (4, 48), (4, 7), (4, 7)),
        TestParams(np.float32, np.uint32, 'tile_diff_32k',
            (67, 128), (67, 120), (67, 112), (67, 104), (67, 144), (67, 136), (67, 97), (67, 97)),
        TestParams(np.float32, np.uint32, 'tile_diff_32k_row_diff_0',
            (67, 128), (67, 120), (61, 112), (61, 104), (67, 144), (67, 136), (61, 97), (67, 97)),
        TestParams(np.float32, np.uint32, 'tile_diff_32k_row_diff_1',
            (67, 128), (67, 120), (67, 112), (67, 104), (61, 144), (61, 136), (67, 97), (61, 97)),
        TestParams(np.float32, np.uint32, 'tile_diff_32k_col_diff_0',
            (67, 128), (67, 120), (67, 112), (67, 104), (67, 144), (67, 136), (67, 97), (67, 101)),
        TestParams(np.float32, np.uint32, 'tile_diff_32k_col_diff_1',
            (67, 128), (67, 120), (67, 112), (67, 104), (67, 144), (67, 136), (67, 101), (67, 97)),
        TestParams(np.float32, np.uint32, 'tile_diff_32k_small_0',
            (67, 128), (67, 120), (67, 112), (67, 104), (67, 144), (67, 136), (61, 97), (67, 101)),
        TestParams(np.float32, np.uint32, 'tile_diff_32k_small_1',
            (67, 128), (67, 120), (67, 112), (67, 104), (67, 144), (67, 136), (67, 101), (61, 97)),
    ]

    for case in case_list:
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        orig_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(orig_dir)
