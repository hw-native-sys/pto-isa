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


def gen_golden_data(case_name, param):
    dtype = param.dtype

    dst_tile_row, dst_tile_col = param.dst_tile_row, param.dst_tile_col
    src0_tile_row, src0_tile_col = param.src0_tile_row, param.src0_tile_col
    src1_tile_row, src1_tile_col = param.src1_tile_row, param.src1_tile_col
    v_valid_row, v_valid_col0, v_valid_col1 = param.valid_row, param.valid_col0, param.valid_col1
    idx_tiles = param.idx_tiles
    if (idx_tiles != 'none'):
        idxtype = param.idxtype
        src0_idx_row, src0_idx_col = param.src0_idx_row, param.src0_idx_col
        src1_idx_row, src1_idx_col = param.src1_idx_row, param.src1_idx_col
    if (idx_tiles == 'dst'):
        dst_idx_row, dst_idx_col = param.dst_idx_row, param.dst_idx_col

    # Generate input arrays
    input1_valid = np.random.uniform(-1000, 1000, size=(v_valid_row, v_valid_col0)).astype(dtype)
    input2_valid = np.random.uniform(-1000, 1000, size=(v_valid_row, v_valid_col1)).astype(dtype)
    input1 = np.zeros([src0_tile_row, src0_tile_col]).astype(dtype)
    input2 = np.zeros([src1_tile_row, src1_tile_col]).astype(dtype)
    input1[0:v_valid_row, 0:v_valid_col0] = input1_valid
    input2[0:v_valid_row, 0:v_valid_col1] = input2_valid
    if (idx_tiles != 'none'):
        input1_idx_valid = np.random.uniform(0, v_valid_col0, size=(v_valid_row, 1)).astype(idxtype)
        input2_idx_valid = np.random.uniform(0, v_valid_col1, size=(v_valid_row, 1)).astype(idxtype)
        input1_idx = np.zeros([src0_idx_row, src0_idx_col]).astype(idxtype)
        input2_idx = np.zeros([src1_idx_row, src1_idx_col]).astype(idxtype)
        input1_idx[0:v_valid_row, 0:1] = input1_idx_valid
        input2_idx[0:v_valid_row, 0:1] = input2_idx_valid

    # Perform the concat operation
    golden = np.zeros([dst_tile_row, dst_tile_col]).astype(dtype)
    if (idx_tiles == 'dst'):
        golden_idx = np.zeros([dst_idx_row, dst_idx_col]).astype(idxtype)

    if (idx_tiles == 'none'):
        golden[:v_valid_row, :v_valid_col0 + v_valid_col1] = np.concatenate([
            input1[:v_valid_row, :v_valid_col0],
            input2[:v_valid_row, :v_valid_col1]
        ], axis=1)
    else:
        if (idx_tiles == 'dst'):
            golden_idx[:v_valid_row, :1] = (input1_idx + input2_idx)[:v_valid_row, :1]
        for i in range(v_valid_row):
            col1_len = input1_idx[i, 0]
            col2_len = input2_idx[i, 0]

            golden[i, :col1_len] = input1[i, :col1_len]
            golden[i, col1_len:col1_len + col2_len] = input2[i, :col2_len]


    # Save the input and golden data to binary files
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")
    if (idx_tiles != 'none'):
        input1_idx.tofile("input1_idx.bin")
        input2_idx.tofile("input2_idx.bin")
    if (idx_tiles == 'dst'):
        golden_idx.tofile("golden_idx.bin")


class TConcatParams:
    def __init__(self, dtype, valid_row, valid_col0, valid_col1, idxtype=None, idx_tiles='none', dst_h=None,
                 dst_w=None, src0_h=None, src0_w=None, src1_h=None, src1_w=None, dst_idx_h=None, dst_idx_w=None,
                 src0_idx_h=None, src0_idx_w=None, src1_idx_h=None, src1_idx_w=None):
        self.dtype = dtype
        self.idxtype = idxtype
        self.idx_tiles = idx_tiles
        self.valid_row = valid_row
        self.valid_col0 = valid_col0
        self.valid_col1 = valid_col1
        self.dst_tile_row = self._default_if_none(dst_h, valid_row)
        self.dst_tile_col = self._default_if_none(dst_w, valid_col0 + valid_col1)
        self.src0_tile_row = self._default_if_none(src0_h, valid_row)
        self.src0_tile_col = self._default_if_none(src0_w, valid_col0)
        self.src1_tile_row = self._default_if_none(src1_h, valid_row)
        self.src1_tile_col = self._default_if_none(src1_w, valid_col1)

        self.dst_idx_row = self._default_if_none(dst_idx_h, self.dst_tile_row)
        self.dst_idx_col = self._default_if_none(dst_idx_w, self.dst_tile_col)
        self.src0_idx_row = self._default_if_none(src0_idx_h, self.src0_tile_row)
        self.src0_idx_col = self._default_if_none(src0_idx_w, self.src0_tile_col)
        self.src1_idx_row = self._default_if_none(src1_idx_h, self.src1_tile_row)
        self.src1_idx_col = self._default_if_none(src1_idx_w, self.src1_tile_col)

    @staticmethod
    def _default_if_none(value, default):
        return default if value is None else value


def generate_case_name(param):
    types_dict = {
        np.float32: 'float',
        np.float16: 'half',
        np.int8: 'int8',
        np.int32: 'int32',
        np.int16: 'int16'
    }
    dtype_str = types_dict.get(param.dtype)
    idxtype_str = types_dict.get(param.idxtype)
    idx_tiles = param.idx_tiles
    sizes_str = f"{param.dst_tile_row}x{param.dst_tile_col}_{param.src0_tile_row}x{param.src0_tile_col}_\
{param.src1_tile_row}x{param.src1_tile_col}_{param.valid_row}x{param.valid_col0}_{param.valid_row}x{param.valid_col1}"
    idxsizes_str = f"{param.dst_idx_row}x{param.dst_idx_col}_{param.src0_idx_row}x{param.src0_idx_col}_\
{param.src1_idx_row}x{param.src1_idx_col}"

    if (idx_tiles == 'none'):
        return f"TCONCATTest.case_{dtype_str}_{sizes_str}"
    return f"TCONCATTest.case_{dtype_str}_{idxtype_str}_{idx_tiles}_{sizes_str}_{idxsizes_str}"


if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TConcatParams(np.float32, 64, 64, 64),
        TConcatParams(np.int32, 64, 64, 64),
        TConcatParams(np.float16, 16, 128, 128),
        TConcatParams(np.float32, 16, 32, 32),
        TConcatParams(np.int16, 32, 128, 128),
        TConcatParams(np.float16, 16, 63, 64, src0_w=64, dst_w=128),
        TConcatParams(np.float32, 16, 31, 32, src0_w=32, dst_w=64),
        TConcatParams(np.int16, 32, 127, 128, src0_w=128, dst_w=256),
        TConcatParams(np.float32, 64, 64, 64, np.int32, 'src'),
        TConcatParams(np.int32, 64, 64, 64, np.int32, 'src'),
        TConcatParams(np.float16, 16, 128, 128, np.int32, 'src'),
        TConcatParams(np.float32, 16, 32, 32, np.int32, 'src'),
        TConcatParams(np.int16, 32, 128, 128, np.int32, 'src'),
        TConcatParams(np.float16, 16, 63, 64, np.int32, 'src', src0_w=64, dst_w=128),
        TConcatParams(np.float32, 16, 31, 32, np.int32, 'src', src0_w=32, dst_w=64),
        TConcatParams(np.int16, 32, 127, 128, np.int32, 'src', src0_w=128, dst_w=256),
        TConcatParams(np.float32, 64, 64, 64, np.int32, 'dst'),
        TConcatParams(np.int32, 64, 64, 64, np.int32, 'dst'),
        TConcatParams(np.float16, 16, 128, 128, np.int32, 'dst'),
        TConcatParams(np.float32, 16, 32, 32, np.int32, 'dst'),
        TConcatParams(np.int16, 32, 128, 128, np.int32, 'dst'),
        TConcatParams(np.float16, 16, 63, 64, np.int32, 'dst', src0_w=64, dst_w=128),
        TConcatParams(np.float32, 16, 31, 32, np.int32, 'dst', src0_w=32, dst_w=64),
        TConcatParams(np.int16, 32, 127, 128, np.int32, 'dst', src0_w=128, dst_w=256),
    ]

    for param in case_params_list:
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, param)
        os.chdir(original_dir)
