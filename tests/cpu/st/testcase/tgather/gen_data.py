#!/user/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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


P0101 = 1
P1010 = 2
P0001 = 3
P0010 = 4
P0100 = 5
P1000 = 6
P1111 = 7

# 需要合tgather_common.h里的对应一致
HALF_P0101_ROW = 5
HALF_P0101_COL = 128
HALF_P1010_ROW = 7
HALF_P1010_COL = 1024
HALF_P0001_ROW = 3
HALF_P0001_COL = 1056
HALF_P0010_ROW = 4
HALF_P0010_COL = 128
HALF_P0100_ROW = 5
HALF_P0100_COL = 256
HALF_P1000_ROW = 6
HALF_P1000_COL = 288
HALF_P1111_ROW = 7
HALF_P1111_COL = 320

FLOAT_P0101_ROW = 4
FLOAT_P0101_COL = 64
FLOAT_P1010_ROW = 7
FLOAT_P1010_COL = 1024
FLOAT_P0001_ROW = 3
FLOAT_P0001_COL = 1056
FLOAT_P0010_ROW = 4
FLOAT_P0010_COL = 128
FLOAT_P0100_ROW = 5
FLOAT_P0100_COL = 256
FLOAT_P1000_ROW = 6
FLOAT_P1000_COL = 288
FLOAT_P1111_ROW = 7
FLOAT_P1111_COL = 320

COL_HALF_P0101_ROW = 8
COL_HALF_P0101_COL = 128
COL_HALF_P1010_ROW = 14
COL_HALF_P1010_COL = 64

COL_HALF_P0001_ROW = 12
COL_HALF_P0001_COL = 256
COL_HALF_P0010_ROW = 16
COL_HALF_P0010_COL = 128
COL_HALF_P0100_ROW = 20
COL_HALF_P0100_COL = 64
COL_HALF_P1000_ROW = 24
COL_HALF_P1000_COL = 96

COL_HALF_P1111_ROW = 7
COL_HALF_P1111_COL = 320

B8_P0101_ROW = 5
B8_P0101_COL = 128
B8_P1010_ROW = 7
B8_P1010_COL = 1024
B8_P0001_ROW = 3
B8_P0001_COL = 1056
B8_P0010_ROW = 4
B8_P0010_COL = 128
B8_P0100_ROW = 5
B8_P0100_COL = 256
B8_P1000_ROW = 6
B8_P1000_COL = 288
B8_P1111_ROW = 7
B8_P1111_COL = 320


class TGatherParamsBase:
    def __init__(self, name):
        self.testname = name


class TGatherParamsMasked(TGatherParamsBase):
    def __init__(self, name, dst_type, src_type, row, col, pattern):
        super().__init__(name)
        self.dst_type = dst_type
        self.src_type = src_type
        self.row = row
        self.col = col
        self.pattern = pattern


class TGatherParamsColMasked(TGatherParamsBase):
    def __init__(self, name, dst_type, src_type, row, col, pattern):
        super().__init__(name)
        self.dst_type = dst_type
        self.src_type = src_type
        self.row = row
        self.col = col
        self.pattern = pattern
        self.gather_col = True


class TGatherParams1D(TGatherParamsBase):
    def __init__(self, name, src_type, src_row, src_col, dst_row, dst_col):
        super().__init__(name)
        self.src_type = src_type
        self.src_row = src_row
        self.src_col = src_col
        self.dst_row = dst_row
        self.dst_col = dst_col


def gather1d(src, indices):
    output = np.zeros_like(indices, dtype=src.dtype)
    for i in range(indices.shape[0]):
        output[i] = src[indices[i]]
    return output


def gen_golden_data(param: TGatherParamsBase):
    if isinstance(param, TGatherParamsColMasked):
        src_type = param.src_type
        dst_type = param.dst_type
        row = param.row
        col = param.col
        pattern = param.pattern
        x1_gm = np.random.randint(1, 100, [row, col]).astype(src_type)
        x1_gm.tofile("./x1_gm.bin")
        if pattern == P0101:
            res = x1_gm[0::2, :]
        elif pattern == P1010:
            res = x1_gm[1::2, :]
        elif pattern == P0001:
            res = x1_gm[0::4, :]
        elif pattern == P0010:
            res = x1_gm[1::4, :]
        elif pattern == P0100:
            res = x1_gm[2::4, :]
        elif pattern == P1000:
            res = x1_gm[3::4, :]
        elif pattern == P1111:
            res = x1_gm[:, :]

        golden = res.flatten()

        x1_gm.tofile("./x1_gm.bin")
        golden.tofile("./golden.bin")
        os.chdir(original_dir)
    elif isinstance(param, TGatherParamsMasked):
        src_type = param.src_type
        dst_type = param.dst_type
        row = param.row
        col = param.col
        pattern = param.pattern
        x1_gm = np.random.randint(1, 100, [row, col]).astype(src_type)
        x1_gm.tofile("./x1_gm.bin")
        res = np.zeros((row, col))
        if pattern == P0101:
            res = x1_gm[:, 0::2]
        elif pattern == P1010:
            res = x1_gm[:, 1::2]
        elif pattern == P0001:
            res = x1_gm[:, 0::4]
        elif pattern == P0010:
            res = x1_gm[:, 1::4]
        elif pattern == P0100:
            res = x1_gm[:, 2::4]
        elif pattern == P1000:
            res = x1_gm[:, 3::4]
        elif pattern == P1111:
            res = x1_gm[:, :]

        golden = res.flatten()

        x1_gm.tofile("./x1_gm.bin")
        golden.tofile("./golden.bin")
        os.chdir(original_dir)
    elif isinstance(param, TGatherParams1D):
        output = np.zeros([param.dst_row * param.dst_col]
                          ).astype(param.src_type)
        src_data = np.random.randint(-20, 20, (param.src_row *
                                     param.src_col)).astype(param.src_type)
        src_data.tofile("./src0.bin")
        indices = np.random.randint(
            0, param.src_row * param.src_col, (param.dst_row * param.dst_col)).astype(np.int32)
        indices.tofile("./src1.bin")
        golden = gather1d(src_data, indices)
        golden.tofile("./golden.bin")


if __name__ == "__main__":
    case_params_list = [
        TGatherParamsMasked("TGATHERTest.case1_float_P0101",
                            np.float32, np.float32, FLOAT_P0101_ROW, FLOAT_P0101_COL, P0101),
        TGatherParamsMasked("TGATHERTest.case1_float_P1010",
                            np.float32, np.float32, FLOAT_P1010_ROW, FLOAT_P1010_COL, P1010),
        TGatherParamsMasked("TGATHERTest.case1_float_P0001",
                            np.float32, np.float32, FLOAT_P0001_ROW, FLOAT_P0001_COL, P0001),
        TGatherParamsMasked("TGATHERTest.case1_float_P0010",
                            np.float32, np.float32, FLOAT_P0010_ROW, FLOAT_P0010_COL, P0010),
        TGatherParamsMasked("TGATHERTest.case1_float_P0100",
                            np.float32, np.float32, FLOAT_P0100_ROW, FLOAT_P0100_COL, P0100),
        TGatherParamsMasked("TGATHERTest.case1_float_P1000",
                            np.float32, np.float32, FLOAT_P1000_ROW, FLOAT_P1000_COL, P1000),
        TGatherParamsMasked("TGATHERTest.case1_float_P1111",
                            np.float32, np.float32, FLOAT_P1111_ROW, FLOAT_P1111_COL, P1111),
        TGatherParamsMasked("TGATHERTest.case1_half_P0101", np.half,
                            np.half, HALF_P0101_ROW, HALF_P0101_COL, P0101),
        TGatherParamsMasked("TGATHERTest.case1_half_P1010", np.half,
                            np.half, HALF_P1010_ROW, HALF_P1010_COL, P1010),
        TGatherParamsMasked("TGATHERTest.case1_half_P0001", np.half,
                            np.half, HALF_P0001_ROW, HALF_P0001_COL, P0001),
        TGatherParamsMasked("TGATHERTest.case1_half_P0100", np.half,
                            np.half, HALF_P0100_ROW, HALF_P0100_COL, P0100),
        TGatherParamsMasked("TGATHERTest.case1_half_P1000", np.half,
                            np.half, HALF_P1000_ROW, HALF_P1000_COL, P1000),

        TGatherParamsMasked("TGATHERTest.case1_U16_P0101",
                            np.uint16, np.uint16, HALF_P0101_ROW, HALF_P0101_COL, P0101),
        TGatherParamsMasked("TGATHERTest.case1_U16_P1010",
                            np.uint16, np.uint16, HALF_P1010_ROW, HALF_P1010_COL, P1010),
        TGatherParamsMasked("TGATHERTest.case1_I16_P0001",
                            np.int16, np.int16, HALF_P0001_ROW, HALF_P0001_COL, P0001),
        TGatherParamsMasked("TGATHERTest.case1_I16_P0010",
                            np.int16, np.int16, HALF_P0010_ROW, HALF_P0010_COL, P0010),
        TGatherParamsMasked("TGATHERTest.case1_U32_P0100",
                            np.uint32, np.uint32, FLOAT_P0100_ROW, FLOAT_P0100_COL, P0100),
        TGatherParamsMasked("TGATHERTest.case1_I32_P1000",
                            np.int32, np.int32, FLOAT_P1000_ROW, FLOAT_P1000_COL, P1000),
        TGatherParamsMasked("TGATHERTest.case1_I32_P1111",
                            np.int32, np.int32, FLOAT_P1111_ROW, FLOAT_P1111_COL, P1111),

        TGatherParamsColMasked("TGATHERTest.case_col_float_P0101",
                               np.float32, np.float32, COL_HALF_P0101_ROW, COL_HALF_P0101_COL, P0101),
        TGatherParamsColMasked("TGATHERTest.case_col_float_P1010",
                               np.float32, np.float32, COL_HALF_P1010_ROW, COL_HALF_P1010_COL, P1010),
        TGatherParamsColMasked("TGATHERTest.case_col_float_P0001",
                               np.float32, np.float32, COL_HALF_P0001_ROW, COL_HALF_P0001_COL, P0001),
        TGatherParamsColMasked("TGATHERTest.case_col_float_P0010",
                               np.float32, np.float32, COL_HALF_P0010_ROW, COL_HALF_P0010_COL, P0010),
        TGatherParamsColMasked("TGATHERTest.case_col_float_P0100",
                               np.float32, np.float32, COL_HALF_P0100_ROW, COL_HALF_P0100_COL, P0100),
        TGatherParamsColMasked("TGATHERTest.case_col_float_P1000",
                               np.float32, np.float32, COL_HALF_P1000_ROW, COL_HALF_P1000_COL, P1000),
        TGatherParamsColMasked("TGATHERTest.case_col_float_P1111",
                               np.float32, np.float32, COL_HALF_P1111_ROW, COL_HALF_P1111_COL, P1111),

        TGatherParamsColMasked("TGATHERTest.case_col_half_P0101",
                               np.half, np.half, COL_HALF_P0101_ROW, COL_HALF_P0101_COL, P0101),
        TGatherParamsColMasked("TGATHERTest.case_col_half_P1010",
                               np.half, np.half, COL_HALF_P1010_ROW, COL_HALF_P1010_COL, P1010),
        TGatherParamsColMasked("TGATHERTest.case_col_half_P0001",
                               np.half, np.half, COL_HALF_P0001_ROW, COL_HALF_P0001_COL, P0001),
        TGatherParamsColMasked("TGATHERTest.case_col_half_P0010",
                               np.half, np.half, COL_HALF_P0010_ROW, COL_HALF_P0010_COL, P0010),
        TGatherParamsColMasked("TGATHERTest.case_col_half_P0100",
                               np.half, np.half, COL_HALF_P0100_ROW, COL_HALF_P0100_COL, P0100),
        TGatherParamsColMasked("TGATHERTest.case_col_half_P1000",
                               np.half, np.half, COL_HALF_P1000_ROW, COL_HALF_P1000_COL, P1000),
        TGatherParamsColMasked("TGATHERTest.case_col_half_P1111",
                               np.half, np.half, COL_HALF_P1111_ROW, COL_HALF_P1111_COL, P1111),

        TGatherParamsMasked("TGATHERTest.case1_i8_P0101",
                            np.int8, np.int8, B8_P0101_ROW, B8_P0101_COL, P0101),
        TGatherParamsMasked("TGATHERTest.case1_i8_P1010",
                            np.int8, np.int8, B8_P1010_ROW, B8_P1010_COL, P1010),
        TGatherParamsMasked("TGATHERTest.case1_i8_P0001",
                            np.int8, np.int8, B8_P0001_ROW, B8_P0001_COL, P0001),
        TGatherParamsMasked("TGATHERTest.case1_i8_P0010",
                            np.int8, np.int8, B8_P0010_ROW, B8_P0010_COL, P0010),
        TGatherParamsMasked("TGATHERTest.case1_i8_P0100",
                            np.int8, np.int8, B8_P0100_ROW, B8_P0100_COL, P0100),
        TGatherParamsMasked("TGATHERTest.case1_i8_P1000",
                            np.int8, np.int8, B8_P1000_ROW, B8_P1000_COL, P1000),
        TGatherParamsMasked("TGATHERTest.case1_i8_P1111",
                            np.int8, np.int8, B8_P1111_ROW, B8_P1111_COL, P1111),

        TGatherParamsMasked("TGATHERTest.case1_u8_P0101",
                            np.uint8, np.uint8, B8_P0101_ROW, B8_P0101_COL, P0101),
        TGatherParamsMasked("TGATHERTest.case1_u8_P1010",
                            np.uint8, np.uint8, B8_P1010_ROW, B8_P1010_COL, P1010),
        TGatherParamsMasked("TGATHERTest.case1_u8_P0001",
                            np.uint8, np.uint8, B8_P0001_ROW, B8_P0001_COL, P0001),
        TGatherParamsMasked("TGATHERTest.case1_u8_P0010",
                            np.uint8, np.uint8, B8_P0010_ROW, B8_P0010_COL, P0010),
        TGatherParamsMasked("TGATHERTest.case1_u8_P0100",
                            np.uint8, np.uint8, B8_P0100_ROW, B8_P0100_COL, P0100),
        TGatherParamsMasked("TGATHERTest.case1_u8_P1000",
                            np.uint8, np.uint8, B8_P1000_ROW, B8_P1000_COL, P1000),
        TGatherParamsMasked("TGATHERTest.case1_u8_P1111",
                            np.uint8, np.uint8, B8_P1111_ROW, B8_P1111_COL, P1111),

        # Test cases for Tgather1D
        TGatherParams1D("TGATHERTest.case_1D_float_32x1024_16x64",
                        np.float32, 32, 1024, 16, 64),
        TGatherParams1D("TGATHERTest.case_1D_int32_32x512_16x256",
                        np.int32, 32, 512, 16, 256),
        TGatherParams1D("TGATHERTest.case_1D_half_16x1024_16x128",
                        np.float16, 16, 1024, 16, 128),
        TGatherParams1D("TGATHERTest.case_1D_int16_32x256_32x64",
                        np.int16, 32, 256, 32, 64),
        TGatherParams1D("TGATHERTest.case_1D_int8_16x1024_16x128",
                        np.int8, 16, 1024, 16, 128),
        TGatherParams1D("TGATHERTest.case_1D_uint8_32x256_32x64",
                        np.uint8, 32, 256, 32, 64),
    ]

    for case in case_params_list:
        if not os.path.exists(case.testname):
            os.makedirs(case.testname)
        original_dir = os.getcwd()
        os.chdir(case.testname)
        gen_golden_data(case)
        os.chdir(original_dir)
