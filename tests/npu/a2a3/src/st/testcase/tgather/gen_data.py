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

# should match those in tgather_common.h
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


class TGatherParamsBase:
    def __init__(self, name):
        self.testname = name


class TGatherParam(TGatherParamsBase):
    def __init__(self, name, dst_type, src_type, row, col, pattern):
        super().__init__(name)
        self.dst_type = dst_type
        self.src_type = src_type
        self.row = row
        self.col = col
        self.pattern = pattern


class TGatherParams1D(TGatherParamsBase):
    def __init__(self, name, src_type, src_row, src_col, dst_row, dst_col):
        super().__init__(name)
        self.src_type = src_type
        self.src_row = src_row
        self.src_col = src_col
        self.dst_row = dst_row
        self.dst_col = dst_col


class TGatherParamsCmp(TGatherParamsBase):
    def __init__(self, name, src_type, dst_type, src_row, src_col, kvalue, i_offset, k, cmpmode=0):
        super().__init__(name)
        self.src_type = src_type
        self.dst_type = dst_type
        self.src_row = src_row
        self.src_col = src_col
        self.kvalue = kvalue
        self.i_offset = i_offset
        self.k = k
        self.cmpmode = cmpmode  # 0 for gt, 1 for eq


def gather_1dim(src, indices):
    output = np.zeros_like(indices, dtype=src.dtype)
    for i in range(indices.shape[0]):
        output[i] = src[indices[i]]
    return output


def gen_golden_data(param: TGatherParamsBase):
    if isinstance(param, TGatherParam):
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
        output = np.zeros([param.dst_row * param.dst_col]).astype(param.src_type)
        src_data = np.random.randint(-20, 20, (param.src_row * param.src_col)).astype(param.src_type)
        src_data.tofile("./src0.bin")
        indices = np.random.randint(0, param.src_row * param.src_col, (param.dst_row * param.dst_col)).astype(np.int32)
        indices.tofile("./src1.bin")
        golden = gather_1dim(src_data, indices)
        golden.tofile("./golden.bin")
    else:
        if param.cmpmode == 0:
            assert param.kvalue > 50, (
                "k-value is not supposed to be too small to make sure golden will not exceed src data"
            )
        src_type = param.src_type
        dst_type = param.dst_type
        src_row = param.src_row
        src_col = param.src_col
        dst_row = param.src_row
        dst_col = param.k
        kvalue = param.kvalue
        i_offset = param.i_offset
        cmpmode = param.cmpmode
        src_data = np.random.randint(0, 100, [src_row, src_col]).astype(src_type)
        golden = np.zeros((dst_row, dst_col)).astype(dst_type)
        # 0x7F800001转float比较时为nan，保证尾块对比通过
        golden[:dst_row][:dst_col] = 0x7F800001
        if cmpmode == 0:
            for i in range(src_row):
                k = 0
                for j in range(src_col):
                    idx = i * src_col + j
                    if src_data[i, j] > kvalue:
                        golden[i, k] = idx
                        k = k + 1
        elif cmpmode == 1:
            for i in range(src_row):
                k = 0
                for j in range(src_col):
                    idx = i * src_col + j
                    if src_data[i, j] == kvalue:
                        golden[i, k] = idx
                        k = k + 1
        else:
            assert False, "not implemented"

        src_data1 = np.array(kvalue).astype(src_type)

        src_data.tofile("./src.bin")
        src_data1.tofile("./src1.bin")
        golden.tofile("./golden.bin")
        os.chdir(original_dir)


if __name__ == "__main__":
    case_params_list = [
        TGatherParam("TGATHERTest.case1_float_P0101", np.float32, np.float32, FLOAT_P0101_ROW, FLOAT_P0101_COL, P0101),
        TGatherParam("TGATHERTest.case1_float_P1010", np.float32, np.float32, FLOAT_P1010_ROW, FLOAT_P1010_COL, P1010),
        TGatherParam("TGATHERTest.case1_float_P0001", np.float32, np.float32, FLOAT_P0001_ROW, FLOAT_P0001_COL, P0001),
        TGatherParam("TGATHERTest.case1_float_P0010", np.float32, np.float32, FLOAT_P0010_ROW, FLOAT_P0010_COL, P0010),
        TGatherParam("TGATHERTest.case1_float_P0100", np.float32, np.float32, FLOAT_P0100_ROW, FLOAT_P0100_COL, P0100),
        TGatherParam("TGATHERTest.case1_float_P1000", np.float32, np.float32, FLOAT_P1000_ROW, FLOAT_P1000_COL, P1000),
        TGatherParam("TGATHERTest.case1_float_P1111", np.float32, np.float32, FLOAT_P1111_ROW, FLOAT_P1111_COL, P1111),
        TGatherParam("TGATHERTest.case1_half_P0101", np.half, np.half, HALF_P0101_ROW, HALF_P0101_COL, P0101),
        TGatherParam("TGATHERTest.case1_half_P1010", np.half, np.half, HALF_P1010_ROW, HALF_P1010_COL, P1010),
        TGatherParam("TGATHERTest.case1_half_P0001", np.half, np.half, HALF_P0001_ROW, HALF_P0001_COL, P0001),
        TGatherParam("TGATHERTest.case1_half_P0010", np.half, np.half, HALF_P0010_ROW, HALF_P0010_COL, P0010),
        TGatherParam("TGATHERTest.case1_half_P0100", np.half, np.half, HALF_P0100_ROW, HALF_P0100_COL, P0100),
        TGatherParam("TGATHERTest.case1_half_P1000", np.half, np.half, HALF_P1000_ROW, HALF_P1000_COL, P1000),
        TGatherParam("TGATHERTest.case1_half_P1111", np.half, np.half, HALF_P1111_ROW, HALF_P1111_COL, P1111),
        TGatherParam("TGATHERTest.case1_U16_P0101", np.uint16, np.uint16, HALF_P0101_ROW, HALF_P0101_COL, P0101),
        TGatherParam("TGATHERTest.case1_U16_P1010", np.uint16, np.uint16, HALF_P1010_ROW, HALF_P1010_COL, P1010),
        TGatherParam("TGATHERTest.case1_I16_P0001", np.int16, np.int16, HALF_P0001_ROW, HALF_P0001_COL, P0001),
        TGatherParam("TGATHERTest.case1_I16_P0010", np.int16, np.int16, HALF_P0010_ROW, HALF_P0010_COL, P0010),
        TGatherParam("TGATHERTest.case1_U32_P0100", np.uint32, np.uint32, FLOAT_P0100_ROW, FLOAT_P0100_COL, P0100),
        TGatherParam("TGATHERTest.case1_I32_P1000", np.int32, np.int32, FLOAT_P1000_ROW, FLOAT_P1000_COL, P1000),
        TGatherParam("TGATHERTest.case1_I32_P1111", np.int32, np.int32, FLOAT_P1111_ROW, FLOAT_P1111_COL, P1111),
        # Test cases for Tgather1D
        TGatherParams1D("TGATHERTest.case_1D_float_32x1024_16x64", np.float32, 32, 1024, 16, 64),
        TGatherParams1D("TGATHERTest.case_1D_int32_32x512_16x256", np.int32, 32, 512, 16, 256),
        TGatherParams1D("TGATHERTest.case_1D_half_16x1024_16x128", np.float16, 16, 1024, 16, 128),
        TGatherParams1D("TGATHERTest.case_1D_int16_32x256_32x64", np.int16, 32, 256, 32, 64),
        TGatherParams1D("TGATHERTest.case_1D_half_1x16_1x16", np.float16, 1, 16, 1, 16),
        TGatherParams1D("TGATHERTest.case_1D_half_1x32_1x32", np.float16, 1, 32, 1, 32),
        TGatherParams1D("TGATHERTest.case_1D_half_1x64_1x64", np.float16, 1, 64, 1, 64),
        TGatherParams1D("TGATHERTest.case_1D_half_1x128_1x128", np.float16, 1, 128, 1, 128),
        TGatherParams1D("TGATHERTest.case_1D_half_1x128_1x64", np.float16, 1, 128, 1, 64),
        TGatherParams1D("TGATHERTest.case_1D_float_1024x16_1024x16", np.float32, 1024, 16, 1024, 16),
        TGatherParams1D("TGATHERTest.case_1D_float_16x16_32x32", np.float32, 16, 16, 32, 32),
        TGatherParams1D("TGATHERTest.case_1D_half_16x16_32x32", np.float16, 16, 16, 32, 32),
        # Test cases for topk
        TGatherParamsCmp("TGATHERTest.case1_float_topk", np.float32, np.uint32, 16, 64, 80, 0, 32, 0),
        TGatherParamsCmp("TGATHERTest.case2_s32_topk", np.int32, np.uint32, 8, 128, 80, 0, 64, 1),
        TGatherParamsCmp("TGATHERTest.case3_float_topk", np.float32, np.uint32, 4, 256, 30, 0, 64, 1),
        TGatherParamsCmp("TGATHERTest.case4_half_topk", np.half, np.uint32, 2, 256, 100, 0, 32, 0),
        TGatherParamsCmp("TGATHERTest.case5_half_topk", np.half, np.uint32, 8, 128, 40, 0, 32, 1),
    ]

    for case in case_params_list:
        if not os.path.exists(case.testname):
            os.makedirs(case.testname)
        original_dir = os.getcwd()
        os.chdir(case.testname)
        gen_golden_data(case)
        os.chdir(original_dir)
