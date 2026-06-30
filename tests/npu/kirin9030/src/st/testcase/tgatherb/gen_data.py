#!/usr/bin/python3
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


def gen_golden_data_tgatherb(case_name, param):
    dtype = param.dtype
    data_size = 1
    if dtype == np.float32 or dtype == np.int32 or dtype == np.uint32:
        data_size = 4
    elif dtype == np.float16 or dtype == np.int16 or dtype == np.uint16:
        data_size = 2
    elif dtype == np.int8 or dtype == np.uint8:
        data_size = 1
    else:
        ValueError(f"{dtype} unsupported data type!!")
    block_size_elem = int(32 / data_size)

    src_shape = [param.src_s1, param.src_s0]
    dst_shape = [param.dst_s1, param.dst_s0]
    offset_col = int(param.dst_s0 / block_size_elem)
    offset_shape = [param.dst_s1, offset_col]
    offset_elt_num = param.dst_s1 * offset_col
    dst_elt_num = param.dst_s1 * param.dst_s0

    src = np.arange(param.src_s1 * param.src_s0).astype(dtype)
    offset = np.zeros(offset_elt_num)
    for i in range(offset_elt_num):
        offset[i] = i * 32
    offset = offset.astype(np.uint32)

    golden = np.zeros(dst_elt_num).astype(dtype)
    output = np.zeros(dst_elt_num).astype(dtype)
    count = 0
    for i in range(offset_elt_num):
        for j in range(int(32 / data_size)):
            golden[count] = src[int(offset[i] / data_size + j)]
            count += 1
    golden.reshape((dst_elt_num)).astype(np.uint32)

    src.tofile(str("x.bin"))
    offset.tofile(str("offset.bin"))
    golden.tofile(str("golden.bin"))

    src_addr = 0x0
    return output, src, offset, src_addr, golden


class TGatherBParam:
    def __init__(self, dtype, dst_s1, dst_s0, offset_s1, offset_s0, src_s1, src_s0):
        self.dtype = dtype
        self.dst_s1 = dst_s1
        self.dst_s0 = dst_s0
        self.offset_s1 = offset_s1
        self.offset_s0 = offset_s0
        self.src_s1 = src_s1
        self.src_s0 = src_s0


def generate_case_name(param):
    dtype_str = {
        np.float32: "float",
        np.int32: "int32",
        np.uint32: "uint32",
        np.int16: "int16",
        np.uint16: "uint16",
        np.float16: "half",
        np.int8: "int8",
        np.uint8: "uint8",
    }[param.dtype]
    return (
        f"TGATHERBTest.case_{dtype_str}_{param.dst_s1}x{param.dst_s0}_{param.offset_s1}x{param.offset_s0}_"
        f"{param.src_s1}x{param.src_s0}"
    )


if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TGatherBParam(np.float32, 2, 128, 2, 16, 2, 128),
        TGatherBParam(np.int32, 2, 128, 2, 16, 2, 128),
        TGatherBParam(np.uint32, 2, 128, 2, 16, 2, 128),
        TGatherBParam(np.int16, 1, 16384, 1, 2048, 1, 16384),
        TGatherBParam(np.uint16, 129, 128, 129, 8, 129, 128),
        TGatherBParam(np.float16, 1, 16384, 1, 2048, 1, 16384),
        TGatherBParam(np.int8, 2, 256, 2, 8, 2, 256),
    ]

    for _, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tgatherb(case_name, param)
        os.chdir(original_dir)
