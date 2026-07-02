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
from enum import Enum

np.random.seed(19)
np.set_printoptions(threshold=np.inf)


class DataFormat(Enum):
    ND2NZ = 1
    ND2ND = 3
    NZ2NZ = 4
    NC1HWC02NC1HWC0 = 7
    FZ2FZ = 8
    FZ4D2FZ4D = 9


def gen_golden_data(case_name, param):
    src_type = param.atype
    shape0 = param.shape0
    shape1 = param.shape1
    shape2 = param.shape2
    shape3 = param.m
    shape4 = param.k
    whole_shape0 = param.ws0
    whole_shape1 = param.ws1
    whole_shape2 = param.ws2
    whole_shape3 = param.ws3
    whole_shape4 = param.ws4
    convtile_formats = {DataFormat["NC1HWC02NC1HWC0"].value, DataFormat["FZ2FZ"].value, DataFormat["FZ4D2FZ4D"].value}

    m, k, basem, basek = param.m, param.k, param.basem, param.basek
    c0_size = 16
    if src_type == np.float32:
        c0_size = 8
    elif src_type == np.int8 or src_type == np.uint8:
        c0_size = 32

    x1_gm = np.random.randint(1, 5, [m, k]).astype(src_type)
    golden = np.zeros([basem, basek]).astype(src_type)

    if param.load_type == DataFormat["ND2NZ"].value:
        x1_gm = np.random.randint(1, 5, [whole_shape3, whole_shape4]).astype(src_type)
        golden = np.zeros([basem, basek]).astype(src_type)  # L1中Tile大小
        min_m = min(m, golden.shape[0])
        min_k = min(k, golden.shape[1])
        golden[:min_m, :min_k] = x1_gm[:min_m, :min_k]
    elif param.load_type == DataFormat["ND2ND"].value:
        x1_gm = np.random.randint(1, 5, [whole_shape0, whole_shape1, whole_shape2, whole_shape3, whole_shape4]).astype(
            src_type
        )
        golden = np.zeros([basem, basek]).astype(src_type)  # L1中Tile大小

        submatrix = x1_gm[
            0:shape0,  # d0: 截取第shape0个元素（对应 shape[0]=1）
            0:shape1,  # d1: 截取前shape1个元素（对应目标 d1=2）
            0:shape2,  # d2: 截取前shape2个元素（对应目标 d2=3）
            0:m,  # d3: 截取前M个元素（对应目标 d3=64）
            0:k,  # d4: 截取K个元素（对应目标 d4=128）
        ]
        flattened_submatrix = submatrix.reshape(basem, k)
        min_m = min(flattened_submatrix.shape[0], golden.shape[0])
        min_k = min(flattened_submatrix.shape[1], golden.shape[1])
        golden[:min_m, :min_k] = flattened_submatrix[:min_m, :min_k]
    elif param.load_type == DataFormat["NZ2NZ"].value:
        x1_gm = np.random.randint(1, 5, [whole_shape0, whole_shape1, whole_shape2, whole_shape3, whole_shape4]).astype(
            src_type
        )

        submatrix = x1_gm[
            0:shape0,  # d0: 截取第shape0个元素（对应 shape[0]=1）
            0:shape1,  # d1: 截取前shape1个元素（对应目标 d1=2）
            0:shape2,  # d2: 截取前shape2个元素（对应目标 d2=4）
            0:m,  # d3: 截取前M个元素（对应目标 d3=16）
            0:k,  # d4: 截取K个元素（对应目标 d4=8）
        ]
        new_submatrix = submatrix.reshape(
            submatrix.shape[0] * submatrix.shape[1], submatrix.shape[2], submatrix.shape[3], submatrix.shape[4]
        )

        golden = np.zeros([basem, basek]).astype(src_type)  # L1中Tile大小 [80,48]
        c0Size = 16
        if src_type == np.float32:
            c0Size = 8
        elif src_type == np.int8 or src_type == np.uint8:
            c0Size = 32
        assert (basek % c0Size) == 0, "BASEK should be c0Size aligned when matrix is NZ format"
        assert (basem % 16) == 0, "BASEM should be 16 aligned when matrix is NZ format"
        golden = (
            golden.reshape((int(basem / 16), 16, int(basek / c0Size), c0Size)).transpose(2, 0, 1, 3).astype(src_type)
        )  # [80,48] -> [6,5,16,8]

        golden[
            : new_submatrix.shape[0], : new_submatrix.shape[1], : new_submatrix.shape[2], : new_submatrix.shape[3]
        ] = new_submatrix
    elif param.load_type in convtile_formats:
        x1_gm = np.random.randint(
            -5, 5, size=(whole_shape0, whole_shape1, whole_shape2, whole_shape3, whole_shape4)
        ).astype(src_type)
        golden = np.zeros(shape=(shape0, shape1, shape2, shape3, shape4), dtype=src_type)
        golden = x1_gm[0:shape0, 0:shape1, 0:shape2, 0:shape3, 0:shape4]

    x2_gm = np.random.randint(1, 5, [m, k]).astype(src_type)
    if param.load_type == DataFormat["ND2NZ"].value:
        assert (basem % 16) == 0, "BASEM should be 16 aligned when matrix A is NZ format"
        assert (basek % c0_size) == 0, "BASEK should be c0_size aligned when matrix A is NZ format"
        golden = (
            golden.reshape((int(basem / 16), 16, int(basek / c0_size), c0_size)).transpose(2, 0, 1, 3).astype(src_type)
        )

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")
    golden.tofile("./golden.bin")


class TloadParams:
    def __init__(self, atype, shape0, shape1, shape2, m, k, ws0, ws1, ws2, ws3, ws4, basem, basek, load_type):
        self.atype = atype
        self.m = m
        self.k = k
        self.shape0 = shape0
        self.shape1 = shape1
        self.shape2 = shape2

        self.ws0 = ws0
        self.ws1 = ws1
        self.ws2 = ws2
        self.ws3 = ws3
        self.ws4 = ws4

        self.basem = basem  # L1 row
        self.basek = basek  # L1 col
        self.load_type = load_type


if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TLOADMIXTest.1_1_1_128_128_half_ND2NZ",
        "TLOADMIXTest.1_1_1_128_128_int8_t_ND2NZ",
        "TLOADMIXTest.1_1_1_128_128_float_ND2NZ",
        "TLOADMIXTest.1_1_1_63_127_half_ND2NZ",
        "TLOADMIXTest.1_1_1_128_128_float_ND2ND",
        "TLOADMIXTest.1_1_1_37_126_int8_t_ND2ND",
        "TLOADMIXTest.1_2_3_64_128_1_3_4_128_128_384_128_half_ND2ND",
        "TLOADMIXTest.1_2_3_33_99_1_2_3_33_99_int8_t_ND2ND",
        "TLOADMIXTest.1_1_1_33_99_1_1_1_64_128_48_112_half_ND2NZ",
        "TLOADMIXTest.1_1_1_59_119_1_1_1_64_128_64_128_int8_t_ND2NZ",
        "TLOADMIXTest.2_2_4_16_8_2_2_4_16_8_80_48_float_NZ2NZ",
        "TLOADMIXTest.1_10_8_16_16_1_11_9_16_16_128_160_half_NZ2NZ",
        "TLOADMIXTest.1_8_4_16_32_1_9_4_16_32_80_256_int8_t_NZ2NZ",
        "TLOADMIXTest.1_1_1_59_119_1_1_1_59_124_59_120_int64_t_ND2ND",
        "TLOADMIXTest.1_2_1_64_128_1_3_4_128_128_128_128_uint64_t_ND2ND",
        "TLOADMIXTest.NC1HWC02NC1HWC0_int8_t_1_3_16_128_32_3_4_1024_1024_32",  # cut N H
        "TLOADMIXTest.NC1HWC02NC1HWC0_int8_t_3_2_128_8_32_3_2_128_128_32",  # cut W
        "TLOADMIXTest.NC1HWC02NC1HWC0_int8_t_3_2_8_128_32_3_8_8_128_32",  # cut C1
        "TLOADMIXTest.NC1HWC02NC1HWC0_float16_1_6_10_100_16_1_6_100_100_16",  # cut H
        "TLOADMIXTest.NC1HWC02NC1HWC0_float16_10_16_16_2_16_256_16_100_16_16",  # cut N C1 W
        "TLOADMIXTest.NC1HWC02NC1HWC0_float16_1_1_1_8192_16_8_16_16_8192_16",  # cut N C1 H
        "TLOADMIXTest.NC1HWC02NC1HWC0_float_1_1_56_112_8_2_3_224_224_8",  # cut N C1 H W
        "TLOADMIXTest.FZ2FZ_float16_1_7_7_20_16_3_7_7_100_16",  # cut N C1
        "TLOADMIXTest.FZ2FZ_float16_64_7_7_2_16_256_7_7_16_16",  # cut N C1
        "TLOADMIXTest.FZ2FZ_float16_96_3_3_8_16_256_3_3_8_16",  # cut C1
        "TLOADMIXTest.FZ2FZ_int8_t_1_3_3_64_32_3_3_3_128_32",  # cut N C1
        "TLOADMIXTest.FZ2FZ_int8_t_8_5_5_32_32_8_5_5_128_32",  # cut N
        "TLOADMIXTest.FZ2FZ_float_70_7_7_2_8_256_7_7_256_8",  # cut C1 N
        "TLOADMIXTest.FZ4D2FZ4D_float16_1_49_7_16_16_1_980_32_16_16",  # cut C1HW N
        "TLOADMIXTest.FZ4D2FZ4D_float16_1_81_3_16_16_1_90_3_16_16",  # cut C1HW
        "TLOADMIXTest.FZ4D2FZ4D_int8_t_1_63_3_16_32_1_63_9_16_32",  # cut N
        "TLOADMIXTest.FZ4D2FZ4D_int8_t_1_125_3_16_32_1_250_5_16_32",  # cut C1HW N
        "TLOADMIXTest.FZ4D2FZ4D_float_1_126_3_16_8_1_4704_7_16_8",  # cut C1HW N
    ]

    case_params_list = [
        TloadParams(np.float16, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128, DataFormat["ND2NZ"].value),
        TloadParams(np.int8, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128, DataFormat["ND2NZ"].value),
        TloadParams(np.float32, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128, DataFormat["ND2NZ"].value),
        TloadParams(np.float16, 1, 1, 1, 63, 127, 1, 1, 1, 63, 127, 64, 128, DataFormat["ND2NZ"].value),
        TloadParams(np.float32, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128, DataFormat["ND2ND"].value),
        TloadParams(np.int8, 1, 1, 1, 37, 126, 1, 1, 1, 37, 126, 37, 128, DataFormat["ND2ND"].value),
        TloadParams(np.float16, 1, 2, 3, 64, 128, 1, 3, 4, 128, 128, 384, 128, DataFormat["ND2ND"].value),
        TloadParams(np.int8, 1, 2, 3, 33, 99, 1, 2, 3, 33, 99, 198, 128, DataFormat["ND2ND"].value),
        TloadParams(np.float16, 1, 1, 1, 33, 99, 1, 1, 1, 64, 128, 48, 112, DataFormat["ND2NZ"].value),
        TloadParams(np.int8, 1, 1, 1, 59, 119, 1, 1, 1, 64, 128, 64, 128, DataFormat["ND2NZ"].value),
        TloadParams(np.float32, 2, 2, 4, 16, 8, 2, 2, 4, 16, 8, 80, 48, DataFormat["NZ2NZ"].value),
        TloadParams(np.float16, 1, 10, 8, 16, 16, 1, 11, 9, 16, 16, 128, 160, DataFormat["NZ2NZ"].value),
        TloadParams(np.int8, 1, 8, 4, 16, 32, 1, 9, 4, 16, 32, 80, 256, DataFormat["NZ2NZ"].value),
        TloadParams(np.int64, 1, 1, 1, 59, 119, 1, 1, 1, 59, 124, 59, 120, DataFormat["ND2ND"].value),
        TloadParams(np.uint64, 1, 2, 1, 64, 128, 1, 3, 4, 128, 128, 128, 128, DataFormat["ND2ND"].value),
        TloadParams(np.int8, 1, 3, 16, 128, 32, 3, 4, 1024, 1024, 32, 1, 1, DataFormat["NC1HWC02NC1HWC0"].value),
        TloadParams(np.int8, 3, 2, 128, 8, 32, 3, 2, 128, 128, 32, 1, 1, DataFormat["NC1HWC02NC1HWC0"].value),
        TloadParams(np.int8, 3, 2, 8, 128, 32, 3, 8, 8, 128, 32, 1, 1, DataFormat["NC1HWC02NC1HWC0"].value),
        TloadParams(np.float16, 1, 6, 10, 100, 16, 1, 6, 100, 100, 16, 1, 1, DataFormat["NC1HWC02NC1HWC0"].value),
        TloadParams(np.float16, 10, 16, 16, 2, 16, 256, 16, 100, 16, 16, 1, 1, DataFormat["NC1HWC02NC1HWC0"].value),
        TloadParams(np.float16, 1, 1, 1, 8192, 16, 8, 16, 16, 8192, 16, 1, 1, DataFormat["NC1HWC02NC1HWC0"].value),
        TloadParams(np.float32, 1, 1, 56, 112, 8, 2, 3, 224, 224, 8, 1, 1, DataFormat["NC1HWC02NC1HWC0"].value),
        TloadParams(np.float16, 1, 7, 7, 20, 16, 3, 7, 7, 100, 16, 1, 1, DataFormat["FZ2FZ"].value),
        TloadParams(np.float16, 64, 7, 7, 2, 16, 256, 7, 7, 16, 16, 1, 1, DataFormat["FZ2FZ"].value),
        TloadParams(np.float16, 96, 3, 3, 8, 16, 256, 3, 3, 8, 16, 1, 1, DataFormat["FZ2FZ"].value),
        TloadParams(np.int8, 2, 3, 3, 64, 32, 3, 3, 3, 128, 32, 1, 1, DataFormat["FZ2FZ"].value),
        TloadParams(np.int8, 8, 5, 5, 32, 32, 8, 5, 5, 128, 32, 1, 1, DataFormat["FZ2FZ"].value),
        TloadParams(np.float32, 70, 7, 7, 2, 8, 256, 7, 7, 256, 8, 1, 1, DataFormat["FZ2FZ"].value),
        TloadParams(np.float16, 1, 49, 7, 16, 16, 1, 980, 32, 16, 16, 1, 1, DataFormat["FZ4D2FZ4D"].value),
        TloadParams(np.float16, 1, 81, 3, 16, 16, 1, 90, 3, 16, 16, 1, 1, DataFormat["FZ4D2FZ4D"].value),
        TloadParams(np.int8, 1, 63, 3, 16, 32, 1, 63, 9, 16, 32, 1, 1, DataFormat["FZ4D2FZ4D"].value),
        TloadParams(np.int8, 1, 125, 3, 16, 32, 1, 250, 5, 16, 32, 1, 1, DataFormat["FZ4D2FZ4D"].value),
        TloadParams(np.float32, 1, 126, 3, 16, 8, 1, 4704, 7, 16, 8, 1, 1, DataFormat["FZ4D2FZ4D"].value),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden_data(case_name, case_params_list[i])

        os.chdir(original_dir)
