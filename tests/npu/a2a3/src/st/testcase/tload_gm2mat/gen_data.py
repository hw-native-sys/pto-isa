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
import struct
import ctypes
import numpy as np

np.random.seed(19)

def gen_golden_data(case_name, gInfo):
    data_type = gInfo.datatype
    gShape0 = gInfo.gShape0
    gShape1 = gInfo.gShape1
    gShape2 = gInfo.gShape2
    gShape3 = gInfo.gShape3
    gShape4 = gInfo.gShape4
    gWholeShape0 = gInfo.gWholeShape0
    gWholeShape1 = gInfo.gWholeShape1
    gWholeShape2 = gInfo.gWholeShape2
    gWholeShape3 = gInfo.gWholeShape3
    gWholeShape4 = gInfo.gWholeShape4
    consecutive_formats = {"ND", "NZ", "NC1HWC02NC1HWC0", "FZ2FZ", "FZ4D2FZ4D"}
    if gInfo.format in consecutive_formats:
        input_arr = np.random.randint(-5, 5, size=(gWholeShape0, gWholeShape1,
                                    gWholeShape2, gWholeShape3, gWholeShape4)).astype(data_type)
        output_arr = np.zeros(shape=(gShape0, gShape1, gShape2, gShape3, gShape4), dtype=data_type)
        output_arr = input_arr[0:gShape0, 0:gShape1, 0:gShape2, 0:gShape3, 0:gShape4]
    elif gInfo.format == "NDC1HWC02NDC1HWC0":
        c0_size = 32 // np.dtype(data_type).itemsize
        input_arr = np.random.randint(-5, 5, size=(gWholeShape0, gWholeShape1,
                                    gWholeShape2, gWholeShape3, gWholeShape4, c0_size)).astype(data_type)
        output_arr = np.zeros(shape=(gShape0, gShape1, gShape2, gShape3, gShape4, c0_size), dtype=data_type)
        output_arr = input_arr[0:gShape0, 0:gShape1, 0:gShape2, 0:gShape3, 0:gShape4, :]
    elif gInfo.format == "DN":
        input_arr = np.random.randint(-5, 5, size=(gWholeShape0, gWholeShape1,
                                    gWholeShape2, gWholeShape4, gWholeShape3)).astype(data_type)
        output_arr = np.zeros(shape=(gShape0, gShape1, gShape2, gShape4, gShape3), dtype=data_type)
        output_arr = input_arr[0:gShape0, 0:gShape1, 0:gShape2, 0:gShape4, 0:gShape3]
    elif gInfo.format == "ND2NZ":
        input_arr = np.random.randint(-5, 5, size=(gWholeShape0, gWholeShape1,
                                    gWholeShape2, gWholeShape3, gWholeShape4)).astype(data_type)
        c0_size = 32 // np.dtype(data_type).itemsize
        gShape4Align = (gShape4 + c0_size - 1) // c0_size * c0_size
        output_arr = np.zeros(
            shape=(gShape0, gShape1, gShape2, gShape3, gShape4Align), dtype=data_type)
        output_arr[0:gShape0, 0:gShape1, 0:gShape2, 0:gShape3,
                   0:gShape4] = input_arr[0:gShape0, 0:gShape1, 0:gShape2, 0:gShape3, 0:gShape4]
        output_arr = output_arr.reshape(gShape0, gShape1, gShape2, gShape3,
            gShape4Align // c0_size, c0_size).transpose(4, 0, 1, 2, 3, 5)
    elif gInfo.format == "DN2ZN":
        input_arr = np.random.randint(-5, 5, size=(gWholeShape0, gWholeShape1,
                                    gWholeShape2, gWholeShape4, gWholeShape3)).astype(data_type)
        c0_size = 32 // np.dtype(data_type).itemsize
        gShape3Align = (gShape3 + c0_size - 1) // c0_size * c0_size
        output_arr = np.zeros(
            shape=(gShape0, gShape1, gShape2, gShape4, gShape3Align), dtype=data_type)
        output_arr[0:gShape0, 0:gShape1, 0:gShape2, 0:gShape4,
                   0:gShape3] = input_arr[0:gShape0, 0:gShape1, 0:gShape2, 0:gShape4, 0:gShape3]
        output_arr = output_arr.reshape(gShape0, gShape1, gShape2, gShape4,
            gShape3Align // c0_size, c0_size).transpose(0, 1, 2, 4, 3, 5)

    input_arr.tofile("./input.bin")
    output_arr.tofile("./golden.bin")

class GlobalTensorInfo:
    def __init__(self, datatype, format, gShape0, gShape1, gShape2, gShape3, gShape4,
                 gWholeShape0, gWholeShape1, gWholeShape2, gWholeShape3, gWholeShape4):
        self.datatype = datatype
        # 0:ND2ND, 1:DN2DN, 2:NZ2NZ, 3:ND2NZ, 4:DN2ZN 5:NC1HWC02NC1HWC0 6:FZ2FZ
        self.format = format
        self.gShape0 = gShape0
        self.gShape1 = gShape1
        self.gShape2 = gShape2
        self.gShape3 = gShape3
        self.gShape4 = gShape4
        self.gWholeShape0 = gWholeShape0
        self.gWholeShape1 = gWholeShape1
        self.gWholeShape2 = gWholeShape2
        self.gWholeShape3 = gWholeShape3
        self.gWholeShape4 = gWholeShape4

if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TLoadGM2L1Test.ND_float_1_1_1_3_128_3_3_3_32_128",
        "TLoadGM2L1Test.ND_int16_t_2_2_1_2_32_3_3_3_111_64",
        "TLoadGM2L1Test.ND_int8_t_1_2_1_11_32_1_3_2_93_32",
        "TLoadGM2L1Test.ND_int8_t_1_1_1_1_201_1_1_1_1_201",
        "TLoadGM2L1Test.DN_float_1_1_1_128_3_3_3_3_128_32",
        "TLoadGM2L1Test.DN_int16_t_2_2_1_32_2_3_3_3_64_111",
        "TLoadGM2L1Test.DN_int8_t_1_2_1_32_11_1_3_2_32_93",
        "TLoadGM2L1Test.DN_float_1_1_1_156_1_1_1_1_156_1",
        "TLoadGM2L1Test.NZ_float_1_5_21_16_8_1_5_21_16_8",
        "TLoadGM2L1Test.NZ_int16_t_2_16_11_16_16_3_23_13_16_16",
        "TLoadGM2L1Test.NZ_int8_t_1_16_32_16_32_1_32_32_16_32",
        "TLoadGM2L1Test.ND2NZ_float_t_1_1_1_49_35_1_1_1_49_35",
        "TLoadGM2L1Test.ND2NZ_int16_t_1_1_1_155_250_1_1_1_752_1000",
        "TLoadGM2L1Test.ND2NZ_int8_t_1_1_1_1023_511_1_1_1_1024_1024",
        "TLoadGM2L1Test.ND2NZ_bfloat16_t_1_1_1_1023_51_1_1_1_1024_1024",
        "TLoadGM2L1Test.ND_bfloat16_t_1_1_1_128_128_1_1_1_256_256",
        "TLoadGM2L1Test.DN_bfloat16_t_1_2_2_128_311_4_3_3_256_400",
        "TLoadGM2L1Test.NZ_bfloat16_t_2_4_5_16_16_7_7_7_16_16",
        "TLoadGM2L1Test.ND2NZ_bfloat16_t_1_1_1_1_1_1_1_1_1_1",
        "TLoadGM2L1Test.ND2NZ_bfloat16_t_1_1_1_1_1_1_1_1_16_16",
        "TLoadGM2L1Test.ND2NZ_bfloat16_t_1_1_1_256_1024_1_1_1_256_1024",
        "TLoadGM2L1Test.ND_int64_1_1_1_3_128_3_3_3_32_128",
        "TLoadGM2L1Test.ND_uint64_2_2_1_2_32_3_3_3_111_64",
        "TLoadGM2L1Test.ND_int64_1_2_1_11_32_1_3_2_93_32",
        "TLoadGM2L1Test.DN_uint64_1_1_1_128_3_3_3_3_128_32",
        "TLoadGM2L1Test.DN_int64_2_2_1_32_2_3_3_3_64_111",
        "TLoadGM2L1Test.DN_uint64_1_2_1_32_11_1_3_2_32_93",
        "TLoadGM2L1Test.DN2ZN_bfloat16_t_1_1_1_256_1024_1_1_1_256_1024",
        "TLoadGM2L1Test.DN2ZN_float_t_1_1_1_49_35_1_1_1_49_35",
        "TLoadGM2L1Test.DN2ZN_int16_t_1_1_1_155_250_1_1_1_752_1000",
        "TLoadGM2L1Test.DN2ZN_int8_t_1_1_1_1023_511_1_1_1_1024_1024",

        "TLoadGM2L1Test.NC1HWC02NC1HWC0_int8_t_2_3_16_128_32_3_4_1024_1024_32", # cut N H
        "TLoadGM2L1Test.NC1HWC02NC1HWC0_int8_t_3_4_128_8_32_3_4_128_128_32", # cut W
        "TLoadGM2L1Test.NC1HWC02NC1HWC0_int8_t_3_4_8_128_32_3_8_8_128_32", # cut C1
        "TLoadGM2L1Test.NC1HWC02NC1HWC0_bfloat16_1_16_10_100_16_1_16_100_100_16", # cut H
        "TLoadGM2L1Test.NC1HWC02NC1HWC0_bfloat16_10_16_16_2_16_256_16_100_16_16", # cut N C1 W
        "TLoadGM2L1Test.NC1HWC02NC1HWC0_bfloat16_1_1_1_8192_16_8_16_16_8192_16", # cut N C1 H
        "TLoadGM2L1Test.NC1HWC02NC1HWC0_float_1_1_112_112_8_2_3_224_224_8", # cut N C1 H W
        
        "TLoadGM2L1Test.FZ2FZ_bfloat16_1_7_7_20_16_3_7_7_100_16", # cut N C1
        "TLoadGM2L1Test.FZ2FZ_bfloat16_128_7_7_2_16_256_7_7_16_16", # cut N C1
        "TLoadGM2L1Test.FZ2FZ_bfloat16_192_3_3_8_16_256_3_3_8_16", # cut C1
        "TLoadGM2L1Test.FZ2FZ_int8_t_2_3_3_64_32_3_3_3_128_32", # cut N C1
        "TLoadGM2L1Test.FZ2FZ_int8_t_3_5_5_128_32_8_5_5_128_32", # cut N
        "TLoadGM2L1Test.FZ2FZ_float_96_7_7_2_8_256_7_7_256_8", # cut C1 N

        "TLoadGM2L1Test.FZ4D2FZ4D_bfloat16_1_49_7_16_16_1_980_32_16_16", # cut C1HW N
        "TLoadGM2L1Test.FZ4D2FZ4D_bfloat16_1_81_3_16_16_1_90_3_16_16", # cut C1HW
        "TLoadGM2L1Test.FZ4D2FZ4D_int8_t_1_63_3_16_32_1_63_9_16_32", # cut N
        "TLoadGM2L1Test.FZ4D2FZ4D_int8_t_1_125_3_16_32_1_250_5_16_32", # cut C1HW N
        "TLoadGM2L1Test.FZ4D2FZ4D_float_1_256_3_16_8_1_4704_7_16_8", # cut C1HW N

        "TLoadGM2L1Test.NDC1HWC02NDC1HWC0_int8_t_1_2_3_16_128_2_3_4_1024_1024", # cut N H
        "TLoadGM2L1Test.NDC1HWC02NDC1HWC0_int8_t_1_3_4_128_8_2_3_4_128_128", # cut W
        "TLoadGM2L1Test.NDC1HWC02NDC1HWC0_int8_t_1_3_4_8_128_2_3_8_8_128", # cut C1
        "TLoadGM2L1Test.NDC1HWC02NDC1HWC0_bfloat16_1_1_16_10_100_2_1_16_100_100", # cut H
        "TLoadGM2L1Test.NDC1HWC02NDC1HWC0_bfloat16_1_1_10_16_2_2_2_256_16_100", # cut N D C1 W
        "TLoadGM2L1Test.NDC1HWC02NDC1HWC0_bfloat16_1_1_1_1_8192_2_8_16_16_8192", # cut N D C1 H
        "TLoadGM2L1Test.NDC1HWC02NDC1HWC0_float_1_1_1_112_112_2_2_3_224_224", # cut N D C1 H W
    ]

    case_params_list = [
        GlobalTensorInfo(np.float32, "ND", 1, 1, 1, 3, 128, 3, 3, 3, 32, 128),
        GlobalTensorInfo(np.int16, "ND", 2, 2, 1, 2, 32, 3, 3, 3, 111, 64),
        GlobalTensorInfo(np.int8, "ND", 1, 2, 1, 11, 32, 1, 3, 2, 93, 32),
        GlobalTensorInfo(np.int8, "ND", 1, 1, 1, 1, 201, 1, 1, 1, 1, 201),
        GlobalTensorInfo(np.float32, "DN", 1, 1, 1, 128, 3, 3, 3, 3, 128, 32),
        GlobalTensorInfo(np.int16, "DN", 2, 2, 1, 32, 2, 3, 3, 3, 64, 111),
        GlobalTensorInfo(np.int8, "DN", 1, 2, 1, 32, 11, 1, 3, 2, 32, 93),
        GlobalTensorInfo(np.float32, "DN", 1, 1, 1, 156, 1, 1, 1, 1, 156, 1),
        GlobalTensorInfo(np.float32, "NZ", 1, 5, 21, 16, 8, 1, 5, 21, 16, 8),
        GlobalTensorInfo(np.int16, "NZ", 2, 15, 11, 16, 16, 3, 23, 13, 16, 16),
        GlobalTensorInfo(np.int8, "NZ", 1, 16, 32, 16, 32, 1, 32, 32, 16, 32),
        GlobalTensorInfo(np.float32, "ND2NZ", 1, 1, 1, 49, 35, 1, 1, 1, 49, 35),
        GlobalTensorInfo(np.int16, "ND2NZ", 1, 1, 1, 155, 250, 1, 1, 1, 752, 1000),
        GlobalTensorInfo(np.int8, "ND2NZ", 1, 1, 1, 1023, 511, 1, 1, 1, 1024, 1024),
        GlobalTensorInfo(np.float16, "ND2NZ", 1, 1, 1, 1023, 51, 1, 1, 1, 1024, 1024),
        GlobalTensorInfo(np.float16, "ND", 1, 1, 1, 128, 128, 1, 1, 1, 256, 256),
        GlobalTensorInfo(np.float16, "DN", 1, 2, 2, 128, 311, 4, 3, 3, 256, 400),
        GlobalTensorInfo(np.float16, "NZ", 2, 4, 5, 16, 16, 7, 7, 7, 16, 16),
        GlobalTensorInfo(np.float16, "ND2NZ", 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        GlobalTensorInfo(np.float16, "ND2NZ", 1, 1, 1, 1, 1, 1, 1, 1, 16, 16),
        GlobalTensorInfo(np.float16, "ND2NZ", 1, 1, 1, 256, 1024, 1, 1, 1, 256, 1024),
        GlobalTensorInfo(np.int64, "ND", 1, 1, 1, 3, 128, 3, 3, 3, 32, 128),
        GlobalTensorInfo(np.uint64, "ND", 2, 2, 1, 2, 32, 3, 3, 3, 111, 64),
        GlobalTensorInfo(np.int64, "ND", 1, 2, 1, 11, 32, 1, 3, 2, 93, 32),
        GlobalTensorInfo(np.uint64, "DN", 1, 1, 1, 128, 3, 3, 3, 3, 128, 32),
        GlobalTensorInfo(np.int64, "DN", 2, 2, 1, 32, 2, 3, 3, 3, 64, 111),
        GlobalTensorInfo(np.uint64, "DN", 1, 2, 1, 32, 11, 1, 3, 2, 32, 93),
        GlobalTensorInfo(np.float16, "DN2ZN", 1, 1, 1, 256, 1024, 1, 1, 1, 256, 1024),
        GlobalTensorInfo(np.float32, "DN2ZN", 1, 1, 1, 49, 35, 1, 1, 1, 49, 35),
        GlobalTensorInfo(np.int16, "DN2ZN", 1, 1, 1, 155, 250, 1, 1, 1, 752, 1000),
        GlobalTensorInfo(np.int8, "DN2ZN", 1, 1, 1, 1023, 511, 1, 1, 1, 1024, 1024),

        GlobalTensorInfo(np.int8, "NC1HWC02NC1HWC0", 2, 3, 16, 128, 32, 3, 4, 1024, 1024, 32),
        GlobalTensorInfo(np.int8, "NC1HWC02NC1HWC0", 3, 4, 128, 8, 32, 3, 4, 128, 128, 32),
        GlobalTensorInfo(np.int8, "NC1HWC02NC1HWC0", 3, 4, 8, 128, 32, 3, 8, 8, 128, 32),
        GlobalTensorInfo(np.float16, "NC1HWC02NC1HWC0", 1, 16, 10, 100, 16, 1, 16, 100, 100, 16),
        GlobalTensorInfo(np.float16, "NC1HWC02NC1HWC0", 10, 16, 16, 2, 16, 256, 16, 100, 16, 16),
        GlobalTensorInfo(np.float16, "NC1HWC02NC1HWC0", 1, 1, 1, 8192, 16, 8, 16, 16, 8192, 16),
        GlobalTensorInfo(np.float32, "NC1HWC02NC1HWC0", 1, 1, 112, 112, 8, 2, 3, 224, 224, 8),

        GlobalTensorInfo(np.float16, "FZ2FZ", 1, 7, 7, 20, 16, 3, 7, 7, 100, 16),
        GlobalTensorInfo(np.float16, "FZ2FZ", 128, 7, 7, 2, 16, 256, 7, 7, 16, 16),
        GlobalTensorInfo(np.float16, "FZ2FZ", 192, 3, 3, 8, 16, 256, 3, 3, 8, 16),
        GlobalTensorInfo(np.int8, "FZ2FZ", 2, 3, 3, 64, 32, 3, 3, 3, 128, 32),
        GlobalTensorInfo(np.int8, "FZ2FZ", 3, 5, 5, 128, 32, 8, 5, 5, 128, 32),
        GlobalTensorInfo(np.float32, "FZ2FZ", 96, 7, 7, 2, 8, 256, 7, 7, 256, 8),

        GlobalTensorInfo(np.float16, "FZ4D2FZ4D", 1, 49, 7, 16, 16, 1, 980, 32, 16, 16),
        GlobalTensorInfo(np.float16, "FZ4D2FZ4D", 1, 81, 3, 16, 16, 1, 90, 3, 16, 16),
        GlobalTensorInfo(np.int8, "FZ4D2FZ4D", 1, 63, 3, 16, 32, 1, 63, 9, 16, 32),
        GlobalTensorInfo(np.int8, "FZ4D2FZ4D", 1, 125, 3, 16, 32, 1, 250, 5, 16, 32),
        GlobalTensorInfo(np.float32, "FZ4D2FZ4D", 1, 256, 3, 16, 8, 1, 4704, 7, 16, 8),

        GlobalTensorInfo(np.int8, "NDC1HWC02NDC1HWC0", 1, 2, 3, 16, 128, 2, 3, 4, 1024, 1024),
        GlobalTensorInfo(np.int8, "NDC1HWC02NDC1HWC0", 1, 3, 4, 128, 8, 2, 3, 4, 128, 128),
        GlobalTensorInfo(np.int8, "NDC1HWC02NDC1HWC0", 1, 3, 4, 8, 128, 2, 3, 8, 8, 128),
        GlobalTensorInfo(np.float16, "NDC1HWC02NDC1HWC0", 1, 1, 16, 10, 100, 2, 1, 16, 100, 100),
        GlobalTensorInfo(np.float16, "NDC1HWC02NDC1HWC0", 1, 1, 10, 16, 2, 2, 2, 256, 16, 100),
        GlobalTensorInfo(np.float16, "NDC1HWC02NDC1HWC0", 1, 1, 1, 1, 8192, 2, 8, 16, 16, 8192),
        GlobalTensorInfo(np.float32, "NDC1HWC02NDC1HWC0", 1, 1, 1, 112, 112, 2, 2, 3, 224, 224),

    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)
