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

from typing import Tuple
import numpy as np
np.random.seed(19)


class ConvTestParams:
    def __init__(
        self,
        input_shape_nc1hwc0: Tuple[int, int, int, int, int],
        weight_shape: Tuple[int, int, int, int, int],  # (C1, H_k, W_k, N, C0)
        stride: Tuple[int, int],  # (stride_h, stride_w)
        dilation: Tuple[int, int],  # (dilation_h, dilation_w)
        padding: Tuple[int, int, int, int],  # (top, bottom, left, right)
        dtype: type = np.float32,
    ):
        self.input_shape_nc1hwc0 = input_shape_nc1hwc0
        self.weight_shape = weight_shape
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.dtype = dtype


def nhwc_to_nc1hwc0(input_nhwc, c0=16):
    n, h, w, c_in = input_nhwc.shape
    c1 = (c_in + c0 - 1) // c0
    if c_in % c0 != 0:
        pad_size = c1 * c0 - c_in
        input_padded = np.pad(
            input_nhwc,
            pad_width=((0, 0), (0, 0), (0, 0), (0, pad_size)),
            mode='constant',
            constant_values=0
        )
    else:
        input_padded = input_nhwc
    output = input_padded.reshape(n, h, w, c1, c0).transpose(0, 3, 1, 2, 4)
    return output, c1


def img2col_nhwc(input_data, kernel_size, stride=(1, 1), dilation=(1, 1), padding=(0, 0, 0, 0)):
    """
    Translate the input feature map in NHWC format to an img2col matrix.
    output shape : [C_in*H_k*W_k, N*H_out*W_out]
    """
    n, h, w, c_in = input_data.shape
    h_k, w_k = kernel_size
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    pad_top, pad_bottom, pad_left, pad_right = padding

    h_out = (h + pad_top + pad_bottom - dilation_h * (h_k - 1) - 1) // stride_h + 1
    w_out = (w + pad_left + pad_right - dilation_w * (w_k - 1) - 1) // stride_w + 1

    #padding
    input_padded = np.pad(
        input_data,
        pad_width=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode='constant',
        constant_values=0
    )
    col_matrix = np.zeros((c_in * h_k * w_k, n * h_out * w_out), dtype=input_data.dtype)
    # img2col
    for n_idx in range(n):
        for h_out_idx in range(h_out):
            for w_out_idx in range(w_out):
                col_idx = n_idx * h_out * w_out + h_out_idx * w_out + w_out_idx
                col = np.zeros((c_in, h_k, w_k), dtype=input_data.dtype)

                for hk in range(h_k):
                    for wk in range(w_k):
                        h_in = h_out_idx * stride_h + hk * dilation_h
                        w_in = w_out_idx * stride_w + wk * dilation_w
                        if 0 <= h_in < (h + pad_top + pad_bottom) and 0 <= w_in < (w + pad_left + pad_right):
                            col[:, hk, wk] = input_padded[n_idx, h_in, w_in, :]
                col_matrix[:, col_idx] = col.flatten()
    return col_matrix, (n, h_out, w_out)


def kernel2matrix_new(weight):
    """
    Convert the convolutional kernel into a matrix.
    Output shape: [C_out, C_in*H_k*W_k]
    """
    c_out, c_in, h_k, w_k = weight.shape
    return weight.reshape(c_out, c_in * h_k * w_k)


def conv2d_matmul_nhwc_float(input_data, weight, stride=(1, 1), dilation=(1, 1), padding=(0, 0, 0, 0)):
    """
    Implement floating-point convolution in NHWC format via matrix multiplication.
    Return: (output feature map [N, H_out, W_out, C_out], col_matrix, kernel_matrix)
    """
    h_k, w_k = weight.shape[2], weight.shape[3]

    col_matrix, (n, h_out, w_out) = img2col_nhwc(input_data, (h_k, w_k), stride, dilation, padding)
    kernel_matrix = kernel2matrix_new(weight)

    output_flat = np.dot(
        kernel_matrix.astype(np.float32),
        col_matrix.astype(np.float32)
    )
    output = output_flat.reshape(weight.shape[0], n, h_out, w_out).transpose(1, 2, 3, 0)
    return output, col_matrix, kernel_matrix


def gen_golden_data(params: ConvTestParams):
    # input
    n, c1_input, h, w, c0_input = params.input_shape_nc1hwc0
    c_in = c1_input * c0_input

    # weight
    c1_weight, h_k, w_k, n_out, c0_weight = params.weight_shape
    dtype = params.dtype

    # 1. Generate an input tensor (in NC1HWC0 format) using the provided data type.
    input_nc1hwc0 = np.random.uniform(-5, 5, size=params.input_shape_nc1hwc0).astype(dtype)

    # 2. Generate a weight tensor using the provided data type.
    weight = np.random.uniform(-5, 5, size=params.weight_shape).astype(dtype)
    if n_out % 16 != 0:
        raise ValueError(f"The 4D weight format requires that N({n_out}) must be a multiple of 16.")
    weight_reshaped = weight.reshape(c1_weight * h_k * w_k, n_out, c0_weight)
    weight_to_save = weight_reshaped.reshape(c1_weight * h_k * w_k, n_out // 16, 16, c0_weight)
    # 3. Convert the input from NC1HWC0 format to NHWC format for convolution computation.
    input_nhwc_temp = input_nc1hwc0.transpose(0, 2, 3, 1, 4)
    input_nhwc = input_nhwc_temp.reshape(n, h, w, c_in)
    # 4. Perform the convolution computation.
    # Convert the weights from [C1, H_k, W_k, N, C0] to [N, C1*C0, H_k, W_k] for computation.
    weight_for_calc = weight.transpose(3, 0, 4, 1, 2).reshape(n_out, c1_weight * c0_weight, h_k, w_k)

    output_nhwc, col_matrix, kernel_matrix = conv2d_matmul_nhwc_float(
        input_nhwc, weight_for_calc,
        params.stride, params.dilation,
        params.padding
    )
    # 5. Convert the output to NC1HWC0 format.
    output_nc1hwc0, c1_out = nhwc_to_nc1hwc0(output_nhwc, c0_input)
    output_nc1hwc0 = output_nc1hwc0.astype(np.float16)

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    input_nc1hwc0.tofile("./input/x1_gm.bin")
    weight_to_save.tofile("./input/x2_gm.bin")
    output_nc1hwc0.tofile("./output/golden.bin")

if __name__ == "__main__":
    # Define a list of test cases.
    case_name_list = [
        ConvTestParams(
            input_shape_nc1hwc0=(4, 32, 16, 96, 16),  # NC1HWC0
            weight_shape=(32, 3, 3, 6144, 16),  # C1HWNC0
            stride=(1, 1),
            dilation=(1, 1),
            padding=(1, 1, 1, 1),
            dtype=np.float16,
        ),
    ]
    gen_golden_data(case_name_list[0])
