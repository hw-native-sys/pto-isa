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
        weight_output_format: str = "5d"
    ):
        self.input_shape_nc1hwc0 = input_shape_nc1hwc0
        self.weight_shape = weight_shape
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.dtype = dtype
        self.weight_output_format = weight_output_format

        n, c1, h, w, c0_input = input_shape_nc1hwc0
        if dtype == np.float32:
            dtype_size = 4
        elif dtype == np.int8:
            dtype_size = 1
        else:
            dtype_size = 2
        expected_c0 = 32 // dtype_size
        # Allow C0 values that are divisors of expected_c0 (for testing smaller blocks)
        if c0_input > expected_c0 or (expected_c0 % c0_input != 0 and c0_input % expected_c0 != 0):
            raise ValueError(f"对于{dtype}类型，C0应为{expected_c0}的整数倍或约数，但输入为{c0_input}")


# calculate function
def calculate_output_shape(input_shape, weight_shape, stride=(1, 1), dilation=(1, 1), padding=(0, 0, 0, 0)):
    n, h, w, c_in = input_shape
    c_out, c_in_w, h_k, w_k = weight_shape
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    pad_top, pad_bottom, pad_left, pad_right = padding
    
    if c_in != c_in_w:
        raise ValueError("输入通道数不匹配: 输入有%d个通道，但权重有%d个输入通道" % (c_in, c_in_w))
    
    h_out = (h + pad_top + pad_bottom - dilation_h * (h_k - 1) - 1) // stride_h + 1
    w_out = (w + pad_left + pad_right - dilation_w * (w_k - 1) - 1) // stride_w + 1
    return (n, h_out, w_out, c_out)


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


def nc1hwc0_to_nhwc(input_nc1hwc0, original_c_in):
    n, c1, h, w, c0 = input_nc1hwc0.shape
    output = input_nc1hwc0.transpose(0, 2, 3, 1, 4).reshape(n, h, w, c1 * c0)
    output = output[:, :, :, :original_c_in]
    return output


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
    
    # matmul
    output_flat = np.dot(
        kernel_matrix.astype(np.float32), 
        col_matrix.astype(np.float32)
    )
    # reshape
    output = output_flat.reshape(weight.shape[0], n, h_out, w_out).transpose(1, 2, 3, 0)
    return output, col_matrix, kernel_matrix


def conv2d_matmul_nhwc_int8(input_data, weight, stride=(1, 1), dilation=(1, 1), padding=(0, 0, 0, 0)):
    """
    Implement int8 convolution in NHWC format via matrix multiplication.
    Return: (output feature map [N, H_out, W_out, C_out], col_matrix, kernel_matrix)
    """
    h_k, w_k = weight.shape[2], weight.shape[3]

    col_matrix, (n, h_out, w_out) = img2col_nhwc(input_data, (h_k, w_k), stride, dilation, padding)
    kernel_matrix = kernel2matrix_new(weight)
    # matmuls
    output_flat = np.dot(
        kernel_matrix.astype(np.int32), 
        col_matrix.astype(np.int32)
    )
    # reshape
    output = output_flat.reshape(weight.shape[0], n, h_out, w_out).transpose(1, 2, 3, 0)
    return output, col_matrix, kernel_matrix


def save_matrix_bin(matrix, filepath):
    dirname = os.path.dirname(filepath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(filepath, 'wb') as f:
        matrix.flatten().tofile(f)
    return filepath


def gen_golden_data(case_name: str, params: ConvTestParams):
    #input
    n, c1_input, h, w, c0_input = params.input_shape_nc1hwc0
    c_in = c1_input * c0_input
    
    # weight
    c1_weight, h_k, w_k, n_out, c0_weight = params.weight_shape
    dtype = params.dtype
    
    if c1_input != c1_weight or c0_input != c0_weight:
        raise ValueError(f"输入通道分块不匹配: 输入有(C1={c1_input}, C0={c0_input})，但权重有(C1={c1_weight}, C0={c0_weight})")
    
    # 1. Generate an input tensor (in NC1HWC0 format) using the provided data type.
    if dtype == np.int8:
        input_nc1hwc0 = np.random.randint(-128, 128, size=params.input_shape_nc1hwc0, dtype=np.int8)
    else:
        input_nc1hwc0 = np.random.uniform(-5, 5, size=params.input_shape_nc1hwc0).astype(dtype)
    input_nhwc_path_bin = "x1_gm.bin"
    save_matrix_bin(input_nc1hwc0, input_nhwc_path_bin)
    
    # 2. Generate a weight tensor using the provided data type.
    if dtype == np.int8:
        weight = np.random.randint(-128, 128, size=params.weight_shape, dtype=np.int8)
    else:
        weight = np.random.uniform(-5, 5, size=params.weight_shape).astype(dtype)

    if params.weight_output_format == "5d":
        weight_to_save = weight
    else:
        if n_out % 16 != 0:
            raise ValueError(f"4维权重格式要求N({n_out})必须是16的倍数")
        weight_reshaped = weight.reshape(c1_weight * h_k * w_k, n_out, c0_weight)
        weight_to_save = weight_reshaped.reshape(c1_weight * h_k * w_k, n_out // 16, 16, c0_weight)
    
    weight_path_bin = "x2_gm.bin"
    save_matrix_bin(weight_to_save, weight_path_bin)

    # 3. Convert the input from NC1HWC0 format to NHWC format for convolution computation.
    input_nhwc_temp = input_nc1hwc0.transpose(0, 2, 3, 1, 4)
    input_nhwc = input_nhwc_temp.reshape(n, h, w, c_in)
    # 4. Perform the convolution computation.
    # Convert the weights from [C1, H_k, W_k, N, C0] to [N, C1*C0, H_k, W_k] for computation.
    weight_for_calc = weight.transpose(3, 0, 4, 1, 2).reshape(n_out, c1_weight * c0_weight, h_k, w_k)
    output_shape = calculate_output_shape(
        (n, h, w, c_in), (n_out, c1_weight * c0_weight, h_k, w_k), 
        params.stride, params.dilation, 
        params.padding
    )
    n_out_calc, h_out, w_out, c_out_calc = output_shape
    if dtype == np.int8:
        output_nhwc, col_matrix, kernel_matrix = conv2d_matmul_nhwc_int8(
            input_nhwc, weight_for_calc, 
            params.stride, params.dilation, 
            params.padding
        )
    else:
        output_nhwc, col_matrix, kernel_matrix = conv2d_matmul_nhwc_float(
            input_nhwc, weight_for_calc, 
            params.stride, params.dilation, 
            params.padding
        )
    # 5. Convert the output to NC1HWC0 format.
    output_nc1hwc0, c1_out = nhwc_to_nc1hwc0(output_nhwc, c0_input)
    output_nc1hwc0_path_bin = "golden_NC1HWC0.bin"
    save_matrix_bin(output_nc1hwc0, output_nc1hwc0_path_bin)
    # 6. Compute and save the 2D matrix (in M×N format).
    if dtype == np.int8:
        output_2d = np.dot(
            kernel_matrix.astype(np.int32), 
            col_matrix.astype(np.int32)
        )
    else:
        output_2d = np.dot(
            kernel_matrix.astype(np.float32), 
            col_matrix.astype(np.float32)
        )
    # Transpose the matrix to the [M, N_out_ch] format.
    output_2d_transposed = output_2d.T
    output_2d_path_bin = "golden.bin"
    save_matrix_bin(output_2d_transposed, output_2d_path_bin)

if __name__ == "__main__":
    # Define a list of test cases.
    case_name_list = [
        "TIMG2COLTest.case2_float16", 
        "TIMG2COLTest.case3_float16", 
        "TIMG2COLTest.case4_int8", 
        "TIMG2COLTest.case6_float16_splitk", 
        "TIMG2COLTest.case7_float16_splitk",
        "TIMG2COLTest.case8_int8_splitk",
        "TIMG2COLTest.case10_float16_fractalZ4d", 
        "TIMG2COLTest.case11_float16_fractalZ4d",
        "TIMG2COLTest.case12_int8_fractalZ4d",
    ]
    # Define the parameters for the test cases.
    case_params_list = [
        ConvTestParams(
            input_shape_nc1hwc0=(1, 4, 3, 8, 16),  # NC1HWC0
            weight_shape=(4, 3, 3, 16, 16),  # C1HWNC0
            stride=(1, 1),
            dilation=(2, 1),
            padding=(1, 1, 1, 1),
            dtype=np.float16
        ),
        ConvTestParams(
            input_shape_nc1hwc0=(1, 4, 8, 16, 8),  # NC1HWC0
            weight_shape=(4, 3, 3, 16, 8),  # C1HWNC0
            stride=(2, 2),
            dilation=(1, 1),
            padding=(1, 1, 1, 1),
            dtype=np.float16
        ),
        ConvTestParams(
            input_shape_nc1hwc0=(1, 1, 4, 8, 32),  # NC1HWC0
            weight_shape=(1, 3, 3, 16, 32),  # C1HWNC0
            stride=(1, 1),
            dilation=(1, 1),
            padding=(1, 1, 1, 1),
            dtype=np.int8
        ),
        ConvTestParams(
            input_shape_nc1hwc0=(1, 4, 17, 5, 16),  # NC1HWC0
            weight_shape=(4, 3, 3, 16, 16),  # C1HWNC0
            stride=(2, 1),
            dilation=(1, 2),
            padding=(1, 1, 1, 1),
            dtype=np.float16
        ),
        ConvTestParams(
            input_shape_nc1hwc0=(1, 2, 10, 12, 8),  # NC1HWC0
            weight_shape=(2, 4, 4, 16, 8),  # C1HWNC0
            stride=(2, 2),
            dilation=(1, 1),
            padding=(1, 2, 3, 0),
            dtype=np.float16
        ),
        ConvTestParams(
            input_shape_nc1hwc0=(1, 2, 9, 20, 32),  # NC1HWC0
            weight_shape=(2, 2, 2, 16, 32),  # C1HWNC0
            stride=(2, 2),
            dilation=(2, 2),
            padding=(1, 1, 1, 0),
            dtype=np.int8
        ),
        ConvTestParams(
            input_shape_nc1hwc0=(1, 4, 17, 5, 16),  # NC1HWC0
            weight_shape=(4, 3, 3, 64, 16),  # C1HWNC0
            stride=(2, 1),
            dilation=(1, 2),
            padding=(1, 1, 1, 1),
            dtype=np.float16,
            weight_output_format="4d"
        ),
        ConvTestParams(
            input_shape_nc1hwc0=(1, 2, 10, 12, 8),  # NC1HWC0
            weight_shape=(2, 4, 4, 32, 8),  # C1HWNC0
            stride=(2, 2),
            dilation=(1, 1),
            padding=(1, 2, 3, 0),
            dtype=np.float16,
            weight_output_format="4d"
        ),
        ConvTestParams(
            input_shape_nc1hwc0=(1, 2, 9, 20, 32),  # NC1HWC0
            weight_shape=(2, 2, 2, 64, 32),  # C1HWNC0
            stride=(2, 2),
            dilation=(2, 2),
            padding=(1, 1, 1, 0),
            dtype=np.int8,
            weight_output_format="4d"
        ),
    ]
    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)