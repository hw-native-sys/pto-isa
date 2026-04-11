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

import torch
from torch_npu.testing.testcase import TestCase, run_tests

import op_extension  # noqa: F401


class TestCustomAdd(TestCase):
    def test_add_custom_ops(self):
        # Define the tensor size
        length = [20, 2048]
        # Create random input tensors on CPU with float16 data type
        x = torch.rand(length, device="cpu", dtype=torch.float16)
        y = torch.rand(length, device="cpu", dtype=torch.float16)

        x_npu = x.npu()
        y_npu = y.npu()
        # Call the custom my_add operator
        output = torch.ops.npu.my_add(x_npu, y_npu)
        # Compute the expected result using standard addition
        cpuout = torch.add(x, y)

        # Validate the results
        self.assertRtolEqual(output, cpuout)


if __name__ == "__main__":
    run_tests()
