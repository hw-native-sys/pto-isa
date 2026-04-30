#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the
# terms and conditions of CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance
# with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY,
# OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import os

DEVICE_ENV_VAR = "PTO_TEST_DEVICE_ID"
DEFAULT_DEVICE_ID = "0"
DEVICE_PREFIX = "npu:"


def get_test_device() -> str:
    device_id = os.getenv(DEVICE_ENV_VAR)
    if not device_id:
        print(
            f"Warning: {DEVICE_ENV_VAR} is not set; defaulting to {DEFAULT_DEVICE_ID}."
        )
        device_id = DEFAULT_DEVICE_ID

    if device_id.startswith(DEVICE_PREFIX):
        return device_id
    return f"{DEVICE_PREFIX}{device_id}"
