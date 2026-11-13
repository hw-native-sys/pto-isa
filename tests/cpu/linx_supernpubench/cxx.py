#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Preserve C++ main semantics while exposing its symbol to the PE launcher."""

import os
import subprocess
import sys

args = sys.argv[1:]
code = subprocess.call([os.environ["LINX_HOST_CXX"], *args])
if code == 0 and "-c" in args and "-o" in args:
    code = subprocess.call(
        [
            os.environ.get("LINX_HOST_OBJCOPY", "objcopy"),
            "--redefine-sym",
            "main=supernpubench_main",
            args[args.index("-o") + 1],
        ]
    )
raise SystemExit(code)
