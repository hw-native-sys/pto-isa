# --------------------------------------------------------------------------------
# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import os
from setuptools import setup, find_packages


def get_include_files():
    include_dir = "include"
    data_files = []

    for root, _, files in os.walk(include_dir):
        if files:
            install_dir = os.path.join("share/pto-isa", root)
            file_paths = [os.path.join(root, f) for f in files]
            data_files.append((install_dir, file_paths))

    return data_files

setup(
    name="pto-isa",
    version="9.1.0",
    description="PTO Tile Library - A tile-based programming library for AI Core",
    author="Huawei Technologies Co., Ltd.",
    license="CANN Open Software License Agreement Version 2.0",
    packages=find_packages(exclude=["tests*", "demos*", "scripts*", "kernels*"]),
    data_files=get_include_files(),
    include_package_data=True,
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Topic :: Software Development :: Libraries",
    ],
    keywords=["ascend", "npu", "tile", "programming", "ai-core"],
)
