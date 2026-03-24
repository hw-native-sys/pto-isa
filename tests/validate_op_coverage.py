#!/usr/bin/env python3
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------
"""
Check for missing test operations in run_st.sh.

This script compares test directories (containing main.cpp) against the operations
referenced in run_st.sh to identify missing test coverage.

Usage:
    python3 tests/validate_op_coverage.py

Exit codes:
    0 - All ops are covered
    1 - Some ops are missing
"""

import os
import re
import sys
from pathlib import Path


# Mapping from version flag to test directory path
VERSION_PATHS = {
    "a3": "tests/npu/a2a3/src/st/testcase",
    "a5": "tests/npu/a5/src/st/testcase",
    "kirin9030": "tests/npu/kirin9030/src/st/testcase",
}


def get_test_directories(base_path: str) -> set:
    """
    Get all test directory names that contain a main.cpp file.

    Args:
        base_path: Path to the testcase directory

    Returns:
        Set of test directory names
    """
    test_dirs = set()
    testcase_path = Path(base_path)

    if not testcase_path.exists():
        return test_dirs

    for item in testcase_path.iterdir():
        if item.is_dir() or item.is_symlink():
            # Check if main.cpp exists (follow symlinks)
            main_cpp = item / "main.cpp"
            if item.is_symlink():
                resolved = item.resolve()
                main_cpp = resolved / "main.cpp"
            if main_cpp.exists():
                test_dirs.add(item.name)

    return test_dirs


def get_ops_in_script(script_path: str, version: str) -> set:
    """
    Extract all operation names referenced in a script for a specific version.

    Args:
        script_path: Path to the script file (e.g., run_st.sh)
        version: Version flag to filter by (e.g., "a3", "a5")

    Returns:
        Set of operation names
    """
    content = Path(script_path).read_text()
    # Pattern: -v <version> -t <opname>
    pattern = rf"-v\s+{re.escape(version)}\s+-t\s+(\S+)"
    return set(re.findall(pattern, content))


def check_missing_ops(script_path: str = "tests/run_st.sh") -> dict:
    """
    Check for missing operations in the script for all versions.

    Args:
        script_path: Path to the script to check

    Returns:
        Dictionary mapping version to set of missing operation names
    """
    missing_by_version = {}

    for version, base_path in VERSION_PATHS.items():
        test_dirs = get_test_directories(base_path)
        ops_in_script = get_ops_in_script(script_path, version)
        missing = test_dirs - ops_in_script

        if missing:
            missing_by_version[version] = sorted(missing)

    return missing_by_version


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)  # Ensure we're in project root

    print("=" * 80)
    print("Checking for missing test operations in run_st.sh")
    print("=" * 80)

    missing = check_missing_ops("tests/run_st.sh")

    if not missing:
        print("\nAll operations are covered!")
        return 0

    total_missing = sum(len(ops) for ops in missing.values())
    print(f"\nFound {total_missing} missing operations across {len(missing)} version(s)")

    for version, ops in sorted(missing.items()):
        print(f"\n### {version.upper()} ({len(ops)} missing) ###")
        for op in ops:
            print(f"  - {op}")

    return 1


if __name__ == "__main__":
    sys.exit(main())
