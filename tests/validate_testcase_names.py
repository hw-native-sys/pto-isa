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
Validate test case names in run_st.sh and run_pipeline.sh against main.cpp definitions.

This script ensures that test case names referenced in shell scripts actually exist
in the corresponding main.cpp files. It handles C preprocessor macros including:
  - Simple macros: #define CASENAME TADD_TDIV
  - CONCAT macros: CONCAT(CASENAME, Test) -> TADD_TDIVTest
  - Token pasting: case_##type_name##_1x128 -> case_fp32_fp16_1x128
  - Macro generators: GENERATE_TCVT_TESTS(dst, src, name)

Usage:
    python3 tests/validate_testcase_names.py

Exit codes:
    0 - All test case names are valid
    1 - Some test case names are invalid or missing
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# =============================================================================
# Configuration
# =============================================================================

# Mapping from version flag to test directory path
VERSION_PATHS: Dict[str, str] = {
    "a3": "tests/npu/a2a3/src/st/testcase",
    "a5": "tests/npu/a5/src/st/testcase",
    "kirin9030": "tests/npu/kirin9030/src/st/testcase",
}

# Scripts to validate
SCRIPTS_TO_CHECK: List[str] = ["tests/run_st.sh", "tests/run_pipeline.sh"]


# =============================================================================
# Path Resolution
# =============================================================================


def get_test_dir(version: str, testname: str) -> Optional[Path]:
    """
    Get the test directory path for a given version and test name.

    Follows symlinks to resolve the actual directory location.

    Args:
        version: Version flag (e.g., "a3", "a5", "kirin9030")
        testname: Test operation name (e.g., "trowsum", "tadd")

    Returns:
        Path to the test directory, or None if version is unknown
    """
    base = VERSION_PATHS.get(version)
    if base is None:
        return None

    test_dir = Path(base) / testname

    # Follow symlinks
    if test_dir.is_symlink():
        test_dir = test_dir.resolve()

    return test_dir


# =============================================================================
# C Preprocessor Macro Handling
# =============================================================================


class MacroExpander:
    """
    Handles C preprocessor macro expansion for test case extraction.

    Supports:
        - Simple object-like macros: #define CASENAME TADD_TDIV
        - CONCAT function-like macro: CONCAT(a, b) -> ab
        - Token pasting operator: a##b -> ab
        - Function-like macro generators that produce TEST_F calls
    """

    def __init__(self, content: str):
        """
        Initialize the macro expander with source content.

        Args:
            content: The C++ source file content
        """
        self.content = content
        self.simple_macros: Dict[str, str] = {}
        self.macro_generators: Dict[str, Dict] = {}

        self._extract_simple_macros()
        self._extract_macro_generators()

    def _extract_simple_macros(self) -> None:
        """Extract simple object-like macro definitions (#define NAME value)."""
        pattern = r"#define\s+(\w+)\s+(\w+)"
        for match in re.finditer(pattern, self.content):
            self.simple_macros[match.group(1)] = match.group(2)

    def _extract_macro_generators(self) -> None:
        """Extract function-like macros that generate TEST_F calls."""
        # Pattern matches multi-line macro definitions with line continuation
        pattern = r"#define\s+(\w+)\s*\(([^)]*)\)\s*\\?\s*\n((?:.*\\\n)*.*?)(?=#define|\n\n|$)"

        for match in re.finditer(pattern, self.content, re.MULTILINE):
            macro_name = match.group(1)
            args_str = match.group(2)
            body = match.group(3)

            if "TEST_F" in body:
                arg_names = [a.strip() for a in args_str.split(",")]
                self.macro_generators[macro_name] = {"args": arg_names, "body": body}

    def expand_concat(self, text: str) -> str:
        """
        Expand CONCAT(a, b) macro calls to concatenated string.

        Args:
            text: Text containing CONCAT macro calls

        Returns:
            Text with CONCAT macros expanded
        """
        pattern = r"CONCAT\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)"

        while True:
            match = re.search(pattern, text)
            if not match:
                break

            # Resolve macro values if available
            first = self.simple_macros.get(match.group(1), match.group(1))
            second = match.group(2)
            expanded = first + second

            text = text[: match.start()] + expanded + text[match.end() :]

        return text

    def expand_token_paste(self, text: str, args: Dict[str, str]) -> str:
        """
        Expand token pasting operators (##) with argument substitution.

        Args:
            text: Text containing ## operators
            args: Mapping of macro argument names to values

        Returns:
            Text with token pasting expanded
        """
        pattern = r"(\w+)?##(\w+)?"

        while True:
            match = re.search(pattern, text)
            if not match:
                break

            left = match.group(1) or ""
            right = match.group(2) or ""

            # Substitute macro arguments
            left_val = args.get(left, left)
            right_val = args.get(right, right)

            text = text[: match.start()] + left_val + right_val + text[match.end() :]

        return text


# =============================================================================
# Test Case Extraction
# =============================================================================


def parse_test_f_call(content: str, start_pos: int) -> Optional[Tuple[str, str]]:
    """
    Parse a TEST_F macro call and extract suite and case names.

    Args:
        content: Source content
        start_pos: Position right after "TEST_F("

    Returns:
        Tuple of (suite_name, case_name) or None if parsing fails
    """
    # Find matching closing parenthesis
    paren_count = 1
    i = start_pos

    while i < len(content) and paren_count > 0:
        if content[i] == "(":
            paren_count += 1
        elif content[i] == ")":
            paren_count -= 1
        i += 1

    if paren_count != 0:
        return None

    inner = content[start_pos : i - 1]

    # Split by comma at top level (not inside nested parentheses)
    paren_depth = 0
    split_pos = -1

    for j, ch in enumerate(inner):
        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth -= 1
        elif ch == "," and paren_depth == 0:
            split_pos = j
            break

    if split_pos < 0:
        return None

    suite = inner[:split_pos].strip()
    case = inner[split_pos + 1 :].strip()

    return suite, case


def extract_test_cases_from_main(main_cpp: Path) -> Set[str]:
    """
    Extract all test case names from a main.cpp file.

    Handles C preprocessor macros including CONCAT and token pasting.

    Args:
        main_cpp: Path to the main.cpp file

    Returns:
        Set of test case names in format "TestSuite.TestCaseName"
    """
    if not main_cpp.exists():
        return set()

    content = main_cpp.read_text()
    expander = MacroExpander(content)
    result: Set[str] = set()

    # Extract direct TEST_F calls
    test_f_pattern = r"TEST_F\s*\("

    for match in re.finditer(test_f_pattern, content):
        parsed = parse_test_f_call(content, match.end())
        if parsed is None:
            continue

        suite, case = parsed

        # Skip if case contains ## (it's inside a macro definition)
        if "##" in case:
            continue

        suite = expander.expand_concat(suite)
        result.add(f"{suite}.{case}")

    # Expand macro generators
    for macro_name, macro_info in expander.macro_generators.items():
        call_pattern = rf"\b{re.escape(macro_name)}\s*\(([^)]+)\)"

        for call_match in re.finditer(call_pattern, content):
            args = [a.strip() for a in call_match.group(1).split(",")]

            if len(args) != len(macro_info["args"]):
                continue

            arg_map = dict(zip(macro_info["args"], args))
            body = expander.expand_token_paste(macro_info["body"], arg_map)

            for test_match in re.finditer(test_f_pattern, body):
                parsed = parse_test_f_call(body, test_match.end())
                if parsed is None:
                    continue

                suite, case = parsed
                suite = expander.expand_concat(suite)
                result.add(f"{suite}.{case}")

    return result


# =============================================================================
# Script Validation
# =============================================================================


def extract_test_refs_from_script(script_path: str) -> List[Tuple[str, str, str]]:
    """
    Extract test references from a shell script.

    Args:
        script_path: Path to the shell script

    Returns:
        List of (version, testname, testcase) tuples
    """
    content = Path(script_path).read_text()

    # Pattern: -v <version> -t <testname> -g <testcase>
    pattern = r"-v\s+(\S+)\s+-t\s+(\S+)\s+-g\s+(\S+)"

    return re.findall(pattern, content)


def check_script(script_path: str) -> List[Dict]:
    """
    Validate all test case references in a script.

    Args:
        script_path: Path to the shell script

    Returns:
        List of issue dictionaries for invalid test cases
    """
    script = Path(script_path)
    if not script.exists():
        print(f"Script not found: {script_path}")
        return []

    test_refs = extract_test_refs_from_script(script_path)
    issues: List[Dict] = []
    checked: Set[Tuple[str, str, str]] = set()

    for version, testname, testcase in test_refs:
        # Skip duplicates
        key = (version, testname, testcase)
        if key in checked:
            continue
        checked.add(key)

        # Get test directory
        test_dir = get_test_dir(version, testname)
        if test_dir is None:
            continue

        # Check main.cpp exists
        main_cpp = test_dir / "main.cpp"
        if not main_cpp.exists():
            issues.append(
                {
                    "file": script_path,
                    "version": version,
                    "testname": testname,
                    "testcase": testcase,
                    "existing_cases": [],
                }
            )
            continue

        # Check test case exists
        existing_cases = extract_test_cases_from_main(main_cpp)
        if testcase not in existing_cases:
            issues.append(
                {
                    "file": script_path,
                    "version": version,
                    "testname": testname,
                    "testcase": testcase,
                    "existing_cases": sorted(existing_cases),
                }
            )

    return issues


# =============================================================================
# Output Formatting
# =============================================================================


def print_issues(issues: List[Dict]) -> None:
    """
    Print validation issues in a human-readable format.

    Args:
        issues: List of issue dictionaries
    """
    for i, issue in enumerate(issues, 1):
        print(f"\n[{i}] {Path(issue['file']).name} | {issue['version']} | {issue['testname']}")
        print(f"    Script test case:  {issue['testcase']}")

        existing = issue["existing_cases"]

        if not existing:
            print("    Available cases: (main.cpp not found)")
        elif len(existing) > 10:
            print(f"    Available cases ({len(existing)} total):")
            for case in existing[:5]:
                print(f"      - {case}")
            print(f"      ... ({len(existing) - 5} more)")
        else:
            print(f"    Available cases ({len(existing)} total):")
            for case in existing:
                print(f"      - {case}")


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> int:
    """
    Main entry point for the validation script.

    Returns:
        0 if all test cases are valid, 1 otherwise
    """
    # Ensure we're in project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)

    print("=" * 80)
    print("Checking test case names in scripts...")
    print("=" * 80)

    all_issues: List[Dict] = []

    for script in SCRIPTS_TO_CHECK:
        print(f"\nChecking {script}...")
        issues = check_script(script)
        all_issues.extend(issues)

    print("\n" + "=" * 80)
    print(f"Total issues found: {len(all_issues)}")
    print("=" * 80)

    if all_issues:
        print_issues(all_issues)
        return 1
    else:
        print("\nAll test cases are valid!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
