# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise CostModel batch discovery without building the actual test suites."""

import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "run_costmodel_tests.sh"
BASH = shutil.which("bash")
TOOLS_AVAILABLE = all(shutil.which(tool) for tool in ("bash", "find", "sort", "basename"))


@unittest.skipUnless(TOOLS_AVAILABLE, "Bash and POSIX find/sort/basename are required")
class TestCostmodelPerfDiscovery(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pto-costmodel-discovery-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tests = self.root / "tests"
        for suite in ("st", "st_fit", "st_a5_fit"):
            (self.tests / "costmodel" / suite / "testcase").mkdir(parents=True)
        self.script = self.tests / SCRIPT.name
        self.write_text(self.script, SCRIPT.read_text(encoding="utf-8"))
        self.perf_dir = self.tests / "costmodel" / "perf_sim_st" / "build" / "bin"
        self.perf_dir.mkdir(parents=True)
        self.write_text(self.perf_dir / "README.txt", "not an executable\n")
        tools = self.root / "tools"
        tools.mkdir()
        self.env = os.environ.copy()
        self.env.update(
            PATH=str(tools) + os.pathsep + self.env.get("PATH", ""),
            PTO_PERF_RUN_LOG=(self.root / "perf-runs.txt").as_posix(),
            PTO_ST_RUN_LOG=(self.root / "st-runs.txt").as_posix(),
        )
        # Ordinary suites are stand-ins: only perf discovery and runner exit handling are under test.
        python = tools / "python3"
        self.write_text(python, '#!/bin/sh\nprintf "%s\\n" "$*" >> "$PTO_ST_RUN_LOG"\nexit 0\n')
        python.chmod(0o755)

    @staticmethod
    def write_text(path, content):
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)

    def add_perf(self, name="sample_perf", exit_code=0, mode=0o755):
        path = self.perf_dir / name
        self.write_text(path, f'#!/bin/sh\nprintf "{name}\\n" >> "$PTO_PERF_RUN_LOG"\nexit {exit_code}\n')
        path.chmod(mode)
        return path

    def add_ordinary_suite(self):
        (self.tests / "costmodel" / "st" / "testcase" / "sample_st").mkdir()

    def run_cli(self):
        return subprocess.run(
            [BASH, str(self.script)],
            cwd=self.root,
            env=self.env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )

    def assert_summary(self, result, total, passed, failed):
        self.assertEqual(result.returncode, 1 if failed else 0, result.stdout + result.stderr)
        self.assertRegex(result.stdout, rf"Total:\s+{total}\b")
        self.assertRegex(result.stdout, rf"Passed:\s+{passed}\b")
        self.assertRegex(result.stdout, rf"Failed:\s+{failed}\b")
        self.assertEqual(result.stderr, "")
        self.assertEqual((self.root / "perf-runs.txt").read_text().splitlines(), ["sample_perf"])

    def test_perf_only_and_non_executable_file(self):
        self.add_perf()
        self.assert_summary(self.run_cli(), total=1, passed=1, failed=0)
        self.assertFalse((self.root / "st-runs.txt").exists())

    def test_mixed_suites_include_perf_binary(self):
        self.add_ordinary_suite()
        self.add_perf()
        self.assert_summary(self.run_cli(), total=2, passed=2, failed=0)
        self.assertEqual(len((self.root / "st-runs.txt").read_text().splitlines()), 1)

    def test_failed_perf_only_changes_exit_status(self):
        self.add_perf(exit_code=7)
        self.assert_summary(self.run_cli(), total=1, passed=0, failed=1)

    def test_failed_perf_in_mixed_suites_changes_exit_status(self):
        self.add_ordinary_suite()
        self.add_perf(exit_code=7)
        self.assert_summary(self.run_cli(), total=2, passed=1, failed=1)

    def test_any_executable_permission_bit_is_discovered(self):
        for bits in range(8):
            executable_bits = (0o100 if bits & 4 else 0) | (0o010 if bits & 2 else 0) | (0o001 if bits & 1 else 0)
            mode = 0o600 | executable_bits
            path = self.add_perf(name=f"mode_{bits}", mode=mode)
            if stat.S_IMODE(path.stat().st_mode) != mode:
                self.skipTest("The filesystem does not preserve POSIX permission bits")
        result = self.run_cli()
        # Files with only group/other execute bits may fail to run as their owner; they must still be discovered.
        found = set(re.findall(r"Start Test Case:perf_sim_st:(mode_[0-7])", result.stdout))
        self.assertEqual(found, {f"mode_{bits}" for bits in range(1, 8)}, result.stdout + result.stderr)
        self.assertRegex(result.stdout, r"Total:\s+7\b")


if __name__ == "__main__":
    unittest.main()
