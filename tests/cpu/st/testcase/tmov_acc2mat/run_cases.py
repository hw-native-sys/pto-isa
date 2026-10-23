#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run each GTest in a separate process so CPU assertions do not stop the suite."""

import argparse
import json
import os
import signal
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path, help="Built CPU tmov_acc2mat GTest executable")
    parser.add_argument("--log-dir", type=Path, help="Default: <build>/logs/tmov_acc2mat")
    parser.add_argument("--timeout", type=float, default=30, help="Per-case timeout in seconds")
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"binary does not exist: {binary}")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if os.name == "posix":
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    log_dir = args.log_dir.resolve() if args.log_dir else binary.parent.parent / "logs/tmov_acc2mat"
    log_dir.mkdir(parents=True, exist_ok=True)
    listing = subprocess.run(
        [str(binary), "--gtest_list_tests"],
        cwd=binary.parent,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    suite = ""
    cases = []
    for line in listing.stdout.splitlines():
        line = line.split("#", 1)[0].rstrip()
        if line.startswith("  ") and suite == "TMOVTest.":
            cases.append(suite + line.strip())
        elif line.endswith(".") and not line.startswith(" "):
            suite = line
    if not cases:
        parser.error("no TMOVTest cases found in binary")
    results = []
    for case in cases:
        try:
            run = subprocess.run(
                [str(binary), f"--gtest_filter={case}", "--gtest_color=no"],
                cwd=binary.parent,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                check=False,
            )
            output = run.stdout
            if run.returncode == 0:
                status = "skipped" if "[  SKIPPED ]" in output else "passed"
            elif run.returncode == -signal.SIGABRT:
                status = "assertion" if "Assertion" in output else "aborted"
            else:
                status = "failed"
            returncode = run.returncode
        except subprocess.TimeoutExpired as error:
            output = error.stdout or ""
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
            status = "timeout"
            returncode = None
        (log_dir / f"{case}.log").write_text(output, encoding="utf-8")
        results.append({"case": case, "status": status, "returncode": returncode})
        print(f"{status.upper():9} {case} exit={returncode}", flush=True)
    statuses = sorted({result["status"] for result in results})
    summary = {status: sum(result["status"] == status for result in results) for status in statuses}
    (log_dir / "results.json").write_text(
        json.dumps({"binary": str(binary), "summary": summary, "results": results}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"SUMMARY total={len(results)} {summary}", flush=True)
    print(f"Logs: {log_dir}")
    return int(any(result["status"] not in ("passed", "skipped") for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
