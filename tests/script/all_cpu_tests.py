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

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from cpu_bfloat16 import detect_bfloat16_cxx, derive_cc_from_cxx


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and run all CPU-SIM STs.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print configure/build and passing test output.")
    parser.add_argument(
        "-b",
        "--build-folder",
        default="build/cpu_st_all",
        help="Build folder to use. Relative paths are resolved from the repo root.",
    )
    parser.add_argument(
        "-c",
        "--compiler",
        required=False,
        help="Optional C++ compiler path or name. When omitted, the current CXX environment or default compiler is used.",
    )
    parser.add_argument(
        "--enable-bf16",
        action="store_true",
        help="Enable BF16 CPU-SIM coverage. This switches to a compiler that supports std::bfloat16_t and C++23.",
    )
    parser.add_argument("-g", "--generator", required=False, help="Optional CMake generator, for example Ninja.")
    parser.add_argument("-j", "--jobs", type=int, default=max(1, os.cpu_count() or 1), help="Parallel build jobs.")
    parser.add_argument("--timeout", type=int, default=30, help="Per-test timeout in seconds.")
    return parser.parse_args()


def color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def green(text: str) -> str:
    return color(text, "32")


def red(text: str) -> str:
    return color(text, "31")


def run_command(
    cmd: list[str], *, cwd: Path, env: dict[str, str], verbose: bool, check: bool = True
) -> subprocess.CompletedProcess[str]:
    if verbose:
        print(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if verbose or proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)
    return proc


def build_all_cpu_tests(repo_root: Path, build_dir: Path, args: argparse.Namespace) -> None:
    tests_path = repo_root / "tests" / "cpu" / "st"
    env = os.environ.copy()
    cc = None
    if args.enable_bf16:
        cxx = detect_bfloat16_cxx(args.compiler)
        cc = derive_cc_from_cxx(cxx)
        env["CXX"] = cxx
        if cc:
            env["CC"] = cc
    elif args.compiler:
        env["CXX"] = args.compiler

    configure_cmd = [
        "cmake",
        "-S",
        str(tests_path),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DPTO_CPU_SIM_ENABLE_BF16={'ON' if args.enable_bf16 else 'OFF'}",
    ]
    if cc:
        configure_cmd.append(f"-DCMAKE_C_COMPILER={cc}")
        configure_cmd.append(f"-DCMAKE_CXX_COMPILER={env['CXX']}")
    if args.generator:
        configure_cmd.extend(["-G", args.generator])
    run_command(configure_cmd, cwd=repo_root, env=env, verbose=args.verbose)

    build_cmd = ["cmake", "--build", str(build_dir), "-j", str(args.jobs)]
    run_command(build_cmd, cwd=repo_root, env=env, verbose=args.verbose)


def generate_test_data(repo_root: Path, build_dir: Path, args: argparse.Namespace) -> None:
    testcase_src_root = repo_root / "tests" / "cpu" / "st" / "testcase"
    gen_env = os.environ.copy()
    gen_env["PYTHONPATH"] = str(repo_root) + os.pathsep + gen_env.get("PYTHONPATH", "")
    if args.enable_bf16:
        gen_env["PTO_CPU_SIM_ENABLE_BF16"] = "1"
    copied_scripts: list[Path] = []
    try:
        for script in sorted(testcase_src_root.glob("*/gen_data.py")):
            dst = build_dir / f"{script.parent.name}_gen_data.py"
            copied_scripts.append(dst)
            dst.write_text(script.read_text(encoding="utf-8"), encoding="utf-8")
            run_command([sys.executable, str(dst.name)], cwd=build_dir, env=gen_env, verbose=args.verbose)
    finally:
        for script_path in copied_scripts:
            script_path.unlink(missing_ok=True)


def run_binaries(repo_root: Path, build_dir: Path, args: argparse.Namespace) -> int:
    bin_dir = build_dir / "bin"
    binaries = sorted(path for path in bin_dir.iterdir() if path.is_file())
    total = 0
    failed = 0

    for binary in binaries:
        cwd = binary.parent
        if os.name == "nt" and binary.parent.name.lower() == "release":
            cwd = binary.parent.parent

        total += 1
        start = time.time()
        try:
            proc = subprocess.run(
                [str(binary)],
                cwd=cwd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
            )
            duration = time.time() - start
            passed = proc.returncode == 0
            status = green("PASS") if passed else red("FAIL")
            print(f"{status} {binary.name} rc={proc.returncode} dur={duration:.2f}s")
            if args.verbose or not passed:
                if proc.stdout:
                    print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
            if not passed:
                failed += 1
        except subprocess.TimeoutExpired as exc:
            duration = time.time() - start
            print(red(f"FAIL {binary.name} rc=124 dur={duration:.2f}s"))
            captured = exc.stdout if isinstance(exc.stdout, str) else ""
            if captured:
                print(captured, end="" if captured.endswith("\n") else "\n")
            print("[TIMEOUT]")
            failed += 1

    summary = f"SUMMARY total={total} pass={total - failed} fail={failed}"
    print(green(summary) if failed == 0 else red(summary))
    return 0 if failed == 0 else 1


def main() -> int:
    args = parse_arguments()
    repo_root = Path(__file__).resolve().parents[2]
    build_dir = Path(args.build_folder)
    if not build_dir.is_absolute():
        build_dir = repo_root / build_dir
    build_dir.mkdir(parents=True, exist_ok=True)

    build_all_cpu_tests(repo_root, build_dir, args)
    generate_test_data(repo_root, build_dir, args)
    return run_binaries(repo_root, build_dir, args)


if __name__ == "__main__":
    sys.exit(main())
