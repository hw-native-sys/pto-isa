# --------------------------------------------------------------------------------
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import argparse
import os
import subprocess
import sys
import time
import multiprocessing
from pathlib import Path
from functools import partial

from cpu_bfloat16 import detect_bfloat16_cxx, derive_cc_from_cxx


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and run all CPU-SIM STs.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print configure/build and passing test output.")
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
    parser.add_argument(
        "--trace-mode",
        action="store_true",
        help="Enable PTO instruction tracing in CPU-SIM via PTO_CPU_SIM_TRACE_MODE.",
    )
    parser.add_argument("-g", "--generator", required=False,
                        help="Optional CMake generator, for example Ninja.")
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Parallel build jobs.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Per-test timeout in seconds.",
    )
    parser.add_argument(
        "--build-folder",
        required=False,
        help=(
            "Optional build root used to isolate generated build artifacts. "
            "When set, each CPU test suite builds under this directory using "
            "its default leaf name."
        ),
    )
    return parser.parse_args()


def color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def green(text: str) -> str:
    return color(text, "32")


def red(text: str) -> str:
    return color(text, "31")


g_lock = multiprocessing.Lock()


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    verbose: bool,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, cwd=cwd, env=env,
                          text=True, stdout=subprocess.PIPE)
    with g_lock:
        if verbose:
            print(f"$ {' '.join(cmd)}")
        if verbose or proc.returncode != 0:
            if proc.stdout:
                print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, output=proc.stdout)
    return proc


def build_single_test(repo_root: Path, tests_path: Path, build_dir: Path, args: argparse.Namespace) -> None:
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
        f"-DPTO_CPU_SIM_TRACE_MODE={'ON' if args.trace_mode else 'OFF'}",
    ]
    if cc:
        configure_cmd.append(f"-DCMAKE_C_COMPILER={cc}")
        configure_cmd.append(f"-DCMAKE_CXX_COMPILER={env['CXX']}")
    if args.generator:
        configure_cmd.extend(["-G", args.generator])
    run_command(configure_cmd, cwd=repo_root, env=env, verbose=args.verbose)

    build_cmd = ["cmake", "--build", str(build_dir), "-j", str(args.jobs)]
    run_command(build_cmd, cwd=repo_root, env=env, verbose=args.verbose)


TEST_SOURCES = [
    ("tests/cpu/st", "build/cpu_st"),
    ("tests/cpu/comm/st", "build/cpu_st_comm"),
]


def resolve_build_dir(repo_root: Path, build_rel: str, args: argparse.Namespace) -> Path:
    if not args.build_folder:
        return repo_root / build_rel

    base = Path(args.build_folder)
    if not base.is_absolute():
        base = repo_root / base
    return base / Path(build_rel).name


def build_all_cpu_tests(repo_root: Path, args: argparse.Namespace) -> None:
    for src_rel, build_rel in TEST_SOURCES:
        tests_path = repo_root / src_rel
        this_build_dir = resolve_build_dir(repo_root, build_rel, args)
        if not tests_path.exists():
            print(f"Skipping non-existent source: {tests_path}")
            continue
        print(f"Building {src_rel} -> {build_rel}")
        build_single_test(repo_root, tests_path, this_build_dir, args)


def generate_test_data(repo_root: Path, args: argparse.Namespace) -> None:
    gen_env = os.environ.copy()
    if args.enable_bf16:
        gen_env["PTO_CPU_SIM_ENABLE_BF16"] = "1"

    for src_rel, build_rel in TEST_SOURCES:
        testcase_src_root = repo_root / src_rel / "testcase"
        if not testcase_src_root.exists():
            continue
        testcase_build_root = resolve_build_dir(repo_root, build_rel, args) / "testcase"
        testcase_build_root.mkdir(parents=True, exist_ok=True)

        env = gen_env.copy()
        env["PYTHONPATH"] = str(repo_root / src_rel) + os.pathsep + str(repo_root) + \
            os.pathsep + env.get("PYTHONPATH", "")

        with multiprocessing.Pool(processes=args.jobs) as pool:
            run_args = [[sys.executable, str(script)] for script in sorted(testcase_src_root.glob("*/gen_data.py"))]
            results = pool.map(partial(run_command, cwd=testcase_build_root,
                                env=env, verbose=args.verbose), run_args)


def run_binaries(repo_root: Path, args: argparse.Namespace) -> int:
    total = 0
    failed = 0

    for src_rel, build_rel in TEST_SOURCES:
        name = src_rel.split("/")[-2].upper()
        print("=" * 60 + f" {name} " + "=" * 60)
        build_dir = resolve_build_dir(repo_root, build_rel, args)
        bin_dir = build_dir / "bin"
        if not bin_dir.exists():
            continue
        testcase_build_root = build_dir / "testcase"
        binaries = sorted(path for path in bin_dir.iterdir() if path.is_file())

        for binary in binaries:
            cwd = testcase_build_root / binary.name
            if not cwd.is_dir():
                cwd = repo_root

            total += 1
            start = time.time()
            try:
                proc = subprocess.run(
                    [str(binary)],
                    cwd=cwd,
                    text=True,
                    stdout=subprocess.PIPE,
                    timeout=args.timeout,
                )
                duration = time.time() - start
                passed = proc.returncode == 0
                status = green("PASS:") if passed else red("FAIL:")
                print(
                    f"{status} {binary.name:<10} (RC={proc.returncode:<3} Duration={duration:.2f}s)")
                if args.verbose or not passed:
                    if proc.stdout:
                        print(proc.stdout, end="\n" if proc.stdout.endswith(
                            "\n") else "\n\n")
                if not passed:
                    failed += 1
            except subprocess.TimeoutExpired as exc:
                duration = time.time() - start
                print(red("FAIL:") +
                      f" {binary.name:<10} (RC=124 Duration={duration:.2f}s)")
                captured = exc.stdout if isinstance(exc.stdout, str) else ""
                if captured:
                    print(captured, end="" if captured.endswith("\n") else "\n")
                print("[TIMEOUT]")
                failed += 1

    summary = f"SUMMARY: TOTAL:{total} PASSED:{total - failed} FAILED:{failed}"
    print(green(summary) if failed == 0 else red(summary))
    return 0 if failed == 0 else 1


def main() -> int:
    args = parse_arguments()
    repo_root = Path(__file__).resolve().parents[2]

    for src_rel, build_rel in TEST_SOURCES:
        build_dir = repo_root / build_rel
        build_dir.mkdir(parents=True, exist_ok=True)

    build_all_cpu_tests(repo_root, args)
    generate_test_data(repo_root, args)
    return run_binaries(repo_root, args)


if __name__ == "__main__":
    sys.exit(main())
